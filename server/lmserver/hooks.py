"""Everything attached to the bundled MLX server (mlx_lm.server), from outside.

No file of the model runner is edited. At load the model gets our capture and the
tokenizer learns Inkling's reply format. Around each generation we clear the capture,
save a cache checkpoint once the prompt has been read, and hand the result to the recorder."""

import copy
from collections import deque
import logging
import threading
import time
from typing import Callable

import numpy as np

from lmserver import inkling
from lmserver.recorder import RequestRecord

PHASES = ("thinking", "answer")
log = logging.getLogger("lmserver")
_state: dict = {"installed": False, "busy": False, "model": None}
_lock = threading.Lock()  # held while the model generates, so nothing else computes with it


KIND_OF_MARKER = {
    "<|content_thinking|>": "thinking",
    "<|content_text|>": "answer",
    inkling.TOOL_CALL_START: "tool",
    "<|message_model|>": "header",  # what follows, up to the next marker, is a tool's name
}


class WordBuilder:
    """Pairs generated tokens with the forward calls that read them, as they arrive.

    The runner first reads the prompt tokens the cache did not already hold (`uncached`
    of them, in calls of any size); every call after that read exactly one generated
    word, in order (lmserver.align states the same rule). Records are used and let go
    at once, so a long reply does not pile up states in memory."""

    def __init__(self, tokenizer, live, uncached: int):
        self._tok, self._live, self._uncached = tokenizer, live, uncached
        self._kind = "header"
        self.texts: list[str] = []
        self.ids: list[int] = []
        self.words: list[dict] = []
        self._kinds: list[str] = []
        self._used = 0  # prompt tokens accounted for so far
        self._queue: deque = deque()  # records that read a generated word not yet added
        self._sum: np.ndarray | None = None
        self._n = np.zeros(len(PHASES), dtype=np.int64)

    def add_token(self, token: int) -> None:
        text = self._tok.decode([token])
        marker = text.startswith("<|") and text.endswith("|>")
        if marker:
            self._kind = KIND_OF_MARKER.get(text, "header")
        self.texts.append(text)
        self.ids.append(token)
        self._kinds.append("marker" if marker else self._kind)

    def held(self) -> int:
        return len(self._queue)

    def update(self, new_records: list) -> list[dict]:
        """Take the forward calls captured since the last update. Returns the new words."""
        for rec in new_records:
            if self._used < self._uncached:
                self._used += rec.n_tokens  # still the prompt
            else:
                self._queue.append(rec)
        new = []
        while self._queue and len(self.words) < len(self.texts):
            i = len(self.words)
            rec = self._queue.popleft()
            b2, b3 = np.stack(rec.block2), np.stack(rec.block3)
            word = {"i": i, "phase": self._kinds[i], **self._live.describe(b2, b3)}
            if self._kinds[i] in PHASES:
                if self._sum is None:
                    self._sum = np.zeros((len(PHASES), 2, *b2.shape), dtype=np.float64)
                p = PHASES.index(self._kinds[i])
                self._sum[p, 0] += b2
                self._sum[p, 1] += b3
                self._n[p] += 1
            self.words.append(word)
            new.append(word)
        return new

    def finish(self) -> list[dict]:
        """Tokens the model never read back (usually the last one) are kept without a state."""
        new = []
        for i in range(len(self.words), len(self.texts)):
            word = {"i": i, "phase": self._kinds[i], "cells": None}
            self.words.append(word)
            new.append(word)
        self._queue.clear()
        return new

    def mean_states(self) -> np.ndarray | None:
        if self._sum is None:
            return None
        with np.errstate(invalid="ignore", divide="ignore"):
            mean = self._sum / self._n[:, None, None, None]
        return mean.astype(np.float32)


def _record(meta, prompt, uncached, builder, timing, finish) -> RequestRecord:
    return RequestRecord(
        completion_id=meta.get("completion_id") or f"local-{timing['time']:.3f}",
        time=timing["time"],
        client=meta.get("client", ""),
        messages=meta.get("messages"),
        reply_tokens=list(builder.texts),
        words=list(builder.words),
        mean_states=builder.mean_states(),
        settings=meta.get("settings", {}),
        prompt_tokens=len(prompt),
        cached_tokens=len(prompt) - uncached,
        first_token_s=timing["first_token_s"],
        tokens_per_s=timing["tokens_per_s"],
        finish=finish,
    )


def build_record(meta, prompt, uncached, tokens, records, tokenizer, live, timing, finish) -> RequestRecord:
    """The whole record from finished lists. Pure: needs no model."""
    builder = WordBuilder(tokenizer, live, uncached)
    for token in tokens:
        builder.add_token(token)
    builder.update(records)
    builder.finish()
    return _record(meta, prompt, uncached, builder, timing, finish)


def pin_model(handler) -> None:
    """One model, loaded once. A request naming another model, an adapter or a draft
    model would make the bundled server unload Inkling or load something beside it."""
    handler.requested_model = "default_model"
    handler.adapter = None
    handler.requested_draft_model = "default_model"


def refuse_remote_images(messages) -> None:
    """The bundled server downloads http(s) image URLs before reading a message.
    Nothing is fetched from the internet here; images must be sent inline."""
    for message in messages or []:
        content = message.get("content") if isinstance(message, dict) else None
        for part in content if isinstance(content, list) else []:
            if not isinstance(part, dict) or part.get("type") != "image_url":
                continue
            url = part.get("image_url")
            url = url.get("url") if isinstance(url, dict) else url
            if isinstance(url, str) and url.strip().lower().startswith(("http://", "https://")):
                raise ValueError("This server does not fetch images from the internet; send the image inline as a data: URL.")


def busy() -> bool:
    return _state["busy"]


def model():
    return _state["model"]


def model_lock() -> threading.Lock:
    return _lock


def install(recorder, live, on_word: Callable[[dict], None] | None = None) -> None:
    import mlx_lm.server as S
    from latent_capture.capture import capturing

    if _state["installed"]:
        raise RuntimeError("lmserver hooks are already installed")
    _state["installed"] = True
    emit = on_word or (lambda event: None)

    class Provider(S.ModelProvider):
        def _load(self, *args, **kwargs):
            super()._load(*args, **kwargs)
            self.is_batchable = False  # capture reads one request at a time
            _state["cap"] = capturing(self.model, max_records=20000)
            _state["cap"].__enter__()
            inkling.configure(self.tokenizer)
            _state["model"] = self.model

    original_tokenize = S.ResponseGenerator._tokenize

    def tokenize(self, tokenizer, request, args):
        out = original_tokenize(self, tokenizer, request, args)
        _state.update(prompt=list(out[0]), generator=self, meta=getattr(request, "lm_meta", {}), stopped=False)
        return out

    original_stop = S.GenerationContext.stop

    def stop(self):
        _state["stopped"] = True  # the client went away
        return original_stop(self)

    original_stream = S.stream_generate

    def stream_generate(model, tokenizer, prompt, **kw):
        cap, full, generator, meta = _state["cap"], _state.get("prompt") or [], _state.get("generator"), _state.get("meta") or {}
        cap.clear()
        uncached = len(prompt)
        builder = WordBuilder(tokenizer, live, uncached)
        cid = meta.get("completion_id")
        progress, cache = kw.get("prompt_progress_callback"), kw.get("prompt_cache")

        # This model's cache cannot be shortened, so the entry the server stores after
        # the reply is never a prefix of a later prompt. Keep a copy from the moment the
        # prompt has been read: the next turn of the conversation extends it.
        def checkpoint(done, total):
            if progress:
                progress(done, total)
            if total > 1 and done == total - 1 and generator is not None and cache is not None and len(full) > 1:
                generator.prompt_cache.insert_cache(
                    generator.model_provider.model_key, full[:-1], copy.deepcopy(cache), cache_type="user"
                )

        kw["prompt_progress_callback"] = checkpoint
        started, first, reason, failed = time.time(), None, None, False
        _state["busy"] = True
        _lock.acquire()

        def take():
            records = cap.records  # only this thread adds to it
            cap.clear()
            return records

        def send(words, done=False):
            try:
                for word in words:
                    emit({"id": cid, "text": builder.texts[word["i"]], **word})
                if done:
                    emit({"id": cid, "done": True})
            except Exception as err:  # a listener must never break a reply
                log.warning("live listener failed: %s", type(err).__name__)

        try:
            emit({"id": cid, "start": True})
            for gen in original_stream(model=model, tokenizer=tokenizer, prompt=prompt, **kw):
                first = first or time.time()
                builder.add_token(int(gen.token))
                reason = gen.finish_reason or reason
                try:
                    send(builder.update(take()))
                except Exception as err:
                    log.warning("describing a word failed: %s", type(err).__name__)
                yield gen
        except GeneratorExit:
            raise
        except BaseException:
            failed = True
            raise
        finally:
            _lock.release()
            _state["busy"] = False
            _state["last"] = (list(full), list(builder.ids))  # token ids of the last request, for the alignment test
            try:
                last_words = builder.update(take()) + builder.finish()
                n, now = len(builder.texts), time.time()
                if failed:
                    finish = "error"
                elif reason is None and _state.get("stopped"):
                    finish = "cancelled"
                elif reason != "length" and inkling.TOOL_CALL_START in builder.texts:
                    finish = "tool_calls"
                else:
                    finish = reason or "stop"
                timing = {
                    "time": started,
                    "first_token_s": (first or now) - started,
                    "tokens_per_s": n / max(now - (first or now), 1e-6) if n > 1 else 0.0,
                }
                recorder.save(_record(meta, full, uncached, builder, timing, finish))
            except Exception as err:  # recording must never break a reply; no text in the log
                log.warning("recording a request failed: %s", type(err).__name__)
            send(last_words if "last_words" in locals() else [], done=True)  # after the save: the page fetches the record on "done"

    S.ModelProvider = Provider
    S.ResponseGenerator._tokenize = tokenize
    S.GenerationContext.stop = stop
    S.stream_generate = stream_generate


def make_handler():
    """The bundled request handler, tagged so the recorder knows who asked."""
    import mlx_lm.server as S

    class Handler(S.APIHandler):
        def log_message(self, *args):  # keep requests out of the terminal log
            pass

        def _meta(self, request, messages):
            body = self.body
            request.lm_meta = {
                "completion_id": self.request_id,
                "client": self.headers.get("User-Agent", ""),
                "messages": messages,
                "settings": {
                    "max_tokens": self.max_tokens,
                    "temperature": self.temperature,
                    "reasoning_effort": (self.chat_template_kwargs or {}).get("reasoning_effort"),
                    "tools": [
                        (t.get("function") or t).get("name") for t in body.get("tools") or [] if isinstance(t, dict)
                    ],
                    "stream": bool(self.stream),
                },
            }
            return request

        def handle_chat_completions(self):
            pin_model(self)
            refuse_remote_images(self.body.get("messages"))
            # The standard field is `reasoning_effort`; the model's template reads it as a template argument.
            effort = self.body.get("reasoning_effort")
            if effort is not None and "reasoning_effort" not in (self.chat_template_kwargs or {}):
                self.chat_template_kwargs = {**(self.chat_template_kwargs or {}), "reasoning_effort": effort}
            request = super().handle_chat_completions()
            return self._meta(request, self.body.get("messages"))

        def handle_text_completions(self):
            pin_model(self)
            return self._meta(super().handle_text_completions(), None)

    return Handler
