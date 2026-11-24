"""Step 0 feasibility spike (throwaway): the bundled MLX server with our capture installed.

Run:  PYTHONPATH=../../spike ../../spike/.venv/bin/python launch.py --model thinkingmachines/Inkling-2-Mini --port 8766

For every request it appends one line to step0_log.jsonl (git-ignored) with the sizes
of the forward calls, and an alignment check of the captured states against a plain
pass over the same tokens with no cache.
"""

import copy
import json
import os
import time

import mlx.core as mx
import numpy as np
import mlx_lm.server as S
from latent_capture.capture import capturing

LOG = os.path.join(os.path.dirname(__file__), "step0_log.jsonl")
VERIFY_MAX_TOKENS = 1500  # the plain pass is skipped for longer texts
state = {"cap": None, "prompt": None}


def parse_tool_call(text, tools=None):
    call = json.loads(text.strip())
    return {"name": call["name"], "arguments": call.get("args", {})}


class Provider(S.ModelProvider):
    def _load(self, *a, **k):
        super()._load(*a, **k)
        self.is_batchable = False  # capture reads one request at a time
        state["cap"] = capturing(self.model, max_records=20000)
        state["cap"].__enter__()
        t = self.tokenizer
        # Inkling's reply format: <|message_model|>[tool name]<|content_KIND|>...<|end_message|>
        t._response_format = {
            "initial_state": "header",
            "transitions": {
                "header": [
                    ("<|content_thinking|>", "reasoning"),
                    ("<|content_text|>", "normal"),
                    ("<|content_invoke_tool_json|>", "tool"),
                ],
                "reasoning": [("<|end_message|>", "header")],
                "normal": [("<|end_message|>", "header")],
            },
        }
        t._tool_parser = parse_tool_call
        t._tool_call_start = "<|content_invoke_tool_json|>"
        t._tool_call_end = "<|end_message|>"
        t._tool_call_start_tokens = tuple(t.encode(t._tool_call_start, add_special_tokens=False))
        print(
            "STEP0 loaded:",
            json.dumps(
                {
                    "has_thinking": t.has_thinking,
                    "think_start": t.think_start,
                    "has_tool_calling": t.has_tool_calling,
                    "tool_parser": getattr(t.tool_parser, "__module__", None),
                }
            ),
            flush=True,
        )


_tokenize = S.ResponseGenerator._tokenize


def tokenize(self, tokenizer, request, args):
    out = _tokenize(self, tokenizer, request, args)
    state["prompt"] = list(out[0])
    state["rg"] = self
    return out


S.ResponseGenerator._tokenize = tokenize


def cos(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def verify(model, full, n_prompt, states):
    """states[i] is the captured block 2 pass 3 state assigned to generated token i.
    Compare with a plain pass at the same position and at the neighbouring ones."""
    cap = state["cap"]
    cap.__exit__()
    try:
        with capturing(model, all_positions=True) as ref:
            mx.eval(model(mx.array([full])))
            b2 = ref.records[-1].block2[-1]
    finally:
        cap.__enter__()
    own, before, after = [], [], []
    for i, s in enumerate(states):
        p = n_prompt + i
        own.append(cos(s, b2[p]))
        before.append(cos(s, b2[p - 1]))
        if p + 1 < len(b2):
            after.append(cos(s, b2[p + 1]))
    best = sum(o > b and (j >= len(after) or o > after[j]) for j, (o, b) in enumerate(zip(own, before)))
    return {
        "own_mean": float(np.mean(own)),
        "own_min": float(np.min(own)),
        "before_mean": float(np.mean(before)),
        "after_mean": float(np.mean(after)) if after else None,
        "own_is_best": int(best),
        "of": len(own),
    }


_stream = S.stream_generate


def stream_generate(model, tokenizer, prompt, **kw):
    cap = state["cap"]
    cap.clear()
    rest = list(prompt)
    toks, t0, first = [], time.time(), None
    # This model's cache cannot be trimmed, so the server's own cache entry (prompt
    # plus reply) is never reusable. Save a copy when the prompt has been read, before
    # the reply starts: the next turn's prompt extends it.
    progress = kw.get("prompt_progress_callback")
    full_prompt, rg, cache = state["prompt"], state.get("rg"), kw.get("prompt_cache")

    def checkpoint(done, total):
        if progress:
            progress(done, total)
        if total > 1 and done == total - 1 and rg is not None and cache is not None:
            rg.prompt_cache.insert_cache(
                rg.model_provider.model_key, full_prompt[:-1], copy.deepcopy(cache), cache_type="user"
            )

    kw["prompt_progress_callback"] = checkpoint
    try:
        for gen in _stream(model=model, tokenizer=tokenizer, prompt=prompt, **kw):
            first = first or time.time()
            toks.append(int(gen.token))
            yield gen
    finally:
        recs = cap.records
        sizes = [r.n_tokens for r in recs]
        # Rule under test: forward calls first consume the uncached prompt tokens;
        # every call after that consumed one generated token, in order.
        used, k = 0, 0
        while k < len(recs) and used < len(rest):
            used += recs[k].n_tokens
            k += 1
        gen_recs = recs[k:]
        line = {
            "prompt_tokens": len(state["prompt"] or []),
            "uncached": len(rest),
            "generated": len(toks),
            "call_sizes_head": sizes[:6],
            "calls": len(sizes),
            "prompt_calls": k,
            "prompt_consumed": used,
            "gen_calls": len(gen_recs),
            "all_gen_calls_single": all(r.n_tokens == 1 for r in gen_recs),
            "first_token_s": round((first or time.time()) - t0, 2),
            "tps": round(len(toks) / max(time.time() - (first or t0), 1e-6), 1),
        }
        full = (state["prompt"] or []) + toks
        if gen_recs and state["prompt"] and len(full) <= VERIFY_MAX_TOKENS:
            try:
                n = min(len(gen_recs), len(toks))
                line["align"] = verify(model, full, len(state["prompt"]), [r.block2[-1] for r in gen_recs[:n]])
            except Exception as e:  # spike: report, do not hide
                line["align_error"] = repr(e)
        with open(LOG, "a") as f:
            f.write(json.dumps(line) + "\n")
        print("STEP0", json.dumps(line), flush=True)


S.stream_generate = stream_generate
S.ModelProvider = Provider

if __name__ == "__main__":
    S.main()
