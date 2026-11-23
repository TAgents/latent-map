"""Throwaway prototype server: one page, one model, one generation at a time.

Run from prototype/:
    PYTHONPATH=../spike:../basemap ../spike/.venv/bin/python server.py
then open http://127.0.0.1:8765

Nothing leaves this machine: the server listens on 127.0.0.1 only.
"""

import json
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx.utils import tree_flatten
from mlx_lm import stream_generate

from latent_capture.capture import capturing
from latent_capture.loader import load_model
from livemap.live import LiveMap, PhaseTracker

HERE = Path(__file__).parent
BUNDLE = HERE.parent / "basemap" / "bundle_v0"
PORT = 8765
MAX_TOKENS_LIMIT = 800
HEARTBEAT_SECONDS = 10  # read every weight this often while idle, to keep the model in memory; 0 turns it off
HEARTBEAT_MAX_SECONDS = 120

MODEL = TOKENIZER = LIVE = None


def map_payload() -> dict:
    b = np.load(BUNDLE / "block2.npz")
    info = json.loads((BUNDLE / "block2.json").read_text())
    xy = b["xy"].astype(float)
    lo, span = xy.min(axis=0), np.ptp(xy, axis=0)
    counts = np.bincount(b["cell"][:, -1], minlength=len(xy))
    areas = []
    for a in sorted(info["broad_of_fine"], key=int):
        areas.append({
            "id": int(a),
            "name": info.get("names", {}).get(a, ""),
            "score": info.get("name_scores", {}).get(a, 0.0),
            "words": info["words"].get(a, [])[:10],
            "answer_share": info["area_mix"][a]["answer_share"],
            "top_form": (info["area_mix"][a]["top_form"] or [None])[0],
            "points": info["area_points"][a],
        })
    return {
        "cells": [{"x": float((p[0] - lo[0]) / span[0]), "y": float((p[1] - lo[1]) / span[1]), "area": int(f), "n": int(n)}
                  for p, f, n in zip(xy, b["fine"], counts)],
        "areas": areas,
        "norms": LIVE.norms,
    }


def generate_events(prompt: str, effort: float, max_tokens: int):
    """Yields one dict per event: a 'token' when the model emits a word, a 'state'
    when the forward call that consumed that word has been captured, then 'done'."""
    head = TOKENIZER.apply_chat_template(
        [{"role": "user", "content": prompt}], add_generation_prompt=True, tokenize=False, reasoning_effort=effort
    )
    phases = PhaseTracker()
    started, emitted, described, seen = time.time(), 0, 0, 0
    first_token_at = last = None
    # mlx-lm (pinned commit) feeds the last prompt token in a call of its own, so the
    # first one-token record is the prompt's, not a generated word's. Skip it.
    skipped_prompt_tail = False
    with capturing(MODEL, max_records=4096) as cap:

        def drain():
            nonlocal described, seen, skipped_prompt_tail
            records = cap.records
            new, seen = records[seen:], len(records)
            for rec in new:
                if rec.n_tokens != 1:
                    continue  # a prompt chunk, not a generated word
                if not skipped_prompt_tail:
                    skipped_prompt_tail = True
                    continue
                state = LIVE.describe(np.stack(rec.block2), np.stack(rec.block3))
                yield {"type": "state", "i": described, **state}
                described += 1

        for r in stream_generate(MODEL, TOKENIZER, head, max_tokens=max_tokens):
            if first_token_at is None:
                first_token_at = time.time() - started
            last = r
            text = TOKENIZER.decode([r.token])
            phase, marker = phases.step(text)
            yield {"type": "token", "i": emitted, "text": text, "phase": phase, "marker": marker}
            emitted += 1
            yield from drain()
        yield from drain()
    seconds = time.time() - started
    yield {
        "type": "done", "tokens": emitted, "states": described, "seconds": seconds,
        "wait": first_token_at or 0.0,  # time before the first word: prompt reading, and paging if the model was idle
        "tps": float(last.generation_tps) if last is not None else 0.0,  # words per second once it is writing
    }


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):  # keep prompts out of the terminal log
        pass

    def _send(self, code: int, body: bytes, kind: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self._send(200, (HERE / "static" / "index.html").read_bytes(), "text/html; charset=utf-8")
        elif self.path == "/api/map":
            self._send(200, json.dumps(map_payload()).encode(), "application/json")
        else:
            self._send(404, b"not found", "text/plain")

    def do_POST(self):
        if self.path != "/api/generate":
            return self._send(404, b"not found", "text/plain")
        try:
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0"))))
            prompt = str(body["prompt"]).strip()
            effort = min(0.99, max(0.0, float(body.get("effort", 0.2))))
            max_tokens = min(MAX_TOKENS_LIMIT, max(1, int(body.get("max_tokens", 300))))
            if not prompt:
                raise ValueError("empty prompt")
        except (KeyError, ValueError, json.JSONDecodeError) as e:
            return self._send(400, json.dumps({"error": str(e)}).encode(), "application/json")
        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        try:
            for event in generate_events(prompt, effort, max_tokens):
                self.wfile.write((json.dumps(event) + "\n").encode())
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass  # the page was closed or the request cancelled


def main() -> None:
    global MODEL, TOKENIZER, LIVE
    print("loading the model (about a minute)...", flush=True)
    MODEL, TOKENIZER = load_model()
    LIVE = LiveMap.load(BUNDLE, HERE / "norms.json")
    for _ in generate_events("Hello", 0.0, 8):  # warm up so the first real prompt is not the slow one
        pass
    params = [p for _, p in tree_flatten(MODEL.parameters())]

    def touch() -> float:
        """Read every weight once. On a Mac under memory pressure, macOS compresses and
        swaps out the model while it sits idle, and the next prompt then waits tens of
        seconds for its first word. (Raising MLX's wired limit does not prevent this:
        memory is only wired while a computation is using it.)"""
        t = time.time()
        mx.eval(mx.stack([p.sum() for p in params]))
        return time.time() - t

    print(f"ready: http://127.0.0.1:{PORT}", flush=True)
    server = HTTPServer(("127.0.0.1", PORT), Handler)
    # When a pass is slow the Mac is too short of memory to hold the model, and reading it
    # back every few seconds only makes the machine swap continuously. Back off instead.
    interval = HEARTBEAT_SECONDS
    server.timeout = interval or None
    touched = time.time()
    while True:
        server.handle_request()  # returns after one request, or after the timeout when idle
        if interval and time.time() - touched >= interval:
            took = touch()
            touched = time.time()
            interval = min(HEARTBEAT_MAX_SECONDS, interval * 2) if took > 3 else HEARTBEAT_SECONDS
            server.timeout = interval
            if took > 3:
                print(f"keep-warm pass took {took:.1f}s: this Mac is short of memory and paged the model out; "
                      f"next pass in {interval}s. Closing other large apps is the real fix.", flush=True)


if __name__ == "__main__":
    main()
