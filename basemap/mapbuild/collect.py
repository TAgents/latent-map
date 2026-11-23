"""Stack per-text captures into flat arrays with one row per token position."""

import numpy as np


def stack(records: list[dict]) -> dict:
    """Each record: block2 and block3 as (passes, tokens, d), tokens, topic, form,
    text_id, n_prompt, and optionally phase (one label per token). Returns arrays of
    shape (N, passes, d) plus per-token lists."""
    records = [r for r in records if len(r["tokens"]) > 0]
    out = {"token": [], "topic": [], "form": [], "text_id": []}
    reply = []
    for r in records:
        n = len(r["tokens"])
        out["token"] += list(r["tokens"])
        for key in ("topic", "form", "text_id"):
            out[key] += [r[key]] * n
        reply.append(np.arange(n) >= r["n_prompt"])
    for key in ("block2", "block3"):
        out[key] = np.concatenate([np.asarray(r[key]).transpose(1, 0, 2) for r in records]).astype(np.float16)
    out["is_reply"] = np.concatenate(reply)
    if records and all("phase" in r for r in records):
        out["phase"] = [p for r in records for p in r["phase"]]
    return out
