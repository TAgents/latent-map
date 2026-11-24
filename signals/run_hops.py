"""Unwritten-step test: answer each question with thinking skipped, then read every pass.

Run: PYTHONPATH=../spike:../basemap ../spike/.venv/bin/python run_hops.py
"""

import json
import time
from pathlib import Path

import numpy as np
from mlx_lm import generate

from latent_capture.loader import load_model
from latent_capture.readout import pass_readouts
from siglib.items_hops import build
from siglib.prompting import head, split_reply, word_ids

OUT = Path(__file__).parent / "results"


def subject_end(tokenizer, ids: list[int], subject: str) -> int:
    """Index of the token at which the subject's name is complete."""
    for p in range(len(ids)):
        if subject in tokenizer.decode(ids[: p + 1]):
            return p
    raise ValueError(f"subject {subject!r} not found in the prompt")


def main() -> None:
    model, tokenizer = load_model()
    items = build()
    family_ids = {}
    for it in items:  # every item's readout tracks every bridge and answer word of its family
        fam = family_ids.setdefault(it["family"], {})
        fam.setdefault(it["bridge"], word_ids(tokenizer, it["bridge"].split()[0]))
        fam.setdefault(it["answer"], word_ids(tokenizer, it["answer"].split()[0]))
    rows, arrays, t0 = [], {}, time.time()
    for n, it in enumerate(items, 1):
        said = {k: split_reply(generate(model, tokenizer, head(tokenizer, it[k]), max_tokens=20, verbose=False))[1]
                for k in ("two_step", "first_step", "second_step")}
        ids = tokenizer.encode(head(tokenizer, it["two_step"]))
        words = family_ids[it["family"]]
        track = sorted({t for v in words.values() for t in v})
        out = pass_readouts(model, ids, track=track, topk=5)
        start = subject_end(tokenizer, ids, it["subject"])
        rows.append({**it, "said": said, "subject_end": start, "n_tokens": len(ids), "track": track,
                     "word_ids": words, "top_last": [[tokenizer.decode([int(t)]) for t in out["top"][k, -1]] for k in range(3)]})
        arrays[f"{it['id']}/rank"] = out["rank"].astype(np.int32)                       # (passes, tokens, n_track)
        arrays[f"{it['id']}/pass_end"] = out["pass_end"][:, [start, len(ids) - 1]].astype(np.float16)   # for the swap test
        if n % 20 == 0:
            print(f"{n}/{len(items)} items, {time.time() - t0:.0f}s", flush=True)
    OUT.mkdir(exist_ok=True)
    (OUT / "hops_replies.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    np.savez_compressed(OUT / "hops_states.npz", **arrays)
    print(f"wrote {len(rows)} items")


if __name__ == "__main__":
    main()
