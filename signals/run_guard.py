"""Guard probe: run the 24 prompt triples and save each reply with its map coordinates.

Run: PYTHONPATH=../spike:../basemap ../spike/.venv/bin/python run_guard.py
Outputs stay in signals/results/ on this Mac (the texts are git-ignored).
"""

import json
import time
from pathlib import Path

import numpy as np

from latent_capture.loader import load_model
from mapbuild.paths import paths
from siglib.prompts_guard import build
from siglib.run import load_map, reply

OUT = Path(__file__).parent / "results"
EFFORT, MAX_TOKENS, TAG = 0.2, 350, "v1"


def main() -> None:
    model, tokenizer = load_model()
    bundle = paths(TAG)["bundle"]
    maps = {b: load_map(bundle, b) for b in ("block2", "block3")}
    rows, arrays, t0 = [], {}, time.time()
    for n, item in enumerate(build(), 1):
        r = reply(model, tokenizer, item["prompt"], maps, EFFORT, MAX_TOKENS)
        rows.append({**item, **{k: r[k] for k in ("tokens", "phase", "marker", "answer", "thinking")}})
        for key in ("block2_cell", "block2_z", "block3_cell", "block3_z"):
            arrays[f"{item['id']}/{key}"] = r[key]
        if n % 12 == 0:
            print(f"{n}/72 replies, {time.time() - t0:.0f}s", flush=True)
    OUT.mkdir(exist_ok=True)
    (OUT / "guard_replies.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    np.savez_compressed(OUT / "guard_states.npz", **arrays)
    print(f"wrote {len(rows)} replies")


if __name__ == "__main__":
    main()
