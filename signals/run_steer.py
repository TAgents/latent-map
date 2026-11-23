"""Steering test: push block 2's output toward code during ordinary chat prompts.

Run: PYTHONPATH=../spike:../basemap ../spike/.venv/bin/python run_steer.py
"""

import json
import time
from pathlib import Path

import numpy as np

from latent_capture.loader import load_model
from mapbuild.areas import is_marker
from mapbuild.paths import paths
from siglib.prompts_steer import PROMPTS
from siglib.run import load_map, reply

OUT = Path(__file__).parent / "results"
EFFORT, MAX_TOKENS, TAG, COORD = 0.2, 160, "v1", 2027
STRENGTHS = (0.1, 0.25, 0.5)
PASS_SETS = {"pass1": (0,), "pass2": (1,), "pass3": (2,), "all": (0, 1, 2)}


def scaled(v: np.ndarray) -> np.ndarray:
    """Zero the loop-signal coordinate and give the vector the length of a normalised state."""
    v = v.astype(np.float64).copy()
    v[COORD] = 0.0
    return (v / np.linalg.norm(v) * np.sqrt(len(v))).astype(np.float32)


def code_directions() -> tuple[dict, dict]:
    """Mean block 2 state of code answers minus mean of other answers, per pass, from the v1 corpus."""
    where = paths(TAG)
    states = np.load(where["states"])
    lists = json.loads(where["tokens"].read_text())
    keep = states["is_reply"] & (np.array(lists["phase"]) == "answer") & ~np.array([is_marker(t) for t in lists["token"]])
    code = keep & (np.array(lists["form"]) == "code")
    other = keep & ~code
    x = states["block2"]
    raw = [x[code, k].astype(np.float64).mean(0) - x[other, k].astype(np.float64).mean(0) for k in range(3)]
    info = {"code_tokens": int(code.sum()), "other_tokens": int(other.sum()),
            "raw_length_by_pass": [float(np.linalg.norm(np.delete(v, COORD))) for v in raw],
            "state_length": float(np.sqrt(x.shape[-1])),
            "cosine_between_passes": [float(scaled(raw[a]) @ scaled(raw[b]) / x.shape[-1]) for a, b in ((0, 1), (1, 2), (0, 2))]}
    vectors = {"pass1": scaled(raw[0]), "pass2": scaled(raw[1]), "pass3": scaled(raw[2]), "all": scaled(sum(raw))}
    return vectors, info


def main() -> None:
    vectors, info = code_directions()
    print("direction:", json.dumps(info), flush=True)
    random = scaled(np.random.default_rng(0).normal(size=len(vectors["all"])))
    settings = [("none", 0.0, "none", None)]
    for s in STRENGTHS:
        settings += [("code", s, name, {"vector": vectors[name], "strength": s, "passes": PASS_SETS[name]}) for name in PASS_SETS]
        settings.append(("random", s, "all", {"vector": random, "strength": s, "passes": PASS_SETS["all"]}))
    model, tokenizer = load_model()
    bundle = paths(TAG)["bundle"]
    maps = {"block2": load_map(bundle, "block2")}
    fine = maps["block2"]["fine"]
    rows, t0 = [], time.time()
    for kind, strength, passes, steer in settings:
        for i, prompt in enumerate(PROMPTS):
            r = reply(model, tokenizer, prompt, maps, EFFORT, MAX_TOKENS, steer=steer)
            ans = np.array([p == "answer" and not m for p, m in zip(r["phase"], r["marker"])])
            areas = fine[r["block2_cell"][ans, -1]] if ans.any() else np.array([], dtype=int)
            rows.append({"prompt": i, "kind": kind, "strength": strength, "passes": passes, "answer": r["answer"],
                         "thinking": r["thinking"], "answer_tokens": [t for t, a in zip(r["tokens"], ans) if a],
                         "area_counts": {str(int(a)): int((areas == a).sum()) for a in np.unique(areas)}})
        print(f"{kind} {strength} {passes}: done, {time.time() - t0:.0f}s", flush=True)
    OUT.mkdir(exist_ok=True)
    (OUT / "steer_replies.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (OUT / "steer_direction.json").write_text(json.dumps(info, indent=1))
    print(f"wrote {len(rows)} replies")


if __name__ == "__main__":
    main()
