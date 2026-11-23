"""Controls added after an independent review of the Guard probe: how much of the
result can be explained by the wording of the prompts or of the thinking text,
and what the "safety policy" area actually contains. Writes results/guard_controls.json.

Run: PYTHONPATH=../spike:../basemap ../spike/.venv/bin/python controls_guard.py
"""

import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

from mapbuild.paths import paths
from siglib.analysis import paired, probe_auc

OUT = Path(__file__).parent / "results"
AREA = 6
SURFACE = {
    "without": r"\bwithout\b", "secret": r"\bsecret", "fake": r"\bfake\b",
    "noticing/caught/detected/discreet": r"notic|caught|detect|discreet",
    "own": r"\bown\b", "legitimising word": r"properly|honestly|legally|safely|polite|real\b|legitimate|correct",
}


def main() -> None:
    rows = [json.loads(line) for line in (OUT / "guard_replies.jsonl").open()]
    arr = np.load(OUT / "guard_states.npz")
    fine = np.load(paths("v1")["bundle"] / "block2.npz")["fine"]
    by = {(r["triple"], r["kind"]): r for r in rows}
    triples = sorted({r["triple"] for r in rows})
    kinds = ("harmful", "topic", "benign")
    think = lambda r: np.flatnonzero([p == "thinking" and not m for p, m in zip(r["phase"], r["marker"])])
    in_area = lambda r, idx: fine[arr[f"{r['id']}/block2_cell"][idx, 2]] == AREA
    out = {}

    # 1. Is the prompt wording alone enough?
    out["prompt_surface_counts"] = {name: {k: sum(bool(re.search(rx, by[t, k]["prompt"], re.I)) for t in triples) for k in kinds}
                                    for name, rx in SURFACE.items()}
    out["prompt_words_mean"] = {k: float(np.mean([len(by[t, k]["prompt"].split()) for t in triples])) for k in kinds}
    y, groups = np.tile([1, 0], len(triples)), np.repeat(np.arange(len(triples)), 2)
    for name, field in (("bag_of_words_on_prompts", "prompt"), ("bag_of_words_on_thinking_text", "thinking")):
        texts = [by[t, k][field] for t in triples for k in ("harmful", "topic")]
        x = CountVectorizer(binary=True).fit_transform(texts).toarray().astype(float)
        out[name] = probe_auc(x, y, groups, n_null=40)

    # 2. Behaviour never varies apart from the prompt type.
    out["state_at_reply_start_marker_in_area"] = {
        k: sum(bool(fine[arr[f"{by[t, k]['id']}/block2_cell"][0, 2]] == AREA) for t in triples) for k in kinds}

    # 3. Same words, different place? The first thinking tokens are the same text in every reply.
    firsts = [tuple(by[t, k]["tokens"][i] for i in think(by[t, k])[:3]) for t in triples for k in kinds]
    out["first_three_thinking_tokens"] = {"distinct_openings": [list(o) for o in sorted(set(firsts))], "replies": len(firsts)}
    out["first_thinking_tokens_in_area"] = {
        str(n): {k: sum(bool(in_area(by[t, k], think(by[t, k])[n:n + 1]).any()) for t in triples) for k in kinds} for n in range(3)}

    # 4. What is in the area?
    def tokens_in(kind):
        inside, total = Counter(), Counter()
        for t in triples:
            r = by[t, kind]; idx = think(r); hit = in_area(r, idx)
            for i, h in zip(idx, hit):
                w = r["tokens"][i].strip().lower()
                total[w] += 1; inside[w] += int(h)
        return inside, total
    stats = {k: tokens_in(k) for k in kinds}
    out["full_stop"] = {k: {"rate_in_area": stats[k][0]["."] / max(1, stats[k][1]["."]),
                            "share_of_area_tokens": stats[k][0]["."] / max(1, sum(stats[k][0].values()))} for k in kinds}
    out["top_area_tokens"] = {k: [[w, c, stats[k][1][w]] for w, c in stats[k][0].most_common(12)] for k in kinds}

    def share(r, keep):
        idx = np.array([i for i in think(r) if keep(r["tokens"][i].strip().lower())], dtype=int)
        return float(in_area(r, idx).mean()) if len(idx) else 0.0
    shared = {w for w in stats["harmful"][1] if w in stats["topic"][1]}
    late = lambda r: float(in_area(r, think(r)[30:]).mean()) if len(think(r)) > 30 else 0.0
    out["controls_harmful_vs_topic"] = {
        "all thinking tokens": paired([share(by[t, "harmful"], lambda w: True) for t in triples], [share(by[t, "topic"], lambda w: True) for t in triples]),
        "without full stops": paired([share(by[t, "harmful"], lambda w: w != ".") for t in triples], [share(by[t, "topic"], lambda w: w != ".") for t in triples]),
        "only words used in both groups": paired([share(by[t, "harmful"], lambda w: w in shared) for t in triples], [share(by[t, "topic"], lambda w: w in shared) for t in triples]),
        "after the first 30 thinking tokens": paired([late(by[t, "harmful"]) for t in triples], [late(by[t, "topic"]) for t in triples]),
    }
    (OUT / "guard_controls.json").write_text(json.dumps(out, indent=1))
    print("surface:", json.dumps(out["prompt_surface_counts"])); print("prompt words:", out["prompt_words_mean"])
    for k in ("bag_of_words_on_prompts", "bag_of_words_on_thinking_text"):
        print(k, f"{out[k]['auc']:.2f} null {out[k]['null_mean']:.2f}±{out[k]['null_sd']:.2f}")
    print("reply-start marker in area:", out["state_at_reply_start_marker_in_area"])
    print("openings:", out["first_three_thinking_tokens"]); print("first tokens in area:", out["first_thinking_tokens_in_area"])
    print("full stop:", json.dumps(out["full_stop"]))
    for k in kinds: print(k, "top area tokens (token, in area, total):", out["top_area_tokens"][k])
    for name, p in out["controls_harmful_vs_topic"].items():
        print(f"{name:36s} H {p['mean_a']:.3f} T {p['mean_b']:.3f}  H>T {p['a_higher']} T>H {p['b_higher']} p={p['p']:.4f}")


if __name__ == "__main__":
    main()
