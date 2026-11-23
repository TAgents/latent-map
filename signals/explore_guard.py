"""Follow-up analyses run after seeing the Guard results; exploratory, not criteria.
How early does the signal appear, and does the thinking text already say it?
Writes results/guard_explore.json.

Run: PYTHONPATH=../spike:../basemap ../spike/.venv/bin/python explore_guard.py
"""

import json
import re
from pathlib import Path

import numpy as np

from mapbuild.paths import paths
from siglib.analysis import paired, probe_auc

OUT = Path(__file__).parent / "results"
AREA = 6
KEYWORDS = re.compile(r"disallow|illegal|refus|policy|harm|cannot|can't|wrongdoing|not allowed|unethical|malicious|fraud|decline", re.I)


def main() -> None:
    rows = [json.loads(line) for line in (OUT / "guard_replies.jsonl").open()]
    arr = np.load(OUT / "guard_states.npz")
    fine = np.load(paths("v1")["bundle"] / "block2.npz")["fine"]
    by = {(r["triple"], r["kind"]): r for r in rows}
    triples = sorted({r["triple"] for r in rows})
    think = lambda r: np.flatnonzero([p == "thinking" and not m for p, m in zip(r["phase"], r["marker"])])

    def first_keyword(r):
        seen = ""
        for n, i in enumerate(think(r)):
            seen += r["tokens"][i]
            if KEYWORDS.search(seen):
                return n
        return None

    rate = lambda r: len(KEYWORDS.findall(r["thinking"])) / max(1, len(think(r)))
    firsts = [first_keyword(by[t, "harmful"]) for t in triples]
    out = {
        "keyword_rate_harmful_vs_topic": paired([rate(by[t, "harmful"]) for t in triples], [rate(by[t, "topic"]) for t in triples]),
        "first_keyword_token_in_harmful_thinking": sorted(x for x in firsts if x is not None),
        "harmful_without_keyword": sum(x is None for x in firsts),
        "first_n_thinking_tokens": {},
    }
    for n in (1, 3, 5, 10):
        def share(r):
            idx = think(r)[:n]
            return float((fine[arr[f"{r['id']}/block2_cell"][idx, 2]] == AREA).mean()) if len(idx) else 0.0

        entry = {"area_share": paired([share(by[t, "harmful"]) for t in triples], [share(by[t, "topic"]) for t in triples]), "probe": {}}
        for block, k in (("block2", 0), ("block2", 2), ("block3", 2)):
            x = np.stack([arr[f"{by[t, kind]['id']}/{block}_z"][think(by[t, kind])[:n], k].astype(float).mean(0)
                          for t in triples for kind in ("harmful", "topic")])
            entry["probe"][f"{block}/pass{k + 1}"] = probe_auc(x, np.tile([1, 0], len(triples)), np.repeat(np.arange(len(triples)), 2), n_null=40)
        out["first_n_thinking_tokens"][str(n)] = entry
    (OUT / "guard_explore.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out["first_keyword_token_in_harmful_thinking"]), out["harmful_without_keyword"])
    for n, e in out["first_n_thinking_tokens"].items():
        a = e["area_share"]
        print(n, f"H {a['mean_a']:.2f} T {a['mean_b']:.2f} H>T {a['a_higher']} T>H {a['b_higher']}", {k: round(v["auc"], 2) for k, v in e["probe"].items()})


if __name__ == "__main__":
    main()
