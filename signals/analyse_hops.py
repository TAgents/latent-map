"""Read the unwritten-step results against T1 and T2 in
docs/superpowers/plans/2026-11-24-hidden-tests.md. Writes results/hops.json.

Run: PYTHONPATH=../spike:../basemap ../spike/.venv/bin/python analyse_hops.py
"""

import json
from collections import Counter
from pathlib import Path

import numpy as np

from siglib.hops import first_pass_in_top, matches, step_readable

OUT = Path(__file__).parent / "results"


def best(rank, track, ids, start):
    """Best rank of any of a word's token ids, per pass, over positions from `start` on. (passes,)"""
    cols = [track.index(t) for t in ids]
    return rank[:, start:, cols].min(axis=(1, 2))


def main() -> None:
    rows = [json.loads(line) for line in (OUT / "hops_replies.jsonl").open()]
    arr = np.load(OUT / "hops_states.npz")
    for r in rows:
        s = r["said"]
        r["ok_first"], r["ok_second"], r["ok_two"] = matches(s["first_step"], r["bridge"]), matches(s["second_step"], r["answer"]), matches(s["two_step"], r["answer"])
        r["step_written"] = matches(s["two_step"], r["bridge"])
        r["kept"] = r["ok_first"] and r["ok_second"] and r["ok_two"] and not r["step_written"] and not r["seen_in_step0"]
        rank = arr[f"{r['id']}/rank"]
        r["bridge_by_pass"] = best(rank, r["track"], r["word_ids"][r["bridge"]], r["subject_end"]).tolist()
        r["answer_by_pass"] = best(rank, r["track"], r["word_ids"][r["answer"]], r["subject_end"]).tolist()
        last = rank[:, -1:, :]
        r["bridge_last"] = best(last, r["track"], r["word_ids"][r["bridge"]], 0).tolist()
        r["answer_last"] = best(last, r["track"], r["word_ids"][r["answer"]], 0).tolist()
    by_family = {}
    for r in rows:
        by_family.setdefault(r["family"], []).append(r)
    for r in rows:
        # The same word's best rank in other items of the family that have a different step and answer.
        others = [o for o in by_family[r["family"]] if o["id"] != r["id"] and r["bridge"] not in (o["bridge"], o["answer"])]
        r["others_best"] = [int(best(arr[f"{o['id']}/rank"], o["track"], r["word_ids"][r["bridge"]], o["subject_end"]).min()) for o in others]
        r["own_best"] = int(min(r["bridge_by_pass"]))
        r["t1"] = step_readable(r["own_best"], r["others_best"])
        b, a = first_pass_in_top(r["bridge_by_pass"]), first_pass_in_top(r["answer_by_pass"])
        r["t2"] = r["t1"] and b is not None and (a is None or b < a)
    kept = [r for r in rows if r["kept"]]
    t1 = [r for r in kept if r["t1"]]
    out = {
        "items": len(rows), "kept": len(kept),
        "dropped": dict(Counter("seen in step 0" if r["seen_in_step0"] else "first step wrong" if not r["ok_first"] else
                                "second step wrong" if not r["ok_second"] else "two-step wrong" if not r["ok_two"] else "step written out"
                                for r in rows if not r["kept"])),
        "T1": {"readable": len(t1), "of": len(kept), "share": len(t1) / max(1, len(kept)), "needed": 0.5, "met": len(t1) / max(1, len(kept)) >= 0.5},
        "T2": {"step_first": sum(r["t2"] for r in t1), "of": len(t1), "met": bool(t1) and sum(r["t2"] for r in t1) / len(t1) > 0.5},
        "by_family": {f: {"kept": sum(r["kept"] for r in rs), "t1": sum(r["kept"] and r["t1"] for r in rs)} for f, rs in by_family.items()},
        "median_best_rank_by_pass": {"step": np.median([r["bridge_by_pass"] for r in kept], axis=0).tolist(),
                                     "answer": np.median([r["answer_by_pass"] for r in kept], axis=0).tolist(),
                                     "step_in_other_items": float(np.median([np.median(r["others_best"]) for r in kept if r["others_best"]]))},
        "at_the_moment_of_answering": {
            f"pass{k + 1}": {"answer_is_top": sum(r["answer_last"][k] == 0 for r in kept),
                             "step_in_top10": sum(r["bridge_last"][k] < 10 for r in kept), "of": len(kept)} for k in range(3)},
        "per_item": [{k: r[k] for k in ("id", "kept", "said", "bridge_by_pass", "answer_by_pass", "own_best", "t1", "t2", "top_last")} for r in rows],
    }
    (OUT / "hops.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in ("items", "kept", "dropped", "T1", "T2", "by_family", "median_best_rank_by_pass", "at_the_moment_of_answering")}, indent=1))


if __name__ == "__main__":
    main()
