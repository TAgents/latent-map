"""Context for the map's headline numbers: chance levels, thinking against
answer, agreement between passes, and how pass-specific the cells are.

Run: ../spike/.venv/bin/python map_stats.py [tag] [states_tag]
"""

import json
import sys
from collections import Counter

import numpy as np

from mapbuild.areas import is_marker, lift
from mapbuild.paths import paths
from mapbuild.stats import lift_null, pass_agreement



def auc(score: np.ndarray, positive: np.ndarray) -> float:
    """Chance that a random positive point has a higher score than a random negative one."""
    order = np.argsort(score, kind="stable")
    ranks = np.empty(len(score)); ranks[order] = np.arange(1, len(score) + 1)
    for v in np.unique(score):  # average ranks over ties
        tie = score == v
        ranks[tie] = ranks[tie].mean()
    n_pos, n_neg = positive.sum(), (~positive).sum()
    return float((ranks[positive].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def main() -> None:
    tag = sys.argv[1] if len(sys.argv) > 1 else "v1"
    source = paths(sys.argv[2] if len(sys.argv) > 2 else tag)
    BUNDLE = paths(tag)["bundle"]
    states = np.load(source["states"])
    lists = json.loads(source["tokens"].read_text())
    # Markers are left out, as in build_map.py (v0 bundles kept them in; the v0 report's numbers include them).
    reply = states["is_reply"] & ~np.array([is_marker(t) for t in lists["token"]]) if tag != "v0" else states["is_reply"]
    pick = lambda key: np.array([v for v, r in zip(lists[key], reply) if r])
    form, topic, text, phase = pick("form"), pick("topic"), pick("text_id"), pick("phase")
    answer = phase == "answer"
    for block in ("block2", "block3"):
        b = np.load(BUNDLE / f"{block}.npz")
        cell, fine = b["cell"][reply], b["fine"]
        area = fine[cell[:, -1]]
        out = {"lift": {}}
        for name, sel in (("all", np.ones(len(area), bool)), ("thinking", ~answer), ("answer", answer)):
            for label, values, strata in (("form", form, topic), ("topic", topic, form)):
                seen = lift(area[sel], values[sel].tolist())
                mean, sd = lift_null(area[sel], values[sel], text[sel], strata=strata[sel])
                out["lift"][f"{label}_{name}"] = {"observed": seen, "null_mean": mean, "null_sd": sd, "over_null": seen / mean}
        cell_share = np.bincount(cell[:, -1], weights=answer, minlength=len(fine)) / np.maximum(np.bincount(cell[:, -1], minlength=len(fine)), 1)
        area_share = {int(a): float(answer[area == a].mean()) for a in np.unique(area)}
        out["phase"] = {
            "answer_share": float(answer.mean()),
            "phase_lift": lift(area, phase.tolist()),
            "auc_by_area": auc(np.array([area_share[int(a)] for a in area]), answer),
            "auc_by_cell": auc(cell_share[cell[:, -1]], answer),
            "answer_tokens_by_form": dict(Counter(form[answer].tolist())),
            "texts_without_answer": int(len(set(text.tolist())) - len(set(text[answer].tolist()))),
        }
        out["passes"] = pass_agreement(cell, fine)
        out["passes"]["form_lift"] = [lift(fine[cell[:, k]], form.tolist()) for k in range(3)]
        out["passes"]["cells_used"] = [int(len(np.unique(cell[:, k]))) for k in range(3)]
        counts = np.stack([np.bincount(cell[:, k], minlength=len(fine)) for k in range(3)], axis=1)
        used = counts.sum(1) > 0
        top = counts.max(1) / np.maximum(counts.sum(1), 1)
        out["passes"]["cells_95pct_one_pass"] = float((top[used] >= 0.95).mean())
        out["passes"]["cells_last_pass_only_or_never"] = float(
            ((counts[used, 2] == 0) | (counts[used, :2].sum(1) == 0)).mean()
        )
        (BUNDLE / f"{block}_stats.json").write_text(json.dumps(out, indent=1))
        print(tag, block)
        for k, v in out["lift"].items():
            print(f"  {k:15s} observed {v['observed']:.2f}  null {v['null_mean']:.2f} ± {v['null_sd']:.2f}  x{v['over_null']:.2f}")
        print("  phase:", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in out["phase"].items()})
        print("  passes:", {k: ([round(x, 2) for x in v] if isinstance(v, list) else round(v, 2)) for k, v in out["passes"].items()})


if __name__ == "__main__":
    main()
