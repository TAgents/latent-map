"""Fit a map for block 2 and for block 3, group cells into areas, and record
how well the areas line up with the corpus's text forms and topics.

Chat-format marker tokens are placed on the map but left out of the fit and
of every statistic (in v0 one whole area was a single marker). Block 3 is
fitted with per-pass scaling, because its last pass spreads much wider than
the first two and would otherwise take most of the cells.

Run: ../spike/.venv/bin/python build_map.py [tag] [states_tag]
`states_tag` lets a new map version be fitted on an older version's states.
Running this resets area names; run name_areas.py afterwards.
"""

import json
import sys
from collections import Counter

import numpy as np

from mapbuild.areas import distinctive_words, group_cells, is_marker, lift
from mapbuild.fit import fit_map, place
from mapbuild.paths import paths

N_FINE, N_BROAD, WORDS = 12, 4, 16
SCALE_PASSES = {"block2": False, "block3": True}


def main() -> None:
    tag = sys.argv[1] if len(sys.argv) > 1 else "v1"
    source = paths(sys.argv[2] if len(sys.argv) > 2 else tag)
    bundle = paths(tag)["bundle"]
    bundle.mkdir(exist_ok=True)
    states = np.load(source["states"])
    lists = json.loads(source["tokens"].read_text())
    use = states["is_reply"] & ~np.array([is_marker(t) for t in lists["token"]])
    pick = lambda key: [v for v, u in zip(lists[key], use) if u]
    tokens, forms, topics, phases = pick("token"), pick("form"), pick("topic"), pick("phase")
    answer = np.array(phases) == "answer"
    for block in ("block2", "block3"):
        x = states[block]
        m = fit_map(x[use], scale_passes=SCALE_PASSES[block])
        cell = place(m, x)
        fine, broad = group_cells(m["centres"], N_FINE, N_BROAD)
        last = cell[use, -1]
        area = fine[last]
        ids = sorted(set(fine.tolist()))
        words = distinctive_words(tokens, area, top=WORDS, areas=ids, method="logodds")
        info = {
            "tag": tag,
            "states_from": source["states"].name,
            "points": int(use.sum()),
            "cells": int(len(m["centres"])),
            "occupied_cells": int(len(set(last.tolist()))),
            "scale_passes": SCALE_PASSES[block],
            "form_lift": lift(area, forms),
            "topic_lift": lift(area, topics),
            "words": {str(a): w for a, w in words.items()},
            "broad_of_fine": {str(f): int(broad[fine == f][0]) for f in ids},
            "area_points": {str(a): int((area == a).sum()) for a in ids},
            "area_mix": {
                str(a): {
                    "answer_share": float(answer[area == a].mean()) if (area == a).any() else 0.0,
                    "top_form": Counter(np.array(forms)[area == a].tolist()).most_common(1)[0] if (area == a).any() else None,
                }
                for a in ids
            },
        }
        np.savez(bundle / f"{block}.npz", **m, fine=fine, broad=broad, cell=cell, used=use)
        (bundle / f"{block}.json").write_text(json.dumps(info, indent=1))
        print(f"{tag} {block}: form lift {info['form_lift']:.2f}, topic lift {info['topic_lift']:.2f}, "
              f"occupied cells {info['occupied_cells']}/{info['cells']} (raw lifts; run map_stats.py for chance levels)")


if __name__ == "__main__":
    main()
