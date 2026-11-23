"""Fit a map for block 2 and for block 3, group cells into areas, and measure
how well the areas line up with the corpus's text forms and topics.

Run: ../spike/.venv/bin/python build_map.py
"""

import json
from collections import Counter
from pathlib import Path

import numpy as np

from mapbuild.areas import distinctive_words, group_cells, lift
from mapbuild.fit import fit_map, place

HERE = Path(__file__).parent
DATA, BUNDLE = HERE / "data", HERE / "bundle_v0"
N_FINE, N_BROAD, WORDS = 12, 4, 16


def main() -> None:
    BUNDLE.mkdir(exist_ok=True)
    states = np.load(DATA / "states.npz")
    lists = json.loads((DATA / "tokens.json").read_text())
    reply = states["is_reply"]
    tokens = [t for t, r in zip(lists["token"], reply) if r]
    forms = [t for t, r in zip(lists["form"], reply) if r]
    topics = [t for t, r in zip(lists["topic"], reply) if r]
    phases = [t for t, r in zip(lists["phase"], reply) if r]
    answer = np.array(phases) == "answer"
    for block in ("block2", "block3"):
        x = states[block]
        m = fit_map(x[reply])  # fit on reply tokens only: the chat template repeats in every text
        cell = place(m, x)
        fine, broad = group_cells(m["centres"], N_FINE, N_BROAD)
        last = cell[reply, -1]
        area = fine[last]
        words = distinctive_words(tokens, area, top=WORDS)
        info = {
            "points": int(reply.sum()),
            "cells": int(len(m["centres"])),
            "occupied_cells": int(len(set(last.tolist()))),
            "form_lift": lift(area, forms),
            "topic_lift": lift(area, topics),
            "phase_lift": lift(area, phases),
            "answer_only": {
                "points": int(answer.sum()),
                "form_lift": lift(area[answer], [f for f, a in zip(forms, answer) if a]),
                "topic_lift": lift(area[answer], [t for t, a in zip(topics, answer) if a]),
            },
            "area_mix": {
                str(a): {
                    "answer_share": float(answer[area == a].mean()) if (area == a).any() else 0.0,
                    "top_form": Counter(np.array(forms)[area == a].tolist()).most_common(1)[0] if (area == a).any() else None,
                }
                for a in sorted(set(fine.tolist()))
            },
            "words": {str(a): w for a, w in words.items()},
            "broad_of_fine": {str(f): int(broad[fine == f][0]) for f in sorted(set(fine.tolist()))},
            "area_points": {str(a): int((area == a).sum()) for a in sorted(set(fine.tolist()))},
        }
        np.savez(BUNDLE / f"{block}.npz", **m, fine=fine, broad=broad, cell=cell)
        (BUNDLE / f"{block}.json").write_text(json.dumps(info, indent=1))
        print(f"{block}: form lift {info['form_lift']:.2f}, topic lift {info['topic_lift']:.2f}, "
              f"phase lift {info['phase_lift']:.2f}, occupied cells {info['occupied_cells']}/{info['cells']}; "
              f"answer tokens only: form {info['answer_only']['form_lift']:.2f}, topic {info['answer_only']['topic_lift']:.2f}")


if __name__ == "__main__":
    main()
