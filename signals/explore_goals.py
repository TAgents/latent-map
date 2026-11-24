"""Exploratory follow-up to the hidden-goal test, written after seeing the results.
H3's fixed measure saturated (the goal's words reach the top 50 somewhere in almost
every answer, with or without a goal), so this compares the best rank itself, pairing
each quiet goal reply with the control reply to the same question.
Writes results/goals_explore.json.

Run: PYTHONPATH=../spike:../basemap ../spike/.venv/bin/python explore_goals.py
"""

import json
from pathlib import Path

import numpy as np

from siglib.analysis import paired
from siglib.items_goals import GOALS

OUT = Path(__file__).parent / "results"


def main() -> None:
    rows = [json.loads(line) for line in (OUT / "goals_replies.jsonl").open()]
    arr = np.load(OUT / "goals_states.npz")
    key = json.loads((OUT / "goals_blind_key.json").read_text())
    guess = {key[k]: v for k, v in json.loads((OUT / "goals_blind_labels.json").read_text()).items()}
    names = list(GOALS)
    control = {r["qid"]: r for r in rows if r["condition"] == "control"}
    out = {}
    for label, pick in (("quiet goal replies", lambda r: guess[r["id"]] == "none"), ("goal replies that show the goal", lambda r: guess[r["id"]] == r["condition"])):
        sel = [r for r in rows if r["condition"] != "control" and pick(r)]
        entry = {"n": len(sel)}
        for k in range(3):
            g = [int(arr[f"{r['id']}/word_rank"][k, names.index(r["condition"])]) for r in sel]
            c = [int(arr[f"{control[r['qid']]['id']}/word_rank"][k, names.index(r["condition"])]) for r in sel]
            p = paired(c, g)  # a_higher = control rank is worse, i.e. the goal reply ranks the word better
            entry[f"pass{k + 1}"] = {"goal_better": p["a_higher"], "control_better": p["b_higher"], "ties": p["ties"], "p": p["p"],
                                     "median_rank_goal": float(np.median(g)), "median_rank_control": float(np.median(c))}
        out[label] = entry
    (OUT / "goals_explore.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
