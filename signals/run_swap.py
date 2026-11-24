"""T3: swap in another item's state where the subject is read, and see whether the answer follows.

Run: PYTHONPATH=../spike:../basemap ../spike/.venv/bin/python run_swap.py
"""

import json
from pathlib import Path

import numpy as np

from latent_capture.loader import load_model
from latent_capture.readout import pass_readouts
from siglib.prompting import head

OUT = Path(__file__).parent / "results"


def main() -> None:
    rows = {r["id"]: r for r in (json.loads(line) for line in (OUT / "hops_replies.jsonl").open())}
    verdict = {p["id"]: p for p in json.loads((OUT / "hops.json").read_text())["as_planned"]["per_item"]}
    arr = np.load(OUT / "hops_states.npz")
    use = [rows[i] for i, p in verdict.items() if p["kept"] and p["t1"]]
    by_family = {}
    for r in use:
        by_family.setdefault(r["family"], []).append(r)
    model, tokenizer = load_model()
    results = []
    for r in use:
        fam = by_family[r["family"]]
        start = fam.index(r)
        donor = next((fam[(start + j) % len(fam)] for j in range(1, len(fam))
                      if fam[(start + j) % len(fam)]["bridge"] != r["bridge"] and fam[(start + j) % len(fam)]["answer"] != r["answer"]), None)
        other = next((o for o in use if o["family"] != r["family"]), None)
        if donor is None or other is None:
            continue
        ids = tokenizer.encode(head(tokenizer, r["two_step"]))
        own_ids, donor_ids = r["word_ids"][r["answer"]], donor["word_ids"][donor["answer"]]
        track = sorted(set(own_ids) | set(donor_ids))
        col = lambda ws: [track.index(t) for t in ws]
        entry = {"id": r["id"], "donor": donor["id"], "control_from": other["id"], "own_answer": r["answer"], "donor_answer": donor["answer"]}
        for name, source in (("none", None), ("donor", donor), ("control", other)):
            replace = None if source is None else {
                "passes": (0, 1, 2), "position": r["subject_end"], "state": arr[f"{source['id']}/pass_end"][:, 0].astype(np.float32)}
            out = pass_readouts(model, ids, track=track, topk=3, replace=replace)
            last = out["rank"][2, -1]
            entry[name] = {"own_rank": int(last[col(own_ids)].min()), "donor_rank": int(last[col(donor_ids)].min()),
                           "top": [tokenizer.decode([int(t)]) for t in out["top"][2, -1]]}
        results.append(entry)
        print(entry["id"], "->", entry["donor_answer"], "|", {k: entry[k]["top"][0] for k in ("none", "donor", "control")}, flush=True)
    n = len(results)
    summary = {
        "swaps": n,
        "donor_answer_is_top": sum(e["donor"]["donor_rank"] == 0 for e in results),
        "donor_answer_is_top_in_control": sum(e["control"]["donor_rank"] == 0 for e in results),
        "donor_answer_was_top_before": sum(e["none"]["donor_rank"] == 0 for e in results),
        "own_answer_top_before": sum(e["none"]["own_rank"] == 0 for e in results),
        "own_answer_top_after_swap": sum(e["donor"]["own_rank"] == 0 for e in results),
        "own_answer_top_after_control": sum(e["control"]["own_rank"] == 0 for e in results),
        "median_donor_rank": {k: float(np.median([e[k]["donor_rank"] for e in results])) for k in ("none", "donor", "control")},
    }
    summary["T3_met"] = bool(n and summary["donor_answer_is_top"] / n >= 1 / 3)
    (OUT / "swap.json").write_text(json.dumps({"summary": summary, "items": results}, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
