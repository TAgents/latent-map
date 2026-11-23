"""Read the steering results against the criteria in
docs/superpowers/plans/2026-11-23-signal-tests.md. Writes results/steer.json.

Run: PYTHONPATH=../spike:../basemap ../spike/.venv/bin/python analyse_steer.py
"""

import json
from pathlib import Path

import numpy as np

from siglib.analysis import distinct_ratio, has_code

OUT = Path(__file__).parent / "results"
CODE_AREAS = ("1", "8")  # map v1, block 2: the two areas whose words are code


def main() -> None:
    rows = [json.loads(line) for line in (OUT / "steer_replies.jsonl").open()]
    settings = sorted({(r["kind"], r["strength"], r["passes"]) for r in rows})
    table = {}
    for kind, strength, passes in settings:
        rs = [r for r in rows if (r["kind"], r["strength"], r["passes"]) == (kind, strength, passes)]
        answered = [bool(r["answer"].strip()) for r in rs]
        ratios = [distinct_ratio(r["answer_tokens"]) for r in rs]
        code_share = [sum(r["area_counts"].get(a, 0) for a in CODE_AREAS) / max(1, sum(r["area_counts"].values())) for r in rs]
        table[f"{kind}/{strength}/{passes}"] = {
            "n": len(rs), "with_code": sum(has_code(r["answer"]) for r in rs), "answered": sum(answered),
            "fluent": sum(a and q >= 0.3 for a, q in zip(answered, ratios)),
            "median_distinct_ratio": float(np.median(ratios)), "mean_code_area_share": float(np.mean(code_share)),
        }
    base = table["none/0.0/none"]
    met = []
    for key, v in table.items():
        kind, strength, passes = key.split("/")
        if kind != "code":
            continue
        ctrl = table[f"random/{strength}/all"]
        s1 = v["with_code"] >= 4 and base["with_code"] == 0 and ctrl["with_code"] <= 1
        s2 = v["fluent"] >= 4
        v["S1"], v["S2"] = s1, s2
        if s1 and s2:
            met.append(key)
    out = {"table": table, "settings_meeting_S1_and_S2": met,
           "S1_met_anywhere": any(v.get("S1") for v in table.values()),
           "direction": json.loads((OUT / "steer_direction.json").read_text())}
    (OUT / "steer.json").write_text(json.dumps(out, indent=1))
    print(f"{'setting':22s} code answered fluent distinct code-area")
    for key, v in table.items():
        flag = ("  S1" if v.get("S1") else "") + (" S2" if v.get("S2") else "")
        print(f"{key:22s} {v['with_code']}/6   {v['answered']}/6     {v['fluent']}/6   {v['median_distinct_ratio']:.2f}    {v['mean_code_area_share']:.2f}{flag}")
    print("settings meeting S1 and S2:", met or "none")


if __name__ == "__main__":
    main()
