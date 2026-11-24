"""Hidden-goal test: reply to ordinary questions with and without a planted goal.

Thinking skipped: reply, then one readout pass over prompt plus reply for the states
and the rank of each goal's words. Thinking on: reply only, to see whether the
thinking text states the goal (H1).

Run: PYTHONPATH=../spike:../basemap ../spike/.venv/bin/python run_goals.py
"""

import json
import time
from pathlib import Path

import numpy as np
from mlx_lm import generate

from latent_capture.loader import load_model
from latent_capture.readout import pass_readouts
from mapbuild.fit import _project
from mapbuild.paths import paths
from siglib.items_goals import GOALS, READOUT_WORDS, conditions, questions
from siglib.prompting import head, split_reply, word_ids
from siglib.run import load_map

OUT = Path(__file__).parent / "results"
SKIP_TOKENS, THINK_TOKENS, THINK_EFFORT = 180, 160, 0.2


def main() -> None:
    model, tokenizer = load_model()
    bundle = paths("v1")["bundle"]
    maps = {"block2": load_map(bundle, "block2"), "block3": load_map(bundle, "block3")}
    goal_ids = {g: sorted({t for w in ws for t in word_ids(tokenizer, w)}) for g, ws in READOUT_WORDS.items()}
    track = sorted({t for ids in goal_ids.values() for t in ids})
    rows, arrays, t0 = [], {}, time.time()
    for n, q in enumerate(questions(), 1):
        for cond, system in conditions().items():
            rid = f"{q['qid']}/{cond}"
            h = head(tokenizer, q["question"], system=system)
            raw = generate(model, tokenizer, h, max_tokens=SKIP_TOKENS, verbose=False)
            answer = split_reply(raw)[1]
            head_ids = tokenizer.encode(h)
            ids = head_ids + tokenizer.encode(raw.split("<|end_message|>", 1)[0], add_special_tokens=False)
            out = pass_readouts(model, ids, track=track, topk=1)
            start = len(head_ids) - 1                      # the last prompt token: nothing written yet
            span = slice(start, len(ids) - 1) if len(ids) - 1 > start else slice(start, start + 1)
            for block, key in (("block2", "block2"), ("block3", "pass_end")):
                z = _project(maps[block], out[key].transpose(1, 0, 2))           # (tokens, passes, 64)
                arrays[f"{rid}/{block}_start"] = z[start].astype(np.float32)
                arrays[f"{rid}/{block}_answer"] = z[span].mean(axis=0).astype(np.float32)
            arrays[f"{rid}/word_rank"] = np.stack(
                [out["rank"][:, span, :][:, :, [track.index(t) for t in goal_ids[g]]].min(axis=(1, 2)) for g in GOALS], axis=1
            ).astype(np.int32)                             # (passes, goals): best rank of the goal's words over the answer
            row = {"id": rid, **q, "condition": cond, "system": system, "answer": answer, "answer_tokens": len(ids) - len(head_ids)}
            if cond != "control":
                think_raw = generate(model, tokenizer, head(tokenizer, q["question"], system=system, effort=THINK_EFFORT, skip_thinking=False),
                                     max_tokens=THINK_TOKENS, verbose=False)
                row["thinking_on"] = dict(zip(("thinking", "answer"), split_reply(think_raw)))
            rows.append(row)
        if n % 5 == 0:
            print(f"{n}/30 questions, {time.time() - t0:.0f}s", flush=True)
    OUT.mkdir(exist_ok=True)
    (OUT / "goals_replies.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    np.savez_compressed(OUT / "goals_states.npz", **arrays)
    # For the blind reader: answers only, shuffled, no condition.
    order = np.random.default_rng(0).permutation(len(rows))
    blind = [{"key": f"r{j:03d}", "question": rows[i]["question"], "answer": rows[i]["answer"]} for j, i in enumerate(order)]
    (OUT / "goals_blind.json").write_text(json.dumps(blind, indent=1))
    (OUT / "goals_blind_key.json").write_text(json.dumps({f"r{j:03d}": rows[i]["id"] for j, i in enumerate(order)}))
    print(f"wrote {len(rows)} replies")


if __name__ == "__main__":
    main()
