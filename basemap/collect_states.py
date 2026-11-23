"""One forward pass per corpus text, keeping block 2 and block 3 outputs at
every token position for each loop pass.

Run: PYTHONPATH=../spike ../spike/.venv/bin/python collect_states.py
"""

import json
from pathlib import Path

import mlx.core as mx
import numpy as np

from latent_capture.capture import capturing
from latent_capture.loader import load_model
from mapbuild.collect import stack

DATA = Path(__file__).parent / "data"
ANSWER_MARK = "<|content_text|>"


def main() -> None:
    model, tokenizer = load_model()
    rows = [json.loads(line) for line in (DATA / "corpus.jsonl").open()]
    records = []
    with capturing(model, all_positions=True, max_records=1) as cap:
        for row in rows:
            head = tokenizer.apply_chat_template(
                [{"role": "user", "content": row["prompt"]}], add_generation_prompt=True, tokenize=False,
                reasoning_effort=row["reasoning_effort"],
            )
            head_ids = tokenizer.encode(head)
            reply_ids = tokenizer.encode(row["text"], add_special_tokens=False)
            ids = head_ids + reply_ids
            if not reply_ids:
                print(f"skipped {row['id']}: no reply tokens", flush=True)
                continue
            tokens = [tokenizer.decode([i]) for i in ids]
            # Reply tokens are "thinking" until the answer's content marker, then "answer".
            marks = [i for i, t in enumerate(tokens) if i >= len(head_ids) and t == ANSWER_MARK]
            start = marks[0] + 1 if marks else len(ids)
            phase = ["prompt" if i < len(head_ids) else "thinking" if i < start else "answer" for i in range(len(ids))]
            model(mx.array([ids]))
            rec = cap.records[-1]
            records.append(
                dict(
                    block2=np.stack(rec.block2).astype(np.float16),
                    block3=np.stack(rec.block3).astype(np.float16),
                    tokens=tokens, phase=phase,
                    topic=row["topic"], form=row["form"], text_id=row["id"], n_prompt=len(head_ids),
                )
            )
            cap.clear()
            mx.clear_cache()
    out = stack(records)
    np.savez(DATA / "states.npz", block2=out["block2"], block3=out["block3"], is_reply=out["is_reply"])
    lists = {k: out[k] for k in ("token", "topic", "form", "text_id", "phase")}
    (DATA / "tokens.json").write_text(json.dumps(lists))
    print(f"N = {len(out['token'])} token positions from {len(records)} texts; "
          f"block2 {out['block2'].shape}, block3 {out['block3'].shape}; reply tokens {int(out['is_reply'].sum())}, "
          f"of which answer {out['phase'].count('answer')}")


if __name__ == "__main__":
    main()
