"""Have the model write the v0 corpus: one short text per topic and form.

The model writes a thinking section before its answer, so the token limit is
high enough for both sections to appear. The chat template's setting is called
reasoning_effort; 0.9 is its default and is what the v0 corpus was written with.

Run: PYTHONPATH=../spike ../spike/.venv/bin/python generate_corpus.py
"""

import json
from pathlib import Path

from mlx_lm import stream_generate

from latent_capture.loader import load_model
from mapbuild.corpus import build_prompts

OUT = Path(__file__).parent / "data" / "corpus.jsonl"
MAX_TOKENS = 300
REASONING_EFFORT = 0.9


def chat_prompt(tokenizer, text: str) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": text}], add_generation_prompt=True, tokenize=False, reasoning_effort=REASONING_EFFORT
    )


def main() -> None:
    model, tokenizer = load_model()
    written = 0
    print("prompt head:", repr(chat_prompt(tokenizer, "x")[:90]), flush=True)
    OUT.parent.mkdir(exist_ok=True)
    with OUT.open("w") as f:
        for item in build_prompts():
            text = "".join(
                r.text for r in stream_generate(model, tokenizer, chat_prompt(tokenizer, item["prompt"]), max_tokens=MAX_TOKENS)
            )
            if not text.strip():
                print(f"skipped {item['id']}: empty generation", flush=True)
                continue
            f.write(json.dumps({**item, "reasoning_effort": REASONING_EFFORT, "text": text}) + "\n")
            f.flush()
            written += 1
            if written % 20 == 0:
                print(f"{written} texts", flush=True)
    print(f"wrote {written} texts to {OUT}")


if __name__ == "__main__":
    main()
