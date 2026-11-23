"""Have the model write a corpus: one short text per topic and form.

The model writes a thinking section before its answer. v0 was written at the
template's default reasoning effort (0.9) with 300 tokens, and 45 of 120 texts
never reached their answer. Later versions use low effort and a higher limit.
The run can be stopped and restarted: texts already written are kept.

Run: PYTHONPATH=../spike ../spike/.venv/bin/python generate_corpus.py [tag]
"""

import json
import sys

from mlx_lm import stream_generate

from latent_capture.loader import load_model
from mapbuild.corpus import build_prompts
from mapbuild.paths import paths

SETTINGS = {"v0": (0.9, 300)}  # tag -> (reasoning effort, token limit)
DEFAULT = (0.2, 500)
ANSWER_MARK, END_MARK = "<|content_text|>", "<|end_message|>"


def chat_prompt(tokenizer, text: str, effort: float) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": text}], add_generation_prompt=True, tokenize=False, reasoning_effort=effort
    )


def main() -> None:
    tag = sys.argv[1] if len(sys.argv) > 1 else "v1"
    effort, max_tokens = SETTINGS.get(tag, DEFAULT)
    out = paths(tag)["corpus"]
    out.parent.mkdir(exist_ok=True)
    done = {json.loads(line)["id"] for line in out.open()} if out.exists() else set()
    todo = [item for item in build_prompts() if item["id"] not in done]
    print(f"{tag}: {len(done)} texts already written, {len(todo)} to go (effort {effort}, limit {max_tokens})", flush=True)
    if not todo:
        return
    model, tokenizer = load_model()
    head = chat_prompt(tokenizer, "x", effort)
    assert f"Thinking effort level: {effort}" in head or effort == 0.0, f"reasoning effort was not applied: {head[:90]!r}"
    with out.open("a") as f:
        for n, item in enumerate(todo, 1):
            text = "".join(
                r.text for r in stream_generate(model, tokenizer, chat_prompt(tokenizer, item["prompt"], effort), max_tokens=max_tokens)
            )
            if not text.strip():
                print(f"skipped {item['id']}: empty generation", flush=True)
                continue
            answered = ANSWER_MARK in text
            finished = answered and text.rstrip().endswith(END_MARK)
            f.write(json.dumps({**item, "reasoning_effort": effort, "answered": answered, "finished": finished, "text": text}) + "\n")
            f.flush()
            if n % 20 == 0:
                print(f"{n}/{len(todo)} texts", flush=True)
    rows = [json.loads(line) for line in out.open()]
    print(f"{out.name}: {len(rows)} texts, {sum(r.get('answered', False) for r in rows)} reached an answer, "
          f"{sum(r.get('finished', False) for r in rows)} finished it")


if __name__ == "__main__":
    main()
