"""Have the model name each area, then score each name with a second run.

Run: PYTHONPATH=../spike ../spike/.venv/bin/python name_areas.py
"""

import json
from pathlib import Path

from mlx_lm import generate

from latent_capture.loader import load_model
from mapbuild.naming import answer_text, name_area, score_name

BUNDLE = Path(__file__).parent / "bundle_v0"
WEAK = 0.5


def main() -> None:
    model, tokenizer = load_model()

    def ask(prompt: str) -> str:
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], add_generation_prompt=True, tokenize=False, reasoning_effort=0.0
        )
        # The model thinks before it answers; allow room for that and keep only the answer.
        return answer_text(generate(model, tokenizer, text, max_tokens=400, verbose=False))

    for block in ("block2", "block3"):
        path = BUNDLE / f"{block}.json"
        info = json.loads(path.read_text())
        # Name from every other word, score on the rest, so the scorer never sees the naming words.
        seen = {a: w[0::2] for a, w in info["words"].items()}
        held = {a: w[1::2] for a, w in info["words"].items()}
        names, scores = {}, {}
        for a in info["words"]:
            names[a] = name_area(ask, seen[a])
            scores[a] = score_name(ask, names[a], held[a], [held[b] for b in held if b != a], seed=int(a))
            print(f"{block} area {a}: {names[a]!r} score {scores[a]:.2f}  <- {', '.join(seen[a][:6])}", flush=True)
        broad_names = {}
        for b in sorted(set(info["broad_of_fine"].values())):
            # Named from the top words of its fine areas, so it does not depend on their names.
            parts = [w for a, bb in info["broad_of_fine"].items() if bb == b for w in info["words"][a][:5]]
            broad_names[str(b)] = name_area(ask, parts)
        info.update(
            names=names, name_scores=scores, broad_names=broad_names,
            weak={a: scores[a] < WEAK for a in names},
            mean_name_score=sum(scores.values()) / len(scores),
        )
        path.write_text(json.dumps(info, indent=1))
        print(f"{block}: mean name score {info['mean_name_score']:.2f}; broad areas {broad_names}", flush=True)


if __name__ == "__main__":
    main()
