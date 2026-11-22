"""Sanity-check captured block 3 states and draw a first rough map."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlx.core as mx
import numpy as np

from latent_capture.capture import capturing
from latent_capture.loader import load_model
from latent_capture.metrics import pass_movement, pca_2d, top_coordinate_share
from prompts import PROMPTS

GOODFIRE_COORDINATE = 2026  # "d_2027" in Goodfire's 1-based numbering; checked both ways below
RESULTS = Path(__file__).parent / "results"


def rms_normalise(x: np.ndarray) -> np.ndarray:
    return x / np.sqrt(np.mean(np.square(x), axis=-1, keepdims=True))


def main() -> None:
    model, tokenizer = load_model()
    topics, states, layout = [], [], None
    with capturing(model) as cap:
        for topic, prompts in PROMPTS.items():
            for text in prompts:
                ids = tokenizer.apply_chat_template(
                    [{"role": "user", "content": text}], add_generation_prompt=True
                )
                model(mx.array([ids]))
                rec = cap.records[-1]
                layout = {
                    "prelude_rows": int(rec.prelude.shape[0]),
                    "pass_rows": [int(p.shape[0]) for p in rec.passes],
                    "final_rows": int(rec.final.shape[0]),
                }
                states.append(np.stack([p[-1] for p in rec.passes]))  # block 3 = last row
                topics.append(topic)
                cap.clear()
    states = np.stack(states)  # (n, 3, d)
    normed = rms_normalise(states)
    flat = normed.reshape(-1, normed.shape[-1])
    idx, share = top_coordinate_share(flat)
    per_pass = [top_coordinate_share(normed[:, k]) for k in range(3)]
    mean_at = {str(i): [float(normed[:, k, i].mean()) for k in range(3)] for i in sorted({idx, 2026, 2027})}
    movement = pass_movement(normed)

    reduced = np.delete(flat, idx, axis=1)
    xy = pca_2d(reduced).reshape(len(topics), 3, 2)
    names = list(PROMPTS)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True)
    for k, ax in enumerate(axes):
        for t in names:
            sel = [i for i, name in enumerate(topics) if name == t]
            ax.scatter(xy[sel, k, 0], xy[sel, k, 1], label=t, s=40)
        ax.set_title(f"block 3 after pass {k + 1}")
    axes[0].legend()
    fig.tight_layout()
    RESULTS.mkdir(exist_ok=True)
    fig.savefig(RESULTS / "pca.png", dpi=120)
    np.savez_compressed(RESULTS / "states.npz", states=states, topics=np.array(topics))

    out = {
        "layout": layout,
        "hidden_size": int(states.shape[-1]),
        "top_coordinate_index_0_based": idx,
        "top_coordinate_share": share,
        "top_coordinate_per_pass": [{"index": i, "share": s} for i, s in per_pass],
        "mean_normalised_value_by_pass": mean_at,
        "pass_movement": [float(v) for v in movement],
    }
    (RESULTS / "states.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
