"""Sanity-check captured block 2 and block 3 states and draw a first rough map.

States are taken at the last prompt token. They are already RMS-normalised by
the model, so a coordinate's share of the squared norm is mean(x_i^2) / d.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlx.core as mx
import numpy as np

from latent_capture.capture import capturing
from latent_capture.loader import load_model
from latent_capture.metrics import pass_movement, pca_2d, top_coordinate_share, topic_separation
from prompts import PROMPTS

RESULTS = Path(__file__).parent / "results"
COORD = 2027  # the dominant coordinate Goodfire reports


def describe(states: np.ndarray, topics: list[str]) -> dict:
    """states: (n, passes, d)"""
    n_pass = states.shape[1]
    rest = np.delete(states, COORD, axis=-1)
    return {
        "top_coordinate_per_pass": [
            dict(zip(("index", "share"), top_coordinate_share(states[:, k]))) for k in range(n_pass)
        ],
        "coord_2027_share_per_pass": [float(np.mean(states[:, k, COORD] ** 2) / states.shape[-1]) for k in range(n_pass)],
        "coord_2027_mean_per_pass": [float(states[:, k, COORD].mean()) for k in range(n_pass)],
        "pass_movement": [float(v) for v in pass_movement(states)],
        "pass_movement_without_2027": [float(v) for v in pass_movement(rest)],
        "topic_separation_per_pass": [topic_separation(rest[:, k], topics) for k in range(n_pass)],
    }


def main() -> None:
    model, tokenizer = load_model()
    topics, block2, block3, layout = [], [], [], None
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
                block2.append(np.stack(rec.block2))
                block3.append(np.stack(rec.block3))
                topics.append(topic)
                cap.clear()
    blocks = {"block2": np.stack(block2), "block3": np.stack(block3)}  # (n, 3, d)

    names = list(PROMPTS)
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    for row, (name, states) in enumerate(blocks.items()):
        rest = np.delete(states, COORD, axis=-1)
        for k in range(3):
            xy = pca_2d(rest[:, k])  # fitted per pass, so axes are not spent on drift between passes
            ax = axes[row, k]
            for t in names:
                sel = [i for i, label in enumerate(topics) if label == t]
                ax.scatter(xy[sel, 0], xy[sel, 1], label=t, s=40)
            ax.set_title(f"{name} output after pass {k + 1}")
    axes[0, 0].legend()
    fig.tight_layout()
    RESULTS.mkdir(exist_ok=True)
    fig.savefig(RESULTS / "pca.png", dpi=110)
    np.savez_compressed(RESULTS / "states.npz", topics=np.array(topics), **blocks)

    out = {"layout": layout, "hidden_size": int(blocks["block3"].shape[-1]), "prompts": len(topics)}
    for name, states in blocks.items():
        out[name] = describe(states, topics)
    (RESULTS / "states.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
