"""Draw each block's map with its named areas, and how cells change by pass.

Run: ../spike/.venv/bin/python draw_map.py [tag] [states_tag]
"""

import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from mapbuild.paths import paths



def label(info: dict, a: str) -> str:
    name = info.get("names", {}).get(a, "")
    if not name or info.get("weak", {}).get(a, True):
        parent = info.get("broad_names", {}).get(str(info["broad_of_fine"][a]), "")
        return f"part of {parent}" if parent else f"area {a}"
    return name


def main() -> None:
    tag = sys.argv[1] if len(sys.argv) > 1 else "v1"
    BUNDLE = paths(tag)["bundle"]
    states = np.load(paths(sys.argv[2] if len(sys.argv) > 2 else tag)["states"])
    first = np.load(BUNDLE / "block2.npz")
    reply = first["used"] if "used" in first else states["is_reply"]
    fig2, axes2 = plt.subplots(2, 3, figsize=(15, 9))
    for row, block in enumerate(("block2", "block3")):
        b = np.load(BUNDLE / f"{block}.npz")
        info = json.loads((BUNDLE / f"{block}.json").read_text())
        xy, fine, cell = b["xy"], b["fine"], b["cell"][reply]
        n_cells = len(xy)
        count = np.bincount(cell[:, -1], minlength=n_cells)
        fig, ax = plt.subplots(figsize=(11, 9))
        cmap = plt.get_cmap("tab20")
        ax.scatter(xy[:, 0], xy[:, 1], s=8 + 3 * np.sqrt(count) * 4, c=[cmap(f % 20) for f in fine],
                   alpha=0.75, linewidths=0.4, edgecolors="white")
        for a in sorted(set(fine.tolist())):
            sel = fine == a
            w = np.maximum(count[sel], 1)
            cx, cy = np.average(xy[sel, 0], weights=w), np.average(xy[sel, 1], weights=w)
            score = info.get("name_scores", {}).get(str(a))
            text = label(info, str(a)) + (f"\n({score:.2f})" if score is not None else "")
            ax.text(cx, cy, text, ha="center", va="center", fontsize=9, weight="bold",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=cmap(a % 20), alpha=0.85))
        ax.set_title(f"Base map {tag}, {block} output at pass 3: {n_cells} cells, sized by reply tokens, "
                     f"coloured by fine area\n(name score in brackets; chance is 0.25)")
        ax.set_xticks([]); ax.set_yticks([])
        fig.tight_layout(); fig.savefig(BUNDLE / f"{block}.png", dpi=130); plt.close(fig)

        for k in range(3):
            moved = np.zeros(n_cells); total = np.zeros(n_cells)
            np.add.at(total, cell[:, k], 1)
            np.add.at(moved, cell[:, k], cell[:, k] != cell[:, -1])
            share = np.divide(moved, total, out=np.zeros(n_cells), where=total > 0)
            ax = axes2[row, k]
            sc = ax.scatter(xy[:, 0], xy[:, 1], s=6 + 10 * np.sqrt(total), c=share, cmap="viridis", vmin=0, vmax=1, alpha=0.85)
            overall = float((cell[:, k] != cell[:, -1]).mean())
            ax.set_title(f"{block}, pass {k + 1}: {overall:.0%} not yet in final cell", fontsize=10)
            ax.set_xticks([]); ax.set_yticks([])
        fig2.colorbar(sc, ax=axes2[row, :], label="share of the cell's tokens that end in a different cell", shrink=0.8)
    fig2.savefig(BUNDLE / "passes.png", dpi=120, bbox_inches="tight")


if __name__ == "__main__":
    main()
