"""Turn the states captured for one generated word into what the page shows."""

import json
from pathlib import Path

import numpy as np

from mapbuild.fit import _prepare, place

COORD = 2027  # the loop-signal coordinate, read on block 3's output


def level(value: float, bands: list[float]) -> str:
    """bands: the 50th, 90th and 99th percentile of movement seen when the map was built."""
    return "usual" if value <= bands[1] else "high" if value <= bands[2] else "far"


class PhaseTracker:
    """Reply text is 'thinking' until the answer's content marker, then 'answer'."""

    def __init__(self) -> None:
        self.phase = "thinking"

    def step(self, text: str) -> tuple[str, bool]:
        """Returns (phase, is_marker) for one token's text."""
        marker = text.startswith("<|") and text.endswith("|>")
        if text == "<|content_text|>":
            self.phase = "answer"
        return self.phase, marker


class LiveMap:
    def __init__(self, m: dict, fine: np.ndarray, norms: dict):
        self.m, self.fine, self.norms = m, fine, norms

    @classmethod
    def load(cls, bundle: Path, norms_path: Path, block: str = "block2") -> "LiveMap":
        b = np.load(bundle / f"{block}.npz")
        m = {k: b[k] for k in ("drop", "pass_means", "components", "centres")}
        return cls(m, b["fine"], json.loads(norms_path.read_text()))

    def describe(self, block2: np.ndarray, block3: np.ndarray) -> dict:
        """block2, block3: (passes, hidden) outputs for one word."""
        cells = place(self.m, block2[None])[0]
        unit = _prepare(block2, int(self.m["drop"]))
        move = np.linalg.norm(unit[1:] - unit[:-1], axis=-1)
        area = int(self.fine[cells[-1]])
        areas = self.fine[cells]
        settled = next(k for k in range(len(cells)) if (areas[k:] == area).all())
        coord = int(self.m["drop"])
        return {
            "cells": [int(c) for c in cells],
            "area": area,
            "move": [float(v) for v in move],
            "level": level(float(move[-1]), self.norms.get(str(area), self.norms["all"])),
            "settled_by": settled + 1,  # first pass from which the word stays in its final area
            "loop": [float(v) for v in np.square(block3[:, coord].astype(np.float64)) / block3.shape[-1]],
        }
