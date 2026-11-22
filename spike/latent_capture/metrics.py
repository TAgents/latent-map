"""Pure functions over measurement numbers and captured states."""

import numpy as np

MIN_SPEED_RATIO = 0.8
MAX_PEAK_GB = 26.0
MAX_SWAP_GROWTH_GB = 0.1


def summarize(runs: list[dict]) -> dict:
    def tps(mode: str) -> float:
        sel = [r for r in runs if r["mode"] == mode]
        return sum(r["tokens"] for r in sel) / sum(r["seconds"] for r in sel)

    off, on = tps("off"), tps("on")
    return {
        "tps_off": off,
        "tps_on": on,
        "speed_ratio": on / off,
        "peak_gb": max(r["peak_gb"] for r in runs),
        "swap_growth_gb": max(r["swap_gb_after"] - r["swap_gb_before"] for r in runs),
    }


def verdict(summary: dict) -> str:
    ok = (
        summary["speed_ratio"] >= MIN_SPEED_RATIO
        and summary["peak_gb"] <= MAX_PEAK_GB
        and summary["swap_growth_gb"] <= MAX_SWAP_GROWTH_GB
    )
    return "go" if ok else "no-go"


def top_coordinate_share(x: np.ndarray) -> tuple[int, float]:
    """Coordinate with the largest mean squared value, and its share of the total."""
    power = np.mean(np.square(x.astype(np.float64)), axis=0)
    idx = int(np.argmax(power))
    return idx, float(power[idx] / power.sum())


def pass_movement(passes: np.ndarray) -> np.ndarray:
    """Mean distance between consecutive passes of unit-normalised states.
    passes: (n, n_passes, d) -> (n_passes - 1,)"""
    p = passes.astype(np.float64)
    p = p / np.linalg.norm(p, axis=-1, keepdims=True)
    return np.linalg.norm(p[:, 1:] - p[:, :-1], axis=-1).mean(axis=0)


def pca_2d(x: np.ndarray) -> np.ndarray:
    centred = x.astype(np.float64) - x.mean(axis=0, keepdims=True)
    u, s, _ = np.linalg.svd(centred, full_matrices=False)
    return u[:, :2] * s[:2]
