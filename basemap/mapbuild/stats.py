"""Statistics that put the map's headline numbers in context: a chance level
for lift that respects tokens being clumped by text, and agreement between passes."""

import numpy as np

from mapbuild.areas import lift


def lift_null(area_of_point, labels, groups, strata=None, n: int = 200, seed: int = 0) -> tuple[float, float]:
    """Mean and standard deviation of lift when whole groups (texts) swap labels.

    Every point in a group shares one label. Labels are shuffled among groups,
    within a stratum when `strata` is given, so the clumping of points by group
    is kept and only the link between label and area is broken."""
    area_of_point = np.asarray(area_of_point)
    groups = np.asarray(groups)
    labels = np.asarray(labels)
    strata = np.zeros(len(groups), dtype=int) if strata is None else np.asarray(strata)
    ids, first, inverse = np.unique(groups, return_index=True, return_inverse=True)
    group_label, group_stratum = labels[first], strata[first]
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(n):
        shuffled = group_label.copy()
        for s in np.unique(group_stratum):
            idx = np.flatnonzero(group_stratum == s)
            shuffled[idx] = group_label[rng.permutation(idx)]
        values.append(lift(area_of_point, shuffled[inverse].tolist()))
    return float(np.mean(values)), float(np.std(values))


def pass_agreement(cell: np.ndarray, fine: np.ndarray) -> dict:
    """Share of points whose cell, and whose fine area, at each pass equals the last pass's.
    cell: (N, passes) cell ids; fine: area id per cell."""
    last = cell[:, -1]
    return {
        "same_cell": [float((cell[:, k] == last).mean()) for k in range(cell.shape[1])],
        "same_area": [float((fine[cell[:, k]] == fine[last]).mean()) for k in range(cell.shape[1])],
    }
