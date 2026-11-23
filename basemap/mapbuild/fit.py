"""Fit the map: drop the loop-signal coordinate, normalise, remove each pass's
mean (and optionally its overall spread), project with PCA, split into cells
with k-means, lay the cells out in 2D."""

import numpy as np
import umap
from sklearn.cluster import KMeans


def _prepare(states: np.ndarray, drop: int) -> np.ndarray:
    x = np.delete(states.astype(np.float32), drop, axis=-1)
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-8)


def _project(m: dict, states: np.ndarray) -> np.ndarray:
    """(N, passes, d) -> (N, passes, dims)"""
    x = _prepare(states, int(m["drop"])) - m["pass_means"]
    if "pass_scales" in m:  # absent in v0 bundles
        x = x / np.asarray(m["pass_scales"], dtype=np.float32)[:, None]
    return x @ m["components"].T


def fit_map(
    states: np.ndarray, dims: int = 64, cells: int = 256, drop: int = 2027, seed: int = 0, scale_passes: bool = False
) -> dict:
    """states: (N, passes, d). With scale_passes, each pass is also divided by its
    root-mean-square distance from its own mean, so a pass that spreads wider than
    the others does not take most of the cells."""
    x = _prepare(states, drop)
    pass_means = x.mean(axis=0)
    centred = x - pass_means
    if scale_passes:
        pass_scales = np.sqrt(np.square(centred).sum(axis=-1).mean(axis=0)).astype(np.float32)
    else:
        pass_scales = np.ones(x.shape[1], dtype=np.float32)
    flat = (centred / pass_scales[:, None]).reshape(-1, x.shape[-1])
    dims = min(dims, flat.shape[1], flat.shape[0])
    cells = min(cells, flat.shape[0])
    _, _, vt = np.linalg.svd(flat - flat.mean(axis=0), full_matrices=False)
    m = {"drop": np.int64(drop), "pass_means": pass_means, "pass_scales": pass_scales, "components": vt[:dims]}
    z = flat @ m["components"].T
    m["centres"] = KMeans(n_clusters=cells, n_init=3, random_state=seed).fit(z).cluster_centers_.astype(np.float32)
    neighbours = max(2, min(15, cells - 1))
    init = "spectral" if cells > 8 else "random"
    m["xy"] = umap.UMAP(n_neighbors=neighbours, min_dist=0.3, init=init, random_state=seed).fit_transform(m["centres"])
    return m


def place(m: dict, states: np.ndarray) -> np.ndarray:
    """Cell id for every point and pass, using the stored pass means. (N, passes)"""
    z = _project(m, states)
    c = m["centres"]
    out = np.empty(z.shape[:2], dtype=np.int32)
    for k in range(z.shape[1]):
        d = (z[:, k] ** 2).sum(1, keepdims=True) - 2 * z[:, k] @ c.T + (c**2).sum(1)
        out[:, k] = d.argmin(axis=1)
    return out
