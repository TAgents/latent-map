"""Group cells into nested areas, find each area's distinctive words, and
measure how well areas line up with known labels."""

from collections import Counter

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage


def group_cells(centres: np.ndarray, n_fine: int = 12, n_broad: int = 4) -> tuple[np.ndarray, np.ndarray]:
    """Fine and broad area id per cell, cut from one Ward tree so fine areas nest."""
    tree = linkage(centres, method="ward")
    fine = fcluster(tree, t=n_fine, criterion="maxclust") - 1
    broad = fcluster(tree, t=n_broad, criterion="maxclust") - 1
    return fine, broad


def is_marker(token: str) -> bool:
    """A chat-format marker such as <|content_text|>, not a word."""
    w = token.strip()
    return w.startswith("<|") and w.endswith("|>")


def _word(token: str) -> str | None:
    w = token.strip().lower()
    if is_marker(w):
        return None
    return w if any(ch.isalpha() for ch in w) else None


def distinctive_words(
    tokens: list[str], area_of_point: np.ndarray, top: int = 12, min_count: int = 3, areas=None, method: str = "share"
) -> dict[int, list[str]]:
    """Per area, its most distinctive words. Pass `areas` to get an entry (possibly
    empty) for areas that hold no points.

    method "share": share inside the area minus share overall. Simple, but common
    words such as "the" rank high wherever they are slightly over-represented.
    method "logodds": how many standard errors the word's log-odds inside the area
    sit above its log-odds everywhere else, so common words need a large excess."""
    words = [_word(t) for t in tokens]
    overall = Counter(w for w in words if w)
    total = sum(overall.values()) or 1
    out = {}
    for a in sorted(set(int(v) for v in area_of_point) | set(int(v) for v in (areas if areas is not None else []))):
        inside = Counter(w for w, v in zip(words, area_of_point) if w and v == a)
        n = sum(inside.values()) or 1
        if method == "logodds":
            scored = []
            for w, c in inside.items():
                if c < min_count:
                    continue
                rest, rest_n = overall[w] - c, total - n
                delta = np.log((c + 0.5) / (n - c + 0.5)) - np.log((rest + 0.5) / (rest_n - rest + 0.5))
                scored.append((float(delta / np.sqrt(1 / (c + 0.5) + 1 / (rest + 0.5))), w))
        else:
            scored = [(c / n - overall[w] / total, w) for w, c in inside.items() if c >= min_count]
        floor = 1.0 if method == "logodds" else 0.0  # log-odds: at least one standard error above the rest
        out[a] = [w for s, w in sorted(scored, key=lambda p: (-p[0], p[1])) if s > floor][:top]
    return out


def lift(area_of_point: np.ndarray, labels: list[str]) -> float:
    """Mean share of the commonest label within an area (weighted by area size),
    divided by the share of the commonest label overall. 1.0 means no better
    than ignoring the areas. Tokens from one text sit together, so even unrelated
    labels score above 1.0; compare against mapbuild.stats.lift_null."""
    if len(labels) == 0:
        return float("nan")
    lab = np.array(labels)
    base = Counter(labels).most_common(1)[0][1] / len(labels)
    hit = sum(Counter(lab[area_of_point == a].tolist()).most_common(1)[0][1] for a in set(area_of_point.tolist()))
    return float(hit / len(labels) / base)
