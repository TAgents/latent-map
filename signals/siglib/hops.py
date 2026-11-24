"""Pure functions for the unwritten-step test."""

import unicodedata

import numpy as np

ALIASES = {
    "united kingdom": ["uk", "u.k.", "britain", "england"], "united states": ["usa", "u.s.", "america", "us"],
    "new delhi": ["delhi"], "yuan": ["renminbi", "rmb"], "england": ["united kingdom", "uk", "britain"],
    "netherlands": ["holland"], "czech republic": ["czechia"], "united arab emirates": ["uae", "u.a.e."],
    "south korea": ["korea"], "pound": ["sterling"], "real": ["reais"], "dollar": ["dollars"], "ruble": ["rouble"],
    "marquez": ["garcia marquez"], "cervantes": ["miguel de cervantes"],
}
TOP = 10


def plain(text: str) -> str:
    text = unicodedata.normalize("NFKD", text)
    return "".join(c for c in text if not unicodedata.combining(c)).lower()


def matches(text: str, target: str) -> bool:
    """Does the text contain the target, ignoring case and accents, or a known alias of it?"""
    t, want = plain(text), plain(target)
    if not t.strip():
        return False
    words = set("".join(c if c.isalnum() or c == "." else " " for c in t).split())
    return want in t or any((a in words) if " " not in a else (a in t) for a in ALIASES.get(want, []))


def step_readable(own_best: int, others_best: list[int]) -> bool:
    """T1 for one item. Ranks are 0 for the most likely word. The step's word must reach the
    top ten, and do at least ten times better than its median best rank in other items."""
    if own_best >= TOP or not len(others_best):
        return False
    return (float(np.median(others_best)) + 1) / (own_best + 1) >= 10


def first_pass_in_top(best_by_pass) -> int | None:
    """Index of the first pass whose best rank is in the top ten, or None."""
    return next((k for k, r in enumerate(best_by_pass) if r < TOP), None)
