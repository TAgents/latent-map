"""Have a model name an area from its words, and score the name by whether a
second run can pick the area's own held-out words from the name alone."""

import random
import re
from typing import Callable

LETTERS = "ABCD"
ANSWER_MARK, END_MARK, THINK_MARK = "<|content_text|>", "<|end_message|>", "<|content_thinking|>"


def answer_text(raw: str) -> str:
    """The answer section of a raw generation. Empty when the model was still
    thinking when it was cut off; the whole text when there are no markers."""
    if ANSWER_MARK in raw:
        return raw.split(ANSWER_MARK, 1)[1].split(END_MARK, 1)[0].strip()
    return "" if THINK_MARK in raw else raw.strip()


def pick_letter(reply: str) -> str | None:
    """The option letter a reply chooses: the first standalone capital A to D,
    or a reply that is nothing but one letter. None when there is no clear pick."""
    found = re.search(r"(?<![A-Za-z])([A-D])(?![A-Za-z])", reply)
    if found:
        return found.group(1)
    bare = re.fullmatch(r"\W*([a-d])\W*", reply.strip())
    return bare.group(1).upper() if bare else None


def name_area(ask: Callable[[str], str], words: list[str]) -> str:
    if len(words) < 3:
        return ""
    reply = ask(
        "These words are typical of one group of contexts in a language model:\n"
        + ", ".join(words)
        + "\nGive a short name for what they have in common, at most five words. Reply with the name only."
    )
    lines = [ln.strip() for ln in reply.strip().splitlines() if ln.strip()]
    if not lines:
        return ""
    name = lines[0].strip(" \"'`*").rstrip(".:;,!").strip(" \"'`*")
    return " ".join(name.split()[:6])


def score_name(ask: Callable[[str], str], name: str, own: list[str], others: list[list[str]], trials: int = 4, seed: int = 0) -> float:
    """Share of trials in which the reply picks the letter of the area's own words.
    This measures whether the name tells this area's words apart from other areas',
    not whether the name is a correct description."""
    others = [o for o in others if o]
    if not name or not own or not others:
        return 0.0
    rng = random.Random(seed)
    right = 0
    for _ in range(trials):
        options = [own] + rng.sample(others, min(3, len(others)))
        rng.shuffle(options)
        answer = LETTERS[options.index(own)]
        listing = "\n".join(f"{LETTERS[i]}. {', '.join(o)}" for i, o in enumerate(options))
        reply = ask(
            f'A group of contexts is named "{name}". Which word list belongs to it?\n{listing}\n'
            "Reply with one letter only."
        )
        right += pick_letter(reply) == answer
    return right / trials
