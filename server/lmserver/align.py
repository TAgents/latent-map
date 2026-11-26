"""Which captured forward calls belong to generated words."""


def first_generated(call_sizes: list[int], uncached: int) -> int:
    """Index of the first forward call that read a generated word.

    The runner first reads the prompt tokens the cache did not already hold
    (`uncached` of them, in calls of any size); every call after that read exactly
    one generated word, in order. Checked against plain passes in server/step0."""
    used = index = 0
    while index < len(call_sizes) and used < uncached:
        used += call_sizes[index]
        index += 1
    return index


class PromptPieces:
    """The sizes of the pieces each prompt was read in.

    The model's output shifts with the size of the pieces a text is read in, so a readout
    has to re-read a prompt in the same pieces as generation did. A prompt that continues
    a saved cache was read partly by an earlier request; this keeps the pieces of the
    latest saved prompts (numbers only, in memory) so they can be put together."""

    def __init__(self, keep: int = 256):
        self._keep = keep
        self._known: dict = {}

    @staticmethod
    def _key(ids):
        return (len(ids), hash(tuple(ids)))

    def remember(self, ids: list[int], pieces: list[int]) -> None:
        self._known.pop(self._key(ids), None)
        self._known[self._key(ids)] = list(pieces)
        while len(self._known) > self._keep:
            del self._known[next(iter(self._known))]

    def forget(self, ids: list[int]) -> None:
        self._known.pop(self._key(ids), None)

    def of(self, prompt: list[int], uncached: int, sizes: list[int]) -> list[int] | None:
        """Pieces of the whole prompt: those of its cached part, then the calls that read
        the rest. None when they are not known."""
        cached = len(prompt) - uncached
        before = self._known.get(self._key(prompt[:cached])) if cached else []
        if before is None or sum(before) != cached or sum(sizes) != uncached:
            return None
        return before + list(sizes)
