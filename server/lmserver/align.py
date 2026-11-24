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
