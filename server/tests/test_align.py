from lmserver.align import first_generated


def test_plain_prompt():
    assert first_generated([23, 1, 1, 1], 24) == 2


def test_long_prompt_in_chunks():
    assert first_generated([2048, 2048, 2048, 1125, 1, 1, 1], 7270) == 5


def test_one_uncached_token():
    assert first_generated([1, 1, 1], 1) == 1


def test_nothing_uncached():
    assert first_generated([1, 1], 0) == 0


def test_fourteen_uncached():
    assert first_generated([13, 1, 1, 1], 14) == 2


def test_no_calls():
    assert first_generated([], 5) == 0


from lmserver.align import PromptPieces


def test_pieces_of_a_fresh_prompt_are_the_calls_that_read_it():
    known = PromptPieces()
    assert known.of(list(range(10)), uncached=10, sizes=[6, 3, 1]) == [6, 3, 1]


def test_a_follow_up_adds_its_calls_to_the_pieces_of_the_cached_part():
    known = PromptPieces()
    first = list(range(10))
    known.remember(first[:-1], known.of(first, 10, [6, 3, 1])[:-1])      # the saved cache holds the prompt less its last token
    second = first + [50, 51, 52, 53]
    assert known.of(second, uncached=5, sizes=[4, 1]) == [6, 3, 4, 1]      # token 9 is read again with the new ones


def test_pieces_are_unknown_when_the_cached_part_is_or_the_prompt_was_not_read_through():
    known = PromptPieces()
    assert known.of(list(range(10)), uncached=4, sizes=[3, 1]) is None     # cached part never seen (e.g. after a restart)
    assert known.of(list(range(10)), uncached=10, sizes=[6]) is None       # stopped while reading the prompt
    known.remember([1, 2, 3], [3])
    assert known.of([1, 2, 4, 9], uncached=1, sizes=[1]) is None           # same length, different tokens


def test_only_the_latest_prompts_are_remembered():
    known = PromptPieces(keep=2)
    for n in (3, 4, 5):
        known.remember(list(range(n)), [n])
    assert known.of(list(range(4)), uncached=1, sizes=[1]) is None         # the first (3 tokens) was dropped
    assert known.of(list(range(5)), uncached=1, sizes=[1]) == [4, 1]


def test_a_forgotten_prompt_no_longer_gives_pieces():
    # The saved cache for a prompt was rebuilt in pieces nobody recorded: the old list must not be reused.
    known = PromptPieces()
    known.remember([1, 2, 3], [3])
    known.forget([1, 2, 3])
    assert known.of([1, 2, 3, 4], uncached=1, sizes=[1]) is None
    known.forget([7, 7])  # unknown: nothing happens
