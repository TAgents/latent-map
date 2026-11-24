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
