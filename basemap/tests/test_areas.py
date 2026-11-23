import numpy as np
import pytest

from mapbuild.areas import distinctive_words, group_cells, lift


def test_fine_areas_nest_inside_broad_areas():
    centres = np.random.default_rng(0).normal(size=(40, 6))
    fine, broad = group_cells(centres, n_fine=8, n_broad=3)
    assert len(set(fine)) == 8 and len(set(broad)) == 3
    assert all(len(set(broad[fine == f])) == 1 for f in set(fine))


def test_distinctive_words_and_empty_area():
    toks = [" def", "def", "return", " the", "the", "the", "!!", "cat", "cat", "cat", "the", "x"]
    area = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 2])
    w = distinctive_words(toks, area, top=2, min_count=2)
    assert w[0][0] == "def" and w[1][0] == "cat" and w[2] == []


def test_lift_is_one_for_random_and_high_for_aligned():
    labels = ["a"] * 50 + ["b"] * 50
    assert lift(np.array([0] * 50 + [1] * 50), labels) == pytest.approx(2.0)
    assert lift(np.array([0, 1] * 50), labels) == pytest.approx(1.0)


def test_distinctive_words_ignores_special_markers():
    toks = ["<|content_text|>"] * 5 + ["cat"] * 5 + ["dog"] * 5
    w = distinctive_words(toks, np.array([0] * 10 + [1] * 5), top=3, min_count=2)
    assert w[0] == ["cat"]


def test_distinctive_words_can_be_asked_for_areas_with_no_points():
    w = distinctive_words(["cat"] * 3, np.array([0, 0, 0]), top=2, min_count=1, areas=[0, 1])
    assert w == {0: [], 1: []} or (w[0] == [] and w[1] == [])
