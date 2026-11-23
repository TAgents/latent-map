import math

import numpy as np
import pytest

from mapbuild.areas import lift
from mapbuild.stats import lift_null, pass_agreement


def clumped(seed=0, texts=40, per=50):
    """Each text sits in one random area, so areas are clumped by text."""
    r = np.random.default_rng(seed)
    groups = np.repeat(np.arange(texts), per)
    area = np.repeat(r.integers(0, 6, size=texts), per)
    return r, groups, area


def test_null_matches_observed_when_labels_are_unrelated_to_areas():
    r, groups, area = clumped()
    labels = np.repeat(r.integers(0, 8, size=40), 50).astype(str).tolist()
    mean, sd = lift_null(area, labels, groups, n=100)
    assert lift(area, labels) > 1.2  # the raw statistic is inflated by clumping
    assert abs(lift(area, labels) - mean) < 3 * sd + 0.05


def test_null_is_far_below_observed_when_labels_follow_areas():
    _, groups, area = clumped()
    labels = area.astype(str).tolist()
    mean, sd = lift_null(area, labels, groups, n=100)
    assert lift(area, labels) > mean + 5 * sd


def test_null_permutes_only_within_strata():
    groups = np.repeat(np.arange(4), 5)
    area = np.repeat([0, 0, 1, 1], 5)
    labels = np.repeat(["a", "b", "c", "d"], 5).tolist()
    strata = np.repeat(["s0", "s1", "s2", "s3"], 5)  # one text per stratum: nothing can move
    mean, sd = lift_null(area, labels, groups, strata=strata, n=20)
    assert mean == pytest.approx(lift(area, labels)) and sd == pytest.approx(0.0)


def test_lift_of_nothing_is_nan():
    assert math.isnan(lift(np.array([], dtype=int), []))


def test_pass_agreement():
    cell = np.array([[0, 1, 1], [2, 2, 2], [3, 0, 1], [0, 0, 0]])
    fine = np.array([0, 0, 1, 1])
    out = pass_agreement(cell, fine)
    assert out["same_cell"] == [0.5, 0.75, 1.0]
    assert out["same_area"] == [0.75, 1.0, 1.0]
