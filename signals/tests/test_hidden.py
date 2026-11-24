import numpy as np
import pytest

from siglib.goals import leave_goal_out_auc, leave_question_out_auc, shown
from siglib.hops import first_pass_in_top, matches, step_readable
from siglib.items_goals import conditions, questions
from siglib.items_hops import build


def test_items_are_complete():
    items = build()
    assert len(items) == 80 and len({i["id"] for i in items}) == 80
    assert all(i["subject"] in i["two_step"] and i["bridge"] in i["second_step"] for i in items)
    assert len(questions()) == 30 and set(conditions()) == {"control", "water", "outdoors", "books", "sleep"}


def test_matches_ignores_case_accents_and_known_aliases():
    assert matches("brasilia.", "Brasília") and matches("It is New Delhi", "New Delhi") and matches("Delhi", "New Delhi")
    assert matches("the UK", "United Kingdom") and matches("USA", "United States") and matches("Gabriel García Marquez", "Márquez")
    assert not matches("Paris", "Warsaw") and not matches("", "Rome")


def test_step_readable_needs_a_top_ten_rank_that_beats_other_items_tenfold():
    assert step_readable(own_best=3, others_best=[400, 900, 80])          # rank 3 against a median of 400
    assert not step_readable(own_best=3, others_best=[5, 12, 30])         # the word ranks high everywhere
    assert not step_readable(own_best=25, others_best=[4000, 9000])       # never in the top ten
    assert not step_readable(own_best=3, others_best=[])                  # nothing to compare with


def test_first_pass_in_top():
    assert first_pass_in_top([40, 7, 0]) == 1 and first_pass_in_top([2, 7, 0]) == 0 and first_pass_in_top([40, 70, 11]) is None


def test_shown_reads_goal_words():
    assert shown("water", "Keep a glass of water nearby.") and shown("water", "Stay hydrated.")
    assert not shown("water", "Practise chords daily.") and shown("books", "Try reading a chapter.") and not shown("books", "Already done.")
    assert shown("sleep", "Get a good night's sleep.") and shown("outdoors", "Step outside for a bit.")


def make(seed=0, signal=True):
    r = np.random.default_rng(seed)
    goals = np.array(["control", "water", "books", "sleep", "outdoors"] * 30)
    qfold = np.repeat(np.arange(30) % 2, 5)
    x = r.normal(size=(150, 12))
    if signal:
        x[:, 0] += (goals != "control") * 3.0          # one shared "a goal is in play" axis
    return x, goals, qfold


def test_leave_goal_out_auc_finds_a_shared_goal_axis_and_not_noise():
    x, goals, qfold = make()
    assert leave_goal_out_auc(x, goals, qfold) > 0.9
    x, goals, qfold = make(signal=False)
    assert 0.3 < leave_goal_out_auc(x, goals, qfold) < 0.7


def test_leave_goal_out_auc_does_not_reward_goal_specific_features():
    r = np.random.default_rng(1)
    x, goals, qfold = make(signal=False)
    for j, g in enumerate(["water", "books", "sleep", "outdoors"]):
        x[:, j] += (goals == g) * 5.0                  # each goal has its own axis: nothing transfers
    assert leave_goal_out_auc(x, goals, qfold) < 0.75


def test_leave_question_out_auc():
    r = np.random.default_rng(2)
    qids = np.repeat(np.arange(30), 4)
    y = r.integers(0, 2, size=120)
    x = r.normal(size=(120, 8)); x[:, 0] += y * 3.0
    assert leave_question_out_auc(x, y, qids) > 0.9
    assert leave_question_out_auc(r.normal(size=(120, 8)), y, qids) < 0.7
