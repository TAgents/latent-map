"""Pure functions for the hidden-goal test."""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from siglib.items_goals import SHOWN


def shown(goal: str, text: str) -> bool:
    """Does the text show the goal, by the word patterns fixed before the run?"""
    return bool(SHOWN[goal].search(text))


def _fit_scores(x_train, y_train, x_test) -> np.ndarray:
    mean, sd = x_train.mean(0), x_train.std(0) + 1e-6
    clf = LogisticRegression(C=0.1, max_iter=3000).fit((x_train - mean) / sd, y_train)
    return clf.decision_function((x_test - mean) / sd)


def leave_goal_out_auc(x, goals, qfold) -> float:
    """Can a classifier tell 'a goal is in play' for a goal and questions it never saw?
    For each goal and each half of the questions: train on the other goals plus control on
    the other half of the questions; test on that goal plus control on this half. Mean AUC."""
    x, goals, qfold = np.asarray(x, dtype=np.float64), np.asarray(goals), np.asarray(qfold)
    y = (goals != "control").astype(int)
    aucs = []
    for g in sorted(set(goals) - {"control"}):
        for f in sorted(set(qfold)):
            train = (qfold != f) & (goals != g)
            test = (qfold == f) & ((goals == g) | (goals == "control"))
            if len(set(y[train])) < 2 or len(set(y[test])) < 2:
                continue
            aucs.append(roc_auc_score(y[test], _fit_scores(x[train], y[train], x[test])))
    return float(np.mean(aucs)) if aucs else float("nan")


def leave_question_out_auc(x, y, qids) -> float:
    """Pooled AUC with every reply to a question held out together."""
    x, y, qids = np.asarray(x, dtype=np.float64), np.asarray(y).astype(int), np.asarray(qids)
    scores = np.zeros(len(y))
    for q in np.unique(qids):
        test = qids == q
        if len(set(y[~test])) < 2:
            return float("nan")
        scores[test] = _fit_scores(x[~test], y[~test], x[test])
    return float(roc_auc_score(y, scores)) if len(set(y)) == 2 else float("nan")
