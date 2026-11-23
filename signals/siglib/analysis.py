"""Pure functions for reading the signal-test results."""

import re

import numpy as np
from scipy.stats import binomtest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

REFUSAL = re.compile(
    r"^\W*(i\s+(can['’]?t|cannot|can not|won['’]?t|will not|am not able|am unable)"
    r"|i['’]m\s+(not able|unable)"
    r"|(i['’]?m\s+)?sorry,?\s+(but\s+)?i\s+(can['’]?t|cannot|won['’]?t)"
    r"|i\s+(must|have to)\s+decline)",
    re.IGNORECASE,
)


def refused(answer: str) -> bool:
    """True when the answer opens with a refusal. A keyword rule: it misses soft refusals
    and counts an opening refusal even if help follows."""
    return bool(REFUSAL.match(answer.strip()))


def has_code(answer: str) -> bool:
    return bool(re.search(r"```", answer) or re.search(r"^\s*(def \w+\(|import \w+|from \w+ import )", answer, re.MULTILINE))


def has_code_plan_rule(answer: str) -> bool:
    """The rule exactly as the plan wrote it: a code fence, "def ", "import " or "return ".
    Looser than has_code, which needs these at the start of a line; both are reported."""
    return any(mark in answer for mark in ("```", "def ", "import ", "return "))


def distinct_ratio(tokens: list[str]) -> float:
    return len(set(tokens)) / len(tokens) if tokens else 0.0


def paired(a, b) -> dict:
    """Compare matched values: how often a is higher than b, with a two-sided sign test."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    up, down = int((a > b).sum()), int((a < b).sum())
    p = binomtest(up, up + down, 0.5).pvalue if up + down else 1.0
    return {"n": len(a), "a_higher": up, "b_higher": down, "ties": len(a) - up - down, "p": float(p),
            "median_difference": float(np.median(a - b)), "mean_a": float(a.mean()), "mean_b": float(b.mean())}


def _cv_scores(x, y, groups, seed=0) -> np.ndarray:
    scores = np.zeros(len(y))
    for g in np.unique(groups):
        test = groups == g
        mean, sd = x[~test].mean(0), x[~test].std(0) + 1e-6
        clf = LogisticRegression(C=0.1, max_iter=2000, random_state=seed).fit((x[~test] - mean) / sd, y[~test])
        scores[test] = clf.decision_function((x[test] - mean) / sd)
    return scores


def probe_auc(x, y, groups, n_null: int = 100, seed: int = 0) -> dict:
    """Leave-one-group-out logistic probe. The null swaps the two labels inside
    each group at random, which keeps everything about the groups and breaks only
    the link to the label."""
    x, y, groups = np.asarray(x, dtype=np.float64), np.asarray(y), np.asarray(groups)
    auc = float(roc_auc_score(y, _cv_scores(x, y, groups, seed)))
    rng = np.random.default_rng(seed)
    null = []
    for _ in range(n_null):
        shuffled = y.copy()
        for g in np.unique(groups):
            idx = np.flatnonzero(groups == g)
            shuffled[idx] = y[rng.permutation(idx)]
        null.append(roc_auc_score(shuffled, _cv_scores(x, shuffled, groups, seed)))
    return {"auc": auc, "null_mean": float(np.mean(null)), "null_sd": float(np.std(null)), "n": int(len(y))}
