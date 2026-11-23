import numpy as np
import pytest

from siglib.analysis import has_code, paired, probe_auc, refused, distinct_ratio
from siglib.prompts_guard import build


def test_prompt_set_is_24_complete_triples():
    items = build()
    assert len(items) == 72 and len({i["id"] for i in items}) == 72
    assert {i["kind"] for i in items} == {"harmful", "topic", "benign"}
    assert all(sum(j["triple"] == i["triple"] for j in items) == 3 for i in items)


def test_paired_counts_wins_and_gives_a_sign_test():
    out = paired([0.5] * 20 + [0.1] * 4, [0.2] * 24)
    assert out["a_higher"] == 20 and out["b_higher"] == 4 and out["ties"] == 0 and out["n"] == 24
    assert out["p"] < 0.01 and out["median_difference"] == pytest.approx(0.3)
    assert paired([0.2, 0.2], [0.2, 0.2])["p"] == 1.0  # all ties: no evidence


def test_refused_reads_common_refusal_openings_only():
    assert refused("I can't help with that request.") and refused("Sorry, but I cannot assist with breaking into an account.")
    assert refused("I won’t provide instructions for that.")
    assert not refused("Sure. First, check that you can't see the router.") and not refused("")


def test_has_code_and_distinct_ratio():
    assert has_code("Here:\n```python\nx = 1\n```") and has_code("def brew(tea):\n    return tea")
    assert not has_code("I would define a good day by how I return home.")
    assert distinct_ratio(["a", "b", "c", "a"]) == 0.75 and distinct_ratio([]) == 0.0


def test_probe_separates_real_structure_and_not_noise():
    r = np.random.default_rng(0)
    groups = np.repeat(np.arange(24), 2)
    y = np.tile([1, 0], 24)
    signal = r.normal(size=(48, 16)); signal[:, 0] += y * 3.0
    good = probe_auc(signal, y, groups, n_null=20)
    noise = probe_auc(r.normal(size=(48, 16)), y, groups, n_null=20)
    assert good["auc"] > 0.9 and good["auc"] > good["null_mean"] + 3 * good["null_sd"]
    assert noise["auc"] < 0.75
    assert 0.3 < good["null_mean"] < 0.7
