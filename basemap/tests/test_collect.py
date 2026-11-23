import numpy as np

from mapbuild.collect import stack


def rec(n, n_prompt, text_id="a"):
    return dict(block2=np.ones((3, n, 4)), block3=np.zeros((3, n, 4)), tokens=[f"t{i}" for i in range(n)],
                topic="x", form="y", text_id=text_id, n_prompt=n_prompt)


def test_stack_concatenates_and_marks_reply_tokens():
    out = stack([rec(5, 2), rec(3, 1, "b")])
    assert out["block2"].shape == (8, 3, 4) and out["block2"].dtype == np.float16
    assert out["is_reply"].tolist() == [False, False, True, True, True, False, True, True]
    assert out["text_id"] == ["a"] * 5 + ["b"] * 3 and len(out["token"]) == 8


def test_stack_skips_empty_records():
    assert stack([rec(0, 0), rec(2, 1)])["block3"].shape == (2, 3, 4)


def test_stack_carries_an_optional_phase_per_token():
    a, b = rec(3, 1), rec(2, 1, "b")
    a["phase"], b["phase"] = ["prompt", "thinking", "answer"], ["prompt", "answer"]
    assert stack([a, b])["phase"] == ["prompt", "thinking", "answer", "prompt", "answer"]
    assert "phase" not in stack([rec(2, 1)])
