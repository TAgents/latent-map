import numpy as np

from livemap.live import LiveMap, PhaseTracker, level


def tiny_map():
    """Two cells on the first kept axis, in a 5-coordinate space where coordinate 2 is dropped."""
    m = {
        "drop": np.int64(2),
        "pass_means": np.zeros((3, 4), dtype=np.float32),
        "components": np.eye(4, dtype=np.float32)[:2],
        "centres": np.array([[1.0, 0.0], [-1.0, 0.0]], dtype=np.float32),
    }
    return LiveMap(m, fine=np.array([7, 9]), norms={"7": [0.1, 0.5, 1.0], "9": [0.1, 0.5, 1.0], "all": [0.1, 0.5, 1.0]})


def test_describe_places_each_pass_and_measures_movement():
    lm = tiny_map()
    right = np.array([1.0, 0, 99.0, 0, 0])   # the huge dropped coordinate must not matter
    left = np.array([-1.0, 0, 0, 0, 0])
    block3 = np.zeros((3, 5)); block3[:, 2] = [2.0, 1.0, 0.0]
    d = lm.describe(np.stack([left, right, right]), block3)
    assert d["cells"] == [1, 0, 0] and d["area"] == 7
    assert d["move"][0] == 2.0 and d["move"][1] == 0.0
    assert d["loop"] == [4 / 5, 1 / 5, 0.0]
    assert d["level"] == "usual" and d["settled_by"] == 2


def test_level_bands():
    assert [level(v, [0.1, 0.5, 1.0]) for v in (0.2, 0.5, 0.7, 1.5)] == ["usual", "usual", "high", "far"]


def test_phase_tracker_follows_the_chat_markers():
    t = PhaseTracker()
    seq = ["<|message_model|>", "<|content_thinking|>", "The", " user", "<|end_message|>", "<|message_model|>",
           "<|content_text|>", "Hello", "<|end_message|>"]
    out = [t.step(s) for s in seq]
    assert out == [("thinking", True), ("thinking", True), ("thinking", False), ("thinking", False), ("thinking", True),
                   ("thinking", True), ("answer", True), ("answer", False), ("answer", True)]
