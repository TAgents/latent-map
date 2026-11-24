from pathlib import Path

import numpy as np
import pytest

from latent_capture.capture import StepCapture
from livemap.live import LiveMap
from lmserver.hooks import build_record

ROOT = Path(__file__).resolve().parents[2]
HIDDEN = 2304
TEXT = {8: "read_file", 9: "<|content_invoke_tool_json|>", 10: "{}", 1: "<|message_model|>", 2: "<|content_thinking|>", 3: "Hmm", 4: "<|end_message|>", 5: "<|content_text|>", 6: "Hi", 7: "."}


class Tok:
    def decode(self, ids):
        return "".join(TEXT[i] for i in ids)


@pytest.fixture(scope="module")
def live():
    return LiveMap.load(ROOT / "basemap" / "bundle_v1", ROOT / "prototype" / "norms_v1.json")


def cap(seed, n_tokens=1):
    rng = np.random.default_rng(seed)
    return StepCapture(
        prelude=np.zeros((2, HIDDEN), np.float32),
        passes=[rng.normal(size=(3, HIDDEN)).astype(np.float32) for _ in range(3)],
        pass_end=[rng.normal(size=HIDDEN).astype(np.float32) for _ in range(3)],
        n_tokens=n_tokens,
    )


META = {"completion_id": "chatcmpl-x", "client": "pytest", "messages": [{"role": "user", "content": "q"}], "settings": {}}
TIMING = {"time": 1.0, "first_token_s": 0.1, "tokens_per_s": 9.0}


def build(live, sizes, uncached, tokens, finish="stop"):
    records = [cap(i, n) for i, n in enumerate(sizes)]
    prompt = list(range(100, 100 + uncached + 2))
    return build_record(META, prompt, uncached, tokens, records, Tok(), live, TIMING, finish), records


def test_words_pair_with_generated_tokens_after_uncached_prompt(live):
    rec, records = build(live, [3, 1, 1, 1, 1], 4, [3, 6, 7])
    assert len(rec.words) == 3 and rec.reply_tokens == ["Hmm", "Hi", "."]
    expect = live.describe(np.stack(records[2].block2), np.stack(records[2].block3))
    assert rec.words[0]["cells"] == expect["cells"] and rec.words[0]["i"] == 0
    assert rec.prompt_tokens == 6 and rec.cached_tokens == 2


def test_last_token_without_a_record_is_kept_without_state(live):
    rec, _ = build(live, [3, 1, 1, 1], 4, [3, 6, 7])
    assert len(rec.words) == 3 and rec.words[2]["cells"] is None and rec.words[1]["cells"] is not None


def test_phases_follow_the_markers(live):
    rec, _ = build(live, [4] + [1] * 7, 4, [1, 2, 3, 4, 1, 5, 6])
    assert [w["phase"] for w in rec.words] == ["marker", "marker", "thinking", "marker", "marker", "marker", "answer"]


def test_mean_states_exclude_markers_and_are_nan_for_an_empty_phase(live):
    rec, records = build(live, [4] + [1] * 3, 4, [5, 6, 7])
    m = rec.mean_states
    assert m.shape == (2, 2, 3, HIDDEN) and m.dtype == np.float32
    assert np.isnan(m[0]).all()  # no thinking words
    want = np.mean([np.stack(records[2].block2), np.stack(records[3].block2)], axis=0)
    assert np.allclose(m[1, 0], want, atol=1e-5)
    want3 = np.mean([np.stack(records[2].block3), np.stack(records[3].block3)], axis=0)
    assert np.allclose(m[1, 1], want3, atol=1e-5)


def test_cancelled_finish_is_carried(live):
    rec, _ = build(live, [4, 1], 4, [6], finish="cancelled")
    assert rec.finish == "cancelled" and rec.completion_id == "chatcmpl-x" and rec.client == "pytest"


def test_tool_call_words_are_not_counted_as_thinking_or_answer(live):
    rec, _ = build(live, [4] + [1] * 8, 4, [1, 2, 3, 4, 1, 8, 9, 10])
    assert [w["phase"] for w in rec.words] == ["marker", "marker", "thinking", "marker", "marker", "header", "marker", "tool"]
    assert np.isnan(rec.mean_states[1]).all() and not np.isnan(rec.mean_states[0]).any()
