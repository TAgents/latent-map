import mlx.core as mx
import numpy as np
import pytest

from latent_capture.capture import capturing


def ids(tokenizer, text):
    return mx.array([tokenizer.encode(text)])


def test_one_record_per_forward_with_three_passes(loaded):
    model, tok = loaded
    with capturing(model) as cap:
        model(ids(tok, "March comes before April"))
    assert len(cap.records) == 1
    rec = cap.records[0]
    assert len(rec.passes) == 3
    assert rec.passes[0].shape[-1] == 2304
    assert rec.passes[0].dtype == np.float32


def test_each_pass_adds_two_block_rows(loaded):
    model, tok = loaded
    with capturing(model) as cap:
        model(ids(tok, "March comes before April"))
    rec = cap.records[0]
    assert all(p.shape[0] == rec.prelude.shape[0] + 2 for p in rec.passes)


def test_keeps_only_last_position(loaded):
    model, tok = loaded
    with capturing(model) as cap:
        model(ids(tok, "one two three four five six seven eight"))
    assert cap.records[0].passes[0].ndim == 2


def test_capture_does_not_change_logits(loaded):
    model, tok = loaded
    x = ids(tok, "The answer is")
    plain = model(x)
    with capturing(model):
        hooked = model(x)
    assert mx.allclose(plain, hooked, atol=1e-4).item()


def test_hook_removed_after_exception(loaded):
    model, _ = loaded
    lm = model.model.language_model
    before = lm._run_loop
    with pytest.raises(RuntimeError):
        with capturing(model):
            raise RuntimeError("boom")
    assert lm._run_loop == before


def test_records_are_capped(loaded):
    model, tok = loaded
    with capturing(model, max_records=2) as cap:
        for _ in range(4):
            model(ids(tok, "hello"))
    assert len(cap.records) == 2


def test_pass_end_is_block3_output(loaded):
    """The state leaving the looped layers is not in block_states until the
    final block appends it, so the last pass's end state must equal final[-1]."""
    model, tok = loaded
    with capturing(model) as cap:
        model(ids(tok, "March comes before April"))
    rec = cap.records[0]
    assert len(rec.pass_end) == 3
    assert rec.pass_end[0].shape == (2304,)
    assert np.allclose(rec.pass_end[2], rec.final[-1], atol=1e-3)
    assert not np.allclose(rec.pass_end[0], rec.pass_end[2], atol=1e-3)


def test_nested_capture_is_refused(loaded):
    model, _ = loaded
    lm = model.model.language_model
    before = lm._run_loop
    with capturing(model):
        with pytest.raises(RuntimeError):
            with capturing(model):
                pass
    assert lm._run_loop == before


def test_all_positions_keeps_a_token_axis_and_matches_last_token(loaded):
    model, tokenizer = loaded
    ids = mx.array([tokenizer.encode("The capital of France is Paris.")])
    with capturing(model, all_positions=True) as cap:
        model(ids)
    full = cap.records[-1]
    with capturing(model) as cap:
        model(ids)
    last = cap.records[-1]
    n = ids.shape[1]
    assert full.block2[0].shape == (n, 2304)
    assert full.block3[2].shape == (n, 2304)
    np.testing.assert_allclose(full.block2[1][-1], last.block2[1], rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(full.block3[2][-1], last.block3[2], rtol=1e-4, atol=1e-4)
