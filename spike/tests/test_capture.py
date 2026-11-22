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
