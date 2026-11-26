import mlx.core as mx
import numpy as np

from latent_capture.readout import pass_readouts


def test_last_pass_readout_is_the_models_own_prediction(loaded):
    model, tokenizer = loaded
    ids = tokenizer.encode("The capital of France is")
    out = pass_readouts(model, ids, track=[5, 99], topk=5)
    real = np.array(model(mx.array([ids]))[0].astype(mx.float32))
    assert out["top"].shape == (3, len(ids), 5) and out["rank"].shape == (3, len(ids), 2)
    np.testing.assert_array_equal(out["top"][2, :, 0], real.argmax(-1))          # same top word at every position
    want = (real > real[:, [5]]).sum(-1)                                          # rank 0 = most likely
    assert np.abs(out["rank"][2, :, 0] - want).max() <= 2                         # ties in low precision aside


def test_block2_matches_capture_and_replacing_a_state_with_itself_changes_nothing(loaded):
    from latent_capture.capture import capturing

    model, tokenizer = loaded
    ids = tokenizer.encode("The capital of France is")
    base = pass_readouts(model, ids, track=[5], topk=5)
    with capturing(model, all_positions=True) as cap:
        model(mx.array([ids]))
    np.testing.assert_allclose(base["block2"][1], cap.records[-1].block2[1], rtol=1e-3, atol=1e-3)
    np.testing.assert_allclose(base["pass_end"][2], cap.records[-1].block3[2], rtol=1e-3, atol=1e-3)
    same = pass_readouts(model, ids, topk=5, replace={"passes": (1,), "position": 3, "state": base["pass_end"][1, 3]})
    np.testing.assert_array_equal(same["top"][2, :, 0], base["top"][2, :, 0])
    other = pass_readouts(model, ids, topk=5, replace={"passes": (0, 1, 2), "position": len(ids) - 1, "state": base["pass_end"][2, 0]})
    assert (other["top"][2, -1] != base["top"][2, -1]).any()
    per_pass = pass_readouts(model, ids, topk=5, replace={"passes": (0, 1, 2), "position": 3, "state": base["pass_end"][:, 3]})
    np.testing.assert_array_equal(per_pass["top"][2, :, 0], base["top"][2, :, 0])   # each pass's own state back in: no change


def test_reading_out_chosen_positions_matches_the_full_readout(loaded):
    model, tokenizer = loaded
    ids = tokenizer.encode("The capital of France is Paris, and the capital of Italy is")
    full = pass_readouts(model, ids, track=[5], topk=5)
    part = pass_readouts(model, ids, track=[5], topk=5, positions=[2, len(ids) - 1], states=False)
    assert part["top"].shape == (3, 2, 5) and part["prob"].shape == (3, 2, 5) and part["rank"].shape == (3, 2, 1)
    assert "pass_end" not in part and "block2" not in part
    np.testing.assert_array_equal(part["top"][:, :, 0], full["top"][:, [2, len(ids) - 1], 0])
    np.testing.assert_allclose(part["prob"], full["prob"][:, [2, len(ids) - 1]], atol=2e-2)
    assert (full["prob"] >= 0).all() and (full["prob"][..., 0] >= full["prob"][..., 1]).all() and (full["prob"].sum(-1) <= 1.001).all()


def test_probabilities_are_the_models_own_and_chunking_changes_nothing(loaded):
    model, tokenizer = loaded
    ids = tokenizer.encode("The capital of France is Paris, and the capital of Italy is")
    real = mx.softmax(model(mx.array([ids]))[0].astype(mx.float32), axis=-1)
    real = np.array(real)
    out = pass_readouts(model, ids, topk=5, states=False)
    want = np.take_along_axis(real, out["top"][2], axis=-1)
    np.testing.assert_allclose(out["prob"][2], want, atol=5e-3)   # includes the model's cap on its output scores
    small = pass_readouts(model, ids, track=[5], topk=5, states=False, chunk=3)
    big = pass_readouts(model, ids, track=[5], topk=5, states=False)
    np.testing.assert_array_equal(small["top"][:, :, 0], big["top"][:, :, 0])
    np.testing.assert_allclose(small["prob"], big["prob"], atol=1e-4)
    np.testing.assert_array_equal(small["rank"], big["rank"])
    assert (np.diff(big["prob"], axis=-1) <= 1e-6).all()            # most likely first
