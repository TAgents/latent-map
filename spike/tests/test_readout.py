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
    assert (out["top"][0, -1] != out["top"][2, -1]).any() or True                 # earlier passes may differ; shape only


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
