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


def test_reading_a_reply_in_pieces_with_the_cache_matches_the_whole_text_at_once(loaded):
    from latent_capture.capture import capturing
    from latent_capture.readout import reply_readouts

    model, tokenizer = loaded
    ids = tokenizer.encode("The capital of France is Paris, and the capital of Italy is Rome. The capital of Spain is Madrid, "
                           "and the capital of Austria is Vienna, where Mozart lived for many years.")
    for n_prompt in (1, 12):
        prompt, reply = ids[:n_prompt], ids[n_prompt:]
        positions = [n_prompt - 1 + i for i in range(len(reply))]
        whole = pass_readouts(model, ids, topk=5, positions=positions, states=False)
        with capturing(model) as cap:
            pieces = reply_readouts(model, prompt, reply, topk=5, prefill=4, chunk=3)
            assert cap.records == []                                   # a capture in place records nothing
        usual = reply_readouts(model, prompt, reply, topk=5, chunk=256)
        # The model itself gives slightly different numbers when a text is read in pieces
        # (up to about 0.07 in the top probability on this text), so piecewise and whole
        # agree on clear cases and closely, not exactly.
        for got in (pieces, usual):
            assert got["top"].shape == (3, len(reply), 5) and got["prob"].shape == (3, len(reply), 5)
            sure = whole["prob"][..., 0] - whole["prob"][..., 1] > 0.2
            np.testing.assert_array_equal(got["top"][..., 0][sure], whole["top"][..., 0][sure])
            assert (got["top"][..., 0] == whole["top"][..., 0]).mean() >= 0.97
            assert np.abs(got["prob"][..., 0] - whole["prob"][..., 0]).mean() < 0.02
    # Read one token at a time, as generation does, the last pass is exactly the model's own prediction.
    cache, real = model.make_cache(), []
    model(mx.array([ids[:11]]), cache=cache)
    for t in ids[11:-1]:
        real.append(mx.softmax(model(mx.array([[t]]), cache=cache)[0, -1].astype(mx.float32), axis=-1))
    real = np.array(mx.stack(real))
    single = reply_readouts(model, ids[:12], ids[12:], topk=5)
    np.testing.assert_array_equal(single["top"][2, :, 0], real.argmax(-1))
    np.testing.assert_allclose(single["prob"][2], np.take_along_axis(real, single["top"][2], axis=-1), atol=1e-3)
    # The prompt read in given pieces, as an earlier generation read it: again exact.
    cache, real = model.make_cache(), []
    model(mx.array([ids[:4]]), cache=cache)
    model(mx.array([ids[4:11]]), cache=cache)
    for t in ids[11:-1]:
        real.append(mx.softmax(model(mx.array([[t]]), cache=cache)[0, -1].astype(mx.float32), axis=-1))
    real = np.array(mx.stack(real))
    same = reply_readouts(model, ids[:12], ids[12:], topk=5, pieces=[4, 7, 1])
    np.testing.assert_allclose(same["prob"][2], np.take_along_axis(real, same["top"][2], axis=-1), atol=1e-3)
    short = reply_readouts(model, ids[:12], ids[12:], topk=5, chunk=1, pieces=[4])      # pieces that stop early: the rest by default
    assert short["top"].shape == same["top"].shape


def test_equal_scores_are_ordered_as_the_sampler_picks_them():
    # At temperature 0 the runner takes the first of equal scores, which is the lowest token id.
    from latent_capture.readout import _by_score_then_id

    ids = np.array([[9, 4, 7, 2], [5, 6, 1, 3]])
    scores = np.array([[1.0, 2.0, 2.0, 0.5], [3.0, 3.0, 3.0, 4.0]], dtype=np.float32)
    np.testing.assert_array_equal(_by_score_then_id(ids, scores), [[4, 7, 9, 2], [3, 1, 5, 6]])
