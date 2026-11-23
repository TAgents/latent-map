import mlx.core as mx
import numpy as np

from latent_capture.capture import capturing
from latent_capture.gen import generate_with_states


def direction(d=2304, seed=0):
    v = np.random.default_rng(seed).normal(size=d).astype(np.float32)
    return v / np.linalg.norm(v) * np.sqrt(d)  # same length as a normalised state


def test_zero_strength_changes_nothing_and_steering_moves_block2_toward_the_vector(loaded):
    model, tokenizer = loaded
    ids = mx.array([tokenizer.encode("The capital of France is Paris.")])
    v = direction()
    with capturing(model) as cap:
        base = np.array(model(ids)[0, -1].astype(mx.float32))
        plain = cap.records[-1]
    with capturing(model, steer={"vector": v, "strength": 0.0, "passes": (0, 1, 2)}) as cap:
        same = np.array(model(ids)[0, -1].astype(mx.float32))
    with capturing(model, steer={"vector": v, "strength": 1.0, "passes": (1,)}) as cap:
        moved = np.array(model(ids)[0, -1].astype(mx.float32))
        steered = cap.records[-1]
    np.testing.assert_allclose(same, base, rtol=1e-3, atol=1e-3)
    assert np.abs(moved - base).max() > 1e-2
    cos = lambda a: float(a @ v / (np.linalg.norm(a) * np.linalg.norm(v)))
    assert cos(steered.block2[1]) > cos(plain.block2[1]) + 0.3   # the steered pass moved toward the vector
    np.testing.assert_allclose(steered.block2[0], plain.block2[0], rtol=1e-3, atol=1e-3)  # the pass before it did not


def test_generate_with_states_gives_one_state_per_generated_token(loaded):
    model, tokenizer = loaded
    head = tokenizer.apply_chat_template([{"role": "user", "content": "Say hello."}], add_generation_prompt=True,
                                         tokenize=False, reasoning_effort=0.0)
    out = list(generate_with_states(model, tokenizer, head, max_tokens=12))
    assert len(out) == 12
    assert all(rec.n_tokens == 1 and len(rec.block2) == 3 for _, _, rec in out)
    assert out[0][1] == "<|message_model|>"
    # The state attached to a token is the one from the call that consumed it: in one
    # forward pass over the prompt plus the tokens, it must match its own position far
    # better than the neighbouring ones. (Step-by-step and one-shot runs differ slightly
    # in low precision, so this compares directions, not exact values.)
    n = len(tokenizer.encode(head))
    with capturing(model, all_positions=True) as cap:
        model(mx.array([tokenizer.encode(head) + [tok for tok, _, _ in out]]))
    full = cap.records[-1].block2[2]
    cos = lambda a, b: float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
    for k in (1, 3, 6):
        own = cos(out[k][2].block2[2], full[n + k])
        assert own > 0.95
        assert own > cos(out[k][2].block2[2], full[n + k - 1]) + 0.05
        assert own > cos(out[k][2].block2[2], full[n + k + 1]) + 0.05
