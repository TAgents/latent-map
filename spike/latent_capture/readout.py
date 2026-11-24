"""What the model would say if it stopped after each loop pass.

After each pass, the model's own last block, final normalisation and output
layer are run on that pass's state. For the last pass this is exactly the
model's real prediction. Run on a whole text at once, without a cache, so the
extra runs of the last block cannot disturb a generation in progress.
"""

import mlx.core as mx
import numpy as np
from mlx_lm.models.inkling2.layers import unweighted_rms_norm


def pass_readouts(model, ids: list[int], track: list[int] | None = None, topk: int = 10, replace: dict | None = None) -> dict:
    """ids: token ids of the text. Returns
      "top":  (passes, tokens, topk) most likely next-token ids per pass and position
      "rank": (passes, tokens, len(track)) rank of each tracked token id, 0 = most likely

    replace: {"passes": (k, ...), "position": p, "state": (hidden,) array}. Before each listed pass's state
    is handed on (to the next pass, or to the last block), the hidden state at position p
    is overwritten with this vector, rescaled to that position's own size. Used to test
    whether a state is what the model actually uses."""
    lm = model.model.language_model
    core = lm.args.loop_core
    track = list(track or [])
    hidden = lm.embed(mx.array([ids]))
    dtype = hidden.dtype

    def run(layers, hidden_states, block_states, bank=None):
        for i, layer in enumerate(layers):
            messages = None if bank is None else lm.message_input(bank, i)
            hidden_states, block_states = layer(hidden_states, block_states, cache=None, conv_mask=None, messages=messages)
        return hidden_states, block_states

    def logits_from(hidden_states, block_states):
        hs, bs = run(lm.layers[core.stop :], hidden_states, block_states)
        normed = unweighted_rms_norm(hs.astype(mx.float32))
        out = lm.norm(lm.final_input(bs, normed)[..., 0, :]).astype(dtype)
        return model.lm_head(out).astype(mx.float32)[0, :, : lm.args.unpadded_vocab_size]

    prelude = run(lm.layers[: core.start], hidden, None)
    hidden_states, block_states = prelude
    tops, ranks, ends, block2 = [], [], [], []
    for loop_pass in range(lm.args.loop_config["num_passes"]):
        bank = None
        if loop_pass > 0:
            bank = (block_states[prelude[1].shape[0] + 1 :], unweighted_rms_norm(hidden_states.astype(mx.float32)))
            hidden_states, block_states = prelude
        hidden_states, block_states = run(lm.layers[core.start : core.stop], hidden_states, block_states, bank)
        if replace is not None and loop_pass in replace["passes"]:
            h = hidden_states.astype(mx.float32)
            p = replace["position"]
            new = mx.array(np.asarray(replace["state"], dtype=np.float32))
            size = mx.sqrt(mx.mean(mx.square(h[0, p])))
            new = new / mx.sqrt(mx.mean(mx.square(new))) * size
            mask = mx.array((np.arange(h.shape[1]) == p).astype(np.float32))[None, :, None]
            hidden_states = (h * (1 - mask) + new[None, None, :] * mask).astype(dtype)
        ends.append(np.array(unweighted_rms_norm(hidden_states.astype(mx.float32))[0]))
        logits = logits_from(hidden_states, block_states)
        tops.append(np.array(mx.argsort(-logits, axis=-1)[:, :topk]))
        block2.append(np.array(block_states[-1][0].astype(mx.float32)))          # this pass's block 2 output
        if track:
            cols = [np.array((logits > logits[:, t : t + 1]).sum(axis=-1)) for t in track]   # one word at a time: small memory
            ranks.append(np.stack(cols, axis=-1))
        mx.clear_cache()
    return {
        "top": np.stack(tops),
        "rank": np.stack(ranks) if track else np.zeros((len(tops), len(ids), 0), dtype=np.int64),
        "pass_end": np.stack(ends),  # (passes, tokens, hidden): block 3's output after each pass, normalised
        "block2": np.stack(block2),  # (passes, tokens, hidden): block 2's output on each pass
    }
