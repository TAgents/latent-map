"""What the model would say if it stopped after each loop pass.

After each pass, the model's own last block, final normalisation and output
layer are run on that pass's state. For the last pass this is exactly the
model's real prediction. pass_readouts runs a whole text at once, without a cache;
reply_readouts reads a long text in pieces with a cache of its own. Neither touches the
cache of a generation in progress.
"""

import mlx.core as mx
import numpy as np
from mlx_lm.models.inkling2.layers import unweighted_rms_norm


def pass_readouts(model, ids: list[int], track: list[int] | None = None, topk: int = 10, replace: dict | None = None,
                  positions: list[int] | None = None, states: bool = True, chunk: int = 256) -> dict:
    """ids: token ids of the text. Returns
      "top":  (passes, tokens, topk) most likely next-token ids per pass and position
      "rank": (passes, tokens, len(track)) rank of each tracked token id, 0 = most likely

    replace: {"passes": (k, ...), "position": p, "state": (hidden,) or (passes, hidden) array}. Before each listed pass's state
    is handed on (to the next pass, or to the last block), the hidden state at position p
    is overwritten with this vector, rescaled to that position's own size. Used to test
    whether a state is what the model actually uses.

    positions: read out only these token positions ("top", "prob" and "rank" then have
    len(positions) rows); the whole text is still run. states=False leaves out the two
    state arrays, which are large for long texts.
    "prob": (passes, tokens, topk) probability of each of the "top" ids, with the
    model's own cap on its output scores applied, so the last pass equals the real one.
    chunk: positions sent through the output layer at a time; keeps memory small."""
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

    def read(hidden_states, block_states):
        """top ids, their probabilities and tracked ranks at the chosen positions."""
        hs, bs = run(lm.layers[core.stop :], hidden_states, block_states)
        out = _final_rows(lm, hs, bs, dtype)
        if positions is not None:
            out = out[mx.array(positions)]
        return _top_words(model, out, topk, track, chunk)

    prelude = run(lm.layers[: core.start], hidden, None)
    hidden_states, block_states = prelude
    tops, probs, ranks, ends, block2 = [], [], [], [], []
    for loop_pass in range(lm.args.loop_config["num_passes"]):
        bank = None
        if loop_pass > 0:
            bank = (block_states[prelude[1].shape[0] + 1 :], unweighted_rms_norm(hidden_states.astype(mx.float32)))
            hidden_states, block_states = prelude
        hidden_states, block_states = run(lm.layers[core.start : core.stop], hidden_states, block_states, bank)
        if replace is not None and loop_pass in replace["passes"]:
            h = hidden_states.astype(mx.float32)
            p = replace["position"]
            state = np.asarray(replace["state"], dtype=np.float32)
            new = mx.array(state[loop_pass] if state.ndim == 2 else state)   # one state for all passes, or one per pass
            size = mx.sqrt(mx.mean(mx.square(h[0, p])))
            new = new / mx.sqrt(mx.mean(mx.square(new))) * size
            mask = mx.array((np.arange(h.shape[1]) == p).astype(np.float32))[None, :, None]
            hidden_states = (h * (1 - mask) + new[None, None, :] * mask).astype(dtype)
        if states:
            ends.append(np.array(unweighted_rms_norm(hidden_states.astype(mx.float32))[0]))
            block2.append(np.array(block_states[-1][0].astype(mx.float32)))          # this pass's block 2 output
        top, prob, rank = read(hidden_states, block_states)
        tops.append(top)
        probs.append(prob)
        if track:
            ranks.append(rank)
        mx.clear_cache()
    out = {
        "top": np.stack(tops),
        "prob": np.stack(probs),
        "rank": np.stack(ranks) if track else np.zeros((len(tops), tops[0].shape[0], 0), dtype=np.int64),
    }
    if states:
        out["pass_end"] = np.stack(ends)  # (passes, tokens, hidden): block 3's output after each pass, normalised
        out["block2"] = np.stack(block2)  # (passes, tokens, hidden): block 2's output on each pass
    return out


def _final_rows(lm, hidden_states, block_states, dtype):
    """The last block's output as the output layer takes it: (tokens, hidden)."""
    normed = unweighted_rms_norm(hidden_states.astype(mx.float32))
    return lm.norm(lm.final_input(block_states, normed)[..., 0, :]).astype(dtype)[0]


def _by_score_then_id(ids: np.ndarray, scores: np.ndarray) -> np.ndarray:
    """Each row's ids, highest score first; equal scores by lowest id, which is the one the
    runner writes at temperature 0 (scores are low precision, so exact ties are common)."""
    order = np.lexsort((ids, -scores), axis=-1)
    return np.take_along_axis(ids, order, axis=-1)


def _top_words(model, rows, topk, track=(), chunk=256):
    """top ids, their probabilities and tracked ranks for each row, with the model's own
    cap on its output scores applied. The output layer is rows x vocabulary, so it is
    done in pieces."""
    args = model.model.language_model.args
    softcap = args.final_logit_softcapping
    top, prob, rank = [], [], []
    for start in range(0, rows.shape[0], chunk):
        logits = model.lm_head(rows[start : start + chunk]).astype(mx.float32)
        logits = (mx.tanh(logits / softcap) * softcap)[:, : args.unpadded_vocab_size]
        cut = mx.argpartition(-logits, topk - 1, axis=-1)[:, :topk]
        ids = _by_score_then_id(np.array(cut), np.array(mx.take_along_axis(logits, cut, axis=-1)))
        best = mx.array(ids)
        top.append(ids)
        prob.append(np.array(mx.take_along_axis(mx.softmax(logits, axis=-1), best, axis=-1)))
        if track:
            rank.append(np.stack([np.array((logits > logits[:, t : t + 1]).sum(axis=-1)) for t in track], axis=-1))
        mx.clear_cache()
    return np.concatenate(top), np.concatenate(prob), (np.concatenate(rank) if track else None)


def reply_readouts(model, prompt_ids: list[int], reply_ids: list[int], topk: int = 10,
                   prefill: int = 2048, chunk: int = 1, pieces: list[int] | None = None) -> dict:
    """The per-pass readout for a reply, for texts too long to run in one go.

    Returns "top" and "prob", each (passes, len(reply_ids), topk): row i is the prediction
    made just before reply token i was written.

    The text is read the way generation reads it, in pieces with the model's cache. The
    last block is run after every pass, and after the earlier passes it gets a cache of
    its own, so on each pass it sees that pass's states for the whole text, as
    pass_readouts does. The prompt is only read (nothing is kept for it); the reply is
    read out. Does not go through the capture hook, so an installed capture records
    nothing.

    The model's output shifts with the size of the pieces a text is read in (measured on
    a 5,700-token prompt: read in pieces of 512 where generation used 2,048, the last pass
    gave the written token for 144 of 150; in the same pieces, 150 of 150). So:
    pieces: the sizes of the pieces generation read the prompt in (a last piece holding
    only the prompt's last token is where the readout starts). Without it the prompt is
    read `prefill` tokens at a time, the runner's default.
    chunk: reply tokens read at a time. Generation reads one at a time, and the default of 1
    reproduces it exactly; larger is several times faster on long replies but near-ties
    come out the other way (5 of 120 tokens at 256 on one request)."""
    lm = model.model.language_model
    core = lm.args.loop_core
    passes = lm.args.loop_config["num_passes"]
    last = len(lm.layers) - core.stop
    cache = model.make_cache()
    own = [model.make_cache()[-last:] for _ in range(passes - 1)]   # the last block's memory after pass 1, pass 2

    def step(tokens, read):
        hidden_states = lm.embed(mx.array([tokens]))
        dtype = hidden_states.dtype
        conv_mask = cache[0][1].make_mask(hidden_states.shape[1])
        caches = iter(cache)

        def run(layers, hidden_states, block_states, caches, bank=None):
            for i, layer in enumerate(layers):
                messages = None if bank is None else lm.message_input(bank, i)
                hidden_states, block_states = layer(hidden_states, block_states, cache=next(caches), conv_mask=conv_mask, messages=messages)
            return hidden_states, block_states

        prelude = run(lm.layers[: core.start], hidden_states, None, caches)
        hidden_states, block_states = prelude
        rows = []
        for loop_pass in range(passes):
            bank = None
            if loop_pass > 0:
                bank = (block_states[prelude[1].shape[0] + 1 :], unweighted_rms_norm(hidden_states.astype(mx.float32)))
                hidden_states, block_states = prelude
            hidden_states, block_states = run(lm.layers[core.start : core.stop], hidden_states, block_states, caches, bank)
            end = run(lm.layers[core.stop :], hidden_states, block_states, caches if loop_pass == passes - 1 else iter(own[loop_pass]))
            if read:
                rows.append(_final_rows(lm, *end, dtype))
        mx.eval([c.state for c in cache] + [c.state for group in own for c in group] + rows)
        mx.clear_cache()
        return rows

    head, start = list(prompt_ids[:-1]), 0
    for size in list(pieces or []) + [prefill] * (len(head) // prefill + 1):   # recorded pieces first, then the default
        if start >= len(head):
            break
        if size > 0:
            step(head[start : start + size], read=False)
            start += size
    tokens = [prompt_ids[-1]] + list(reply_ids[:-1])                 # reply token i is predicted at the token before it
    tops, probs, held = [[] for _ in range(passes)], [[] for _ in range(passes)], [[] for _ in range(passes)]

    def flush():  # the output layer, for the rows gathered so far
        for k in range(passes):
            if held[k]:
                top, prob, _ = _top_words(model, mx.concatenate(held[k]), topk)
                tops[k].append(top)
                probs[k].append(prob)
                held[k].clear()

    for start in range(0, len(tokens), chunk):
        for k, rows in enumerate(step(tokens[start : start + chunk], read=True)):
            held[k].append(rows)
        if sum(r.shape[0] for r in held[0]) >= 256:
            flush()
    flush()
    return {"top": np.stack([np.concatenate(t) for t in tops]), "prob": np.stack([np.concatenate(p) for p in probs])}
