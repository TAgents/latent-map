"""What the model would have written at each word if it had stopped after each loop pass.

The model's own last block and output layer are run on each pass's state (see
latent_capture.readout). The last pass is the model's real prediction. The earlier
passes are a reading the model was never trained to give, so they show what the
state would produce, not how the model reasons.

The text is re-read in pieces with a cache, so long agent requests fit. The model's
numbers shift with the size of the pieces a text is read in, so the prompt is re-read in
the pieces generation used (recorded with the request) and the reply one token at a time,
as generation read it. Then the last pass reproduces what generation computed."""

import time

MAX_TOKENS = 32768  # prompt plus reply
TOPK = 5


class TooLong(ValueError):
    pass


class NotLoaded(RuntimeError):
    pass


def _run(model, prompt_ids, reply_ids, topk, pieces):
    from latent_capture.readout import reply_readouts

    return reply_readouts(model, prompt_ids, reply_ids, topk=topk, pieces=pieces)  # re-read in pieces with a cache of its own


def compute(model, tokenizer, prompt_ids: list[int], reply_ids: list[int], pieces: list[int] | None = None, run=_run) -> dict:
    """{"k", "seconds", "words", "hits", "pieces_known"}: words[i][pass] is a list of [text, probability,
    token id], most likely first, for the prediction made just before reply word i was
    written. hits[i][pass] says whether that pass's top token is the token written
    (compared by id: different tokens can show as the same text). pieces: the sizes of
    the pieces generation read the prompt in, when known; the prompt is re-read the same way.
    pieces_known is False when it was not: the readout is then close to generation, not equal."""
    if not prompt_ids:
        raise ValueError("a readout needs the prompt's tokens")
    if len(prompt_ids) + len(reply_ids) > MAX_TOKENS:
        raise TooLong(f"prompt and reply are {len(prompt_ids) + len(reply_ids)} tokens; the limit is {MAX_TOKENS}")
    started = time.time()
    # Recorded pieces are used only when they are sizes that add up to this prompt.
    known = bool(pieces) and all(isinstance(n, int) and n > 0 for n in pieces) and sum(pieces) == len(prompt_ids)
    pieces = list(pieces) if known else None
    if not reply_ids:
        return {"k": TOPK, "seconds": 0.0, "words": [], "hits": [], "pieces_known": known}
    out = run(model, list(prompt_ids), list(reply_ids), TOPK, pieces)  # row i: the prediction made just before reply word i
    top, prob = out["top"], out["prob"]
    words = [
        [[[tokenizer.decode([int(t)]), round(float(p), 4), int(t)] for t, p in zip(top[k, j], prob[k, j])] for k in range(top.shape[0])]
        for j in range(len(reply_ids))
    ]
    hits = [[int(top[k, j, 0]) == int(reply_ids[j]) for k in range(top.shape[0])] for j in range(len(reply_ids))]
    return {"k": TOPK, "seconds": round(time.time() - started, 2), "words": words, "hits": hits, "pieces_known": known}
