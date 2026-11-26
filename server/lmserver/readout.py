"""What the model would have written at each word if it had stopped after each loop pass.

The model's own last block and output layer are run on each pass's state (see
latent_capture.readout). The last pass is the model's real prediction. The earlier
passes are a reading the model was never trained to give, so they show what the
state would produce, not how the model reasons."""

import time

MAX_TOKENS = 4096  # prompt plus reply; the whole text is re-read in one go, without the cache
TOPK = 5


class TooLong(ValueError):
    pass


def _run(model, ids, topk, positions, states):
    from latent_capture.readout import pass_readouts

    return pass_readouts(model, ids, topk=topk, positions=positions, states=states)


def compute(model, tokenizer, prompt_ids: list[int], reply_ids: list[int], run=_run) -> dict:
    """{"k", "seconds", "words"}: words[i][pass] is a list of [text, probability], most
    likely first, for the prediction made just before reply word i was written."""
    if not prompt_ids:
        raise ValueError("a readout needs the prompt's tokens")
    if len(prompt_ids) + len(reply_ids) > MAX_TOKENS:
        raise TooLong(f"prompt and reply are {len(prompt_ids) + len(reply_ids)} tokens; the limit is {MAX_TOKENS}")
    started = time.time()
    if not reply_ids:
        return {"k": TOPK, "seconds": 0.0, "words": []}
    # The prediction for reply word i is made at the position before it.
    positions = [len(prompt_ids) - 1 + i for i in range(len(reply_ids))]
    out = run(model, list(prompt_ids) + list(reply_ids), TOPK, positions, False)
    top, prob = out["top"], out["prob"]
    words = [
        [[[tokenizer.decode([int(t)]), round(float(p), 4)] for t, p in zip(top[k, j], prob[k, j])] for k in range(top.shape[0])]
        for j in range(len(positions))
    ]
    return {"k": TOPK, "seconds": round(time.time() - started, 2), "words": words}
