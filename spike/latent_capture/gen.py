"""Generate text and hand back, for every generated token, the states captured
when the model consumed that token."""

from mlx_lm import stream_generate

from latent_capture.capture import StepCapture, capturing


def generate_with_states(model, tokenizer, prompt: str, max_tokens: int, steer: dict | None = None):
    """Yields (token_id, token_text, StepCapture) for each generated token, in order.

    The pinned mlx-lm feeds the last prompt token in a forward call of its own, so
    the first one-token record belongs to the prompt and is skipped. After that the
    n-th one-token record is the call that consumed the n-th generated token. The
    final token is yielded only if the model consumed it (it looks one step ahead)."""
    with capturing(model, max_records=8192, steer=steer) as cap:
        tokens: list[tuple[int, str]] = []
        seen, skipped, matched = 0, False, 0

        def drain():
            nonlocal seen, skipped, matched
            records = cap.records
            new, seen = records[seen:], len(records)
            for rec in new:
                if rec.n_tokens != 1:
                    continue
                if not skipped:
                    skipped = True
                    continue
                if matched < len(tokens):
                    yield (*tokens[matched], rec)
                    matched += 1

        for r in stream_generate(model, tokenizer, prompt, max_tokens=max_tokens):
            tokens.append((int(r.token), tokenizer.decode([r.token])))
            yield from drain()
        yield from drain()


__all__ = ["generate_with_states", "StepCapture"]
