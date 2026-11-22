"""Capture Inkling 2 Mini's block states after each loop pass.

The model has no single residual stream. Each block of six layers keeps a
running sum, and `TextModel._run_loop` holds them stacked as `block_states`
with shape (rows, batch, tokens, hidden). This module swaps that method for a
copy that snapshots `block_states` after the prelude, after each pass through
the looped layers, and after the final layers. The computation is unchanged.
"""

from collections import deque
from dataclasses import dataclass, field

import mlx.core as mx
import numpy as np
from mlx_lm.models.inkling2.layers import unweighted_rms_norm


@dataclass
class StepCapture:
    """Block states for one forward call, last token position of batch row 0.
    Every array is float32 with shape (rows, hidden_size)."""

    prelude: np.ndarray
    passes: list[np.ndarray] = field(default_factory=list)
    final: np.ndarray | None = None


def _snapshot(block_states: mx.array) -> np.ndarray:
    return np.array(block_states[:, 0, -1, :].astype(mx.float32))


class capturing:
    """Context manager: `with capturing(model) as cap: ...; cap.records`."""

    def __init__(self, model, max_records: int = 4096):
        self._lm = model.model.language_model
        self._records: deque[StepCapture] = deque(maxlen=max_records)

    @property
    def records(self) -> list[StepCapture]:
        return list(self._records)

    def clear(self) -> None:
        self._records.clear()

    def __enter__(self):
        lm = self._lm
        if lm.args.loop_config is None:
            raise ValueError("Model has no loop configuration; nothing to capture.")
        records = self._records

        # Mirrors TextModel._run_loop at mlx-lm commit e2b5bbd3, plus snapshots.
        def run_loop(hidden_states, cache, conv_mask):
            core = lm.args.loop_core
            caches = iter(cache)

            def run(layers, hidden_states, block_states, bank=None):
                for i, layer in enumerate(layers):
                    messages = None if bank is None else lm.message_input(bank, i)
                    hidden_states, block_states = layer(
                        hidden_states,
                        block_states,
                        cache=next(caches),
                        conv_mask=conv_mask,
                        messages=messages,
                    )
                return hidden_states, block_states

            prelude = run(lm.layers[: core.start], hidden_states, None)
            hidden_states, block_states = prelude
            record = StepCapture(prelude=_snapshot(block_states))
            for loop_pass in range(lm.args.loop_config["num_passes"]):
                bank = None
                if loop_pass > 0:
                    bank = (
                        block_states[prelude[1].shape[0] + 1 :],
                        unweighted_rms_norm(hidden_states.astype(mx.float32)),
                    )
                    hidden_states, block_states = prelude
                hidden_states, block_states = run(
                    lm.layers[core.start : core.stop], hidden_states, block_states, bank
                )
                record.passes.append(_snapshot(block_states))
            out = run(lm.layers[core.stop :], hidden_states, block_states)
            record.final = _snapshot(out[1])
            records.append(record)
            return out

        object.__setattr__(lm, "_run_loop", run_loop)
        return self

    def __exit__(self, *exc):
        object.__delattr__(self._lm, "_run_loop")
        return False
