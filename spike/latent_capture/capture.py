"""Capture Inkling 2 Mini's block states after each loop pass.

The model has no single residual stream. The first layer of each block of six
appends the RMS-normalised state entering that block to `block_states`, shape
(rows, batch, tokens, hidden). With 24 layers and a loop over layers 6-17:

    row 0  the token embedding            (appended by layer 0)
    row 1  output of block 1              (appended by layer 6, same every pass)
    row 2  output of block 2, this pass   (appended by layer 12)
    row 3  output of block 3, last pass   (appended by layer 18, final only)

The output of block 3 for each pass is NOT in `block_states` during the loop;
it is the hidden state leaving the pass. This module swaps `_run_loop` for a
copy that snapshots `block_states` after the prelude, after each pass and after
the final layers, plus the normalised hidden state at the end of each pass.
The computation is unchanged.
"""

from collections import deque
from dataclasses import dataclass, field

import mlx.core as mx
import numpy as np
from mlx_lm.models.inkling2.layers import unweighted_rms_norm


@dataclass
class StepCapture:
    """Block states for one forward call, batch row 0.
    Every array is float32 with shape (rows, hidden_size) for the last token
    position, or (rows, tokens, hidden_size) when captured with all_positions."""

    prelude: np.ndarray
    passes: list[np.ndarray] = field(default_factory=list)
    # Output of block 3 at the end of each pass, RMS-normalised, shape (hidden_size,)
    # or (tokens, hidden_size) with all_positions.
    pass_end: list[np.ndarray] = field(default_factory=list)
    final: np.ndarray | None = None

    @property
    def block2(self) -> list[np.ndarray]:
        """Output of block 2 after each pass."""
        return [p[-1] for p in self.passes]

    @property
    def block3(self) -> list[np.ndarray]:
        """Output of block 3 after each pass."""
        return self.pass_end


def _snapshot(block_states: mx.array, all_positions: bool = False) -> np.ndarray:
    picked = block_states[:, 0] if all_positions else block_states[:, 0, -1, :]
    return np.array(picked.astype(mx.float32))


class capturing:
    """Context manager: `with capturing(model) as cap: ...; cap.records`."""

    def __init__(self, model, max_records: int = 512, all_positions: bool = False):
        self._lm = model.model.language_model
        self._all = all_positions
        self._records: deque[StepCapture] = deque(maxlen=max_records)

    @property
    def records(self) -> list[StepCapture]:
        return list(self._records)

    def clear(self) -> None:
        self._records.clear()

    def __enter__(self):
        lm = self._lm
        if "_run_loop" in vars(lm):
            raise RuntimeError("A capture is already installed on this model.")
        if lm.args.loop_config is None:
            raise ValueError("Model has no loop configuration; nothing to capture.")
        records = self._records
        every = self._all

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
            record = StepCapture(prelude=_snapshot(block_states, every))
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
                record.passes.append(_snapshot(block_states, every))
                end = unweighted_rms_norm(hidden_states.astype(mx.float32))
                record.pass_end.append(np.array(end[0] if every else end[0, -1, :]))
            out = run(lm.layers[core.stop :], hidden_states, block_states)
            record.final = _snapshot(out[1], every)
            records.append(record)
            return out

        object.__setattr__(lm, "_run_loop", run_loop)
        return self

    def __exit__(self, *exc):
        object.__delattr__(self._lm, "_run_loop")
        return False
