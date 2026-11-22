# Capture spike report

Run on 2026-11-22 on an Apple M5 with 32GB, macOS 27. Model: `thinkingmachines/Inkling-2-Mini`, snapshot `b85c3cf`, loaded with mlx-lm at commit `e2b5bbd3` (pull request 2086, unmerged) and MLX 0.32.4.

## Verdict

The automatic verdict in `results/measure.json` is **no-go**, and the only failing check is swap growth. That check turned out not to be a valid signal on this machine (see "Swap" below). Speed and model memory both pass with room to spare. My reading is **go, with a clean re-run of the swap check owed**.

## What was asked and what was found

| Question | Threshold | Measured | Result |
|---|---|---|---|
| Can block states be captured per loop pass? | yes | yes, by wrapping `TextModel._run_loop` | pass |
| Does capture change the model's output? | no | logits equal within 1e-4 | pass |
| Speed with capture on, relative to off | at least 0.80 | 0.90 (16.5 against 18.2 tokens per second) | pass |
| MLX peak memory | at most 26GB | 18.3GB | pass |
| Swap growth during a run | at most 0.1GB | 0.46GB in one of four runs | fail, but see below |

All 17 tests pass.

## Speed and memory

Twenty prompts, 128 new tokens each, run four times (off, on, off, on):

| Run | Capture | Tokens | Seconds | Tokens per second | MLX peak (GB) | Swap before (GB) | Swap after (GB) |
|---|---|---|---|---|---|---|---|
| 1 | off | 2,413 | 131.9 | 18.30 | 18.3 | 31.98 | 29.13 |
| 2 | on | 2,413 | 146.4 | 16.48 | 18.3 | 29.13 | 28.94 |
| 3 | off | 2,413 | 133.0 | 18.14 | 18.3 | 28.94 | 28.94 |
| 4 | on | 2,413 | 146.7 | 16.45 | 18.3 | 28.94 | 29.40 |

Capture costs about 10% of speed. It converts every snapshot to numpy during the forward pass, which is the simplest possible approach; projecting on the GPU, as the design intends, should cost less.

## Swap

The machine already had 29 to 32GB of swap in use before any run, left over from earlier activity, and it moved in both directions during the measurement: down 2.85GB in run 1, up 0.46GB in run 4, down 2.6GB overall. System-wide swap therefore says nothing about this process here. Memory pressure after the runs showed 76% free. A valid check needs a re-run after a restart with other apps closed, or a per-process measure.

## Which checkpoint

The official checkpoint loaded directly with MLX (19GB on disk). No community conversion was needed, so contributors can all run the official weights.

## Block state layout

`block_states` has shape (rows, batch, tokens, 2304). Observed rows: 1 after the prelude (block 1), 3 after each loop pass (blocks 1, 2, 3), 4 after the final layers (blocks 1 to 4). There is no separate row for the token embedding. The spike keeps the last token position only.

## Check against published findings

Measured on the block 3 sum at the last prompt token, RMS-normalised, over 20 prompts. This is a small sample and a different quantity from Goodfire's per-layer analysis, so differences are not contradictions.

| Published (Goodfire) | Measured here |
|---|---|
| Coordinate 2027 dominates, about 46% of the squared norm | Coordinate 2027 (0-based) is the largest overall, with 21% of the squared norm: 26% after pass 2, 22% after pass 3. After pass 1 a different coordinate (2230) is largest at 15%. |
| Its mean normalised value is about +32 | +18.0, +24.4, +22.5 after passes 1, 2, 3 |
| It turns negative at the end of loop 3 | Not seen: it stays positive on the block 3 sum. Goodfire's observation is about layer 18's own output, which the spike does not capture. |
| States settle across passes | Mean movement between passes falls from 0.52 (pass 1 to 2) to 0.36 (pass 2 to 3). |

## First rough map

`results/pca.png` shows a 2D PCA of the same states, one panel per pass, with coordinate 2027 removed.

- Chat prompts separate clearly from the other three topics in every pass.
- After pass 1, dates, code and math are partly separated.
- After pass 3 they are bunched together, not further apart.

So the idea drawn in the designs, that territories pull apart from pass to pass, is **not supported by this check**. With 20 prompts, one token position and plain PCA it is not refuted either, but the base-map work must test it properly before the walk-through and the settling flag rely on it.

## Not checked

- Capture during long conversations or with images.
- Capture of per-layer outputs, cross-loop read weights, or input shares (needed by the loop inspector).
- Memory with a Swift app and a web view running alongside.
- Behaviour when the machine is under real memory pressure.
