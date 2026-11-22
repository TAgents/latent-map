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

All 20 tests pass.

## Speed and memory

Twenty prompts, up to 128 new tokens each (some stop early), run four times (off, on, off, on):

| Run | Capture | Tokens | Seconds | Tokens per second | MLX peak (GB) | Swap before (GB) | Swap after (GB) |
|---|---|---|---|---|---|---|---|
| 1 | off | 2,413 | 131.9 | 18.30 | 18.3 | 31.98 | 29.13 |
| 2 | on | 2,413 | 146.4 | 16.48 | 18.3 | 29.13 | 28.94 |
| 3 | off | 2,413 | 133.0 | 18.14 | 18.3 | 28.94 | 28.94 |
| 4 | on | 2,413 | 146.7 | 16.45 | 18.3 | 28.94 | 29.40 |

Capture costs about 10% of speed. It converts every snapshot to numpy during the forward pass, which forces the model to finish each step before the next can be queued. Projecting on the GPU, as the design intends, should cost less.

These figures were taken before the capture also recorded the end-of-pass state (one more small snapshot per pass); the measurement was not repeated after that change. They were also taken with 29 to 32GB of system swap in use, so absolute speeds may be pessimistic.

## Swap

The machine already had 29 to 32GB of swap in use before any run, left over from earlier activity, and it moved in both directions during the measurement: down 2.85GB in run 1, up 0.46GB in run 4, down 2.6GB overall. System-wide swap therefore says nothing about this process here. A valid check needs a re-run after a restart with other apps closed, or a per-process measure.

## Which checkpoint

The official checkpoint loaded directly with MLX (19GB on disk). No community conversion was needed, so contributors can all run the official weights.

## Block state layout

The first layer of each block of six appends the RMS-normalised state entering that block to `block_states`, shape (rows, batch, tokens, 2304):

| Row | What it is | When it appears |
|---|---|---|
| 0 | the token embedding | after the prelude (1 row) |
| 1 | output of block 1 | in each pass, identical every pass |
| 2 | output of block 2 for this pass | in each pass (3 rows) |
| 3 | output of block 3 for the last pass | after the final layers (4 rows) |

The output of block 3 for passes 1 and 2 is never in `block_states`; it is the hidden state leaving the pass, and the capture records it separately. A test confirms that the recorded end state of pass 3 equals row 3.

An independent review caught that the first version of this spike mislabelled row 2 as block 3. The figures below are from the corrected capture.

## Check against published findings

Measured at the last prompt token over 20 prompts. A small sample.

| Published (Goodfire) | Measured here, block 3 output |
|---|---|
| Coordinate 2027 carries about 46% of the squared norm | 40% after pass 1, 34% after pass 2 |
| Its mean normalised value is about +32 | +30.2 after pass 1, +28.1 after pass 2 |
| It is removed at the end of the last loop | 0.6% of the squared norm after pass 3; mean +1.8 (near zero, not negative) |

So the loop signal reproduces: high after the first two passes and gone after the third.

On the block 2 output the same coordinate is weaker and does not collapse: 14%, 26%, 22% across the three passes.

## Do states settle?

Mean movement between consecutive passes, with coordinate 2027 removed:

| State | Pass 1 to 2 | Pass 2 to 3 |
|---|---|---|
| Block 2 output | 0.55 | 0.41 |
| Block 3 output | 0.39 | 1.21 |

Block 2's output settles. Block 3's output does the opposite: it changes most on the last pass, when it stops writing for another pass and feeds the final block instead. The design maps "block 3 on shared axes across passes" and reads settling from it; on this evidence block 2's output is the better state for a settling measure, and block 3's last pass should be treated as a different kind of state.

## First rough map

`results/pca.png` shows a 2D PCA fitted separately for each pass, coordinate 2027 removed, for both block outputs.

- Chat and code each separate clearly from the rest in every pass.
- Dates and math overlap with each other throughout, which is plausible for related topics.
- Topic separation (distance between topic centres divided by spread within topics, in the full space) rises with each pass: 1.24, 1.36, 1.56 for block 2 and 1.25, 1.33, 1.53 for block 3.

So topics do pull apart from pass to pass, as the designs assume. With 20 prompts and one token position this is an indication, not a result.

## Not checked

- Speed after the end-of-pass snapshot was added.
- Capture during long conversations or with images.
- Capture of per-layer outputs, cross-loop read weights, or input shares (needed by the loop inspector).
- Memory with a Swift app and a web view running alongside.
- Behaviour when the machine is under real memory pressure.
- Mapping records to generated tokens: generation makes extra forward calls for prompt chunks and one lookahead step, so records are not one per generated token.
