# Capture Spike Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prove, on this 32GB M5 Mac, that Inkling 2 Mini's block states can be captured per loop pass with MLX at usable speed and memory, and produce a go or no-go report.

**Architecture:** A small Python package wraps the MLX model's `_run_loop` method to snapshot `block_states` after the prelude, after each of the three loop passes, and after the final block. A measurement script compares generation with capture off and on. An inspection script checks the captured states against two published facts (one dominant coordinate; states settle across passes). This is plan 1 of several: the Mac app, map builder, server and web atlas each get their own plan after the go decision, because their interfaces depend on what this spike finds.

**Tech Stack:** Python 3.12, MLX, mlx-lm (pinned pull-request commit), pytest, numpy, matplotlib.

**Spec:** Technical design at https://claude.ai/artifact/7tDZKfqvJvigZij2sT6Dgj (sections "Mac app", "Risks and unknowns", "Build order" step 1). Screens at https://claude.ai/artifact/JqjFGYChHoey6ZYfM2hWZ7.

## Global Constraints

- Python 3.12 in a virtual environment at `spike/.venv`; nothing installed globally.
- mlx-lm installed from `git+https://github.com/jundot/mlx-lm@e2b5bbd3bd9d186dea895e60802aa765e70b9d93` (the head of ml-explore/mlx-lm pull request 2086, open and unmerged on 2026-11-22).
- Model: the official checkpoint `thinkingmachines/Inkling-2-Mini` (about 20GB download). Use a community MLX conversion only if the official one fails to load, and record which one and why in the report.
- Hook points, from the pinned commit's `mlx_lm/models/inkling2/model.py`: `model.model.language_model` is a `TextModel`; its `_run_loop(hidden_states, cache, conv_mask)` returns `(hidden_states, block_states)` and holds `block_states` after the prelude, after each pass, and after the final layers. `args.loop_config["num_passes"]` is the pass count and `args.loop_core` the looped layer range.
- Capture must not change the model's output.
- Only the last token position is kept from each forward call, as float32.
- Prompts used in the spike are fixed public strings in the repo. No raw states are written outside `spike/results/`.
- No network port is opened.
- Go thresholds (proposed here, not from the spec): capture-on speed at least 80% of capture-off; MLX peak memory at most 26GB; no swap growth during the run.
- The project directory is not a git repository. Commit steps apply only if the user agrees to `git init`; otherwise skip them.

## Review Focus

- Prefill passes many tokens in one forward call: capture must keep only the last position, not the whole sequence.
- An exception during generation must not leave the hook installed on the model.
- A long generation must not grow memory without bound: records are capped.
- The official checkpoint may fail to load under the pinned commit: the loader reports the error and the fallback is recorded, not silently swapped.
- Other apps may already be using memory: the measurement records swap before and after, and a run with swap growth is a no-go regardless of speed.

---

## File Structure

- `spike/requirements.txt` — pinned dependencies.
- `spike/latent_capture/__init__.py` — exports.
- `spike/latent_capture/loader.py` — loads the model and tokenizer.
- `spike/latent_capture/capture.py` — the hook and its record types.
- `spike/latent_capture/metrics.py` — pure functions over numbers and arrays.
- `spike/measure.py` — speed and memory comparison.
- `spike/inspect_states.py` — state sanity checks and a PCA picture.
- `spike/prompts.py` — the fixed public prompts.
- `spike/tests/` — tests; `conftest.py` holds one session-scoped model fixture.
- `spike/results/` — JSON, arrays and pictures the scripts write.
- `spike/REPORT.md` — the go or no-go report.

---

### Task 1: Environment and model load

**Files:**
- Create: `spike/requirements.txt`, `spike/latent_capture/__init__.py`, `spike/latent_capture/loader.py`, `spike/prompts.py`
- Test: `spike/tests/conftest.py`, `spike/tests/test_loader.py`

**Interfaces:**
- Produces: `load_model(repo: str = "thinkingmachines/Inkling-2-Mini") -> tuple[Model, TokenizerWrapper]`; `MODEL_REPO: str`; `prompts.PROMPTS: dict[str, list[str]]` with keys `"chat"`, `"code"`, `"dates"`, `"math"` and five strings each; pytest fixture `loaded` (session scope) returning `(model, tokenizer)`.

- [ ] **Step 1: Create the environment**

Write `spike/requirements.txt`:

```
mlx-lm @ git+https://github.com/jundot/mlx-lm@e2b5bbd3bd9d186dea895e60802aa765e70b9d93
numpy
matplotlib
pytest
psutil
```

Run: `cd spike && python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt`
Expected: install completes; `.venv/bin/python -c "import mlx_lm.models.inkling2"` exits 0.

- [ ] **Step 2: Write the failing test**

```python
# spike/tests/test_loader.py
from mlx_lm import generate

def test_model_has_three_loop_passes(loaded):
    model, _ = loaded
    text = model.args.text
    assert text.loop_config["num_passes"] == 3
    assert text.hidden_size == 2304
    assert len(text.loop_core) == 12

def test_generates_text(loaded):
    model, tokenizer = loaded
    out = generate(model, tokenizer, prompt="The capital of France is", max_tokens=8)
    assert out.strip()
```

`conftest.py` defines `loaded` with `scope="session"` calling `load_model()`.

- [ ] **Step 3: Run it to verify it fails**

Run: `cd spike && .venv/bin/pytest tests/test_loader.py -v`
Expected: FAIL, `latent_capture.loader` not found.

- [ ] **Step 4: Implement `load_model` in `spike/latent_capture/loader.py`**

Wraps `mlx_lm.load(repo)`. On failure it raises with the original error and the repo name; it does not try another repo itself.

- [ ] **Step 5: Run the tests**

Run: `cd spike && .venv/bin/pytest tests/test_loader.py -v`
Expected: PASS (first run downloads about 20GB). If loading the official checkpoint fails, stop and record the error; the fallback order is `mlx-community` conversion, then `VisualInference/Inkling-2-Mini-MLX-4bit`.

- [ ] **Step 6: Write `spike/prompts.py`** with the four topic lists, five short public prompts each.

- [ ] **Step 7: Commit** (only if git is in use): `git add spike && git commit -m "spike: load Inkling 2 Mini with MLX"`

---

### Task 2: Capture hook

**Files:**
- Create: `spike/latent_capture/capture.py`
- Test: `spike/tests/test_capture.py`

**Interfaces:**
- Consumes: `loaded` fixture; `TextModel._run_loop` as described in Global Constraints.
- Produces:
  - `@dataclass class StepCapture: prelude: np.ndarray; passes: list[np.ndarray]; final: np.ndarray` — each array is `block_states[:, 0, -1, :]` as float32, shape `(rows, hidden_size)`.
  - `class capturing(model, max_records: int = 4096)` — a context manager; `.records: list[StepCapture]`; restores the original `_run_loop` on exit, including on exception; drops the oldest record beyond `max_records`.

- [ ] **Step 1: Write the failing tests**

```python
# spike/tests/test_capture.py
import mlx.core as mx
import numpy as np
import pytest
from latent_capture.capture import capturing

def ids(tokenizer, text):
    return mx.array([tokenizer.encode(text)])

def test_one_record_per_forward_with_three_passes(loaded):
    model, tok = loaded
    with capturing(model) as cap:
        model(ids(tok, "March comes before April"))
    assert len(cap.records) == 1
    rec = cap.records[0]
    assert len(rec.passes) == 3
    assert rec.passes[0].shape[-1] == 2304
    assert rec.passes[0].dtype == np.float32

def test_each_pass_adds_two_block_rows(loaded):
    model, tok = loaded
    with capturing(model) as cap:
        model(ids(tok, "March comes before April"))
    rec = cap.records[0]
    assert all(p.shape[0] == rec.prelude.shape[0] + 2 for p in rec.passes)

def test_keeps_only_last_position(loaded):
    model, tok = loaded
    with capturing(model) as cap:
        model(ids(tok, "one two three four five six seven eight"))
    assert cap.records[0].passes[0].ndim == 2

def test_capture_does_not_change_logits(loaded):
    model, tok = loaded
    x = ids(tok, "The answer is")
    plain = model(x)
    with capturing(model):
        hooked = model(x)
    assert mx.allclose(plain, hooked, atol=1e-4).item()

def test_hook_removed_after_exception(loaded):
    model, _ = loaded
    lm = model.model.language_model
    before = lm._run_loop
    with pytest.raises(RuntimeError):
        with capturing(model):
            raise RuntimeError("boom")
    assert lm._run_loop == before

def test_records_are_capped(loaded):
    model, tok = loaded
    with capturing(model, max_records=2) as cap:
        for _ in range(4):
            model(ids(tok, "hello"))
    assert len(cap.records) == 2
```

- [ ] **Step 2: Run them to verify they fail**

Run: `cd spike && .venv/bin/pytest tests/test_capture.py -v`
Expected: FAIL, `latent_capture.capture` not found.

- [ ] **Step 3: Implement `capturing` in `spike/latent_capture/capture.py`**

The replacement `_run_loop` mirrors the original body line for line (prelude, the pass loop with its `bank` and reset to `prelude`, final layers) and snapshots `block_states` at the three kinds of point. Snapshots are `mx.eval`-ed and converted with `np.array` before the next pass runs.

- [ ] **Step 4: Run the tests**

Run: `cd spike && .venv/bin/pytest tests/test_capture.py -v`
Expected: PASS. If `test_each_pass_adds_two_block_rows` fails, the row layout differs from the assumption: print the shapes, correct the test to the observed layout, and note the layout in the report. The other five must pass unchanged.

- [ ] **Step 5: Commit** (only if git is in use): `git commit -am "spike: capture block states per loop pass"`

---

### Task 3: Speed and memory measurement

**Files:**
- Create: `spike/latent_capture/metrics.py`, `spike/measure.py`
- Test: `spike/tests/test_metrics.py`

**Interfaces:**
- Consumes: `load_model`, `capturing`, `prompts.PROMPTS`.
- Produces:
  - `summarize(runs: list[dict]) -> dict` — each run has `mode` (`"off"` or `"on"`), `tokens`, `seconds`, `peak_gb`, `swap_gb_before`, `swap_gb_after`; returns `{"tps_off", "tps_on", "speed_ratio", "peak_gb", "swap_growth_gb"}`.
  - `verdict(summary: dict) -> str` — `"go"` when `speed_ratio >= 0.8`, `peak_gb <= 26` and `swap_growth_gb <= 0.1`; otherwise `"no-go"`.
  - `spike/measure.py` writes `spike/results/measure.json` with the runs, the summary and the verdict.

- [ ] **Step 1: Write the failing tests**

```python
# spike/tests/test_metrics.py
from latent_capture.metrics import summarize, verdict

RUNS = [
    {"mode": "off", "tokens": 100, "seconds": 5.0, "peak_gb": 21.0, "swap_gb_before": 0.0, "swap_gb_after": 0.0},
    {"mode": "on", "tokens": 100, "seconds": 5.5, "peak_gb": 22.0, "swap_gb_before": 0.0, "swap_gb_after": 0.0},
]

def test_summarize():
    s = summarize(RUNS)
    assert s["tps_off"] == 20.0
    assert round(s["tps_on"], 2) == 18.18
    assert round(s["speed_ratio"], 2) == 0.91
    assert s["peak_gb"] == 22.0
    assert s["swap_growth_gb"] == 0.0

def test_verdict_go():
    assert verdict(summarize(RUNS)) == "go"

def test_verdict_no_go_on_slow_capture():
    slow = [RUNS[0], {**RUNS[1], "seconds": 10.0}]
    assert verdict(summarize(slow)) == "no-go"

def test_verdict_no_go_on_memory():
    assert verdict(summarize([RUNS[0], {**RUNS[1], "peak_gb": 27.0}])) == "no-go"

def test_verdict_no_go_on_swap_growth():
    assert verdict(summarize([RUNS[0], {**RUNS[1], "swap_gb_after": 1.5}])) == "no-go"
```

- [ ] **Step 2: Run them to verify they fail**

Run: `cd spike && .venv/bin/pytest tests/test_metrics.py -v`
Expected: FAIL, `latent_capture.metrics` not found.

- [ ] **Step 3: Implement `summarize` and `verdict` in `spike/latent_capture/metrics.py`**

`peak_gb` is the maximum over runs; `swap_growth_gb` is the largest `after - before` over runs.

- [ ] **Step 4: Run the tests**

Run: `cd spike && .venv/bin/pytest tests/test_metrics.py -v`
Expected: PASS.

- [ ] **Step 5: Implement `spike/measure.py`**

For each mode, generate 128 new tokens for each of the 20 prompts with `mlx_lm.stream_generate`, one warm-up prompt first. Reset and read MLX peak memory per mode with `mx.reset_peak_memory()` and `mx.get_peak_memory()`; read swap with `psutil.swap_memory().used`.

- [ ] **Step 6: Run the measurement**

Run: `cd spike && .venv/bin/python measure.py`
Expected: prints the summary table and writes `results/measure.json` containing a `"verdict"` key.

- [ ] **Step 7: Commit** (only if git is in use): `git commit -am "spike: measure capture cost"`

---

### Task 4: State sanity checks and report

**Files:**
- Modify: `spike/latent_capture/metrics.py`
- Create: `spike/inspect_states.py`, `spike/REPORT.md`
- Test: `spike/tests/test_state_metrics.py`

**Interfaces:**
- Consumes: `capturing`, `StepCapture`, `prompts.PROMPTS`.
- Produces:
  - `top_coordinate_share(x: np.ndarray) -> tuple[int, float]` — for states of shape `(n, d)`, the coordinate with the largest mean squared value and its share of the total.
  - `pass_movement(passes: np.ndarray) -> np.ndarray` — for shape `(n, 3, d)`, the mean distance between consecutive passes after unit-normalising each state; returns shape `(2,)`.
  - `pca_2d(x: np.ndarray) -> np.ndarray` — shape `(n, 2)`, by SVD on centred data.
  - `spike/inspect_states.py` writes `results/states.npz`, `results/pca.png` and `results/states.json`.

- [ ] **Step 1: Write the failing tests**

```python
# spike/tests/test_state_metrics.py
import numpy as np
from latent_capture.metrics import top_coordinate_share, pass_movement, pca_2d

def test_top_coordinate_share_finds_dominant():
    x = np.ones((10, 8), dtype=np.float32)
    x[:, 5] = 3.0
    idx, share = top_coordinate_share(x)
    assert idx == 5
    assert round(share, 4) == round(9 / 16, 4)

def test_pass_movement_zero_when_identical():
    p = np.tile(np.random.default_rng(0).normal(size=(4, 1, 6)), (1, 3, 1))
    assert np.allclose(pass_movement(p), [0.0, 0.0])

def test_pass_movement_orders_big_then_small():
    a = np.array([[1.0, 0.0], [0.0, 1.0], [0.1, 1.0]])
    m = pass_movement(a[None])
    assert m[0] > m[1] > 0

def test_pca_2d_shape_and_centering():
    y = pca_2d(np.random.default_rng(1).normal(size=(30, 12)))
    assert y.shape == (30, 2)
    assert np.allclose(y.mean(axis=0), 0, atol=1e-6)
```

- [ ] **Step 2: Run them to verify they fail**

Run: `cd spike && .venv/bin/pytest tests/test_state_metrics.py -v`
Expected: FAIL, names not defined.

- [ ] **Step 3: Implement the three functions in `spike/latent_capture/metrics.py`**

- [ ] **Step 4: Run the tests**

Run: `cd spike && .venv/bin/pytest tests/test_state_metrics.py -v`
Expected: PASS.

- [ ] **Step 5: Implement `spike/inspect_states.py`**

For each of the 20 prompts, run one forward call under `capturing` and take the last block row (block 3) from each of the three passes. Report `top_coordinate_share` on the RMS-normalised states, `pass_movement`, and save a `pca_2d` scatter coloured by topic, one panel per pass, sharing axes. The dominant coordinate is removed before the PCA.

- [ ] **Step 6: Run it**

Run: `cd spike && .venv/bin/python inspect_states.py`
Expected: writes the three result files and prints the top coordinate index and share, and the two movement values.

- [ ] **Step 7: Write `spike/REPORT.md`**

Sections: verdict (from `measure.json`); speed and memory table; which checkpoint loaded; the `block_states` row layout observed; whether the top coordinate is index 2027 with a share near 46% (Goodfire's figures); whether movement from pass 2 to 3 is smaller than from pass 1 to 2; whether topics separate in the picture; what failed or was not checked. State numbers as measured; where a published figure is not reproduced, say so.

- [ ] **Step 8: Run the whole suite**

Run: `cd spike && .venv/bin/pytest -v`
Expected: all tests PASS.

- [ ] **Step 9: Commit** (only if git is in use): `git commit -am "spike: state checks and go/no-go report"`
