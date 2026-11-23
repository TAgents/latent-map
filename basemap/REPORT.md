# Base map v0 — report

Run on 2026-11-23, on the 32GB M5, from the plan in `docs/superpowers/plans/2026-11-23-base-map-v0.md`.

## Verdict

**The thresholds set in the plan are not all met, so by the plan's own rule regions are not yet "visible".** Structure passes clearly on block 2; naming fails.

| Threshold (set before the data) | Block 2 | Block 3 |
| --- | --- | --- |
| Form lift at least 1.5 | **2.51** | **1.98** |
| Topic lift at least 1.5 | **2.16** | 1.21 |
| Mean name score at least 0.5 (chance 0.25) | 0.33 | 0.10 |

What I take from it: block 2 has real, usable regions, and the map of it is worth building on. The naming step works where it produces a name and fails by producing none, for a reason given below. Block 3 needs a different treatment from the one this version gave it.

## What was built

- **Corpus:** 120 texts written by the model itself, 24 topics by 5 forms (explanation, dialogue, code, story opening, numbered list), 300 tokens each.
- **States:** 38,272 token positions, block 2 and block 3 outputs after each of the three passes. 34,977 are reply tokens; 7,503 of those are answer tokens and the rest are the model's thinking.
- **Map, per block:** coordinate 2027 removed, vectors normalised, each pass's mean removed, 64 dimensions by PCA, 256 cells by k-means, cell layout by UMAP, 12 fine areas inside 4 broad areas.
- **Names:** the model names each fine area from half of its distinctive words; a second run must pick the other half out of four word lists.

Files: `bundle_v0/block2.png`, `bundle_v0/block3.png`, `bundle_v0/passes.png`, and the numbers in `bundle_v0/block2.json` and `bundle_v0/block3.json`.

## Findings

### 1. Block 2 has regions that line up with what the text is

Within a fine area, the commonest text form covers 52% of tokens, against 21% for the commonest form overall. For topic it is 9.3% against 4.3% (24 topics, so the base is low and this is the weaker of the two results).

Areas found on block 2, at pass 3:

| Area | Name | Score | Tokens | Answer share | Commonest form | Distinctive words |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | Two-character dialogue crafting | 0.25 | 2,793 | 2% | story | opening, maybe, write, story, dialogue, need, craft, creative |
| 1 | Python code generation | 0.75 | 1,543 | 8% | code | function, comments, with, code, python, maybe, a, include |
| 2 | (none) | 0.00 | 1,571 | 0% | steps | short, paragraph, one, cise, five, clear, keep, list |
| 3 | (none) | 0.00 | 3,119 | 1% | explain | need, user, final, ensure, answer, only, no, request |
| 4 | (none) | 0.00 | 120 | 0% | explain | (only the thinking-start marker) |
| 5 | (none) | 0.00 | 827 | 22% | dialogue | i, such, a, let, in, con, cit, we |
| 6 | Sorting algorithms | 1.00 | 4,818 | 26% | code | n, return, prime, index, if, def, g, time |
| 7 | (none) | 0.00 | 3,027 | 30% | explain | the, roman, and, of, jazz, in, open, e |
| 8 | (none) | 0.00 | 8,127 | 38% | steps | and, to, immune, con, by, the, are, light |
| 9 | Accuracy caveats | 1.00 | 3,652 | 17% | steps | practical, tips, no, facts, matters, matter, related, not |
| 10 | informal chat tokens | 1.00 | 1,731 | 64% | dialogue | i, maya, alex, that, you, sam, yeah, it |
| 11 | (none) | 0.00 | 3,649 | 4% | story | a, the, stake, we, town, because, is, in |

### 2. The map separates thinking from answering

Nobody asked it to. Answer tokens are 21% of the reply tokens overall, but the share per area runs from 0% to 64%. Areas 0 to 4 are almost purely the model planning what to write ("need", "user", "final", "ensure", "answer"); area 10 is mostly finished dialogue. This matters for the Guard goal: where the model is deliberating and where it is committing to output are different places on the map.

### 3. Settling is visible on block 2

| Block 2 | After pass 1 | After pass 2 |
| --- | --- | --- |
| Already in its final cell | 43% | 52% |
| Already in its final fine area | 70% | 77% |

So most words are in the right area from the first pass and tighten within it, which is what the screen designs assume. Form and topic lift barely change by pass on block 2 (form 2.55, 2.52, 2.51; topic 2.04, 2.07, 2.16).

### 4. Block 3 cannot share one map across passes

Only 4% of words are in their final block 3 cell before the last pass, and 74% of block 3's cells are used by essentially one pass. Block 3's output on the last pass is a different kind of state from its output on passes 1 and 2. Removing each pass's mean, which was this version's shortcut, is not enough. Block 3 needs a separate map per pass, as the technical design originally said. Its form lift is 1.20 on passes 1 and 2 and 1.98 on pass 3.

### 5. Model-written names: accurate when given, often not given

- Block 2: 5 of 12 areas got a name. Four of those five scored 0.75 or 1.00.
- Block 3: 2 of 12 got a name.
- The seven unnamed block 2 areas are not a scoring failure. On word lists with no clear common theme, the model thinks for more than 1,200 tokens without answering, even with reasoning effort set to 0. The naming call gives it 400.
- The unnamed areas are mostly the ones whose words are function words or a mix of topics (areas 7 and 8 hold several topics each). "The model cannot name it" is itself a signal that the area is mixed.
- No broad area got a usable name.

### 6. The model thinks first, by default

The chat template adds "Thinking effort level: 0.9" unless told otherwise, and the model then writes a thinking section before its answer. The setting is `reasoning_effort`. An earlier attempt used the wrong keyword, which the template silently ignored, so the whole corpus was written at the default 0.9. That is recorded in each corpus row. The effect is a corpus that is 79% thinking text.

## Limits of this result

- **One small, model-written corpus.** 24 topics, 5 forms, 120 texts. Tokens from the same text sit close together, so some of the area structure may be per-text, not per-form.
- **The topic threshold was easy.** With 24 topics the base rate is 4.3%, so a lift of 2.16 means only 9.3% purity.
- **256 cells, not 4,096.**
- **Area words are dominated by thinking text**, because the corpus is.
- **Nothing here is causal.** Areas line up with form and phase; nothing was changed to see what follows.
- **Names were not reviewed by a person** beyond my reading of the table above.
- **Speed was not re-measured** with every-position capture.

## What this changes for the next step

1. **Use block 2 for the main map.** Give block 3 a map per pass, or leave it to the loop inspector.
2. **Label thinking and answer in the app.** The map already separates them, and the Live screen should show which one a word belongs to.
3. **Make naming cheaper and bounded.** Name only areas that pass a coherence check, cap the thinking, and treat "no name" as "mixed area". Splitting mixed areas further is the other option.
4. **Rebalance the corpus.** Generate with lower reasoning effort so answers are not a fifth of the data, and add texts that are not model-written.
5. **Raise the bar on topic.** Use fewer, broader topic groups or a purity measure that does not depend on the base rate.

## How to run it

From `basemap/`, with the spike's environment:

```
PYTHONPATH=../spike ../spike/.venv/bin/python generate_corpus.py   # about 35 minutes
PYTHONPATH=../spike ../spike/.venv/bin/python collect_states.py    # about 3 minutes, writes 1GB
../spike/.venv/bin/python build_map.py
PYTHONPATH=../spike ../spike/.venv/bin/python name_areas.py        # about 10 minutes
../spike/.venv/bin/python draw_map.py
```
