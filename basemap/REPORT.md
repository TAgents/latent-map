# Base map v0 — report

Run on 2026-11-23, on the 32GB M5, from the plan in `docs/superpowers/plans/2026-11-23-base-map-v0.md`. Revised the same day after an independent review; the review's main corrections are marked.

## Verdict

**The thresholds set in the plan are not all met, so by the plan's own rule regions are not yet "visible".**

| Threshold (set before the data) | Block 2 | Block 3 |
| --- | --- | --- |
| Form lift at least 1.5 | **2.51** | **1.98** |
| Topic lift at least 1.5 | 2.16, but see below | 1.21 |
| Mean name score at least 0.5 (chance 0.25) | 0.33 | 0.09 |

**The lift thresholds were badly chosen (review correction).** Tokens from one text sit close together, so even labels with no link to the map score well above 1.0. Shuffling labels between whole texts gives the real chance level:

| Lift on fine areas | Observed | Chance (mean ± sd) | Times chance |
| --- | --- | --- | --- |
| Block 2, form, all reply tokens | 2.51 | 1.28 ± 0.06 | 1.97 |
| Block 2, form, thinking tokens only | 2.10 | 1.23 ± 0.06 | 1.71 |
| Block 2, form, answer tokens only | 1.46 | 1.14 ± 0.07 | 1.28 |
| Block 2, topic, all reply tokens | 2.16 | 1.72 ± 0.07 | 1.26 |
| Block 2, topic, answer tokens only | 2.38 | 1.75 ± 0.17 | 1.36 |
| Block 3, form, all reply tokens | 1.98 | 1.16 ± 0.04 | 1.71 |
| Block 3, topic, all reply tokens | 1.21 | 1.20 ± 0.02 | 1.01 |

Read honestly:

- **Block 2 carries text form clearly**, at about twice chance. A good part of that comes from the thinking text restating the request ("write a dialogue", "numbered list", "python function"). On answer text alone the effect is modest, 1.28 times chance.
- **Block 2 carries topic weakly**, about 1.3 times chance. The plan's 1.5 threshold sits below what random topic labels score (1.72), so "passing" it means nothing.
- **Block 3 carries form, and no topic at all** on this map.

All of these numbers are written by `map_stats.py` to `bundle_v0/<block>_stats.json`.

## What was built

- **Corpus:** 120 texts written by the model itself, 24 topics by 5 forms (explanation, dialogue, code, story opening, numbered list), capped at 300 tokens each.
- **States:** 38,272 token positions, block 2 and block 3 outputs after each of the three passes. 34,977 are reply tokens; 7,503 of those are answer tokens and the rest are the model's thinking.
- **Map, per block:** coordinate 2027 removed, vectors normalised, each pass's mean removed, 64 dimensions by PCA, 256 cells by k-means, cell layout by UMAP, 12 fine areas inside 4 broad areas.
- **Names:** the model names each fine area from half of its distinctive words; a second run must pick the other half out of four word lists, 8 trials per name.

Pictures: `bundle_v0/block2.png`, `bundle_v0/block3.png`, `bundle_v0/passes.png`.

## Findings

### 1. Areas on block 2

| Area | Name | Score | Tokens | Answer share | Commonest form | Distinctive words |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | Two-character dialogue crafting | 0.25 | 2,793 | 2% | story | opening, maybe, write, story, dialogue, need, craft, creative |
| 1 | Python code generation | 0.88 | 1,543 | 8% | code | function, comments, with, code, python, maybe, a, include |
| 2 | (none) | 0.00 | 1,571 | 0% | steps | short, paragraph, one, cise, five, clear, keep, list |
| 3 | (none) | 0.00 | 3,119 | 1% | explain | need, user, final, ensure, answer, only, no, request |
| 4 | (none) | 0.00 | 120 | 0% | explain | (only the thinking-start marker) |
| 5 | (none) | 0.00 | 827 | 22% | dialogue | i, such, a, let, in, con, cit, we |
| 6 | Sorting algorithms | 1.00 | 4,818 | 26% | code | n, return, prime, index, if, def, g, time |
| 7 | (none) | 0.00 | 3,027 | 30% | explain | the, roman, and, of, jazz, in, open, e |
| 8 | (none) | 0.00 | 8,127 | 38% | steps | and, to, immune, con, by, the, are, light |
| 9 | Accuracy caveats | 1.00 | 3,652 | 17% | steps | practical, tips, no, facts, matters, matter, related, not |
| 10 | informal chat tokens | 0.88 | 1,731 | 64% | dialogue | i, maya, alex, that, you, sam, yeah, it |
| 11 | (none) | 0.00 | 3,649 | 4% | story | a, the, stake, we, town, because, is, in |

One of the twelve areas (4) is spent on a single chat-format marker token. Marker tokens should be left out of the fit.

### 2. Answer tokens are enriched in some areas (review correction: not a clean separation)

- Answer tokens are 21.5% of reply tokens. Their share per fine area runs from 0% to 64%.
- As a classifier this is weak at area level: the area's answer share ranks an answer token above a thinking token 78% of the time (0.5 is chance). At cell level it is 87% on block 2 and 94% on block 3, so block 3 is the better separator.
- Only one block 2 area is majority-answer. The plan's own statistic, phase lift, is 1.02 and 1.00, which says areas do not split the two by majority.
- **It is confounded with text form.** 45 of the 120 texts never reach an answer within 300 tokens, and no story text does. Answer tokens by form: dialogue 2,937, explanation 2,485, list 1,109, code 972, story 0. So "0% answer" in the story areas and "64% answer" in the dialogue area partly restate form.

This is worth a proper test with a balanced corpus. It is not yet something to build a Guard signal on.

### 3. Settling is visible on block 2

| Block 2 | After pass 1 | After pass 2 |
| --- | --- | --- |
| Already in its final cell | 43% | 52% |
| Already in its final fine area | 70% | 77% |

The reviewer checked the area figure against shuffles: 32% when pass-1 areas are shuffled within a text and 12% when fully shuffled, so 70% is real. Whether words then "tighten" inside the area was not measured. Form lift barely changes by pass on block 2 (2.55, 2.52, 2.51).

### 4. Block 3's passes do not share this map (review correction: cause not established)

- Only 4% of words are in their final block 3 cell before the last pass, and 26% in their final area.
- 55% of block 3's used cells take at least 95% of their points from one pass, and 68% are used either only by the last pass or never by it. (An earlier draft said 74%; that figure was not reproducible.)
- Passes 1 and 2 use only 120 and 123 of the 256 cells, because the last pass has more than twice their spread and takes most of the cell centres. Their form lift of 1.20 therefore says little about whether those passes carry form structure.
- So: removing each pass's mean is not enough, and the scale differs by pass too. Per-pass scaling, or a separate fit per pass, was not tried. "Block 3 needs a map per pass" is the likely fix, not a demonstrated one.

### 5. Model-written names

- Block 2: 5 of 12 areas got a name; four of the five scored 0.88 or 1.00. Block 3: 2 of 12.
- **A score shows the name tells this area's words apart from three other areas' words. It does not show the name is right (review correction).** Area 6 is named "Sorting algorithms" and scores 1.00, but its words (`n, return, prime, index, if, def`) describe code in general. Area 0 is named for dialogue and its commonest form is story.
- **Most failures are the model not answering.** On block 2, 10 of 15 naming calls and 6 of 40 scoring calls hit the 400-token limit still thinking, with reasoning effort at 0. In one check outside the committed run, two word lists produced over 1,200 thinking tokens with no answer. Area 0's score of 0.25 is mostly unanswered scoring calls, not wrong picks. Every reply is saved in `bundle_v0/<block>_naming_log.json`.
- Whether "no name" means "mixed area" is not established. Unnamed areas have lower form purity on average, but area 11 is 87% story and unnamed. The word ranking favours function words ("the", "a", "and"), which is at least as likely a cause.
- No broad area on block 2 got a name. One on block 3 was named "English stopwords" and was not scored.

### 6. The model thinks first, by default

The chat template adds "Thinking effort level: 0.9" unless told otherwise, and the model writes a thinking section before its answer. The setting is `reasoning_effort`. The corpus was generated by a version of the script that passed the wrong keyword, which the template silently ignored, so the texts were written at the default 0.9; the corpus rows were corrected afterwards to say so, and the committed script now passes 0.9 explicitly. The result is a corpus that is 79% thinking text.

## Limits of this result

- **One small, model-written corpus.** 24 topics, 5 forms, 120 texts, unbalanced between thinking and answer and with no answers at all for stories.
- **256 cells, not 4,096.** The 64 dimensions keep about 43% of the variance on block 2 (reviewer's figure).
- **Nothing here is causal.** Areas line up with form; nothing was changed to see what follows.
- **Names were not reviewed by a person** beyond my reading of the table above.
- **Speed was not re-measured** with every-position capture.
- **The spike tests were not re-run by the reviewer**, and the reviewer did not look at the pictures.

## What this changes for the next step

1. **Use block 2 for the main map.** Try per-pass scaling or a per-pass fit for block 3 before deciding what to do with it.
2. **Fix the corpus first.** Lower reasoning effort and a higher token limit so every text reaches its answer, balanced across forms, plus texts that are not model-written. Most open questions above wait on this.
3. **Judge structure against the shuffle chance level**, not a fixed threshold.
4. **Leave chat-marker tokens out of the fit**, and rank area words by a measure that does not favour function words.
5. **Bound the naming.** Skip areas that fail a coherence check, give the scorer a same-form distractor, and have a person check names; a high score is not enough.
6. **Test thinking against answer properly** on the balanced corpus, controlling for form, before using it in the app.

## How to run it

From `basemap/`, with the spike's environment:

```
PYTHONPATH=../spike ../spike/.venv/bin/python generate_corpus.py   # about 35 minutes
PYTHONPATH=../spike ../spike/.venv/bin/python collect_states.py    # about 3 minutes, writes 1GB
../spike/.venv/bin/python build_map.py                             # resets names; run name_areas.py after it
PYTHONPATH=../spike ../spike/.venv/bin/python name_areas.py        # about 25 minutes
../spike/.venv/bin/python map_stats.py
../spike/.venv/bin/python draw_map.py
```
