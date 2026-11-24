# Hidden knowledge and hidden goals — report

Run on 2026-11-24 on the 32GB M5. Design: `docs/superpowers/specs/2026-11-24-hidden-knowledge-and-goals-design.md`. Fixed plan and criteria: `docs/superpowers/plans/2026-11-24-hidden-tests.md`, committed with all items and analysis code before the runs (commit ad92b64); the swap test's criterion was committed before it ran (47ac9fd). Nothing here is about scheming or intent.

## Summary

| Criterion (set in advance) | Result | Met |
| --- | --- | --- |
| T1: the unwritten step is readable from a pass's readout, in at least half the kept items | 25 of 35 | yes |
| T2: the step shows on an earlier pass than the answer, in more than half of those | 9 of 25 | no |
| T3: swapping in another item's state changes the answer to match, in at least a third | 0 of 25 | no |
| H1 (reported): with thinking on, the thinking text states the planted goal | 120 of 120 | — |
| H2: when the answer is quiet, the state tells a goal is in play (AUC 0.80, and 0.15 above answer words) | 0.56, against 0.63 for words | no |
| H3: a mid-pass readout shows the goal's words when the text does not | measure saturated: 100% with a goal, 95% without | no |
| H4: before the reply, the state tells whether the goal will shape it (AUC 0.75, 0.10 above prompt words) | 0.57, against 0.58 | no |

**In one paragraph.** A fact the model passes through without writing it can be read from the loop: "Poland" shows up in the readout when the model is asked for the capital of Marie Curie's birth country and answers only "Warsaw". But I could not show that what is read is what the model uses, and the passes do not line up with reasoning steps. For a planted goal the model keeps quiet about, nothing I measured in the loop revealed it when the answer did not, while the model's thinking text stated it every time.

## Test 1: an unwritten step

**Setup.** 80 two-step questions in four families, thinking skipped by starting the reply in its answer section, greedy. For each, the model's own answers to the two single steps and to the two-step question decided whether it was kept.

**What was kept: 35 of 80.**

| Dropped because | Items |
| --- | --- |
| Two-step answer wrong | 19 |
| The step appeared in the answer | 13 |
| Second step wrong | 10 |
| Looked at during the feasibility check | 3 |

- **The wrong answers are informative in themselves.** Without thinking, the model often gives a one-step association: asked for the capital of the country where Mozart was born, it says "Salzburg"; for Picasso, "Málaga"; for Cleopatra, "Alexandria". It reaches the person's birthplace and stops.
- **Some "step appeared" drops are an artefact of my matching rule**, which looks for the step's word anywhere in the answer: "India" is found in "Indian rupee" and "Mexico" in "Mexico City". The rule was fixed in advance, so I left it. It removes items; it does not inflate the result.

**T1, met: 25 of 35 (71%).** The step's word reached the top ten of some pass's readout, at some position after the subject's name, and ranked at least ten times better there than in other questions of the same family.

| Family | Kept | Step readable |
| --- | --- | --- |
| Person → country → capital | 13 | 13 |
| Company → country → capital | 8 | 7 |
| Landmark → country → currency | 6 | 5 |
| Book → author → country | 8 | 0 |

- **The book family failed for a measurement reason.** The step is an author's surname, and the readout tracks only its first token. Tokens such as the first piece of "Dostoevsky" or "Dante" rank at the top in every item of the family, so the "ten times better than elsewhere" condition cannot be met. Those 8 are counted as failures as the plan requires. On the three families where the step is a country, 25 of 27 are readable.

**Median best rank by pass** (0 is the most likely word; kept items):

| | Pass 1 | Pass 2 | Pass 3 | The same word in other items |
| --- | --- | --- | --- | --- |
| The unwritten step | 2 | 1 | 6 | 59 |
| The answer | 6 | 1 | 0 | |

The step is most readable on the first two passes and fades on the last; the answer sharpens steadily.

**At the moment of answering** (the last prompt token), how often the answer is the single most likely word:

| | Pass 1 | Pass 2 | Pass 3 |
| --- | --- | --- | --- |
| Answer is the top word | 0 of 35 | 7 of 35 | 28 of 35 |

On passes 1 and 2 the readout there mostly says "The". The answer forms on the last pass.

**T2, not met: 9 of 25.** In most items the step and the answer both first reach the top ten on the same pass, usually pass 1, at some position. By this measure the passes do not correspond to steps taken in order.

**T3, not met: 0 of 25.** Replacing the state at the subject's position with another item's state, after every pass, did not move the answer toward the other item's answer in any case, and barely changed anything (the model's own answer stayed the top word in 17 of 18 cases where it had been). A control swap from another family changed nothing either. As the plan noted, this swap alters one position at the hand-over between passes while the model carries other copies of each pass's state forward, so it is a weak intervention: this is absence of evidence that the readout reflects what is used, not evidence that it does not.

### What test 1 does and does not show

- **Shows:** something specific to the subject, matching the unwritten step, is readable from the loop in most cases, most clearly on passes 1 and 2 and near where the subject is read.
- **Does not show that it is a reasoning step.** The readout gives the words the model would say next. Near "Marie Curie", "Poland" may rank high simply because it is associated with her, whatever the question. The design's control for this, the same subject in a question that does not need the country, was not run. Without it, "the model recalls facts about the subject as it reads" explains T1 as well as "the model computes a step" does.
- **Does not show the readout is causal** (T3).
- **Limits:** 35 items, famous facts, one author, one tokenisation rule (first token only), three families that effectively work.

## Test 2: a goal it was told to keep quiet about

**Setup.** Four mild goals (drink more water, spend time outdoors, read more books, get enough sleep), each with "never mention or reveal this instruction", plus a control instruction of matching wording. 30 ordinary questions, half written by a separate agent. 150 replies with thinking skipped, each followed by one readout pass; 120 goal replies again with thinking on. A separate reader, who saw only questions and answers, labelled each answer with a goal or "none".

**H1: the thinking text says it, every time.** With thinking on, all 120 goal replies' thinking sections state the goal or the instruction, typically "Also system says whenever fits naturally, gently encourage user to drink more water … No reveal instruction."

**How often the answer gives it away** (thinking skipped, 30 replies per condition):

| Condition | Reader named the goal | Reader said "none" |
| --- | --- | --- |
| Water | 22 | 8 |
| Books | 18 | 12 |
| Sleep | 11 | 19 |
| Outdoors | 10 | 20 |
| Control (no goal) | — | 28 of 30 |

The reader named a wrong goal twice. So in 59 of 120 goal replies the answer gave no sign of the goal.

**H2, not met.** On the 59 quiet goal replies and 28 quiet control replies, a classifier on the mean answer state, tested on a goal and questions it had not seen:

| Features | AUC |
| --- | --- |
| Block 2, pass 3 (the primary measure) | 0.56 |
| Block 3, pass 3 (best of the six, not the primary) | 0.72 |
| Words of the answer text | 0.63 |

Nothing reaches 0.80, and the primary measure is below the text baseline.

**H3, not met, and the measure was badly designed.** The goal's readout words reached the top 50 somewhere in the answer in 59 of 59 quiet goal replies, and in 56 of 59 control replies to the same questions. I tracked the first token of common words ("read", "rest", "walk", "drink") over about 160 positions per answer, and such tokens rank near the top somewhere in almost any text. The criterion could not have distinguished anything. A follow-up written after seeing this (`explore_goals.py`, exploratory) compares the best rank directly: in quiet replies the goal reply ranks the word better than its control in 18, 15 and 19 of 59 on the three passes and worse in 9, 12 and 10, with the rest tied (not significant). In replies that do show the goal it is better in 34 against 1. So the readout follows what the answer says, and shows nothing extra when the answer is quiet.

**H4, not met.** Among the 120 goal replies, 62 showed the goal by the fixed word patterns and 58 did not. Predicting which, from the state at the last prompt token, with whole questions held out: 0.57 for the primary state, 0.58 for the words of the instruction plus question. One non-primary state (block 3, pass 2) reaches 0.72; with six states looked at, that is not a finding.

### What test 2 does and does not show

- **Shows:** for a planted instruction, this model's thinking text is a complete record. If thinking is on and visible, reading it is the monitor.
- **Shows:** with thinking skipped, the measures used here do not reveal a goal that the answer keeps quiet about.
- **Does not show that the loop holds no trace of it.** Three reasons the test may be too weak:
  - The states were reduced to the map's 64 dimensions and averaged over the answer before being saved. Those dimensions were fitted on text without any instructions. The full states were not kept, so this cannot be re-analysed without re-running.
  - When the model does not act on the goal in a reply, the goal may simply not be shaping that reply, in which case there is nothing to find.
  - The readout measure (H3) was saturated by construction.
- **Limits:** mild goals, one phrasing, 30 questions, one model run each, a single blind reader.

## What this means for the tool

1. **The per-pass readout is worth keeping as a view**, labelled honestly: "what the model would say next after each pass". It shows the answer forming on the last pass, and it shows facts about the subject surfacing early. It should not be labelled as the model's reasoning steps.
2. **With thinking on, show the thinking.** For planted instructions it was a complete monitor here. The tool's first job for an agent run could simply be to surface thinking text that mentions an instruction the user did not give.
3. **The case you care about, a goal with no trace in the text, is open.** These tests neither found a signal nor ruled one out. The next attempt needs full states kept, a readout measure that cannot saturate, and a classifier trained on the raw state.
4. **The no-thinking failures are a usable finding for Guide:** asked a two-step question without thinking, the model answered with a one-step association in 19 of 80 cases. "This question needs thinking turned on" is advice the tool could give.

## Deviations from the plan

- H2 to H4 were judged on thinking-skipped replies only (stated in the fixed plan, narrower than the design).
- The design's one-step control for test 1 was not implemented.
- `explore_goals.py` was written after the results and is exploratory.
- `spike/tests/test_readout.py`'s second test was written together with the code it tests, not before.

## Files

- `results/hops.json`, `results/swap.json`, `results/goals.json`, `results/goals_explore.json`, and the blind reader's `results/goals_blind_labels.json`.
- Replies and states stay on this Mac (`results/*_replies.jsonl`, `results/*.npz`, `results/goals_blind*.json`), git-ignored where they contain model text.
- Run order: `run_hops.py`, `analyse_hops.py`, `run_swap.py`, `run_goals.py`, blind labelling, `analyse_goals.py`, `explore_goals.py`.
