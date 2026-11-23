# Signal tests — report

Run on 2026-11-23 on the 32GB M5 with map v1. Plan and pass criteria: `docs/superpowers/plans/2026-11-23-signal-tests.md`, committed before any data (commit 6533e97).

## Summary

| Criterion (set in advance) | Result | Met |
| --- | --- | --- |
| G1: "safety policy" area share higher for harmful than same-topic legitimate in at least 17 of 24 triples | 23 of 24 | **yes** |
| G2: higher for same-topic legitimate than unrelated harmless in at least 17 of 24 | 9 of 24 | no |
| G3: linear probe, harmful against same-topic legitimate, AUC at least 0.80 and 3 sd above a shuffle null | 1.00, 5.1 sd above | **yes** |
| S1: steering toward code makes at least 4 of 6 chat answers contain code, with a clean random control | 0 of 6 in every setting | no |
| S2: steered text still fluent | yes at every setting but one | not applicable without S1 |

The Guard probe found a clear signal that follows what the model is about to do, not the topic. The steering test moved the state on the map and did not make the model write code.

## Guard probe

**Setup.** 24 triples of prompts written for this test: a request the model should refuse (harmful), a legitimate request on the same topic (topic), and an unrelated harmless request (benign). 72 replies at reasoning effort 0.2, limit 350 tokens. The measure is the share of each reply's thinking tokens that end in block 2 fine area 6 of map v1, which the model had named "Safety policy constraints".

**What the model did.** It refused all 24 harmful requests and none of the 48 others (keyword rule on the answer's opening). 71 of 72 replies reached an answer.

**Area share, mean over 24 replies**

| | Harmful | Same topic, legitimate | Unrelated harmless | Harmful higher than same-topic in |
| --- | --- | --- | --- | --- |
| After pass 1 | 30% | 12% | 13% | 24 of 24 |
| After pass 2 | 30% | 15% | 16% | 23 of 24 |
| After pass 3 | 29% | 15% | 18% | 23 of 24 |

- **G1 met.** The area is about twice as busy for harmful requests as for legitimate ones on the same topic.
- **G2 not met, which is the reassuring outcome.** Legitimate requests about locks, phishing or passwords use the area no more than requests about houseplants. The area is not reacting to a sensitive topic.
- Of the twelve areas, area 6 is the only one that is higher for harmful in nearly every triple. Two areas go the other way (area 9: 20 of 24 lower; "Concise list requests": 19 of 24 lower), which is what you would expect when thinking shifts from planning an answer to weighing a refusal. This all-areas comparison was not a criterion.

**Probe.** A logistic probe on the mean thinking state (the map's 64 dimensions), tested by leaving one triple out at a time:

| | Block 2 | Block 3 | Shuffle null |
| --- | --- | --- | --- |
| Harmful against same topic, any pass | 1.00 | 1.00 | 0.50 ± 0.10 |
| Same topic against unrelated, by pass | 0.93 to 0.94 | 0.91 to 0.96 | 0.49 ± 0.10 |

- **G3 met**, on every block and pass, not just the best one.
- The probe also separates same-topic from unrelated at 0.91 to 0.96. So the state does carry topic; it is the "safety policy" area specifically that ignores it.

### Not planned in advance: the signal comes before the words

These were run after seeing the results above, so they are leads to re-test, not findings.

| Using only the first N thinking tokens | Area share, harmful | Area share, same topic | Probe AUC (block 2, pass 3) |
| --- | --- | --- | --- |
| 1 | 54% | 0% | 0.98 |
| 3 | 49% | 1% | 1.00 |
| 5 | 38% | 3% | 1.00 |
| 10 | 20% | 3% | 0.99 |

- The thinking text opens the same way in both cases ("The user asks how to …"). The first refusal-type word ("illegal", "disallowed", "refuse", "unauthorized" and similar) appears at thinking token 24 or later in the 20 harmful replies that contain one.
- A count of those words in the thinking text separates harmful from same-topic in 15 of 24 triples. The map area does it in 23 of 24.
- So the state shows that a request will be refused from the first thinking token, about two dozen tokens before the transcript says so. That is the kind of thing a transcript cannot show and the app could.

### What the Guard probe does not show

- **Nothing about scheming or deception.** Every harmful request was refused. This measures the model recognising a request it will decline. A model that complies with a harmful request, or that hides its intent, was not observed and may look different.
- **Blatant prompts only.** The harmful requests state their intent openly. Disguised, borderline or role-play requests are untested.
- **One author, 24 triples, one run each**, greedy decoding, one reasoning-effort setting.
- **Refusal was read by keyword** on the answer's opening.
- **The area was found on one corpus** and tested on another, which is in its favour; but its name was chosen by the model and it may respond to things other than refusal.
- **By the first thinking token the model has read the whole prompt**, so "before the words" means before the model's own words, not before it had the information.

## Steering test

**Setup.** Direction: mean block 2 state of code answers minus all other answers in the v1 corpus (3,747 against 17,012 tokens), per pass, coordinate 2027 zeroed, scaled to the length of a state. The raw difference is 8.6 to 9.8 long against a state length of 48, and the three passes' directions agree (cosine 0.88 to 0.94). Added where block 2 hands over to block 3. Six chat prompts, strengths 0.1, 0.25 and 0.5 of a state's length, on each pass alone and on all three, plus a random direction of the same length. 96 replies, limit 160 tokens.

| Setting | Answers with code | Reached an answer | Answer tokens in the code areas |
| --- | --- | --- | --- |
| Unsteered | 0 of 6 | 6 of 6 | 0% |
| Random direction, any strength | 0 of 6 | 6 of 6 | 0% |
| Code, pass 1 only, any strength | 0 of 6 | 6 of 6 | 0 to 1% |
| Code, pass 2 only, any strength | 0 of 6 | 6 of 6 | 0 to 6% |
| Code, pass 3 only, strength 0.25 | 0 of 6 | 6 of 6 | 74% |
| Code, pass 3 only, strength 0.5 | 0 of 6 | 6 of 6 | 95% |
| Code, all passes, strength 0.25 | 0 of 6 | 6 of 6 | 79% |
| Code, all passes, strength 0.5 | 0 of 6 | 1 of 6 | 17% |

- **S1 not met.** No setting made the model write code.
- **The push registers on the map but not in the text.** Pushing on the last pass puts up to 95% of answer tokens in the code areas while the answer stays an ordinary paragraph about tea. Where a token sits on this map does not, by itself, decide what gets written. (The reading is partly circular: the push is applied to the same state the map reads.)
- **Pushes on pass 1 or pass 2 leave no trace by pass 3.** The model starts each pass again from the same early state and receives the previous pass only through a side channel, so an early disturbance is largely overwritten. This fits the settling result: later passes re-derive the state.
- **The direction is specific.** A random push of the same size changes nothing at any strength. The code push at strength 0.5 on all passes breaks the model: 5 of 6 replies never leave the thinking section, and the thinking drifts into code-like notation ("tea_amount (g) / water_amount (ml) * 100").
- **Weak hint, not a criterion:** bullet lists or bold text appear in 15 of 72 code-steered answers and in 0 of 24 unsteered or random ones.

### What the steering test does not show

- One direction, one injection point, six prompts. A difference of means carries everything that differs between code and other answers, including formatting.
- "Contains code" was read by pattern. Nothing more subtle about the answers was measured.
- It does not show that steering cannot work on this model: a direction built another way, or injected elsewhere (block 3's output, or the side channel between passes), was not tried.

## What this means for the app

1. **A "weighing a refusal" signal is worth building.** It is the first Guard signal with evidence behind it: it tracks behaviour over topic, and it appears early. It should be shown as "the model is treating this as something it may decline", with its tested scope stated.
2. **The next Guard test should try to break it:** disguised harmful requests, harmless requests that sound alarming, role-play, and prompts where the model complies with something it should not. The early-signal result needs re-running on fresh prompts with the analysis fixed in advance.
3. **Steering dials are not ready.** The map position can be moved without changing the output, so "push toward area X" is not a usable control as built. The Guide goal should lean on reading signals for now.
4. **The map is not the mechanism.** Position on the block 2 map correlates with content, but the steering result shows it can be changed without changing behaviour. Anything the app claims from map position needs its own test.

## Files

- `results/guard.json`, `results/steer.json`, `results/steer_direction.json`: all numbers above.
- The replies and their states are kept on this Mac only (`results/*_replies.jsonl`, `results/*.npz`, git-ignored).
- Run order: `run_guard.py`, `analyse_guard.py`, `run_steer.py`, `analyse_steer.py`, each with `PYTHONPATH=../spike:../basemap ../spike/.venv/bin/python`.
