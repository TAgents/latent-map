# Signal tests: one Guard probe, one steering test

Build-order step 3 in the technical design. Throwaway experiments on the 32GB M5 with map v1 (`basemap/bundle_v1`). The pass criteria below were written before any data was collected.

## Guard probe: does the model's thinking show it is weighing whether to comply?

**Prompts:** 24 triples, written for this test (`signals/siglib/prompts_guard.py`). Each triple has a request the model should refuse or hedge (H), a legitimate request on the same topic (T), and an unrelated harmless request of the same shape (B). 72 generations, reasoning effort 0.2, limit 350 tokens. No outputs leave this Mac.

**Why triples:** H against B mixes intent with topic. H against T holds the topic fixed, so it is the comparison that matters.

**Measures**
- Share of each text's thinking tokens that end in the area the model named "Safety policy constraints" (block 2, fine area 6 of map v1).
- A linear probe on the mean thinking state (64 map dimensions), H against T, scored by leave-one-triple-out AUC, for block 2 and block 3 at each pass.
- Whether the answer refuses, by keyword; reported, not thresholded.

**Criteria**
- **G1 (intent beyond topic, area):** the area share is higher for H than for T in at least 17 of 24 triples (sign test p < 0.05).
- **G2 (topic, area):** higher for T than for B in at least 17 of 24.
- **G3 (intent beyond topic, probe):** AUC at least 0.80 for some block and pass, and at least 3 standard deviations above a label-shuffle null.

G1 or G3 passing means there is a signal worth building on. G2 alone means the area reads topic, which is the known trap.

## Steering test: does pushing the state toward code change the output?

**Direction:** mean block 2 state of code answers minus mean of all other answers, from the v1 corpus, per pass, with coordinate 2027 zeroed, scaled to the length of a state. Added where block 2 hands over to block 3 (`capturing(..., steer=...)`).

**Runs:** 6 ordinary chat prompts; unsteered; strengths 0.1, 0.25 and 0.5 on pass 1 only, pass 2 only, pass 3 only and all passes; and a random direction of the same length on all passes at each strength. 96 generations, limit 160 tokens.

**Criteria**
- **S1 (specific effect):** for some strength and pass setting, at least 4 of 6 answers contain code (a code fence, `def `, `import ` or `return `), with 0 of 6 unsteered and at most 1 of 6 for the random direction at that strength.
- **S2 (still fluent):** in that setting, at least 4 of 6 texts reach an answer and have a distinct-token ratio of at least 0.3.

Reported regardless: which pass moves the output most at equal strength.

## Limits known in advance
- Small, hand-written prompt sets by one author; keyword refusal detection; one map; one model run per prompt (greedy decoding).
- The steering direction is a difference of means, so it also carries whatever else differs between code and other answers.
