# Using the model itself to visualise its latent state — ideas

Brainstormed 2026-11-23. **Nothing here has been tested on Inkling 2 Mini.** These are candidate directions for after the first version, not part of its approved scope.

The ideas are grouped by how much the result can be trusted.

## 1. Signals the model already computes

These are the model's own decisions, so they are faithful by construction and nearly free to capture.

- **Expert routing as a built-in tagging.** Each layer picks 8 of 128 experts per word. That is the model's own classification of what kind of work a word needs.
  - Colour words by expert set.
  - Show which experts change between loop passes.
  - Draw a map of experts that fire together.
  - The counts are small and discrete, which suits the private sharing design better than 4,096 cells.
- **Block attention weights.** The model has a component (`BlockAttentionResidual` in `mlx_lm/models/inkling2/layers.py`) that weighs earlier blocks' outputs against each other. Showing the weights per word would say how much the model leans on the last pass versus earlier ones.
- **Confidence per pass.** How spread out the next-word prediction is after each pass. A second settling signal alongside movement.

Not yet checked: whether the router's choices and the block attention weights are easy to read out in the MLX port, and their exact shapes.

## 2. The model decodes its own states

These need extra forward passes, so they would run on demand for a selected word, not live.

- **"If it stopped here."** Run the final block and output head on the state after pass 1, 2 and 3, and show the three predicted words side by side. Goodfire found this kind of lens (coda or tuned lens) works well on this model where the plain logit lens fails mid-block. The variable-loop sibling `thinkingmachines/TML-d24-varloop-research` could do it natively.
- **Patch and describe.** Insert a captured state into a blank slot in a fresh prompt such as "This is about: ___" and let the model finish the sentence. Gives a plain-language reading of any state at any pass. Related published work: Patchscopes, SelfIE.
- **Patch and continue.** The same, but let the model write freely from the state, to show where that state "wants to go".

## 3. Causal steering

Push the state toward an area and compare the output before and after. This is the only idea here that shows what *changes* the output, not just what correlates with it. It is the most interesting direction because it could tell a user how to use the model better.

### What it could give the user

- **Prompt coaching.** Steering finds which areas improve an answer when pushed toward. The map then shows which wordings land there naturally. The app could say "adding an example moves this prompt toward the area where answers were better".
- **Telling a prompt problem from a model limit.** When an answer is wrong, push toward the relevant area and re-run. If that fixes it, the knowledge was there and the prompt did not reach it. If nothing helps, no rewording will either.
- **Dials in place of prompt-craft.** A few tested directions become sliders such as more concise, more careful or more formal.
- **Which pass matters.** Push on pass 1, 2 or 3 separately. The capture spike suggests topic is set early and tightened later, so early pushes may change what the answer is about and late ones how it is worded. Specific to looped models and little explored.
- **Shared recipes.** The web map could carry directions many people found useful, shared as counts of "this dial helped" and never as text.
- **Self-generated variations (related).** Have the model rewrite the prompt with small edits and see which edits move the state: a sensitivity view over the input words.

### Risks

- **Strength.** Strong pushes usually make output garbled, so each dial needs a safe range.
- **Unclean directions.** An area's centre may mix several things, so a push can have side effects.
- **Thin evidence.** A dial that works on 20 test prompts may not generalise.
- **Loop signal.** Coordinate 2027 dominates the block 3 output on passes 1 and 2 (40% and 34% share in the spike), so pushes may need to leave it alone.

### Cheapest first test

1. Take the spike's chat and code prompts (`spike/prompts.py`).
2. Compute the average difference between their states.
3. Add that difference while running a chat prompt, once per pass, at a few strengths.
4. Read the output and note where it turns toward code and where it breaks.

The capture hook in `spike/latent_capture/capture.py` already sits in the right place (a mirror of `TextModel._run_loop`). It only reads today; the test needs a version that can also add a vector to the state.

## 4. The model as narrator

The least trustworthy group: the model can produce a fluent explanation that is wrong. Anything shown from here should be labelled as the model's guess.

- **Naming areas.** Give the model each area's top words and example contexts and ask for a short name. Done once at map-build time, it replaces the invented area names in the mockups.
- **Explaining a run.** Feed the trajectory as text and ask for a summary.

### Making the names checkable

Naming is the safest use of the narrator, because a name can be tested:

- **Score each name.** Give a second run of the model the name and a mix of words from inside and outside the area, and ask which belong. A name that cannot pick out its own area is dropped or marked uncertain.
- **Name at each level.** Broad areas and the finer areas inside them get separate names, so the nesting on the map is labelled consistently.
- **Name per pass.** If an area's name changes between pass 1 and pass 3, that is itself a finding about what the loop does.
- **Show the evidence.** Clicking a name shows the words and contexts it was made from.
- **Runs at map-build time.** It happens once on the build machine from public text, so it adds nothing to the app's memory use and touches no user data.

## The user's picks (2026-11-23)

The user called all of these useful and singled out two:

- **Causal steering**, because it could show how to use the model better.
- **Model-named areas**, as a good direction.

## Suggested order

1. **Model-named areas** — user pick; solves a problem the map already has, and fits into the base-map build.
2. **Causal steering, cheapest first test** — user pick; highest practical payoff if it works.
3. **Expert routing** — free, faithful and privacy-friendly.
4. **"If it stopped here"** — the most readable answer to "what did each pass add".
5. **Patch and describe** on click — highest upside, highest risk.

## Related notes

- `latent-space-visualisation-and-geometry.md` — methods survey and the Goodfire findings.
- `inkling-2-mini.md` — model architecture.
- `territory-page-future.md` — the deferred regions page.
- `../spike/REPORT.md` — measured results the ideas above lean on.
