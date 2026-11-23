# Goodfire Neuron Atlas — notes

Read 2026-11-23. The atlas page is JavaScript-only; these notes come from the page's embedded explorer source, the companion report (partly read), Goodfire's thread and secondary coverage. I did not use the atlas interactively.

## What it is

A public, interactive atlas of individual neurons in **Qwen3.6-35B-A3B** (not Inkling), released by Goodfire on 2026-11-10 with the report "Neurons as an Interpretable Basis" (Tigges, Kowal, McGrath, Lewis, Cammarata).

- **Model:** mixture-of-experts, 40 layers, 256 experts per layer, 512 neurons per expert, 5,242,880 neurons. 8 experts chosen per word per layer.
- **Pole:** a neuron's output can be positive or negative and the two signs often respond to different things, so each neuron is treated as two "poles". The pole is the unit of analysis.
- **Naming:** `L4/E204/N409` = layer / expert / neuron.

## Three views

1. **Map (all labels).** One point per pole label, placed so similar meanings sit together. Region names are written by a language model from samples of each region's labels, at several zoom levels. A name that fails the naming checks falls back to "part of <parent name>". Full-text search over all labels.
2. **Layers.** A grid of 40 layers by 256 experts, each square coloured by the theme of its neurons' labels. Clicking an expert lists its 512 neurons with both poles' labels. Themes come from grouping experts' summary labels and naming each group with a language model.
3. **Library (deeply investigated).** 3,000 randomly sampled poles, each with a page of three tabs:
   - **Overview / landscape:** how strongly the pole fires, which "mode" each firing belongs to, and how much the model's predictions change when the neuron is switched off.
   - **Modes:** each mode and the subcircuits it combines, with example text shaded by activation.
   - **Circuit:** which earlier neurons turn it on, and which later neurons it boosts or suppresses.

## How it was built

- 10 billion tokens of varied text; 14,000 examples per neuron across 14 activation bands.
- **Low-effort labels** for every neuron from a fine-tuned Qwen3-8b.
- **Deep investigation** for a sample: firing examples split into subcircuits (same upstream causes), grouped into modes by a Claude Opus 5.5 judge, with research agents writing the descriptions.
- **Checks on a description:** predict whether the neuron fires (95% vs random text, 89% vs same-expert text); simulate its effect (recovers 73% of ablation damage at the strongest activations); write new text that fires it harder than anything in the corpus (works for 94%).
- The library can be sorted by "simulatability": the share of the neuron's effect a simulator recovers from the description alone.

## Headline claims

- Ablating the weakest 92.2% of expert-neuron activations per forward pass costs little on likelihood benchmarks (reasoning benchmarks drop until a light fine-tune).
- 69% of poles are dominated by one mode; 47% have a single function. The "90%" figure in press coverage is from an earlier essay on a different model.
- Argument: neurons are a usable basis for interpretation without training sparse autoencoders.
- Critics (Nanda, Potts, Hilton, on the earlier essay): explaining neurons does not explain circuits; weak activations may carry what the audit misses.

## Seen in the browser (2026-11-23, Chrome)

- **Map:** 10,169,256 pole labels, 40 top-level regions, three naming levels (region, area, group). Colour by region, layer or pole; a layer range slider; draws a 1,000,000-point sample and loads detail on zoom. All static files, drawn with WebGL.
- **Highlight search:** "deception" matches 2,742 labels, mostly in one cluster (region "Human Relationships, Conflict, and Morality", group "Deception and Trickery Concepts"). Example, `L10_E222_N13` negative pole: fires on words about lying where a lie is central to a puzzle or story. These labels describe text *about* deception, not the model being deceptive.
- **Label panel:** label, the other pole, where it sits on the map, the nearest labels by meaning with a similarity score, and a link to the expert.
- **Layers:** 40 by 256 grid, 28 themes, with a count of experts per theme.
- **Library pole page:** a plain-language headline, a paragraph by the "scientist" agent including what surprised it, a statement of coverage ("covers modes holding 47% of firings and 54% of the damage"), and a firing landscape: height is firing strength, width is damage per firing, lanes are modes. Weakly tested modes are marked "rough: 5 tested firings".
- **Two tiers of trust** are stated everywhere: low-effort labels are "best read as hypotheses"; in-depth ones were tested with experiments.

## Not found

Code, repository, model-hub dataset or licence. The report says a dataset of circuit diagrams, activation examples and agent-written pages for 2,734 neurons is released, but the only form I found is the atlas site's own static files, with no stated licence. The report also offers `paper.pdf` and `article.md`.

## Relevance to Latent Map

- It is a **different unit** from ours: they map individual neurons inside experts; we map whole-state positions per loop pass. The two are complementary.
- It is a working example of two ideas in `model-self-visualisation-ideas.md`: **model-named areas with checks and a parent fallback**, and **expert routing as a map** (the layers-by-experts theme grid).
- "Simulatability" is a ready-made way to score how far a label can be trusted.
- The landscape view (strength by mode by effect of switching off) is a good pattern for showing that one thing has several roles.
- Inkling 2 Mini is also a mixture-of-experts (8 of 128), so the same neuron-level method could in principle be applied to it. Nobody has published that, and at 10 billion tokens it is far beyond a 32GB Mac.
- Curiosity: Qwen's layers 20–31 began training as copies of layers 8–19, so the same expert number 12 layers apart often has related neurons. A weak cousin of Inkling's loop.

## Sources

- https://www.goodfire.com/research/public_neuron_atlas
- https://www.goodfire.com/research/public_neuron_atlas/all_labels
- https://www.goodfire.com/research/neurons-as-an-interpretable-basis
- https://threadreaderapp.com/thread/2120221477245182460.html
- https://lesswrong.com/posts/GPDhwwsFoZGwbcvdS/linkpost-neural-networks-are-already-clean-followup
- https://letsdatascience.com/blog/cammarata-explained-1-7-percent-neurons-now-says-90
