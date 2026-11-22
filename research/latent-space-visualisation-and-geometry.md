# Latent-space visualisation, live tagging, and the geometry of language

Researched 2026-11-22. Companion to `inkling-2-mini.md`. Most sources are 2026 preprints or lab write-ups; the arXiv papers were read at abstract level unless noted, so treat specific numbers as the authors' claims.

## The short version

- There are four families of method for seeing or labelling what a model is doing as it runs: lenses, probes, feature dictionaries (sparse autoencoders), and trajectory geometry. They differ mainly in cost per token and in whether they need training first.
- For Inkling 2 Mini specifically, only one interpretability study exists (Goodfire, 2026-11-20). It released no code or artifacts, and it found the plain logit lens does not work on mid-block layers.
- The 2026 geometry literature has moved from "concepts are directions" to "concepts are often curved manifolds", and several papers trace that shape back to word co-occurrence statistics.
- Practical constraint: Ollama and llama.cpp do not expose hidden states. Any of this needs the PyTorch/transformers path (or NDIF for remote access). Whether that path runs the NVFP4 weights on a 32GB Mac is unverified.

## 1. Options for reading the latent space during inference

| Family | What you get per token | Needs training? | Cost | Fit for live use |
|---|---|---|---|---|
| Logit lens | Top tokens each layer "would predict" | No | One unembedding per layer | Cheap, but poor on Inkling's mid-block layers |
| Tuned lens | Same, with a learned affine map per layer | Yes, small | Same as logit lens at run time | Good; Goodfire reports it works "very well" on Inkling |
| Coda lens | Hidden state decoded by running the model's own last block | No | One extra block-4 pass per readout | Good; specific to looped models with a decoder block |
| Linear probes | A score per concept you chose in advance (topic, sentiment, truthfulness, safety) | Yes, tiny, needs labelled examples | Negligible | Best option for "tagging" known concepts |
| Manifold probes | Position along a curved concept (time, space) | Yes | Low | Research-grade, single paper |
| Sparse autoencoders / transcoders | Thousands of unsupervised named features, a few active per token | Yes, expensive | Moderate | Best for open-ended discovery; none exist for Inkling yet |
| Attribution graphs | Which features caused which, per prompt | Needs transcoders | High, offline | Not live |
| Chain lens | Short multi-token phrases for a hidden state | No (needs a multi-token-prediction head) | ~100x a single-token lens | Borderline for live use; interim draft |
| Activation oracles / natural language autoencoders | A paragraph of free text describing an activation | Yes, very expensive | Hundreds of generated tokens per activation | Offline only |
| Trajectory geometry (PCA, curvature, distance to fixed point) | A path through space per token, per layer or loop | No | Low | Best for the "map" picture |

Notes on each:

- **Lenses.** Logit lens and tuned lens are the baseline. Patchscopes generalises them by patching an activation into a different prompt.
- **Probes.** Hidden-state probes are already used for streaming moderation, producing per-token scores so output can be stopped early. A 2025 comparison found prompted probing best when compute allows and SAE-based probes better than raw-activation probes when it does not.
- **Sparse autoencoders.** The open tooling is Neuronpedia (feature browser and hosted attribution graphs), circuit-tracer (reference implementation for attribution graphs), EleutherAI's sparsify, and CLT-Forge for cross-layer transcoders. Recent critiques: attribution graphs can flip depending on which features sit near the TopK cutoff, and cross-layer transcoders may learn unfaithful circuits.
- **Turn-averaged SAEs.** Anthropic Fellows trained SAEs on activations averaged over a whole conversation turn, giving a handful of features per turn instead of per token. Useful if you want coarse tags per response.
- **Natural language autoencoders (Anthropic, May 2026).** One model turns an activation into text, a second turns the text back into the activation; trained jointly with RL. Reconstruction reaches 0.6 to 0.8 fraction of variance explained. Stated limits: confabulation and cost. Training code and a Neuronpedia front end are public.
- **Chain lens (LessWrong draft, 2026-11-11).** Rolls out short token chains from a hidden state using the model's multi-token-prediction head. Claimed ~30x cheaper than oracles or NLAs; better on computational intermediates, worse on hallucination.

## 2. Existing visualisation tools

| Tool | What it shows | Backend | Status |
|---|---|---|---|
| model-graph (MIT) | Per-layer norm heatmaps, logit-lens flips, attention arcs, residual trajectory in a fixed random 3D basis, MoE router flow | Python server on PyTorch, any Hugging Face causal LM; `mps` supported | Landing-page visuals are mock data; assumes one residual stream per layer, which Inkling lacks |
| InTraVisTo | Heat maps of hidden states decoded to tokens during generation | Hugging Face models | Academic tool with paper |
| elevate-foundry/visualize-llm | Live activation heatmaps across layers, with directional ablation | Qwen3-0.6B by default | Small project |
| saklas | Steering plus trait probes with a monitoring terminal UI | Hugging Face transformers | On PyPI; details not read (page blocked the fetch) |
| repeng | Lean steering-vector library | Hugging Face transformers | Mentioned in comparison only |
| Neuronpedia | Feature browser, attribution graphs, NLA front end | Hosted, selected models | Mature |
| llm-visuals | Throughput, GPU, real MoE expert routing | llama.cpp | Not activations, but works with GGUF |

None of these is known to support Inkling 2 Mini out of the box.

## 3. What is known about Inkling 2 Mini's internals

From Goodfire, "Interpreting Inkling 2 Mini" (read in full). Thinking Machines gave them early access.

Structure that matters for a map:
- Width 2304. Four blocks of six layers: B1 (1-6) runs once, B2 (7-12) and B3 (13-18) loop three times, B4 (19-24) runs once as the decoder.
- **There is no single residual stream.** Each block keeps a running sum of its own layers' outputs. Each sublayer reads a softmax mix over earlier block sums.
- In loops 2 and 3, each looped layer also reads a message from the previous loop's B2 and B3 sums.
- Each layer's input decomposes exactly into shares of earlier outputs that sum to 1.

Findings:
- Depth-wise attention favours recent layers, like a sliding window.
- Across loops, the model "almost always reads from block 3 in loops 2 and 3".
- One coordinate (2027) carries about 46% of the squared norm. It surges at layer 18 in loops 1 and 2 and spikes negative in loop 3. Goodfire speculates it tags which block later loops should read.
- Logit lens "does not work well, apart from the final few layers per loop". Tuned lens and coda lens work very well.
- Early loops are more interpretable than mid-block layers.

Gaps:
- No code, notebooks, SAEs, probes or Neuronpedia pages were released.
- No tooling is named. NDIF is hosting a "Looped Lens Challenge" and working on researcher access.
- The variable-loop model is not covered by Goodfire. Thinking Machines' own monitorability results on it: no-chain-of-thought time horizon rose from 16s at one pass to 37s at three.
- Goodfire did not use SAEs, probes, PCA, patching or attribution graphs, so all of those are open territory on this model.

## 4. Looped models in general

- **Fixed points (Blayney et al., NeurIPS 2026).** In many looped models each layer in the cycle converges to its own fixed point, so the block traces a stable cycle in latent space, and attention patterns settle with it. Ouro is an exception. Their toolkit: distance to an approximate fixed point per loop, PCA of states across loops, attention-pattern distance between loops, and a classifier for fixed point / orbit / slider behaviour. Code is public.
- **Routing (Wu, Hu, Zhang, Sep 2026).** One loop does not equal one reasoning step. A learned linear layer in front of a frozen loop can select which transition happens next. They visualise by applying the final readout head after each loop. Code is public.
- **Stability theory.** A looped model can be run for extra iterations without "overthinking" only if it has reachable, stable fixed points.
- **Trajectory metrics (Jun 2026).** Probe-free measures over depth: path length, curvature, convergence, layer-to-layer cosine. Reasoning prompts show roughly 2.5x the curvature of lexical variation. Tested only on small models.

None of these studied Inkling 2 Mini.

## 5. Geometry of language in latent space (2025-2026)

Where the field stands:

- **Directions are not enough.** The linear representation hypothesis (a concept is a direction) still describes many features, but 2026 work keeps finding curved structure. "Globally linear representations are often inadequate abstractions" (The Shape of Beliefs, Llama-3.2).
- **Known shapes.** Circles for months and weekdays, one-dimensional curves for years and ordinals, a sphere for cities, clusters for categories, and tree-like splits for hierarchies. These are reported as stable across model families and sizes, and they reshape with context.
- **Where the shape comes from.** Two papers trace it to plain text statistics rather than a special mechanism:
  - Translation symmetry in co-occurrence (how often two months co-occur depends only on the gap between them) provably yields circles and smooth curves. ICML 2026.
  - Hierarchy geometry falls out of the spectrum of word co-occurrence: leading eigenvectors split broad branches first, then finer ones. Shown in word2vec and Gemma 2B.
- **Steering consequences.** Pushing along a straight line moves activations off the manifold and causes side effects. Moving along the curve changes outputs smoothly.
- **What sparse autoencoders miss.** They recover continuous structure poorly, fragmenting a manifold across many features ("dilution"). The proposal is to treat groups of features, not single ones, as the unit.
- **Architecture dependence.** How clean the manifolds are varies by task and by model; some concepts are linearly decodable with no clean shape.
- **Not settled.** Strict manifold assumptions probably do not hold globally, and most results are on narrow concept families (time, space, numbers).

Methods you could reuse: Supervised Multi-Dimensional Scaling for testing a shape hypothesis (code public), and the Manifold Probe for finding curved concepts in superposition.

## 6. What this means for latent-map

My read, not a finding:

- The natural objects to draw for Inkling are the four block sums and the per-layer writes, per loop, not a single residual trajectory.
- The cheapest informative live view is: PCA trajectory of block sums across the three loops, plus tuned-lens or coda-lens token labels, plus the 2027 coordinate and cross-loop attention weights as loop-state signals.
- Tagging known concepts live means training small linear probes on Inkling's block sums. Nobody has published any.
- The variable-loop model is the interesting target for geometry: one to five passes lets you ask whether its loops converge to a cycle, as the fixed-point paper found elsewhere. This is untested on Inkling.
- Everything depends on getting hidden states out locally, which is the first thing to verify.

## Not verified

- Whether transformers on Apple silicon can load the NVFP4/MXFP8 weights within 32GB, or whether NDIF or nnsight supports the model yet.
- The saklas feature list, and the unread parts of the NLA page and the June 2026 Circuits Update.
- Maintenance status of the small visualiser projects.
- Peer-review status of most 2026 preprints.

## Sources

Inkling and looped models
- [Interpreting Inkling 2 Mini — Goodfire](https://www.goodfire.com/research/interpreting-inkling-2-mini)
- [A Mechanistic Analysis of Looped Reasoning Language Models (arXiv 2604.11791)](https://arxiv.org/abs/2604.11791) · [code](https://github.com/HughBlayney/looped-mechanistic-analysis)
- [Shared Weights, Selected Computations (arXiv 2609.39892)](https://arxiv.org/abs/2609.39892) · [code](https://github.com/wjjpku/howloop)
- [Stability and Generalization in Looped Transformers — NeurIPS 2026](https://neurips.cc/virtual/2026/poster/154011)
- [Trajectory Geometry of Transformer Representations Across Layers (arXiv 2606.09287)](https://arxiv.org/abs/2606.09287)

Reading activations
- [Natural Language Autoencoders — Anthropic](https://transformer-circuits.pub/2026/nla/) · [code](https://github.com/kitft/natural_language_autoencoders)
- [Verbalizing Activation Differences with NLAs (arXiv 2610.30934)](https://arxiv.org/abs/2610.30934)
- [Activation Oracles (arXiv 2512.15674)](https://arxiv.org/abs/2512.15674)
- [The Chain Lens, interim — LessWrong](https://www.lesswrong.com/posts/Nr6fksEqqxNpdz6rk/the-chain-lens-interim)
- [WorkspaceBench — AI Alignment Forum](https://www.alignmentforum.org/posts/Zeg2JztbdhguL48uH/workspacebench-evaluating-interpretability-methods-for-the)
- [Circuits Updates, June 2026 — Anthropic](https://transformer-circuits.pub/2026/june-update/index.html)
- [SAE Attribution Graphs Are Decided at the Margin (arXiv 2611.00606)](https://arxiv.org/abs/2611.00606)
- [SAEs for Mechanistic Interpretability in NLP: A Critical Survey (arXiv 2610.20139)](https://arxiv.org/pdf/2610.20139)
- [The Circuits Research Landscape — Neuronpedia](https://www.neuronpedia.org/graph/info)
- [Hidden-State Probes for Streaming Moderation (arXiv 2606.10487)](https://www.alphaxiv.org/abs/2606.10487)
- [Task-specific prompts and SAEs for activation monitoring (arXiv 2504.20271)](https://arxiv.org/html/2504.20271v1)

Tools
- [model-graph](https://model-graph.com/) · [repo](https://github.com/QSchlegel/model-graph)
- [InTraVisTo (arXiv 2507.13858)](https://arxiv.org/pdf/2507.13858v1)
- [elevate-foundry/visualize-llm](https://github.com/elevate-foundry/visualize-llm)
- [saklas on PyPI](https://pypi.org/project/saklas/2.0.2/)
- [llm-visuals](https://github.com/DingoOz/llm-visuals)

Geometry
- [Symmetry in language statistics shapes the geometry of model representations (arXiv 2602.15029)](https://arxiv.org/abs/2602.15029)
- [Hierarchical Concept Geometry Emerges from Word Co-occurrence (arXiv 2605.23821)](https://arxiv.org/abs/2605.23821)
- [The Shape of Beliefs (arXiv 2602.02315)](https://arxiv.org/abs/2602.02315)
- [Do Sparse Autoencoders Capture Concept Manifolds? (arXiv 2604.28119)](https://arxiv.org/abs/2604.28119)
- [Probing for Representation Manifolds in Superposition (arXiv 2605.18537)](https://arxiv.org/abs/2605.18537)
- [Hypothesis-Driven Feature Manifold Analysis via Supervised MDS (arXiv 2510.01025)](https://arxiv.org/abs/2510.01025) · [code](https://github.com/UKPLab/tmlr2026-manifold-analysis)
- [Geometry of Ordinal Representations in Language Models (arXiv 2607.04167)](https://arxiv.org/pdf/2607.04167)
- [Geometry of Human Perceptual Domains Emerges Transiently in LLM Representations (arXiv 2605.27970)](https://arxiv.org/pdf/2605.27970)
