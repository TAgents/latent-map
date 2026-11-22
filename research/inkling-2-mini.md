# Inkling 2 Mini — Thinking Machines' looped model for local machines

Researched 2026-11-22 from the lab's announcement and model cards. Coverage was two days old at the time, so details may have changed since.

## Summary

Inkling 2 Mini is an open-weights looped transformer that Thinking Machines Lab released on 2026-11-20, built to run on a laptop. It is Apache 2.0, about 20GB on disk, and runs on Apple silicon with 32GB or more of memory.

## Architecture

| Property | Value |
|---|---|
| Stored layers | 24 decoder layers, in four blocks of six |
| Looping | Middle 12 layers (blocks 2 and 3) run three times per token on shared weights; first and last blocks run once |
| Layer applications per token | 48 |
| Parameters | 26.4B total; 3.4B active per pass; 5.4B effective active at three passes |
| Mixture of experts | 8 of 128 experts routed per token per layer |
| Attention | Hybrid of local and global layers, with attention residuals |
| Context | Up to 1M tokens (per the model cards; the announcement gives none) |
| Modalities | Text and image in, text out; image sides 40px to 4096px |
| Precision | Expert layers (20 layers) in NVFP4, last four layers in MXFP8; trained and shipped natively in these formats |
| Checkpoint size | About 20GB |
| Licence | Apache 2.0, plus a linked Model Acceptable Use Policy |
| Hugging Face repo | `thinkingmachines/Inkling-2-Mini` |

Training notes from the announcement:
- It started with 384 experts per layer; two-thirds were pruned mid-training, keeping the 128 most used.
- It was then healed on 1T tokens distilled from a larger internal model.
- A reinforcement-learning phase used 2.2T tokens over 2,000 steps.
- No pretraining token count is published.

## Running it locally

Hardware, per the model cards:
- Apple silicon with 32GB or more of memory
- One NVIDIA DGX Spark
- Other GPUs with 32GB or more of VRAM recommended

Runtimes:
- **Ollama:** library page is `https://ollama.com/library/inkling-2-mini`. The official pages do not state the exact tag; `ollama run inkling-2-mini` is inferred from the URL and unconfirmed.
- **GGUF:** quantized versions via Unsloth and llama.cpp (docs at `https://unsloth.ai/docs/inkling-2-mini`, not read). Hugging Face listed 173 community quantizations.
- **vLLM:**
  ```
  pip install vllm
  vllm serve "thinkingmachines/Inkling-2-Mini"
  ```
  Serves an OpenAI-compatible endpoint at `http://localhost:8000/v1/chat/completions`.
- **SGLang:**
  ```
  pip install sglang
  python3 -m sglang.launch_server --model-path "thinkingmachines/Inkling-2-Mini" --host 0.0.0.0 --port 30000
  ```
- **Docker Model Runner:** `docker model run hf.co/thinkingmachines/Inkling-2-Mini`
- **Transformers:** `AutoModelForMultimodalLM.from_pretrained(..., device_map="auto")` with `AutoProcessor`.
- **Others named in coverage:** Exo Labs, NVIDIA Dynamo, TokenSpeed, Llama App. MLX is not mentioned on the official pages.

Speed: no measured tokens-per-second figure is published. The lab's demos ran on an M5 Pro MacBook with 48GB.

### Fit on this machine

Checked 2026-11-22: Apple M5, 32GB memory, 549GB free disk, Ollama and Docker installed, no llama.cpp. This meets the stated minimum with no headroom, so expect memory pressure with other apps open. Nothing has been downloaded or installed.

## Variable-loop sibling: TML-d24-varloop-research

- Repo: `thinkingmachines/TML-d24-varloop-research`, Apache 2.0, same 26.4B parameters and 20GB checkpoint.
- Trained with one to five passes through the middle 12 layers (sampled 1:2:6:2:1), then post-trained at three passes.
- Active parameters: 3.4B at one pass, 5.4B at three, 7.4B at five.
- The loop count "can be chosen at inference time", but no flag, config field or code for setting it is documented.
- Needs at least 32GB of GPU or unified memory.
- Open weights only; only four community quantizations at the time, and runtime support is described as new.
- The lab calls it a research artifact, "not carefully tuned for deployment", that "may behave unpredictably at pass counts other than 3". It was used for the lab's monitorability experiments.

## Benchmarks (the lab's own figures)

| Benchmark | Score |
|---|---|
| DeepSWE v1.1 | 48.2% |
| Vals Index v2.2 | 46.5% |
| SWE-bench Pro v2 (Hard) | 77.9% |
| SWEBench Verified | 79.2% |
| BrowseComp (with compaction) | 79.0% |
| MCP Atlas | 82.7% |

- The lab claims it leads every open-weights model under 100B total parameters on DeepSWE and Vals Index.
- That claim holds for the announcement's comparison set. On the company model card's wider table it is not best on every row (SWEBench Verified is second, behind 80.1%).
- Settings differ between pages: BrowseComp is 76.0% without context management, and EnterpriseOps-Gym is 42.0% or 36.7% depending on the harness.
- Stanford intelligence-per-watt on a DGX Spark: 0.703, against 0.430 for Qwen3.6-35B-A3B, with the comparison models at other precisions.

## Stated weaknesses

- Trails Qwen3.8-27B on multi-turn chat (MultiChallenge 50.0%) and many-constraint instructions (ComplexConstraints v1.1 33.4%).
- Trails larger models on CritPt v2 and HLE-Diamond.
- Weaker factual recall than Inkling 2 Small, and "more likely than larger models to state incorrect information with confidence".
- Sometimes gets stuck in long reasoning loops; the lab recommends limits on output tokens and agent turns.
- Less reliable on detailed tool instructions and may take destructive actions without confirmation.
- Lower output quality outside English.
- Minimum reasoning effort does not fully disable reasoning.

## Not verified

- The exact Ollama tag and the Unsloth/llama.cpp commands (those pages were not read).
- How to set the loop count on the varloop model.
- Any real tokens-per-second figure.
- The MarkTechPost article, which refused the fetch.
- All benchmark numbers are the lab's own.

## Related

Inkling 2 Small is the larger predecessor from earlier in November 2026: a mixture-of-experts model with 594B total parameters and 17B active. It is not a laptop model.

## Sources

- [Introducing Inkling 2 Mini — Thinking Machines Lab](https://thinkingmachines.ai/news/inkling-2-mini/)
- [Inkling 2 Mini model card — Hugging Face](https://huggingface.co/thinkingmachines/Inkling-2-Mini)
- [TML-d24-varloop-research model card — Hugging Face](https://huggingface.co/thinkingmachines/TML-d24-varloop-research)
- [Inkling 2 Mini company model card](https://thinkingmachines.ai/model-card/inkling-2-mini/)
- [TML-d24-varloop-research company model card](https://thinkingmachines.ai/model-card/tml-d24-varloop-research/)
- [Thinking Machines ships variable-loop Inkling research model — AI Weekly](https://aiweekly.co/alerts/thinking-machines-ships-variable-loop-inkling-research-model)
- [Inkling 2 Mini coverage — The Next Web](https://thenextweb.com/news/inkling-2-mini-thinking-machines-looped-open-weights-laptop)
- [Inkling 2 Small — AIwire](https://www.hpcwire.com/aiwire/2026/11/11/thinking-machines-lab-introduces-inkling-2-small-for-coding-and-knowledge-work/)
