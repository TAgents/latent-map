"""Compare generation speed and memory with capture off and on."""

import json
import time
from pathlib import Path

import mlx.core as mx
import psutil
from mlx_lm import stream_generate

from latent_capture.capture import capturing
from latent_capture.loader import MODEL_REPO, load_model
from latent_capture.metrics import summarize, verdict
from prompts import PROMPTS

MAX_TOKENS = 128
GB = 1024**3


def chat_prompt(tokenizer, text: str) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": text}], add_generation_prompt=True, tokenize=False
    )


def generate_count(model, tokenizer, text: str) -> int:
    n = 0
    for _ in stream_generate(model, tokenizer, chat_prompt(tokenizer, text), max_tokens=MAX_TOKENS):
        n += 1
    return n


def run_mode(model, tokenizer, mode: str, prompts: list[str]) -> dict:
    mx.reset_peak_memory()
    swap_before = psutil.swap_memory().used / GB
    tokens = 0
    start = time.perf_counter()
    if mode == "on":
        with capturing(model) as cap:
            for p in prompts:
                tokens += generate_count(model, tokenizer, p)
                cap.clear()
    else:
        for p in prompts:
            tokens += generate_count(model, tokenizer, p)
    seconds = time.perf_counter() - start
    return {
        "mode": mode,
        "tokens": tokens,
        "seconds": seconds,
        "peak_gb": mx.get_peak_memory() / GB,
        "swap_gb_before": swap_before,
        "swap_gb_after": psutil.swap_memory().used / GB,
    }


def main() -> None:
    model, tokenizer = load_model()
    prompts = [p for group in PROMPTS.values() for p in group]
    generate_count(model, tokenizer, "Hello")  # warm-up
    runs = [run_mode(model, tokenizer, mode, prompts) for mode in ("off", "on", "off", "on")]
    summary = summarize(runs)
    result = {"model": MODEL_REPO, "max_tokens": MAX_TOKENS, "runs": runs, "summary": summary, "verdict": verdict(summary)}
    out = Path(__file__).parent / "results" / "measure.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    for k, v in summary.items():
        print(f"{k:16s} {v:.3f}")
    print("verdict", result["verdict"])


if __name__ == "__main__":
    main()
