"""Where each map version keeps its files. v0 predates version tags in file names."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def paths(tag: str) -> dict:
    data = ROOT / "data"
    suffix = "" if tag == "v0" else f"_{tag}"
    return {
        "corpus": data / f"corpus{suffix}.jsonl",
        "states": data / f"states{suffix}.npz",
        "tokens": data / f"tokens{suffix}.json",
        "bundle": ROOT / f"bundle_{tag}",
    }
