"""Load Inkling 2 Mini with MLX."""

from mlx_lm import load

MODEL_REPO = "thinkingmachines/Inkling-2-Mini"


def load_model(repo: str = MODEL_REPO):
    """Return (model, tokenizer). Raises with the repo name if loading fails;
    never falls back to another repo by itself."""
    try:
        return load(repo)
    except Exception as err:
        raise RuntimeError(f"Could not load {repo} with MLX: {err}") from err
