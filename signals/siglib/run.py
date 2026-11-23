"""Generate one reply and return its text with per-token map coordinates."""

import numpy as np

from latent_capture.gen import generate_with_states
from mapbuild.areas import is_marker
from mapbuild.fit import _project, place

ANSWER_MARK = "<|content_text|>"


def load_map(bundle, block: str) -> dict:
    b = np.load(bundle / f"{block}.npz")
    m = {k: b[k] for k in ("drop", "pass_means", "pass_scales", "components", "centres") if k in b}
    m["fine"] = b["fine"]
    return m


def reply(model, tokenizer, prompt: str, maps: dict, effort: float, max_tokens: int, steer: dict | None = None) -> dict:
    """maps: {"block2": map, "block3": map}. Returns the text, and for every generated
    token its phase, whether it is a chat marker, and for each block its cell per pass
    (tokens, passes) and its position in the map's 64 dimensions (tokens, passes, 64)."""
    head = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], add_generation_prompt=True, tokenize=False, reasoning_effort=effort
    )
    texts, states = [], {"block2": [], "block3": []}
    for _, text, rec in generate_with_states(model, tokenizer, head, max_tokens, steer=steer):
        texts.append(text)
        states["block2"].append(np.stack(rec.block2))
        states["block3"].append(np.stack(rec.block3))
    phase, current = [], "thinking"
    for t in texts:
        if t == ANSWER_MARK:
            current = "answer"
        phase.append(current)
    out = {
        "tokens": texts,
        "phase": phase,
        "marker": [is_marker(t) for t in texts],
        "answer": "".join(t for t, p in zip(texts, phase) if p == "answer" and not is_marker(t)),
        "thinking": "".join(t for t, p in zip(texts, phase) if p == "thinking" and not is_marker(t)),
    }
    for block, m in maps.items():
        x = np.stack(states[block]) if states[block] else np.zeros((0, 3, 2304), dtype=np.float32)
        out[f"{block}_cell"] = place(m, x) if len(x) else np.zeros((0, 3), dtype=np.int32)
        out[f"{block}_z"] = _project(m, x).astype(np.float16) if len(x) else np.zeros((0, 3, 64), dtype=np.float16)
    return out
