"""Build prompts, including the form that skips the model's thinking section."""

SKIP_THINKING = "<|message_model|><|content_text|>"  # start the reply for the model, directly in its answer


def head(tokenizer, question: str, system: str | None = None, effort: float = 0.0, skip_thinking: bool = True) -> str:
    messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": question}]
    text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False, reasoning_effort=effort)
    return text + (SKIP_THINKING if skip_thinking else "")


def word_ids(tokenizer, word: str) -> list[int]:
    """First token id of the word with and without a leading space (capitalised and not)."""
    forms = {word, " " + word, word.capitalize(), " " + word.capitalize()}
    return sorted({tokenizer.encode(f, add_special_tokens=False)[0] for f in forms})


def split_reply(raw: str) -> tuple[str, str]:
    """(thinking text, answer text) of a raw generation that may include chat markers."""
    answer = raw.split("<|content_text|>", 1)[1] if "<|content_text|>" in raw else ("" if "<|content_thinking|>" in raw else raw)
    thinking = raw.split("<|content_thinking|>", 1)[1].split("<|end_message|>", 1)[0] if "<|content_thinking|>" in raw else ""
    return thinking.strip(), answer.split("<|end_message|>", 1)[0].strip()
