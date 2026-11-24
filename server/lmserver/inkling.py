"""Inkling's reply format, described for the bundled MLX server.

A reply is a run of messages, each `<|message_model|>[tool name]<|content_KIND|>...<|end_message|>`.
The server has no built-in knowledge of this, so without it the markers leak into the answer."""

import json

TOOL_CALL_START = "<|content_invoke_tool_json|>"
TOOL_CALL_END = "<|end_message|>"

# Text in "header" (the role marker and a tool's name) is dropped by the server.
RESPONSE_FORMAT = {
    "initial_state": "header",
    "transitions": {
        "header": [
            ("<|content_thinking|>", "reasoning"),
            ("<|content_text|>", "normal"),
            (TOOL_CALL_START, "tool"),
        ],
        "reasoning": [("<|end_message|>", "header")],
        "normal": [("<|end_message|>", "header")],
    },
}


def parse_tool_call(text: str, tools=None) -> dict:
    """The model writes {"name": ..., "args": {...}}; the chat API wants "arguments"."""
    call = json.loads(text.strip())
    return {"name": call["name"], "arguments": call.get("args") or {}}


def configure(tokenizer) -> None:
    """Teach an mlx-lm TokenizerWrapper the format above."""
    tokenizer._response_format = RESPONSE_FORMAT
    tokenizer._tool_parser = parse_tool_call
    tokenizer._tool_call_start = TOOL_CALL_START
    tokenizer._tool_call_end = TOOL_CALL_END
    tokenizer._tool_call_start_tokens = tuple(tokenizer.encode(TOOL_CALL_START, add_special_tokens=False))
