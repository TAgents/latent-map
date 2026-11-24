from mlx_lm.generate import TextStateMachine

from lmserver import inkling

THINK_ANSWER = "<|message_model|><|content_thinking|>Plan.<|end_message|><|message_model|><|content_text|>Hi.<|end_message|>"
TOOL = (
    "<|message_model|><|content_thinking|>Read it.<|end_message|><|message_model|>read_file"
    '<|content_invoke_tool_json|>{"name":"read_file","args":{"path":"a.toml"}}<|end_message|>'
)


def read(text, step=None):
    transitions = {k: list(v) for k, v in inkling.RESPONSE_FORMAT["transitions"].items()}
    transitions["tool"] = [(inkling.TOOL_CALL_END, None)]
    reader = TextStateMachine(transitions).reader(state=inkling.RESPONSE_FORMAT["initial_state"])
    step = step or len(text)
    out = TextStateMachine.Delta()
    for i in range(0, len(text), step):
        out = out + reader.feed(text[i : i + step])
    return out


def test_thinking_and_answer_are_split():
    d = read(THINK_ANSWER)
    assert (d.content, d.reasoning, d.tool_calls) == ("Hi.", "Plan.", ())


def test_tool_call_is_extracted():
    d = read(TOOL)
    assert d.content == "" and d.reasoning == "Read it." and len(d.tool_calls) == 1
    assert inkling.parse_tool_call(d.tool_calls[0]) == {"name": "read_file", "arguments": {"path": "a.toml"}}


def test_markers_never_reach_content():
    for text in (THINK_ANSWER, TOOL):
        d = read(text, step=1)
        assert "<|" not in d.content and "<|" not in d.reasoning
    assert read(THINK_ANSWER, step=1).content == "Hi."


def test_parse_tool_call():
    assert inkling.parse_tool_call(' {"name":"read_file","args":{"path":"a.toml"}} ') == {
        "name": "read_file",
        "arguments": {"path": "a.toml"},
    }


def test_parse_tool_call_without_args():
    assert inkling.parse_tool_call('{"name":"ls"}') == {"name": "ls", "arguments": {}}


def test_answer_only_reply():
    assert read("<|message_model|><|content_text|>Hi.<|end_message|>").content == "Hi."
