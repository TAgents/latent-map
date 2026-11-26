import json

from lmserver import app, summary
from lmserver.recorder import Recorder
from test_recorder import make_record


def step(reply, phases, **over):
    """A stored request as Recorder.get returns it. phases: one letter per token:
    | marker, h header, t thinking, a answer, T tool."""
    name = {"|": "marker", "h": "header", "t": "thinking", "a": "answer", "T": "tool"}
    assert len(reply) == len(phases)
    base = dict(completion_id="c1", time=100.0, text_kept=True, reply_tokens=reply,
                words=[{"i": i, "phase": name[p]} for i, p in enumerate(phases)],
                messages=[{"role": "system", "content": "Be brief."}, {"role": "user", "content": "What version is it?"}],
                prompt_tokens=1000, cached_tokens=0, n_tokens=len(reply), first_token_s=2.0, tokens_per_s=10.0,
                finish="stop", readout=None, readout_available=True, settings={})
    base.update(over)
    return base


THINK = ["<|message_model|>", "<|content_thinking|>", "Let", " me", " read", " it", ".", " The", " system", " prompt", " says",
         " never", " mention", " this", ".", "<|end_message|>"]
CALL = ["<|message_model|>", "bash", "<|content_invoke_tool_json|>", '{"name":"bash",', '"args":{"command":"ls"}}', "<|end_message|>"]


def test_a_step_counts_its_parts_and_names_the_tool_it_called():
    s = summary.summarize([step(THINK + CALL, "||ttttttttttttt|" + "|h|TT|", finish="tool_calls")])["steps"][0]
    assert (s["thinking"], s["answer"], s["tool"], s["finish"]) == (13, 0, "bash", "tool_calls")
    assert s["tool_args"] == '{"command":"ls"}'
    assert s["seconds"] == 2.0 + 22 / 10.0
    assert s["asked"] == {"role": "user", "text": "What version is it?"}


def test_thinking_sentences_that_refer_to_instructions_are_listed_with_where_they_start():
    s = summary.summarize([step(THINK, "||ttttttttttttt|")])["steps"][0]
    assert s["mentions"] == [{"i": 8, "text": "The system prompt says never mention this."}]   # points at " system", where the match starts
    plain = summary.summarize([step(["<|content_thinking|>", "The", " file", " system", " is", " fine", "."], "|tttttt")])["steps"][0]
    assert plain["mentions"] == []          # "file system" is not an instruction
    answer_only = summary.summarize([step(["My", " instructions", " say", " no", "."], "aaaaa")])["steps"][0]
    assert answer_only["mentions"] == []    # only the thinking is searched: the answer is already in front of the user


def test_what_led_to_a_step_is_the_last_message_shortened():
    long_result = {"role": "tool", "tool_call_id": "x", "content": "line\n" * 200}
    parts = {"role": "user", "content": [{"type": "text", "text": "look at"}, {"type": "image_url", "image_url": {"url": "data:..."}}]}
    a, b, c = summary.summarize([step(THINK, "||ttttttttttttt|", messages=[long_result]),
                                 step(THINK, "||ttttttttttttt|", messages=[parts]),
                                 step(THINK, "||ttttttttttttt|", messages=[{"role": "user", "content": None}])])["steps"]
    assert a["asked"]["role"] == "tool" and len(a["asked"]["text"]) <= 161 and a["asked"]["text"].endswith("…")
    assert b["asked"] == {"role": "user", "text": "look at [image]"}
    assert c["asked"] == {"role": "user", "text": ""}


def test_readout_state_and_how_often_each_pass_gave_the_token_written():
    hits = [[False, False, True]] * 2 + [[True, False, True], [False, True, True], [True, True, False]] + [[False, False, False]]
    done = step(["<|content_thinking|>", "<|x|>", "a", "b", "c", "<|end_message|>"], "||tta|", readout={"k": 5, "words": [], "hits": hits})
    s = summary.summarize([done, step(THINK, "||ttttttttttttt|"), step(THINK, "||ttttttttttttt|", readout_available=False)])["steps"]
    assert s[0]["readout"] == "stored" and s[0]["hits"] == [2, 2, 2] and s[0]["counted"] == 3   # markers are not counted
    assert s[1]["readout"] == "available" and s[1]["hits"] is None
    assert s[2]["readout"] == "none"


def test_totals_add_up_the_run():
    out = summary.summarize([step(THINK + CALL, "||ttttttttttttt|" + "|h|TT|", finish="tool_calls", time=100.0),
                             step(["<|content_text|>", "It", " is", " 2", "."], "|aaaa", completion_id="c2", time=130.0, cached_tokens=900)])
    t = out["totals"]
    assert (t["steps"], t["tool_calls"], t["thinking"], t["answer"], t["with_mentions"], t["text_kept"]) == (2, 1, 13, 4, 1, True)
    assert t["tokens"] == 27 and t["prompt_tokens"] == 2000 and t["cached_tokens"] == 900
    assert abs(t["seconds"] - (2.0 + 2.2 + 2.0 + 0.5)) < 1e-9 and t["span_seconds"] == 30.0 + 2.5
    assert [s["id"] for s in out["steps"]] == ["c1", "c2"]
    json.dumps(out)


def test_without_kept_text_the_numbers_remain_and_the_text_parts_say_so():
    s = step(THINK, "||ttttttttttttt|", text_kept=False, reply_tokens=None, messages=None, readout_available=False)
    out = summary.summarize([s])
    one = out["steps"][0]
    assert one["thinking"] == 13 and one["mentions"] is None and one["asked"] is None and one["tool"] is None
    assert out["totals"]["text_kept"] is False and out["totals"]["with_mentions"] is None
    assert summary.summarize([]) == {"steps": [], "totals": None}


def test_the_route_gives_a_sessions_summary(tmp_path):
    r = Recorder(tmp_path / "db.sqlite")
    r.save(make_record())
    ctx = app.Ctx(recorder=r, map_payload={}, settings={}, static=tmp_path)
    session = r.sessions()[0]["session"]
    status, _, raw = app.route(ctx, "GET", f"/api/sessions/{session}/summary")
    out = json.loads(raw)
    assert status == 200 and out["totals"]["steps"] == 1 and out["steps"][0]["id"] == "chatcmpl-1"
    assert app.route(ctx, "GET", "/api/sessions/nope/summary")[0] == 404


def test_a_match_far_into_a_long_stretch_is_shown_and_pointed_at():
    words = [f" item{k}" for k in range(80)] + [" the", " instructions", " say", " x", "."]
    s = summary.summarize([step(["<|content_thinking|>"] + words, "|" + "t" * len(words))])["steps"][0]
    (m,) = s["mentions"]
    assert "instructions say x." in m["text"] and len(m["text"]) <= 320 and m["text"].startswith("…")
    assert m["i"] == 1 + 81            # the word that matched, not the start of the stretch


def test_several_tool_calls_in_one_reply_are_kept_apart():
    two = CALL + ["<|message_model|>", "read", "<|content_invoke_tool_json|>", '{"name":"read","args":{"filePath":"a.py"}}', "<|end_message|>"]
    s = summary.summarize([step(two, "|h|TT|" + "|h|T|", finish="tool_calls")])["steps"][0]
    assert s["tool"] == "bash" and s["tool_args"] == '{"command":"ls"}' and s["more_tools"] == ["read"]
    one = summary.summarize([step(CALL, "|h|TT|", finish="tool_calls")])["steps"][0]
    assert one["more_tools"] == []


def test_phrasings_the_pattern_is_meant_to_catch():
    for text in ("I don't mention it", "shouldn’t reveal that", "No need to mention books", "The system says be brief", "was told to hide"):
        assert summary.MENTIONS.search(text), text
    for text in ("the file system is fine", "mention the tests", "an operating system"):
        assert not summary.MENTIONS.search(text), text


def test_a_damaged_row_does_not_take_the_summary_down():
    bad_text = step(THINK, "||ttttttttttttt|", messages=[{"role": "user", "content": [{"type": "text", "text": None}, "loose", {"type": "input_audio"}]}])
    short_hits = step(["a", "b"], "aa", readout={"hits": [[True], []]})
    no_phase = step(["a", "b"], "aa", words=[{"i": 0}, {"i": 1, "phase": "answer"}], messages=["not a dict"])
    broken = step(["a"], "a", words="nonsense")
    out = summary.summarize([bad_text, short_hits, no_phase, broken])
    a, b, c, d = out["steps"]
    assert a["asked"] == {"role": "user", "text": "loose [input_audio]"}
    assert b["hits"] == [1, 0, 0] and b["counted"] == 2
    assert c["answer"] == 1 and c["asked"] == {"role": "", "text": ""}
    assert d["id"] == "c1" and d["damaged"] is True and out["totals"]["steps"] == 4


def test_a_step_says_how_its_readout_should_be_read():
    sampled = step(["a"], "a", settings={"temperature": 0.7}, readout={"hits": [[True, True, True]], "pieces_known": False})
    s = summary.summarize([sampled, step(["a"], "a")])["steps"]
    assert s[0]["temperature"] == 0.7 and s[0]["pieces_known"] is False
    assert s[1]["temperature"] == 0 and s[1]["pieces_known"] is None


def test_mentions_are_counted_over_steps_that_have_thinking():
    out = summary.summarize([step(THINK, "||ttttttttttttt|"), step(["It", "."], "aa", completion_id="c2")])
    assert out["totals"]["with_mentions"] == 1 and out["totals"]["with_thinking"] == 1
