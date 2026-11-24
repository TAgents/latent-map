import numpy as np

from lmserver.recorder import Recorder, RequestRecord

NOW = 1_800_000_000.0
DAY = 86400.0


def make_record(**over):
    base = dict(
        completion_id="chatcmpl-1",
        time=NOW,
        client="pytest",
        messages=[{"role": "user", "content": "Why is the sky blue?"}],
        reply_tokens=["Sun", "light", "."],
        words=[{"phase": "answer", "cells": [1, 2, 3], "area": 4, "move": [0.1, 0.2], "level": "usual", "settled_by": 1, "loop": [0.0, 0.1, 0.2]}] * 3,
        mean_states=np.zeros((2, 2, 3, 8), dtype=np.float32),
        settings={"max_tokens": 100, "tools": []},
        prompt_tokens=10,
        cached_tokens=0,
        first_token_s=0.5,
        tokens_per_s=17.0,
        finish="stop",
    )
    base.update(over)
    return RequestRecord(**base)


def test_save_and_get_round_trip(tmp_path):
    r = Recorder(tmp_path / "db.sqlite")
    r.save(make_record())
    got = r.get("chatcmpl-1")
    assert got["reply_tokens"] == ["Sun", "light", "."]
    assert got["messages"][0]["content"] == "Why is the sky blue?"
    assert got["words"][0]["cells"] == [1, 2, 3]
    assert got["text_kept"] is True and got["finish"] == "stop" and got["client"] == "pytest"
    assert r.get("missing") is None


def test_same_conversation_groups_into_one_session(tmp_path):
    r = Recorder(tmp_path / "db.sqlite")
    first = [{"role": "user", "content": "Why is the sky blue?"}]
    r.save(make_record())
    r.save(make_record(completion_id="chatcmpl-2", time=NOW + 5,
                       messages=first + [{"role": "assistant", "content": "Scattering."}, {"role": "user", "content": "And sunsets?"}]))
    sessions = r.sessions()
    assert len(sessions) == 1 and sessions[0]["requests"] == 2
    assert sessions[0]["title"] == "Why is the sky blue?"
    assert [q["completion_id"] for q in r.requests(sessions[0]["session"])] == ["chatcmpl-1", "chatcmpl-2"]


def test_different_first_message_is_a_new_session(tmp_path):
    r = Recorder(tmp_path / "db.sqlite")
    r.save(make_record())
    r.save(make_record(completion_id="chatcmpl-2", messages=[{"role": "user", "content": "Hello"}]))
    assert len(r.sessions()) == 2


def test_text_completion_without_messages_is_recorded(tmp_path):
    r = Recorder(tmp_path / "db.sqlite")
    r.save(make_record(completion_id="cmpl-9", messages=None))
    assert r.sessions()[0]["session"] == "cmpl-9"
    assert r.get("cmpl-9")["messages"] is None


def test_no_text_mode_leaves_no_text_in_the_file(tmp_path):
    r = Recorder(tmp_path / "db.sqlite", keep_text=False)
    r.save(make_record(
        messages=[{"role": "user", "content": "zebra-quartz question"},
                  {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "t", "arguments": {"q": "zebra-quartz"}}}]}],
        reply_tokens=["zebra-quartz"],
        settings={"max_tokens": 5, "tools": ["zebra-quartz"]},
    ))
    r.close()
    for f in tmp_path.iterdir():
        assert b"zebra-quartz" not in f.read_bytes(), f.name


def test_no_text_mode_reports_text_not_kept(tmp_path):
    r = Recorder(tmp_path / "db.sqlite", keep_text=False)
    r.save(make_record())
    got = r.get("chatcmpl-1")
    assert got["text_kept"] is False and got["reply_tokens"] is None and got["messages"] is None
    assert len(got["words"]) == 3
    assert r.sessions()[0]["title"] == "(text not kept)"


def test_mean_states_round_trip_with_nan(tmp_path):
    r = Recorder(tmp_path / "db.sqlite")
    m = np.random.default_rng(0).normal(size=(2, 2, 3, 8)).astype(np.float32)
    m[0] = np.nan
    r.save(make_record(mean_states=m))
    back = r.mean_states("chatcmpl-1")
    assert back.dtype == np.float32 and np.array_equal(back, m, equal_nan=True)
    r.save(make_record(completion_id="chatcmpl-2", mean_states=None))
    assert r.mean_states("chatcmpl-2") is None


def test_forget_removes_every_request_of_the_session(tmp_path):
    r = Recorder(tmp_path / "db.sqlite")
    r.save(make_record())
    r.save(make_record(completion_id="chatcmpl-2"))
    r.save(make_record(completion_id="chatcmpl-3", messages=[{"role": "user", "content": "Other"}]))
    session = next(s["session"] for s in r.sessions() if s["requests"] == 2)
    assert r.forget(session) == 2
    assert r.get("chatcmpl-1") is None and r.mean_states("chatcmpl-2") is None
    assert r.get("chatcmpl-3") is not None


def test_purge_removes_only_old_records(tmp_path):
    r = Recorder(tmp_path / "db.sqlite", retention_days=5)
    r.save(make_record(completion_id="old", time=NOW - 6 * DAY))
    r.save(make_record(completion_id="new", time=NOW - 4 * DAY))
    assert r.purge(now=NOW) == 1
    assert r.get("old") is None and r.mean_states("old") is None
    assert r.get("new") is not None


def test_sessions_are_newest_first(tmp_path):
    r = Recorder(tmp_path / "db.sqlite")
    r.save(make_record(completion_id="a", time=NOW, messages=[{"role": "user", "content": "A"}]))
    r.save(make_record(completion_id="b", time=NOW + 9, messages=[{"role": "user", "content": "B"}]))
    assert [s["title"] for s in r.sessions()] == ["B", "A"]
