import json

import numpy as np
import pytest

from lmserver import app, readout
from lmserver.recorder import Recorder
from test_recorder import make_record


class Tok:
    def decode(self, ids):
        return "".join(f"w{i}" for i in ids)


def fake_run(model, ids, topk, positions, states):
    """Stands in for the model: at position p the passes predict p+100, p+200, p+300 first."""
    assert states is False
    top = np.zeros((3, len(positions), topk), dtype=np.int64)
    prob = np.zeros((3, len(positions), topk), dtype=np.float32)
    for k in range(3):
        for j, p in enumerate(positions):
            top[k, j] = [p + 100 * (k + 1) + r for r in range(topk)]
            prob[k, j] = [0.5 / (r + 1) for r in range(topk)]
    return {"top": top, "prob": prob}


def test_each_word_gets_the_prediction_made_just_before_it():
    out = readout.compute(None, Tok(), prompt_ids=[1, 2, 3, 4], reply_ids=[10, 11, 12], run=fake_run)
    assert out["k"] == readout.TOPK and len(out["words"]) == 3
    # word 0 was predicted at the last prompt position (3); word 2 at position 5
    assert out["words"][0][0][0] == ["w103", 0.5]
    assert out["words"][0][2][0][0] == "w303"
    assert out["words"][2][1][0][0] == "w205"
    assert len(out["words"][0]) == 3 and len(out["words"][0][0]) == readout.TOPK
    json.dumps(out)  # plain data only


def test_too_long_or_empty_texts_are_refused():
    with pytest.raises(readout.TooLong):
        readout.compute(None, Tok(), [0] * readout.MAX_TOKENS, [1], run=fake_run)
    with pytest.raises(ValueError):
        readout.compute(None, Tok(), [], [1], run=fake_run)
    assert readout.compute(None, Tok(), [1], [], run=fake_run)["words"] == []


def test_recorder_keeps_token_ids_and_readout(tmp_path):
    r = Recorder(tmp_path / "db.sqlite")
    r.save(make_record(prompt_ids=[5, 6, 7], reply_ids=[8, 9, 10]))
    assert r.token_ids("chatcmpl-1") == ([5, 6, 7], [8, 9, 10])
    got = r.get("chatcmpl-1")
    assert got["readout_available"] is True and got["readout"] is None
    r.save_readout("chatcmpl-1", {"k": 5, "words": [["x"]]})
    assert r.get("chatcmpl-1")["readout"] == {"k": 5, "words": [["x"]]}
    r.save(make_record(completion_id="chatcmpl-2"))  # recorded without ids
    assert r.token_ids("chatcmpl-2") is None and r.get("chatcmpl-2")["readout_available"] is False


def test_no_text_mode_keeps_neither_ids_nor_readout(tmp_path):
    r = Recorder(tmp_path / "db.sqlite")
    r.save(make_record(prompt_ids=[5, 6, 7], reply_ids=[8, 9, 10]))
    r.save_readout("chatcmpl-1", {"k": 5, "words": [[[["zebra-quartz", 0.5]]]]})
    r.close()
    r = Recorder(tmp_path / "db.sqlite", keep_text=False)
    assert r.token_ids("chatcmpl-1") is None and r.get("chatcmpl-1")["readout"] is None
    r.save(make_record(completion_id="chatcmpl-2", prompt_ids=[5], reply_ids=[8]))
    assert r.token_ids("chatcmpl-2") is None and r.get("chatcmpl-2")["readout_available"] is False
    r.close()
    for f in tmp_path.iterdir():
        assert b"zebra-quartz" not in f.read_bytes(), f.name


def test_an_older_database_is_upgraded_in_place(tmp_path):
    import sqlite3

    db = sqlite3.connect(tmp_path / "db.sqlite")
    db.execute(
        """CREATE TABLE requests (
            completion_id TEXT PRIMARY KEY, session TEXT, time REAL, client TEXT, title TEXT,
            text_kept INTEGER, messages TEXT, reply_tokens TEXT, words TEXT, settings TEXT,
            prompt_tokens INTEGER, cached_tokens INTEGER, n_tokens INTEGER,
            first_token_s REAL, tokens_per_s REAL, finish TEXT, mean_states BLOB)"""
    )
    db.execute("INSERT INTO requests (completion_id, session, time, text_kept, words, settings, n_tokens) VALUES ('old', 's', 1, 1, '[]', '{}', 0)")
    db.commit()
    db.close()
    r = Recorder(tmp_path / "db.sqlite")
    assert r.get("old")["readout_available"] is False
    r.save(make_record(prompt_ids=[1], reply_ids=[2]))
    assert r.token_ids("chatcmpl-1") == ([1], [2])


@pytest.fixture
def ctx(tmp_path):
    r = Recorder(tmp_path / "db.sqlite")
    r.save(make_record(prompt_ids=[1, 2, 3], reply_ids=[10, 11, 12]))
    r.save(make_record(completion_id="noids", messages=[{"role": "user", "content": "x"}]))
    calls = []

    def run(prompt, reply):
        calls.append(1)
        return readout.compute(None, Tok(), prompt, reply, run=fake_run)

    c = app.Ctx(recorder=r, map_payload={}, settings={}, static=tmp_path, readout=run)
    c.calls = calls
    return c


def test_route_computes_once_then_serves_the_stored_readout(ctx):
    status, _, raw = app.route(ctx, "POST", "/api/requests/chatcmpl-1/readout")
    assert status == 200 and len(json.loads(raw)["words"]) == 3
    assert app.route(ctx, "POST", "/api/requests/chatcmpl-1/readout")[0] == 200
    assert len(ctx.calls) == 1
    assert json.loads(app.route(ctx, "GET", "/api/requests/chatcmpl-1")[2])["readout"]["k"] == readout.TOPK


def test_route_says_why_a_readout_is_not_available(ctx):
    assert app.route(ctx, "POST", "/api/requests/nope/readout")[0] == 404
    status, _, raw = app.route(ctx, "POST", "/api/requests/noids/readout")
    assert status == 409 and "text" in json.loads(raw)["error"]
    ctx.recorder.save(make_record(completion_id="long", prompt_ids=[0] * readout.MAX_TOKENS, reply_ids=[1]))
    status, _, raw = app.route(ctx, "POST", "/api/requests/long/readout")
    assert status == 413 and str(readout.MAX_TOKENS) in json.loads(raw)["error"]
    assert app.route(ctx, "POST", "/api/requests/chatcmpl-1/readout")[0] == 200
    ctx.readout = None
    assert app.route(ctx, "POST", "/api/requests/chatcmpl-1/readout")[0] == 200  # stored one still served
    ctx.recorder.save(make_record(completion_id="fresh", prompt_ids=[1], reply_ids=[2]))
    assert app.route(ctx, "POST", "/api/requests/fresh/readout")[0] == 503

    def not_loaded(prompt, reply):
        raise RuntimeError("model not loaded")

    ctx.readout = not_loaded
    assert app.route(ctx, "POST", "/api/requests/fresh/readout")[0] == 503


def test_built_records_carry_token_ids():
    from pathlib import Path

    from livemap.live import LiveMap
    from lmserver.hooks import build_record
    import test_hooks as th

    root = Path(__file__).resolve().parents[2]
    live = LiveMap.load(root / "basemap" / "bundle_v1", root / "prototype" / "norms_v1.json")
    rec = build_record(th.META, [100, 101, 102, 103], 4, [5, 6], [th.cap(0, 4), th.cap(1), th.cap(2)], th.Tok(), live, th.TIMING, "stop")
    assert rec.prompt_ids == [100, 101, 102, 103] and rec.reply_ids == [5, 6]
