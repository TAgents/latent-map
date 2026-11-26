import json

import numpy as np
import pytest

from lmserver import app, readout
from lmserver.recorder import Recorder
from test_recorder import make_record


class Tok:
    def decode(self, ids):
        return "".join(f"w{i}" for i in ids)


def fake_run(model, prompt_ids, reply_ids, topk, pieces=None):
    """Stands in for the model: at position p the passes predict p+100, p+200, p+300 first."""
    positions = [len(prompt_ids) - 1 + i for i in range(len(reply_ids))]
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
    assert out["words"][0][0][0] == ["w103", 0.5, 103]
    assert out["words"][0][2][0][0] == "w303"
    assert out["words"][2][1][0][0] == "w205"
    assert len(out["words"][0]) == 3 and len(out["words"][0][0]) == readout.TOPK
    json.dumps(out)  # plain data only


def test_an_agent_sized_request_is_accepted():
    assert readout.MAX_TOKENS >= 30000
    out = readout.compute(None, Tok(), [0] * 14600, [1, 2], run=fake_run)
    assert len(out["words"]) == 2 and out["words"][1][0][0][2] == 14700


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

    def run(prompt, reply, pieces=None):
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



def test_built_records_carry_token_ids():
    from pathlib import Path

    from livemap.live import LiveMap
    from lmserver.hooks import build_record
    import test_hooks as th

    root = Path(__file__).resolve().parents[2]
    live = LiveMap.load(root / "basemap" / "bundle_v1", root / "prototype" / "norms_v1.json")
    rec = build_record(th.META, [100, 101, 102, 103], 4, [5, 6], [th.cap(0, 4), th.cap(1), th.cap(2)], th.Tok(), live, th.TIMING, "stop")
    assert rec.prompt_ids == [100, 101, 102, 103] and rec.reply_ids == [5, 6]


# --- findings of the readout review ---

def test_hits_are_decided_by_token_id_not_by_text():
    # fake_run's pass k predicts p+100*(k+1) at position p; make the reply equal pass 3's top at word 1 only
    out = readout.compute(None, Tok(), prompt_ids=[1, 2], reply_ids=[7, 302], run=fake_run)
    assert out["hits"] == [[False, False, False], [False, False, True]]
    assert out["words"][1][2][0] == ["w302", 0.5, 302]  # text, probability, id


def test_requests_with_images_keep_no_tokens_for_a_readout():
    from lmserver.hooks import has_images

    assert has_images([{"role": "user", "content": [{"type": "text", "text": "x"}, {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA"}}]}])
    assert not has_images([{"role": "user", "content": "plain"}, {"role": "user", "content": [{"type": "text", "text": "x"}]}])
    assert not has_images(None)
    from pathlib import Path
    from livemap.live import LiveMap
    from lmserver.hooks import build_record
    import test_hooks as th

    root = Path(__file__).resolve().parents[2]
    live = LiveMap.load(root / "basemap" / "bundle_v1", root / "prototype" / "norms_v1.json")
    meta = {**th.META, "messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64,AA"}}]}]}
    rec = build_record(meta, [100, 101], 2, [5], [th.cap(0, 2), th.cap(1)], th.Tok(), live, th.TIMING, "stop")
    assert rec.prompt_ids is None and rec.reply_ids is None


def test_route_reports_failures_truthfully(ctx):
    ctx.recorder.save(make_record(completion_id="fresh", prompt_ids=[1], reply_ids=[2]))

    def out_of_memory(prompt, reply, pieces=None):
        raise RuntimeError("[metal::malloc] Attempting to allocate")

    ctx.readout = out_of_memory
    status, _, raw = app.route(ctx, "POST", "/api/requests/fresh/readout")
    assert status == 500 and "RuntimeError" in json.loads(raw)["error"] and "not loaded" not in json.loads(raw)["error"]

    def not_loaded(prompt, reply, pieces=None):
        raise readout.NotLoaded()

    ctx.readout = not_loaded
    assert app.route(ctx, "POST", "/api/requests/fresh/readout")[0] == 503
    ctx.recorder.save(make_record(completion_id="noprompt", prompt_ids=[], reply_ids=[2]))
    ctx.readout = lambda p, r, pieces=None: readout.compute(None, Tok(), p, r, run=fake_run)
    assert app.route(ctx, "POST", "/api/requests/noprompt/readout")[0] == 409


def test_the_same_readout_asked_twice_at_once_is_computed_once(ctx):
    import threading
    import time

    def slow(prompt, reply, pieces=None):
        ctx.calls.append(1)
        time.sleep(0.2)
        return readout.compute(None, Tok(), prompt, reply, run=fake_run)

    ctx.readout = slow
    results = []
    threads = [threading.Thread(target=lambda: results.append(app.route(ctx, "POST", "/api/requests/chatcmpl-1/readout")[0])) for _ in range(2)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert results == [200, 200] and len(ctx.calls) == 1


def test_two_servers_opening_an_old_database_at_once_do_not_crash(tmp_path):
    import sqlite3
    import threading

    db = sqlite3.connect(tmp_path / "db.sqlite")
    db.execute(
        """CREATE TABLE requests (
            completion_id TEXT PRIMARY KEY, session TEXT, time REAL, client TEXT, title TEXT,
            text_kept INTEGER, messages TEXT, reply_tokens TEXT, words TEXT, settings TEXT,
            prompt_tokens INTEGER, cached_tokens INTEGER, n_tokens INTEGER,
            first_token_s REAL, tokens_per_s REAL, finish TEXT, mean_states BLOB)"""
    )
    db.commit()
    db.close()
    errors = []

    def open_it():
        try:
            Recorder(tmp_path / "db.sqlite")
        except Exception as err:
            errors.append(err)

    threads = [threading.Thread(target=open_it) for _ in range(6)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert errors == []


def test_the_pieces_the_prompt_was_read_in_are_kept_and_handed_to_the_readout(tmp_path):
    r = Recorder(tmp_path / "db.sqlite")
    r.save(make_record(prompt_ids=[5, 6, 7], reply_ids=[8, 9], pieces=[2, 1]))
    r.save(make_record(completion_id="chatcmpl-2", prompt_ids=[5, 6, 7], reply_ids=[8, 9]))
    assert r.pieces("chatcmpl-1") == [2, 1] and r.pieces("chatcmpl-2") is None and r.pieces("nope") is None
    seen = {}

    def run(model, prompt_ids, reply_ids, topk, pieces):
        seen["pieces"] = pieces
        return fake_run(model, prompt_ids, reply_ids, topk)

    readout.compute(None, Tok(), [5, 6, 7], [8, 9], pieces=[2, 1], run=run)
    assert seen["pieces"] == [2, 1]
    readout.compute(None, Tok(), [5, 6, 7], [8, 9], run=run)
    assert seen["pieces"] is None


def test_the_route_reads_a_request_in_its_recorded_pieces(tmp_path):
    seen = []
    ctx = app.Ctx(recorder=Recorder(tmp_path / "db.sqlite"), map_payload={}, settings={}, static=tmp_path,
                  readout=lambda p, r, pieces: seen.append(pieces) or {"k": 5, "words": [], "hits": []})
    ctx.recorder.save(make_record(prompt_ids=[5, 6, 7], reply_ids=[8, 9], pieces=[2, 1]))
    status, _, _ = app.route(ctx, "POST", "/api/requests/chatcmpl-1/readout")
    assert status == 200 and seen == [[2, 1]]


def test_the_result_says_whether_the_prompt_was_read_in_its_recorded_pieces():
    seen = []

    def run(model, prompt_ids, reply_ids, topk, pieces):
        seen.append(pieces)
        return fake_run(model, prompt_ids, reply_ids, topk)

    assert readout.compute(None, Tok(), [5, 6, 7], [8, 9], pieces=[2, 1], run=run)["pieces_known"] is True
    assert readout.compute(None, Tok(), [5, 6, 7], [8, 9], run=run)["pieces_known"] is False
    for bad in ([2, 2], [3, 0], [4, -1], [2]):  # do not add up to the prompt, or not sizes at all
        assert readout.compute(None, Tok(), [5, 6, 7], [8, 9], pieces=bad, run=run)["pieces_known"] is False
    assert seen == [[2, 1], None, None, None, None, None]


def test_the_record_shows_its_pieces_and_no_text_mode_drops_them(tmp_path):
    r = Recorder(tmp_path / "db.sqlite")
    r.save(make_record(prompt_ids=[5, 6, 7], reply_ids=[8, 9], pieces=[2, 1]))
    assert r.get("chatcmpl-1")["pieces"] == [2, 1]
    r.close()
    r = Recorder(tmp_path / "db.sqlite", keep_text=False)
    assert r.pieces("chatcmpl-1") is None and r.get("chatcmpl-1")["pieces"] is None
