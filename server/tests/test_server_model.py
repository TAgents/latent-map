"""End-to-end checks with the real model. Run alone: pytest -m model tests/test_server_model.py"""

import http.client
import json
import threading
import time
import urllib.request

import mlx.core as mx
import numpy as np
import pytest

pytestmark = pytest.mark.model
PORT = 8799
BASE = f"http://127.0.0.1:{PORT}"
TOOLS = [{"type": "function", "function": {"name": "read_file", "description": "Read a file from the project and return its text.",
          "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}}]


def get(path):
    return json.loads(urllib.request.urlopen(BASE + path, timeout=30).read())


def chat(**body):
    req = urllib.request.Request(BASE + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=300).read())


def record(cid, tries=50):
    for _ in range(tries):
        try:
            return get("/api/requests/" + cid)
        except urllib.error.HTTPError:
            time.sleep(0.2)
    raise AssertionError("no record for " + cid)


@pytest.fixture(scope="module")
def server(tmp_path_factory):
    from lmserver import app

    db = tmp_path_factory.mktemp("db") / "requests.db"
    threading.Thread(target=app.main, args=(["--port", str(PORT), "--db", str(db)],), daemon=True).start()
    for _ in range(150):
        try:
            if get("/health")["status"] == "ok":
                break
        except Exception:
            time.sleep(1)
    chat(messages=[{"role": "user", "content": "Hi"}], max_tokens=4)  # waits for the model to load
    return db


def agreement(rec):
    """Share of words whose stored last-pass cell equals the cell from a plain pass over
    the same tokens, at the word's own position and at the position before it."""
    from latent_capture.capture import capturing
    from lmserver import hooks

    prompt, reply = hooks._state["last"]
    live_map = None
    with hooks.model_lock():
        cap = hooks._state["cap"]
        cap.__exit__()
        try:
            with capturing(hooks.model(), all_positions=True) as ref:
                mx.eval(hooks.model()(mx.array([prompt + reply])))
                plain = ref.records[-1]
        finally:
            cap.__enter__()
    from pathlib import Path
    from livemap.live import LiveMap

    root = Path(__file__).resolve().parents[2]
    live_map = LiveMap.load(root / "basemap" / "bundle_v1", root / "prototype" / "norms_v1.json")
    own = before = n = 0
    for w in rec["words"]:
        if w["cells"] is None:
            continue
        p = len(prompt) + w["i"]
        cell = lambda q: live_map.describe(np.stack([b[q] for b in plain.block2]), np.stack([b[q] for b in plain.block3]))["cells"][-1]
        own += cell(p) == w["cells"][-1]
        before += cell(p - 1) == w["cells"][-1]
        n += 1
    return own / n, before / n, n


def test_chat_reply_is_clean_recorded_and_aligned(server):
    r = chat(messages=[{"role": "user", "content": "Why is the sky blue? Two sentences."}], max_tokens=200, reasoning_effort="low")
    m = r["choices"][0]["message"]
    assert "<|" not in m["content"] and m["content"].strip() and m["reasoning"].strip()
    rec = record(r["id"])
    assert len(rec["words"]) >= r["usage"]["completion_tokens"] - 1
    assert rec["settings"]["reasoning_effort"] == "low" and rec["finish"] == "stop"
    assert {w["phase"] for w in rec["words"]} >= {"thinking", "answer", "marker"}
    own, before, n = agreement(rec)
    assert n > 20 and own >= 0.85 and own - before >= 0.4, (own, before, n)


def test_follow_up_turn_uses_the_cache(server):
    first = [{"role": "user", "content": "Name three rivers in Europe. " + "Keep it short. " * 40}]
    a = chat(messages=first, max_tokens=120, reasoning_effort="low")
    b = chat(messages=first + [{"role": "assistant", "content": a["choices"][0]["message"]["content"]},
                               {"role": "user", "content": "And one in Asia?"}], max_tokens=120, reasoning_effort="low")
    assert b["usage"]["prompt_tokens_details"]["cached_tokens"] > 100
    rec = record(b["id"])
    assert rec["cached_tokens"] > 100 and rec["session"] == record(a["id"])["session"]
    own, before, n = agreement(rec)
    assert own >= 0.85 and own - before >= 0.4, (own, before, n)


def test_tool_call_comes_back_in_standard_form(server):
    r = chat(messages=[{"role": "user", "content": "What version is declared in pyproject.toml? Use the tools."}], tools=TOOLS, max_tokens=300)
    c = r["choices"][0]
    assert c["finish_reason"] == "tool_calls"
    call = c["message"]["tool_calls"][0]["function"]
    assert call["name"] == "read_file" and json.loads(call["arguments"]) == {"path": "pyproject.toml"}
    rec = record(r["id"])
    assert rec["finish"] == "tool_calls" and rec["settings"]["tools"] == ["read_file"]


def test_abandoned_stream_is_recorded_and_the_next_request_works(server):
    conn = http.client.HTTPConnection("127.0.0.1", PORT, timeout=60)
    body = json.dumps({"messages": [{"role": "user", "content": "Write a long essay about rivers."}], "stream": True, "max_tokens": 600})
    conn.request("POST", "/v1/chat/completions", body, {"Content-Type": "application/json"})
    resp, cid, chunks = conn.getresponse(), None, 0
    while chunks < 5:
        line = resp.readline().decode()
        if line.startswith("data: {"):
            cid = json.loads(line[6:])["id"]
            chunks += 1
    resp.close()
    conn.close()
    rec = None
    for _ in range(100):
        try:
            rec = get("/api/requests/" + cid)
            break
        except urllib.error.HTTPError:
            time.sleep(0.3)
    assert rec is not None and rec["finish"] == "cancelled" and 0 < rec["n_tokens"] < 600
    r = chat(messages=[{"role": "user", "content": "Say hello in French, one word."}], max_tokens=150, reasoning_effort="low")
    own, before, n = agreement(record(r["id"]))
    assert own >= 0.85, (own, before, n)


def test_other_web_pages_and_hosts_are_refused(server):
    req = urllib.request.Request(BASE + "/api/sessions", headers={"Origin": "https://example.com"})
    with pytest.raises(urllib.error.HTTPError) as e:
        urllib.request.urlopen(req, timeout=10)
    assert e.value.code == 403
    assert isinstance(get("/api/sessions"), list) and get("/api/settings")["keep_text"] is True


def post(path):
    req = urllib.request.Request(BASE + path, b"", method="POST")
    return json.loads(urllib.request.urlopen(req, timeout=300).read())


def test_readout_last_pass_is_what_the_model_wrote(server):
    r = chat(messages=[{"role": "user", "content": "What is the capital of the country where Mozart was born? One short sentence."}],
             max_tokens=200, reasoning_effort="low", temperature=0)
    rec = record(r["id"])
    assert rec["readout_available"] and rec["readout"] is None
    out = post(f"/api/requests/{r['id']}/readout")
    words, written = out["words"], rec["reply_tokens"]
    assert len(words) == len(written) and out["k"] == 5
    # Greedy generation: the last pass's top word is the word that was written.
    same = sum(h[2] for h in out["hits"])
    assert same >= 0.99 * len(written), (same, len(written))
    assert out["hits"][0][2] and out["hits"][-1][2]  # the first and last word line up too
    assert [w[2][0][0] for w in words[:5]] == written[:5]
    # The earlier passes are a different reading, not a copy of the last one.
    assert sum(w[0][0][0] != w[2][0][0] for w in words) > 0
    assert all(0 <= alt[1] <= 1 for w in words for k in w for alt in k)
    assert record(r["id"])["readout"]["words"] == words  # stored
    assert post(f"/api/requests/{r['id']}/readout")["seconds"] == out["seconds"]  # served from the store, not recomputed
    # The server still generates correctly afterwards.
    again = chat(messages=[{"role": "user", "content": "Say hello."}], max_tokens=80, reasoning_effort="low")
    own, before, n = agreement(record(again["id"]))
    assert own >= 0.85, (own, before, n)


def test_readout_of_a_request_longer_than_one_reading_piece(server):
    # Several thousand tokens: over the earlier 4,096 limit, several prompt pieces, and past the sliding window of 1,024.
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    notes = (root / "signals" / "REPORT_hidden.md").read_text() + (root / "server" / "README.md").read_text()
    r = chat(messages=[{"role": "system", "content": "Notes:\n" + notes},
                       {"role": "user", "content": "In two sentences, what did test 1 show?"}],
             max_tokens=150, reasoning_effort="low", temperature=0)
    rec = record(r["id"])
    assert rec["prompt_tokens"] > 4096, rec["prompt_tokens"]
    out = post(f"/api/requests/{r['id']}/readout")
    assert len(out["words"]) == len(rec["reply_tokens"])
    same = sum(h[2] for h in out["hits"])
    assert same >= 0.99 * len(out["hits"]), (same, len(out["hits"]))
    # A follow-up turn continues the saved cache: its prompt was read partly by the first request,
    # and the readout has to put those pieces together.
    first = [{"role": "system", "content": "Notes:\n" + notes}, {"role": "user", "content": "In two sentences, what did test 1 show?"}]
    r2 = chat(messages=first + [{"role": "assistant", "content": r["choices"][0]["message"]["content"]},
                                {"role": "user", "content": "And test 2, in two sentences?"}],
              max_tokens=150, reasoning_effort="low", temperature=0)
    rec2 = record(r2["id"])
    # The piece records themselves: the first prompt read fresh, the follow-up continuing its saved cache.
    assert sum(rec["pieces"]) == rec["prompt_tokens"] and rec["pieces"][-1] == 1 and rec["pieces"][0] == 2048, rec["pieces"]
    assert rec2["pieces"][: len(rec["pieces"]) - 1] == rec["pieces"][:-1], (rec["pieces"], rec2["pieces"])
    assert sum(rec2["pieces"]) == rec2["prompt_tokens"] and rec2["pieces"][-1] == 1
    assert sum(rec2["pieces"][len(rec["pieces"]) - 1 :]) == rec2["prompt_tokens"] - rec2["cached_tokens"]
    assert rec2["cached_tokens"] > 4096, rec2["cached_tokens"]
    out2 = post(f"/api/requests/{r2['id']}/readout")
    assert out["pieces_known"] is True and out2["pieces_known"] is True
    same2 = sum(h[2] for h in out2["hits"])
    assert same2 >= 0.99 * len(out2["hits"]), (same2, len(out2["hits"]))
    print("READOUT HITS", same, len(out["hits"]), out["seconds"], same2, len(out2["hits"]), out2["seconds"])
    # A client that sends the reply back exactly, thinking included (agents do), continues from the cache
    # saved after the reply: prompt pieces, then one piece per token written.
    said = r["choices"][0]["message"]
    back = {"role": "assistant", "content": said["content"], "reasoning_content": said.get("reasoning_content") or said.get("reasoning")}
    r3 = chat(messages=first + [back, {"role": "user", "content": "And test 2, in one sentence?"}],
              max_tokens=60, reasoning_effort="low", temperature=0)
    rec3 = record(r3["id"])
    print("AFTER-REPLY", rec["prompt_tokens"], rec["n_tokens"], rec3["cached_tokens"], rec3["pieces"][:4], rec3["pieces"].count(1))
    assert rec3["cached_tokens"] > rec["prompt_tokens"], (rec3["cached_tokens"], rec["prompt_tokens"])
    assert rec3["pieces"] is not None and sum(rec3["pieces"]) == rec3["prompt_tokens"]
    assert rec3["pieces"][: len(rec["pieces"])] == rec["pieces"] and rec3["pieces"].count(1) >= rec3["cached_tokens"] - rec["prompt_tokens"]
    out3 = post(f"/api/requests/{r3['id']}/readout")
    same3 = sum(h[2] for h in out3["hits"])
    assert out3["pieces_known"] is True and same3 >= 0.99 * len(out3["hits"]), (same3, len(out3["hits"]))
    again = chat(messages=[{"role": "user", "content": "Say hello."}], max_tokens=80, reasoning_effort="low")
    own, before, n = agreement(record(again["id"]))
    assert own >= 0.85, (own, before, n)


def test_the_next_step_of_a_run_carries_on_from_the_readout_before_it(server):
    # An agent's second step sends the first reply back exactly. Its readout, asked for right after
    # the first step's, should read only what is new and still give what the model wrote.
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    notes = (root / "signals" / "REPORT_hidden.md").read_text() + (root / "server" / "README.md").read_text()
    first = [{"role": "system", "content": "Notes:\n" + notes}, {"role": "user", "content": "In one sentence, what was test 3?"}]
    r = chat(messages=first, max_tokens=1200, reasoning_effort="low", temperature=0)
    assert r["choices"][0]["finish_reason"] == "stop"
    said = r["choices"][0]["message"]
    back = {"role": "assistant", "content": said["content"], "reasoning_content": said.get("reasoning_content") or said.get("reasoning")}
    r2 = chat(messages=first + [back, {"role": "user", "content": "And what did it find, in one sentence?"}],
              max_tokens=1200, reasoning_effort="low", temperature=0)
    rec, rec2 = record(r["id"]), record(r2["id"])
    assert rec2["cached_tokens"] == rec["prompt_tokens"] + rec["n_tokens"], (rec2["cached_tokens"], rec["prompt_tokens"], rec["n_tokens"])
    out = post(f"/api/requests/{r['id']}/readout")
    out2 = post(f"/api/requests/{r2['id']}/readout")
    print("SHARED READOUT", rec["prompt_tokens"], out["seconds"], out["reused_tokens"], "then", rec2["prompt_tokens"], out2["seconds"], out2["reused_tokens"])
    assert out["reused_tokens"] == 0
    assert out2["reused_tokens"] == rec["prompt_tokens"] + rec["n_tokens"], out2["reused_tokens"]
    for o in (out, out2):
        same = sum(h[2] for h in o["hits"])
        assert o["pieces_known"] is True and same >= 0.99 * len(o["hits"]), (same, len(o["hits"]))
    assert out2["seconds"] < out["seconds"]
    # A reply in between drops the kept reading, so a readout after it starts again and is still right.
    r3 = chat(messages=first + [back, {"role": "user", "content": "And test 2, in one sentence?"}],
              max_tokens=1200, reasoning_effort="low", temperature=0)
    out3 = post(f"/api/requests/{r3['id']}/readout")
    same3 = sum(h[2] for h in out3["hits"])
    assert out3["reused_tokens"] == 0 and same3 >= 0.99 * len(out3["hits"]), (out3["reused_tokens"], same3, len(out3["hits"]))
