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
