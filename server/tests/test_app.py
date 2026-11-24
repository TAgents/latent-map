import argparse
import json

import pytest

from lmserver import app
from lmserver.recorder import Recorder
from test_recorder import make_record


@pytest.fixture
def ctx(tmp_path):
    static = tmp_path / "static"
    static.mkdir()
    (static / "index.html").write_text("<html>page</html>")
    r = Recorder(tmp_path / "db.sqlite")
    r.save(make_record())
    r.save(make_record(completion_id="chatcmpl-2", messages=[{"role": "user", "content": "Other"}]))
    return app.Ctx(recorder=r, map_payload={"tag": "v1"}, settings={"keep_text": True, "retention_days": 5, "map_tag": "v1"}, static=static)


def body(result):
    return json.loads(result[2])


def test_check_args_refuses_a_non_local_host():
    with pytest.raises(SystemExit) as e:
        app.check_args(argparse.Namespace(host="0.0.0.0", port=8765, retention_days=5))
    assert "127.0.0.1" in str(e.value)
    app.check_args(argparse.Namespace(host="127.0.0.1", port=8765, retention_days=5))


def test_route_sessions_and_request_round_trip(ctx):
    sessions = body(app.route(ctx, "GET", "/api/sessions"))
    assert len(sessions) == 2
    requests = body(app.route(ctx, "GET", "/api/sessions/" + sessions[0]["session"]))
    status, kind, raw = app.route(ctx, "GET", "/api/requests/" + requests[0]["completion_id"])
    assert status == 200 and kind == "application/json" and json.loads(raw)["words"]
    assert body(app.route(ctx, "GET", "/api/map")) == {"tag": "v1"}


def test_route_unknown_request_is_404(ctx):
    assert app.route(ctx, "GET", "/api/requests/nope")[0] == 404
    assert app.route(ctx, "GET", "/api/whatever")[0] == 404


def test_route_forget_deletes_and_reports_count(ctx):
    session = body(app.route(ctx, "GET", "/api/sessions"))[0]["session"]
    assert app.route(ctx, "GET", f"/api/sessions/{session}/forget")[0] == 404  # deleting needs POST
    assert body(app.route(ctx, "POST", f"/api/sessions/{session}/forget")) == {"deleted": 1}
    assert len(body(app.route(ctx, "GET", "/api/sessions"))) == 1


def test_route_ignores_api_paths_of_the_bundled_server(ctx):
    assert app.route(ctx, "POST", "/v1/chat/completions") is None
    assert app.route(ctx, "GET", "/v1/models") is None
    assert app.route(ctx, "GET", "/health") is None


def test_route_rejects_path_traversal_for_static(ctx):
    assert app.route(ctx, "GET", "/../lmserver/app.py")[0] == 404
    assert app.route(ctx, "GET", "/static/../../x")[0] == 404
    status, kind, raw = app.route(ctx, "GET", "/")
    assert status == 200 and kind.startswith("text/html") and b"page" in raw


def test_settings_report_text_mode(ctx):
    assert body(app.route(ctx, "GET", "/api/settings"))["keep_text"] is True


def test_requests_from_other_web_pages_are_refused():
    ok = lambda **h: app.allowed(h, 8765)
    assert ok(Host="127.0.0.1:8765")  # a coding tool: no Origin
    assert ok(Host="localhost:8765", Origin="http://127.0.0.1:8765")
    assert not ok(Host="127.0.0.1:8765", Origin="https://example.com")
    assert not ok(Host="evil.example:8765")  # a web page that re-pointed its own name at this Mac
    assert not ok(Host="127.0.0.1:8765", Origin="null")


def test_log_filter_drops_request_bodies():
    import logging

    f = app.NoBodies()
    rec = lambda msg: logging.LogRecord("x", logging.ERROR, "", 0, msg, (), None)
    assert not f.filter(rec("JSONDecodeError: boom - Raw body: secret"))
    assert not f.filter(rec("Incoming Request Body: secret"))
    assert f.filter(rec("Starting httpd"))
