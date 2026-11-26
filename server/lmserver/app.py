"""Latent Map local server: Inkling 2 Mini over the standard chat API on this Mac only,
with every request recorded locally and shown in an inspector page.

    ./run.sh            then open http://127.0.0.1:8765
"""

import argparse
import json
import logging
import os
import queue
import re
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
ROOT = HERE.parent
HOST = "127.0.0.1"
HEARTBEAT_SECONDS, HEARTBEAT_MAX_SECONDS = 10, 120
JSON = "application/json"


@dataclass
class Ctx:
    recorder: object
    map_payload: dict
    settings: dict
    static: Path
    readout: object = None  # (prompt_ids, reply_ids, pieces) -> dict, set once the model can be used
    readout_lock: object = field(default_factory=threading.Lock)


def check_args(args) -> None:
    if args.host != HOST:
        raise SystemExit(f"This server only listens on {HOST}; refusing host {args.host!r}.")
    if args.retention_days <= 0:
        raise SystemExit("--retention-days must be above 0.")


def allowed(headers, port: int) -> bool:
    """Only this Mac's own tools and this server's own page may call it. A web page
    elsewhere could otherwise post to the model or delete sessions through the browser."""
    local = {f"127.0.0.1:{port}", f"localhost:{port}"}
    if headers.get("Host") not in local:
        return False
    origin = headers.get("Origin")
    return origin is None or origin in {f"http://{h}" for h in local}


class NoBodies(logging.Filter):
    """The bundled server logs a request's body when it cannot parse it. Keep text out of the terminal."""

    def filter(self, record) -> bool:
        message = str(record.msg)
        return "Raw body" not in message and "Request Body" not in message


def _json(status: int, payload) -> tuple[int, str, bytes]:
    return status, JSON, json.dumps(payload).encode()


def route(ctx: Ctx, method: str, path: str):
    """Answer for one of our paths as (status, content type, body), or None when the
    path belongs to the bundled server."""
    path = path.split("?", 1)[0]
    if ".." in path:
        return _json(404, {"error": "not found"})
    if method == "GET" and path in ("/", "/index.html"):
        return 200, "text/html; charset=utf-8", (ctx.static / "index.html").read_bytes()
    if not path.startswith("/api/"):
        return None
    r = ctx.recorder
    if method == "GET":
        if path == "/api/map":
            return _json(200, ctx.map_payload)
        if path == "/api/settings":
            return _json(200, ctx.settings)
        if path == "/api/sessions":
            return _json(200, r.sessions())
        if m := re.fullmatch(r"/api/sessions/([\w.-]+)", path):
            return _json(200, r.requests(m[1]))
        if m := re.fullmatch(r"/api/sessions/([\w.-]+)/summary", path):
            from lmserver import summary

            found = [r.get(q["completion_id"]) for q in r.requests(m[1])]
            found = [f for f in found if f]  # one may have been forgotten in between
            return _json(200, summary.summarize(found)) if found else _json(404, {"error": "no such session"})
        if m := re.fullmatch(r"/api/requests/([\w.-]+)", path):
            found = r.get(m[1])
            return _json(200, found) if found else _json(404, {"error": "no such request"})
    if method == "POST":
        if m := re.fullmatch(r"/api/sessions/([\w.-]+)/forget", path):
            return _json(200, {"deleted": r.forget(m[1])})
        if m := re.fullmatch(r"/api/requests/([\w.-]+)/readout", path):
            return _readout(ctx, m[1])
    return _json(404, {"error": "not found"})


def _readout(ctx: Ctx, completion_id: str):
    """The stored readout, or a new one computed now (the model re-reads the whole text)."""
    from lmserver import readout

    with ctx.readout_lock:  # the same readout asked for twice is computed once
        found = ctx.recorder.get(completion_id)
        if found is None:
            return _json(404, {"error": "no such request"})
        if found["readout"] is not None:
            return _json(200, found["readout"])
        ids = ctx.recorder.token_ids(completion_id)
        if ids is None or not ids[0]:
            return _json(409, {"error": "Not available: this request was recorded without its text or tokens, "
                                        "or it contains an image, so the model cannot re-read it as it was."})
        if ctx.readout is None:
            return _json(503, {"error": "The model is not loaded yet."})
        try:
            result = ctx.readout(*ids, ctx.recorder.pieces(completion_id))
        except readout.TooLong as err:
            return _json(413, {"error": f"Too long for a readout in this version: {err}."})
        except readout.NotLoaded:
            return _json(503, {"error": "The model is not loaded yet."})
        except Exception as err:  # say what failed; never the text
            return _json(500, {"error": f"The readout failed ({type(err).__name__}). If this Mac is short of memory, close other apps and try again."})
        ctx.recorder.save_readout(completion_id, result)
        return _json(200, result)


class LiveHub:
    """Fans the words of the request being generated out to open inspector pages."""

    def __init__(self):
        self._lock, self._queues = threading.Lock(), set()

    def publish(self, event: dict) -> None:
        with self._lock:
            for q in self._queues:
                try:
                    q.put_nowait(event)
                except queue.Full:
                    pass  # a stalled page misses live words; the stored record is complete

    def subscribe(self) -> queue.Queue:
        q = queue.Queue(maxsize=5000)
        with self._lock:
            self._queues.add(q)
        return q

    def unsubscribe(self, q) -> None:
        with self._lock:
            self._queues.discard(q)


def map_payload(bundle: Path, norms: dict, tag: str) -> dict:
    import numpy as np

    b = np.load(bundle / "block2.npz")
    info = json.loads((bundle / "block2.json").read_text())
    xy = b["xy"].astype(float)
    lo, span = xy.min(axis=0), np.ptp(xy, axis=0)
    counts = np.bincount(b["cell"][:, -1], minlength=len(xy))
    areas = [
        {
            "id": int(a),
            "name": info.get("names", {}).get(a, ""),
            "score": info.get("name_scores", {}).get(a, 0.0),
            "words": info["words"].get(a, [])[:10],
            "answer_share": info["area_mix"][a]["answer_share"],
            "top_form": (info["area_mix"][a]["top_form"] or [None])[0],
            "points": info["area_points"][a],
        }
        for a in sorted(info["broad_of_fine"], key=int)
    ]
    cells = [
        {"x": float((p[0] - lo[0]) / span[0]), "y": float((p[1] - lo[1]) / span[1]), "area": int(f), "n": int(n)}
        for p, f, n in zip(xy, b["fine"], counts)
    ]
    return {"cells": cells, "areas": areas, "norms": norms, "tag": tag}


def make_app_handler(ctx: Ctx, hub: LiveHub, port: int):
    from lmserver import hooks

    class AppHandler(hooks.make_handler()):
        def _refuse(self) -> bool:
            if allowed(self.headers, port):
                return False
            self._reply(403, JSON, b'{"error": "this server only answers its own page and local tools"}')
            return True

        def _reply(self, status, kind, data):
            self.send_response(status)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def _live(self):
            self.send_response(200)
            self.send_header("Content-Type", "application/x-ndjson")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            q = hub.subscribe()
            try:
                while True:
                    try:
                        event = q.get(timeout=15)
                    except queue.Empty:
                        event = {"idle": True}
                    self.wfile.write(json.dumps(event).encode() + b"\n")
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, OSError):
                pass
            finally:
                hub.unsubscribe(q)

        def do_GET(self):
            if self._refuse():
                return
            if self.path == "/api/live":
                return self._live()
            answer = route(ctx, "GET", self.path)
            if answer is None:
                return super().do_GET()
            self._reply(*answer)

        def do_POST(self):
            if self._refuse():
                return
            answer = route(ctx, "POST", self.path)
            if answer is None:
                return super().do_POST()
            self._reply(*answer)

        def do_OPTIONS(self):
            if self._refuse():
                return
            return super().do_OPTIONS()

    return AppHandler


def make_prompt_cache(S, cli_args):
    """The bundled server applies its byte limit only on a path this server does not use."""
    limit = getattr(cli_args, "prompt_cache_bytes", None)
    return S.LRUPromptCache(cli_args.prompt_cache_size, **({"max_bytes": limit} if limit else {}))


def open_recorder(path: Path, keep_text: bool, retention_days: float):
    """The database and SQLite's side files, readable by this user only."""
    from lmserver.recorder import Recorder

    os.umask(0o077)
    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    os.chmod(path.parent, 0o700)
    for extra in ("", "-wal", "-shm"):  # files left by an earlier version
        if os.path.exists(str(path) + extra):
            os.chmod(str(path) + extra, 0o600)
    return Recorder(path, keep_text=keep_text, retention_days=retention_days)


def start_purger(recorder, stop: threading.Event, every: float = 3600) -> threading.Thread:
    """Drop records past retention, whatever else is or is not running."""

    def loop():
        while not stop.wait(every):
            try:
                recorder.purge()
            except Exception as err:
                logging.getLogger("lmserver").warning("purge failed: %s", type(err).__name__)

    thread = threading.Thread(target=loop, daemon=True)
    thread.start()
    return thread


def keep_warm(recorder, stop: threading.Event) -> None:
    """While idle, read every weight now and then so macOS does not page the model out,
    backing off when the Mac is short of memory."""
    import mlx.core as mx
    from mlx.utils import tree_flatten

    from lmserver import hooks

    interval = HEARTBEAT_SECONDS
    while not stop.wait(interval):
        model = hooks.model()
        if model is None or hooks.busy():
            continue
        try:
            with hooks.model_lock():
                t = time.time()
                mx.eval(mx.stack([p.sum() for _, p in tree_flatten(model.parameters())]))
                took = time.time() - t
        except Exception as err:
            logging.getLogger("lmserver").warning("keep-warm pass failed: %s", type(err).__name__)
            continue
        interval = min(HEARTBEAT_MAX_SECONDS, interval * 2) if took > 3 else HEARTBEAT_SECONDS
        if took > 3:
            print(f"keep-warm pass took {took:.1f}s: this Mac is short of memory and paged the model out; "
                  f"next pass in {interval}s. Closing other large apps is the real fix.", flush=True)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Latent Map local server")
    p.add_argument("--host", default=HOST, help=argparse.SUPPRESS)
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--model", default="thinkingmachines/Inkling-2-Mini")
    p.add_argument("--db", default="~/.latent-map/requests.db")
    p.add_argument("--no-text", action="store_true", help="store numbers only, never prompt or reply text")
    p.add_argument("--retention-days", type=float, default=5)
    p.add_argument("--map-tag", default="v1")
    p.add_argument("--cache-bytes", default="2GB", help="limit for saved prompt caches")
    p.add_argument("--no-keep-warm", action="store_true")
    args = p.parse_args(argv)
    check_args(args)

    os.environ["HF_HUB_OFFLINE"] = "1"  # the model is on disk; nothing is fetched or checked online
    import mlx_lm.server as S
    from livemap.live import LiveMap

    from lmserver import hooks

    db = Path(args.db).expanduser()
    recorder = open_recorder(db, keep_text=not args.no_text, retention_days=args.retention_days)
    recorder.purge()

    bundle = ROOT / "basemap" / f"bundle_{args.map_tag}"
    live = LiveMap.load(bundle, ROOT / "prototype" / f"norms_{args.map_tag}.json")
    ctx = Ctx(
        recorder=recorder,
        map_payload=map_payload(bundle, live.norms, args.map_tag),
        settings={"keep_text": not args.no_text, "retention_days": args.retention_days, "map_tag": args.map_tag,
                  "db": str(db)},
        static=HERE / "static",
    )
    hub = LiveHub()

    def read_out(prompt_ids, reply_ids, pieces=None):
        from lmserver import readout

        if hooks.model() is None:
            raise readout.NotLoaded()
        with hooks.model_lock():  # waits for a reply in progress; replies wait for this
            return readout.compute(hooks.model(), hooks.tokenizer(), prompt_ids, reply_ids, pieces)

    ctx.readout = read_out
    hooks.install(recorder, live, on_word=hub.publish)
    handler = make_app_handler(ctx, hub, args.port)
    stop = threading.Event()

    def run(host, port, provider):
        generator = S.ResponseGenerator(provider, make_prompt_cache(S, provider.cli_args))
        start_purger(recorder, stop)
        if not args.no_keep_warm:
            threading.Thread(target=keep_warm, args=(recorder, stop), daemon=True).start()
        print(f"Latent Map server: http://{host}:{port}  (API at /v1; loading the model takes about a minute)", flush=True)
        try:
            S._run_http_server(host, port, generator, handler_class=handler)
        finally:
            stop.set()

    S.run = run
    logging.getLogger().addFilter(NoBodies())
    sys.argv = [
        "lmserver", "--model", args.model, "--host", HOST, "--port", str(args.port),
        "--allowed-origins", f"http://{HOST}:{args.port}", "--max-tokens", "2048",
        "--prompt-cache-bytes", args.cache_bytes, "--log-level", "WARNING",
    ]
    S.main()


if __name__ == "__main__":
    main()
