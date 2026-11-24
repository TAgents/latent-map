"""A local store of what the model did for each request. One SQLite file on this Mac."""

import hashlib
import io
import json
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

NO_TEXT_TITLE = "(text not kept)"


@dataclass
class RequestRecord:
    completion_id: str
    time: float
    client: str
    messages: list | None  # None for text completions
    reply_tokens: list[str]  # one string per generated token
    words: list[dict]  # per generated token: phase, cells, area, move, level, settled_by, loop
    mean_states: np.ndarray | None  # float32 (thinking/answer, block 2/3, pass, hidden); NaN where a phase is empty
    settings: dict
    prompt_tokens: int
    cached_tokens: int
    first_token_s: float
    tokens_per_s: float
    finish: str  # stop, length, tool_calls, cancelled, error


def _text_of(message: dict) -> str:
    content = message.get("content")
    if isinstance(content, list):
        return " ".join(str(p.get("text", "")) for p in content if isinstance(p, dict))
    return content if isinstance(content, str) else ("" if content is None else json.dumps(content, default=str))


def _first(messages: list, role: str) -> str:
    return next((_text_of(m) for m in messages if isinstance(m, dict) and m.get("role") == role), "")


def session_of(rec: RequestRecord) -> str:
    """Requests of one conversation share a client, a system prompt and a first user message."""
    if not rec.messages:
        return rec.completion_id
    key = "\n".join([rec.client, _first(rec.messages, "system"), _first(rec.messages, "user")])
    return hashlib.sha1(key.encode()).hexdigest()[:16]


class Recorder:
    def __init__(self, path: str | Path, keep_text: bool = True, retention_days: float = 5):
        self.keep_text, self.retention_days = keep_text, retention_days
        self._lock = threading.Lock()
        self._db = sqlite3.connect(str(path), check_same_thread=False)
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("PRAGMA secure_delete=ON")  # deleted rows are overwritten, not just unlinked
        self._db.execute(
            """CREATE TABLE IF NOT EXISTS requests (
                completion_id TEXT PRIMARY KEY, session TEXT, time REAL, client TEXT, title TEXT,
                text_kept INTEGER, messages TEXT, reply_tokens TEXT, words TEXT, settings TEXT,
                prompt_tokens INTEGER, cached_tokens INTEGER, n_tokens INTEGER,
                first_token_s REAL, tokens_per_s REAL, finish TEXT, mean_states BLOB)"""
        )
        self._db.execute("CREATE INDEX IF NOT EXISTS by_session ON requests (session, time)")
        self._db.commit()
        if not keep_text:
            self._drop_stored_text()

    def _drop_stored_text(self) -> None:
        """Text kept by an earlier run must not outlive a switch to no-text mode."""
        rows = self._db.execute("SELECT completion_id, settings FROM requests WHERE text_kept = 1").fetchall()
        if not rows:
            return
        for cid, settings in rows:
            kept = json.loads(settings or "{}")
            kept.pop("tools", None)
            self._db.execute(
                "UPDATE requests SET messages = NULL, reply_tokens = NULL, title = ?, text_kept = 0, settings = ? "
                "WHERE completion_id = ?",
                (NO_TEXT_TITLE, json.dumps(kept), cid),
            )
        self._db.commit()
        self._db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        self._db.execute("VACUUM")
        self._db.execute("PRAGMA wal_checkpoint(TRUNCATE)")

    def close(self) -> None:
        with self._lock:
            self._db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            self._db.close()

    def save(self, rec: RequestRecord) -> None:
        settings = dict(rec.settings)
        if self.keep_text:
            title = (_first(rec.messages, "user") if rec.messages else "")[:60]
            messages, reply = json.dumps(rec.messages), json.dumps(rec.reply_tokens)
        else:
            title, messages, reply = NO_TEXT_TITLE, None, None
            settings.pop("tools", None)  # tool names are the user's text too
        blob = None
        if rec.mean_states is not None:
            buf = io.BytesIO()
            np.save(buf, rec.mean_states.astype(np.float32))
            blob = buf.getvalue()
        row = (
            rec.completion_id, session_of(rec), rec.time, rec.client, title, int(self.keep_text), messages, reply,
            json.dumps(rec.words), json.dumps(settings), rec.prompt_tokens, rec.cached_tokens, len(rec.reply_tokens),
            rec.first_token_s, rec.tokens_per_s, rec.finish, blob,
        )
        with self._lock:
            self._db.execute("INSERT OR REPLACE INTO requests VALUES (" + ",".join("?" * len(row)) + ")", row)
            self._db.commit()

    def sessions(self) -> list[dict]:
        with self._lock:
            rows = self._db.execute(
                """SELECT session, MIN(client), MIN(time), MAX(time), COUNT(*),
                          (SELECT title FROM requests r2 WHERE r2.session = r.session ORDER BY time LIMIT 1)
                   FROM requests r GROUP BY session ORDER BY MAX(time) DESC"""
            ).fetchall()
        keys = ("session", "client", "first_time", "last_time", "requests", "title")
        return [dict(zip(keys, row)) for row in rows]

    def requests(self, session: str) -> list[dict]:
        with self._lock:
            rows = self._db.execute(
                "SELECT completion_id, time, n_tokens, finish FROM requests WHERE session = ? ORDER BY time", (session,)
            ).fetchall()
        return [dict(zip(("completion_id", "time", "tokens", "finish"), row)) for row in rows]

    def get(self, completion_id: str) -> dict | None:
        cols = ("completion_id", "session", "time", "client", "title", "text_kept", "messages", "reply_tokens", "words",
                "settings", "prompt_tokens", "cached_tokens", "n_tokens", "first_token_s", "tokens_per_s", "finish")
        with self._lock:
            row = self._db.execute(
                f"SELECT {','.join(cols)} FROM requests WHERE completion_id = ?", (completion_id,)
            ).fetchone()
        if row is None:
            return None
        out = dict(zip(cols, row))
        out["text_kept"] = bool(out["text_kept"])
        for key in ("messages", "reply_tokens", "words", "settings"):
            out[key] = json.loads(out[key]) if out[key] is not None else None
        return out

    def mean_states(self, completion_id: str) -> np.ndarray | None:
        with self._lock:
            row = self._db.execute("SELECT mean_states FROM requests WHERE completion_id = ?", (completion_id,)).fetchone()
        return None if row is None or row[0] is None else np.load(io.BytesIO(row[0]))

    def _delete(self, where: str, args: tuple) -> int:
        with self._lock:
            n = self._db.execute(f"DELETE FROM requests WHERE {where}", args).rowcount
            self._db.commit()
            if n:
                self._db.execute("PRAGMA wal_checkpoint(TRUNCATE)")  # do not leave deleted text in the log file
        return n

    def forget(self, session: str) -> int:
        return self._delete("session = ?", (session,))

    def purge(self, now: float | None = None) -> int:
        cutoff = (time.time() if now is None else now) - self.retention_days * 86400
        return self._delete("time < ?", (cutoff,))
