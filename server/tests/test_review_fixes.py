"""Tests for the findings of the whole-branch review (2026-11-24)."""

import argparse
import os
import stat
import threading
import time

import numpy as np
import pytest

from lmserver import app, hooks
from lmserver.recorder import Recorder
from test_hooks import META, TIMING, Tok, cap, live  # noqa: F401
from test_recorder import NOW, DAY, make_record


class FakeCache:
    def __init__(self, nbytes):
        self.nbytes = nbytes

    def is_trimmable(self):
        return False


def test_prompt_cache_byte_limit_is_enforced():
    import mlx_lm.server as S

    cache = app.make_prompt_cache(S, argparse.Namespace(prompt_cache_size=10, prompt_cache_bytes=3000))
    for i in range(10):
        cache.insert_cache("m", [1, 2, i + 10], [FakeCache(1000)])
    assert cache.nbytes <= 3000


def test_requests_cannot_switch_or_add_models():
    h = argparse.Namespace(requested_model="Qwen/Qwen3-30B", adapter="some/adapter", requested_draft_model="x")
    hooks.pin_model(h)
    assert (h.requested_model, h.adapter, h.requested_draft_model) == ("default_model", None, "default_model")


def test_remote_image_urls_are_refused():
    remote = [{"role": "user", "content": [{"type": "text", "text": "look"}, {"type": "image_url", "image_url": {"url": "https://example.com/a.png"}}]}]
    with pytest.raises(ValueError, match="fetch"):
        hooks.refuse_remote_images(remote)
    with pytest.raises(ValueError):
        hooks.refuse_remote_images([{"role": "user", "content": [{"type": "image_url", "image_url": "HTTP://example.com/a.png"}]}])
    hooks.refuse_remote_images([{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}]}])
    hooks.refuse_remote_images([{"role": "user", "content": "plain https://example.com text is fine"}])
    hooks.refuse_remote_images(None)


def test_purge_runs_on_its_own_timer_and_survives_errors(tmp_path):
    r = Recorder(tmp_path / "db.sqlite", retention_days=5)
    r.save(make_record(completion_id="old", time=time.time() - 6 * DAY))
    calls, real = [], r.purge

    def flaky():
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("boom")
        return real()

    r.purge = flaky
    stop = threading.Event()
    app.start_purger(r, stop, every=0.05)
    for _ in range(100):
        if r.get("old") is None:
            break
        time.sleep(0.02)
    stop.set()
    assert r.get("old") is None and len(calls) >= 2


def test_database_files_are_private(tmp_path):
    r = app.open_recorder(tmp_path / "sub" / "requests.db", keep_text=True, retention_days=5)
    r.save(make_record())
    for f in (tmp_path / "sub").iterdir():
        assert stat.S_IMODE(f.stat().st_mode) & 0o077 == 0, f.name
    assert stat.S_IMODE((tmp_path / "sub").stat().st_mode) == 0o700


def test_switching_to_no_text_removes_text_stored_earlier(tmp_path):
    r = Recorder(tmp_path / "db.sqlite")
    r.save(make_record(messages=[{"role": "user", "content": "zebra-quartz question"}], reply_tokens=["zebra-quartz"],
                       settings={"tools": ["zebra-quartz"], "max_tokens": 5}))
    r.close()
    r = Recorder(tmp_path / "db.sqlite", keep_text=False)
    got = r.get("chatcmpl-1")
    assert got["text_kept"] is False and got["messages"] is None and len(got["words"]) == 3
    r.close()
    for f in tmp_path.iterdir():
        assert b"zebra-quartz" not in f.read_bytes(), f.name


def test_odd_message_content_does_not_lose_the_record(tmp_path):
    r = Recorder(tmp_path / "db.sqlite")
    r.save(make_record(messages=[{"role": "user", "content": {"weird": 1}}, {"role": "user", "content": [{"type": "text", "text": 5}]}]))
    assert r.get("chatcmpl-1") is not None


def test_builder_fed_in_pieces_matches_all_at_once_and_holds_no_records(live):  # noqa: F811
    tokens = [5, 6, 7, 6, 7, 6, 7, 6]  # the answer marker, then answer words
    records = [cap(0, 3), cap(1, 1)] + [cap(i + 2) for i in range(8)]
    whole = hooks.build_record(META, [0] * 6, 4, tokens, records, Tok(), live, TIMING, "stop")
    b = hooks.WordBuilder(Tok(), live, 4)
    b.update(records[:2])  # the prompt
    for i, t in enumerate(tokens):
        b.add_token(t)
        b.update(records[2 + i : 3 + i])
        assert b.held() == 0  # every record is used and let go at once
    b.finish()
    assert [w["cells"] for w in b.words] == [w["cells"] for w in whole.words]
    assert np.allclose(b.mean_states(), whole.mean_states, equal_nan=True)


def test_more_records_than_tokens_are_not_kept_forever(live):  # noqa: F811
    b = hooks.WordBuilder(Tok(), live, 0)
    b.add_token(3)
    b.update([cap(1), cap(2)])
    assert len(b.words) == 1 and b.held() == 1
