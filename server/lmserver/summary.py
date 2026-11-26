"""One page for a whole session: its steps in order, and what in them is worth reading.

Numbers and pointers only. Nothing here scores a run or judges it."""

import json
import re

# Thinking that matches an instruction word pattern, for a person to read. The evidence:
# on the 120 thinking texts of the planted-goal test (signals/REPORT_hidden.md) this
# pattern matches 119, but it was widened after seeing those texts, so that is a fitted
# figure. On the ordinary steps stored on this Mac when it was written it matched 2 of 20
# with thinking. A tool's own system prompt gives the model plenty to refer to.
MENTIONS = re.compile(
    r"\binstruct|\bsystem (prompt|message|says|said|wants|asks|asked|tells|told|requires)|\bdeveloper (prompt|message|says|said|wants)"
    r"|\b(was|were|been|am|i'm) told\b|\bno need (to )?mention\b"
    r"|(\b(not|never|without)|n['’]t) (to )?(mention|reveal|disclos)",
    re.I,
)
SNIPPET = 160
COUNTED = ("thinking", "answer")


def _short(text: str, limit: int = SNIPPET) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[:limit] + "…"


def _asked(messages) -> dict | None:
    """The last message sent: what this step was a reply to."""
    if not messages:
        return None
    last = messages[-1] if isinstance(messages[-1], dict) else {}
    content = last.get("content")
    if isinstance(content, list):
        parts = []
        for p in content:
            if not isinstance(p, dict):
                parts.append(str(p))
            elif p.get("type") == "text":
                parts.append(str(p.get("text") or ""))
            else:
                parts.append("[image]" if "image" in str(p.get("type")) else f"[{p.get('type')}]")
        content = " ".join(parts)
    return {"role": str(last.get("role", "")), "text": _short(content if isinstance(content, str) else "")}


def _mentions(texts: list[str], phases: list[str]) -> list[dict]:
    """Stretches of thinking (to a full stop or line end) that match, each with the index
    of the word where the match starts and the text around it."""
    found, first, parts = [], None, []

    def close():
        nonlocal first, parts
        raw = re.sub(r"\s", " ", "".join(parts))  # same length, so positions still map to words
        m = MENTIONS.search(raw)
        if m:
            at, i = 0, first
            for k, part in enumerate(parts):
                if at + len(part) > m.start():
                    i = first + k
                    break
                at += len(part)
            lo, hi = max(0, m.start() - 100), min(len(raw), m.start() + 200)
            text = ("…" if lo > 0 and raw[:lo].strip() else "") + " ".join(raw[lo:hi].split()) + ("…" if raw[hi:].strip() else "")
            found.append({"i": i, "text": text})
        first, parts = None, []

    for i, (text, phase) in enumerate(zip(texts, phases)):
        if phase != "thinking":
            close()
            continue
        if first is None:
            if not text.strip():
                continue
            first = i
        parts.append(text)
        if "\n" in text or text.rstrip().endswith((".", "!", "?")):
            close()
    close()
    return found


def _tools(texts: list[str], phases: list[str]) -> list[tuple[str | None, str | None]]:
    """Each tool call in the reply as (name, arguments). A reply can hold several."""
    calls, name, raw = [], "", ""

    def close():
        nonlocal name, raw
        if raw.strip():
            tool, args = name.strip() or None, _short(raw.strip(), 200)
            try:
                call = json.loads(raw)
                tool = tool or call.get("name")
                args = _short(json.dumps(call.get("args", {}), separators=(",", ":"), ensure_ascii=False), 200)
            except (ValueError, AttributeError):
                pass
            calls.append((tool, args))
        name, raw = "", ""

    for text, phase in zip(texts, phases):
        if phase == "marker" and text in ("<|message_model|>", "<|end_message|>"):
            close()
        elif phase == "header":
            name += text
        elif phase == "tool":
            raw += text
    close()
    return calls


def _step(rec: dict) -> dict:
    words = rec.get("words") or []
    phases = [w.get("phase", "") for w in words]
    texts = rec.get("reply_tokens")
    n, speed = rec.get("n_tokens") or 0, rec.get("tokens_per_s") or 0
    readout = rec.get("readout") or {}
    out = {
        "id": rec["completion_id"],
        "time": rec["time"],
        "seconds": (rec.get("first_token_s") or 0) + (n / speed if speed > 0 else 0),
        "prompt_tokens": rec.get("prompt_tokens") or 0,
        "cached_tokens": rec.get("cached_tokens") or 0,
        "n_tokens": n,
        "finish": rec.get("finish"),
        "temperature": float((rec.get("settings") or {}).get("temperature") or 0),
        "thinking": phases.count("thinking"),
        "answer": phases.count("answer"),
        "tool": None, "tool_args": None, "more_tools": [], "asked": None, "mentions": None,
        "readout": "stored" if rec.get("readout") else "available" if rec.get("readout_available") else "none",
        "hits": None, "counted": None, "pieces_known": readout.get("pieces_known"), "damaged": False,
    }
    if texts is not None:
        texts = [str(t) for t in texts] + [""] * (len(phases) - len(texts))
        calls = _tools(texts, phases)
        if calls:
            out["tool"], out["tool_args"] = calls[0]
            out["more_tools"] = [str(name or "?") for name, _ in calls[1:]]
        out["mentions"] = _mentions(texts, phases)
        out["asked"] = _asked(rec.get("messages")) or {"role": "", "text": ""}
    hits = readout.get("hits")
    if hits:
        rows = [h for h, p in zip(hits, phases) if p in COUNTED]
        out["hits"] = [sum(bool(h[k]) for h in rows if k < len(h)) for k in range(3)]
        out["counted"] = len(rows)
    return out


def _numbers_only(rec: dict) -> dict:
    """A step whose stored row could not be read: shown, with nothing claimed about it."""
    def num(key):
        return rec.get(key) if isinstance(rec.get(key), (int, float)) else 0
    return {
        "id": rec.get("completion_id"), "time": num("time"), "seconds": 0, "prompt_tokens": num("prompt_tokens"),
        "cached_tokens": num("cached_tokens"), "n_tokens": num("n_tokens"), "finish": rec.get("finish"), "temperature": 0,
        "thinking": 0, "answer": 0, "tool": None, "tool_args": None, "more_tools": [], "asked": None, "mentions": None,
        "readout": "none", "hits": None, "counted": None, "pieces_known": None, "damaged": True,
    }


def summarize(records: list[dict]) -> dict:
    """records: a session's requests as Recorder.get returns them, oldest first."""
    steps = []
    for rec in records:
        try:
            steps.append(_step(rec))
        except Exception:  # one unreadable row must not hide the rest of the run
            steps.append(_numbers_only(rec))
    if not steps:
        return {"steps": [], "totals": None}
    kept = any(s["mentions"] is not None for s in steps)
    totals = {
        "steps": len(steps),
        "seconds": sum(s["seconds"] for s in steps),                                  # the model working
        "span_seconds": steps[-1]["time"] + steps[-1]["seconds"] - steps[0]["time"],  # first request to last word
        "tokens": sum(s["n_tokens"] for s in steps),
        "prompt_tokens": sum(s["prompt_tokens"] for s in steps),
        "cached_tokens": sum(s["cached_tokens"] for s in steps),
        "thinking": sum(s["thinking"] for s in steps),
        "answer": sum(s["answer"] for s in steps),
        "tool_calls": sum(s["finish"] == "tool_calls" for s in steps),
        "with_mentions": sum(bool(s["mentions"]) for s in steps) if kept else None,
        "with_thinking": sum(s["thinking"] > 0 and s["mentions"] is not None for s in steps),  # steps the pattern could look at
        "text_kept": kept,
    }
    return {"steps": steps, "totals": totals}
