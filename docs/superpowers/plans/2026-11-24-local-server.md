# Local Server and Inspector (first part) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** One local process that serves Inkling 2 Mini over the standard chat API (with thinking and tool calls in standard form), records what the loop did for every request, and shows any request in a local inspector page.

**Architecture:** The bundled `mlx_lm.server` is wrapped from outside, never edited: a `ModelProvider` subclass installs capture and the Inkling reply format at load; a replacement for the module's `stream_generate` clears capture, saves a cache checkpoint when the prompt has been read, and hands the finished request to a recorder; an `APIHandler` subclass adds our routes and tags each request. Everything the spike in `server/step0/launch.py` proved is moved into tested modules.

**Tech Stack:** Python 3.12 in `spike/.venv`, mlx-lm pinned at `e2b5bbd3`, `latent_capture` (in `spike/`), `mapbuild` (in `basemap/`), `livemap` (in `prototype/`), SQLite from the standard library, one static HTML page. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-11-24-local-server-and-inspector-design.md` (read "Step 0 results" first).

**Covers** build-order steps 2 and 3, plus tool calling (step 6, shown cheap by step 0). Readout on demand, conversation history in the chat pane, the run summary and a "responses" route for Codex are a second plan.

## Global Constraints

- Listens on `127.0.0.1` only. Starting with any other host is refused with an error.
- No prompt, reply or state leaves the Mac. `HF_HUB_OFFLINE=1` is set before `mlx_lm` is imported. Allowed origin is the server's own `http://127.0.0.1:<port>` only.
- No file under `spike/.venv` is edited.
- Default port `8765`. Default retention 5 days. Text kept by default; `--no-text` stores numbers only.
- Batching is off (`is_batchable = False`): capture reads one request at a time.
- Prompt cache limit 2GB by default.
- Request text never appears in the terminal log.
- Wording in the page follows the spec's table "What the inspector may claim"; every view carries its status line.
- Tests that need the model are marked `@pytest.mark.model` and are not part of the default run.
- Run everything from `server/` with `PYTHONPATH=../spike:../basemap:../prototype` and `../spike/.venv/bin/python`.
- Commits are local only and end with `Co-Authored-By: Claude Fable 5.2 <noreply@anthropic.com>`.

## Review Focus

1. **Client disconnects mid-reply** (coding tools cancel often): the partial reply is still recorded, marked `finish = "cancelled"`, and capture is left clean for the next request. Test in Task 4.
2. **Cache leaves nothing or one token to read:** alignment must hold with `uncached` of 0 and 1. Test in Task 1.
3. **`--no-text`:** no prompt or reply text reaches the database file in any column, including tool-call arguments. Test in Task 3 by searching the raw file bytes.
4. **Text-completion requests (`/v1/completions`) and requests with no user message:** recorded without error, session falls back to the completion id. Test in Task 3.
5. **Retention:** a record older than the limit is gone after purge, including its state blob, and "forget" removes every request of the session. Test in Task 3.

---

## File structure

```
server/
  lmserver/
    __init__.py
    inkling.py     reply format and tool-call parser for Inkling
    align.py       which forward calls belong to generated words
    recorder.py    SQLite store: save, list, get, forget, purge
    hooks.py       everything attached to mlx_lm.server
    app.py         our routes, argument checks, main()
  static/index.html
  tests/
  pytest.ini
  run.sh
  README.md
  step0/           the spike, kept for reference
```

### Task 1: Alignment rule

**Files:** Create `server/lmserver/align.py`, `server/tests/test_align.py`, `server/pytest.ini`, `server/lmserver/__init__.py`

**Interfaces:**
- Produces: `first_generated(call_sizes: list[int], uncached: int) -> int` — index of the first forward call that read a generated word. Forward calls first use up `uncached` prompt tokens; every later call read one generated word.

- [ ] **Step 1: Write the failing tests**

```python
from lmserver.align import first_generated

def test_plain_prompt():            assert first_generated([23, 1, 1, 1], 24) == 2
def test_long_prompt_in_chunks():   assert first_generated([2048, 2048, 2048, 1125, 1, 1, 1], 7270) == 5
def test_one_uncached_token():      assert first_generated([1, 1, 1], 1) == 1
def test_nothing_uncached():        assert first_generated([1, 1], 0) == 0
def test_fourteen_uncached():       assert first_generated([13, 1, 1, 1], 14) == 2
def test_no_calls():                assert first_generated([], 5) == 0
```

- [ ] **Step 2:** Run `pytest tests/test_align.py -q`. Expected: FAIL, module not found.
- [ ] **Step 3:** Implement `first_generated`. `pytest.ini` registers the `model` marker and sets `addopts = -m "not model"`.
- [ ] **Step 4:** Run again. Expected: 6 passed.
- [ ] **Step 5:** Commit.

### Task 2: Inkling reply format and tool parser

**Files:** Create `server/lmserver/inkling.py`, `server/tests/test_inkling.py`

**Interfaces:**
- Produces: `RESPONSE_FORMAT: dict`, `TOOL_CALL_START = "<|content_invoke_tool_json|>"`, `TOOL_CALL_END = "<|end_message|>"`, `parse_tool_call(text: str, tools=None) -> dict` returning `{"name": str, "arguments": dict}`, and `configure(tokenizer) -> None` which sets `_response_format`, `_tool_parser`, `_tool_call_start`, `_tool_call_end`, `_tool_call_start_tokens` on an mlx-lm `TokenizerWrapper` (exactly as `server/step0/launch.py` does).

The format, copied from the spike:

```python
RESPONSE_FORMAT = {
    "initial_state": "header",
    "transitions": {
        "header": [("<|content_thinking|>", "reasoning"), ("<|content_text|>", "normal"),
                   ("<|content_invoke_tool_json|>", "tool")],
        "reasoning": [("<|end_message|>", "header")],
        "normal": [("<|end_message|>", "header")],
    },
}
```

- [ ] **Step 1: Write the failing tests.** No model needed: build `mlx_lm.generate.TextStateMachine` from `RESPONSE_FORMAT["transitions"]` plus `"tool": [(TOOL_CALL_END, None)]`, read from state `"header"`.

```python
THINK_ANSWER = "<|message_model|><|content_thinking|>Plan.<|end_message|><|message_model|><|content_text|>Hi.<|end_message|>"
TOOL = ('<|message_model|><|content_thinking|>Read it.<|end_message|><|message_model|>read_file'
        '<|content_invoke_tool_json|>{"name":"read_file","args":{"path":"a.toml"}}<|end_message|>')

def test_thinking_and_answer_are_split():   # content == "Hi.", reasoning == "Plan.", no tool calls
def test_tool_call_is_extracted():          # content == "", reasoning == "Read it.", one tool call text
def test_markers_never_reach_content():     # "<|" not in content or reasoning, fed one character at a time
def test_parse_tool_call():                 # {"name": "read_file", "arguments": {"path": "a.toml"}}
def test_parse_tool_call_without_args():    # '{"name":"ls"}' -> arguments == {}
def test_answer_only_reply():               # "<|message_model|><|content_text|>Hi.<|end_message|>" -> content "Hi."
```

- [ ] **Step 2:** Run. Expected: FAIL, module not found.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Run. Expected: 6 passed.
- [ ] **Step 5:** Commit.

### Task 3: Recorder

**Files:** Create `server/lmserver/recorder.py`, `server/tests/test_recorder.py`

**Interfaces:**
- Produces:

```python
@dataclass
class RequestRecord:
    completion_id: str            # "chatcmpl-…" from the response
    time: float                   # unix seconds
    client: str                   # user-agent, "" if none
    messages: list | None         # request messages, None for text completions
    reply_tokens: list[str]       # one string per generated token
    words: list[dict]             # per generated token: {"phase", "cells", "area", "move", "level", "settled_by", "loop"}
    mean_states: np.ndarray | None  # float32 (phase 2: thinking, answer; block 2; pass 3; hidden 2304), NaN where a phase is empty
    settings: dict                # max_tokens, temperature, reasoning effort, tools declared (names only)
    prompt_tokens: int; cached_tokens: int; first_token_s: float; tokens_per_s: float
    finish: str                   # "stop", "length", "tool_calls", "cancelled", "error"

class Recorder:
    def __init__(self, path: str | Path, keep_text: bool = True, retention_days: float = 5): ...
    def save(self, rec: RequestRecord) -> None
    def sessions(self) -> list[dict]        # newest first: {"session", "client", "first_time", "last_time", "requests", "title"}
    def requests(self, session: str) -> list[dict]   # {"completion_id", "time", "tokens", "finish"}
    def get(self, completion_id: str) -> dict | None # everything except mean_states; "text_kept": bool
    def mean_states(self, completion_id: str) -> np.ndarray | None
    def forget(self, session: str) -> int   # rows deleted
    def purge(self, now: float | None = None) -> int
```

- Session rule: `session = sha1(client + "\n" + first system message text + "\n" + first user message text)[:16]`; with no messages, the completion id. `title` is the first 60 characters of the first user message, or `"(text not kept)"`.
- One SQLite file, `journal_mode=WAL`, opened with `check_same_thread=False` behind a lock. States stored as a blob with `np.save`. With `keep_text=False`: `messages` and `reply_tokens` are not stored, tool names in `settings` are not stored, and the session id is still computed from the text before it is dropped.
- `mean_states` shape `(2, 2, 3, 2304)` is about 110KB per request; the spec's 55KB counted one phase.

- [ ] **Step 1: Write the failing tests** (`tmp_path` database, a `make_record(**over)` helper):

```python
def test_save_and_get_round_trip()
def test_same_conversation_groups_into_one_session()      # second request extends the first's messages
def test_different_first_message_is_a_new_session()
def test_text_completion_without_messages_is_recorded()   # messages=None -> session == completion_id
def test_no_text_mode_leaves_no_text_in_the_file()        # write "zebra-quartz" in prompt, reply and a tool name; close; bytes of every file in tmp_path lack it
def test_no_text_mode_reports_text_not_kept()             # get()["text_kept"] is False, words still present
def test_mean_states_round_trip_with_nan()
def test_forget_removes_every_request_of_the_session()
def test_purge_removes_only_old_records()                 # times 6 and 4 days before `now`, retention 5 -> one left, its states gone too
def test_sessions_are_newest_first()
```

- [ ] **Step 2:** Run. Expected: FAIL, module not found.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Run. Expected: 10 passed.
- [ ] **Step 5:** Commit.

### Task 4: Hooks on the bundled server

**Files:** Create `server/lmserver/hooks.py`, `server/tests/test_hooks.py`, `server/tests/test_server_model.py`

**Interfaces:**
- Consumes: `first_generated`, `inkling.configure`, `Recorder`, `RequestRecord`, `latent_capture.capture.capturing`, `livemap.live.LiveMap`, `livemap.live.PhaseTracker`.
- Produces:

```python
def install(recorder: Recorder, live: LiveMap, on_word: Callable[[dict], None] | None = None) -> None
    # Replaces mlx_lm.server.ModelProvider, .stream_generate and ResponseGenerator._tokenize. Idempotent: a second call raises.
def build_record(meta: dict, prompt: list[int], uncached: int, tokens: list[int], records: list[StepCapture],
                 tokenizer, live: LiveMap, timing: dict, finish: str) -> RequestRecord   # pure; no model needed
class Handler(mlx_lm.server.APIHandler)   # see below
```

Behaviour, each point taken from the spike unless marked new:
- Provider `_load`: after `super()._load`, set `is_batchable = False`, install `capturing(model, max_records=20000)`, call `inkling.configure(tokenizer)`.
- `_tokenize` wrapper stashes the full prompt tokens and the response generator.
- `stream_generate` wrapper: clear capture; wrap `prompt_progress_callback` so that at `done == total - 1` (and `total > 1`) a deep copy of the cache is inserted under `prompt[:-1]` with `cache_type="user"`; yield tokens; in `finally`, call `build_record` and `recorder.save`. `finish` is `"cancelled"` when the generator is closed early (catch `GeneratorExit`, record, re-raise), `"error"` on any other exception. A failure inside recording is logged without text and never breaks the reply.
- `build_record`: generated records are `records[first_generated(sizes, uncached):]`, paired in order with `tokens` (the last token may have no record). Phase per token from `PhaseTracker`; per-word fields from `live.describe(block2, block3)` with `block2 = np.stack(rec.block2)`, `block3 = np.stack(rec.block3)`; marker tokens get `phase = "marker"` and are left out of the means.
- **New:** `Handler.handle_chat_completions` calls `super()`, then sets `request.lm_meta = {"completion_id": self.request_id, "client": self.headers.get("User-Agent", ""), "settings": {...}}` and, when the body has a top-level `reasoning_effort` and no `chat_template_kwargs.reasoning_effort`, copies it into `self.body["chat_template_kwargs"]` before calling `super()`. Confirm on first run that the server reads `chat_template_kwargs` after this point (`server.py:663-670`); if it is read earlier, move the copy into `do_POST` before delegation and note it in the commit.
- **New:** `Handler.log_message` is silenced, as in the prototype.
- **New:** `on_word`, when given, is called with each word's dict as soon as its record exists (used by Task 5's live stream). Call it from the generation loop, not only at the end.

- [ ] **Step 1: Write the failing tests for `build_record`** with fake `StepCapture` objects (random arrays, `n_tokens` set) and a stub tokenizer and a real `LiveMap.load` from `basemap/bundle_v1` and `prototype/norms_v1.json`:

```python
def test_words_pair_with_generated_tokens_after_uncached_prompt()   # sizes [3,1,1,1,1], uncached 4, 3 tokens -> 3 words
def test_last_token_without_a_record_is_kept_without_state()        # 3 tokens, 2 generated records -> words[2]["cells"] is None
def test_phases_follow_the_markers()                                # thinking, marker, answer
def test_mean_states_exclude_markers_and_are_nan_for_an_empty_phase()
def test_cancelled_finish_is_carried()
```

- [ ] **Step 2:** Run. Expected: FAIL.
- [ ] **Step 3:** Implement `hooks.py`.
- [ ] **Step 4:** Run. Expected: 5 passed.
- [ ] **Step 5: Model test** (`@pytest.mark.model`, `tests/test_server_model.py`): start the server on a free port in a thread with a `tmp_path` database, then
  - one non-streamed chat request: `content` has no `<|`, `reasoning` is non-empty, a record with that completion id exists, `len(words) >= completion_tokens - 1`;
  - alignment: for that record, cosine of each captured block 2 pass 3 state with a plain pass over prompt plus reply is highest at its own position for at least 95% of words (port `verify` from the spike);
  - the same conversation extended by one turn reports `cached_tokens > 0`;
  - a request with a `read_file` tool returns `finish_reason == "tool_calls"` and parsed arguments;
  - a streamed request abandoned after 5 chunks leaves a record with `finish == "cancelled"`, and the next request still succeeds and aligns.

  Run: `pytest -m model tests/test_server_model.py -q`. Expected: 5 passed. Stop any other process holding the model first.
- [ ] **Step 6:** Commit.

### Task 5: App — routes, safety checks, start script

**Files:** Create `server/lmserver/app.py`, `server/run.sh`, `server/tests/test_app.py`

**Interfaces:**
- Consumes: `hooks.install`, `hooks.Handler`, `Recorder`.
- Produces: `main(argv=None)`; `check_args(args) -> None` (raises `SystemExit` with a message); `route(recorder, live_payload, method, path) -> tuple[int, str, bytes] | None` (pure routing for our paths; `None` means "not ours, pass to the bundled handler").

Routes (all JSON unless noted):

| Method and path | Returns |
| --- | --- |
| `GET /` | `static/index.html` |
| `GET /api/map` | the map payload the prototype serves today (`prototype/server.py::map_payload`, moved here) |
| `GET /api/sessions` | `recorder.sessions()` |
| `GET /api/sessions/<id>` | `recorder.requests(id)` |
| `GET /api/requests/<completion_id>` | `recorder.get(...)`, 404 if absent |
| `POST /api/sessions/<id>/forget` | `{"deleted": n}` |
| `GET /api/live` | ndjson stream: one line per word of the request being generated, then `{"done": completion_id}`; idle keep-alive line every 15 s |
| `GET /api/settings` | `{"keep_text", "retention_days", "map_tag"}` |

`main`: sets `HF_HUB_OFFLINE=1` before importing `mlx_lm`; arguments `--port` (8765), `--db` (`~/.latent-map/requests.db`, directory created with mode 700), `--no-text`, `--retention-days` (5), `--map-tag` (v1), `--cache-bytes` (2GB); builds the bundled server's argument namespace with `host="127.0.0.1"`, `allowed_origins=[f"http://127.0.0.1:{port}"]`, `max_tokens=2048`; purges at start and every hour; starts the prototype's keep-warm heartbeat with its back-off (moved from `prototype/server.py`), paused while a request is running.

- [ ] **Step 1: Write the failing tests:**

```python
def test_check_args_refuses_a_non_local_host()            # host "0.0.0.0" -> SystemExit mentioning 127.0.0.1
def test_route_sessions_and_request_round_trip()          # recorder with two records in tmp_path
def test_route_unknown_request_is_404()
def test_route_forget_deletes_and_reports_count()
def test_route_ignores_api_paths_of_the_bundled_server()  # "/v1/chat/completions" -> None
def test_route_rejects_path_traversal_for_static()        # "/../lmserver/app.py" -> 404
def test_settings_report_text_mode()
```

- [ ] **Step 2:** Run. Expected: FAIL.
- [ ] **Step 3:** Implement. `run.sh`: `cd "$(dirname "$0")" && PYTHONPATH=../spike:../basemap:../prototype exec ../spike/.venv/bin/python -m lmserver.app "$@"`.
- [ ] **Step 4:** Run. Expected: 7 passed. Then the whole default suite: `pytest -q`. Expected: 34 passed.
- [ ] **Step 5:** Commit.

### Task 6: The page

**Files:** Create `server/static/index.html`, `server/README.md`. Modify `prototype/README.md` (one line: superseded by `server/`).

**Interfaces:** Consumes the routes of Task 5 and `POST /v1/chat/completions` (streamed).

One page, three columns, adapted from `prototype/static/index.html` (keep its map drawing and word strip code):

```
┌ Sessions ──────────┬ Request ───────────────────────────┬ Map ─────────────┐
│ ● chat  17:02  3   │ Thinking (collapsed, click to open)│  [map of 256     │
│   codex 16:40  12  │ Answer words, coloured by area     │   cells, the     │
│   curl  16:31  1   │ ▂▃▅ three passes under each word   │   selected word's│
│ [Forget session]   │ Tool call: read_file {path: …}     │   three passes]  │
│                    │ ───────────────────────────────    │                  │
│ Keeping text for   │ [ message box            ] [Send]  │ status lines     │
│ 5 days on this Mac │ effort: [low ▾]                    │                  │
└────────────────────┴────────────────────────────────────┴──────────────────┘
```

- Sending a message posts to `/v1/chat/completions` with `stream: true` and the effort setting, shows reasoning and content as they arrive, listens to `/api/live` for the words' map positions, and on `done` loads the stored record so the live view and the stored view are the same code path.
- The chat pane sends only the current message in this plan (conversation history is the next plan); say so under the box.
- Selecting a session lists its requests; selecting a request shows it. Requests from other clients appear without reloading (poll `/api/sessions` every 5 s).
- Status lines, verbatim from the spec: under the map "Position on a map built from 120 texts"; under last-pass movement "Measured; not known to predict anything", in neutral colours.
- A record stored without text shows the word strip as blank boxes and the line "Text was not kept for this request."
- No external scripts, fonts or requests of any kind.

- [ ] **Step 1:** Build the page.
- [ ] **Step 2: Check it by hand with the server running** (`./run.sh`): send a chat message and see words and map fill in; reload and open the same request from the session list and see the same view; run `server/step0/client2.py` against port 8765 and see a second session with a tool call; press Forget and see it go; restart with `--no-text` and confirm the blank-box view. Record what was checked in the commit message.
- [ ] **Step 3:** In the browser's network panel confirm every request goes to `127.0.0.1:8765`.
- [ ] **Step 4:** `README.md`: how to start, the endpoint URL to give a chat app or coding tool (`http://127.0.0.1:8765/v1`, model name `default_model`, any API key), what is stored and where, how to turn text off, how to delete everything (remove `~/.latent-map`).
- [ ] **Step 5:** Commit.

### Task 7: A real client

**Files:** Modify `server/README.md`.

Step 0's fourth check is still open. Codex, the one suitable tool installed, needs a "responses" route the bundled server does not have, so it waits for the second plan unless the user chooses otherwise.

- [ ] **Step 1:** With the user's go-ahead for the install, point one chat-API coding tool at the server and run the read-only task from the spike ("What version is declared in pyproject.toml?") in a scratch folder. If no install is approved, run the three-turn tool loop in `server/step0/client2.py` extended to send the tool result back, and state in the README that no third-party tool has been tested.
- [ ] **Step 2:** Record in `server/README.md` what worked, time per step, and anything the tool sent that the server refused.
- [ ] **Step 3:** Commit.
