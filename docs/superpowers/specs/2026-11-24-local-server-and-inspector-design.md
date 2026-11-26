# Local model server with an inspector — design

Status: approved by the user on 2026-11-24, with the decisions recorded at the end. As of 2026-11-26, all seven build steps are done (see "Build order"). How the built server behaves is described in `server/README.md`, which is the current reference where it differs from this design.

## Why

So far Latent Map is a prototype page that talks to its own small server. To be useful day to day, the model has to be usable from the places people already work: a chat window, and coding tools that call a model over an API. And the part that is specific to this project, seeing what the model did inside its loop, matters most where nobody reads every step: agent runs.

So the product becomes: **a local server that runs Inkling 2 Mini for any client, records what the loop did for every request, and lets you open any request afterwards.** A flight recorder for a local model.

## Goals

1. **One process owns the model** and serves every client. Only one copy fits in 32GB.
2. **A standard API on localhost,** so chat apps and coding tools that accept a custom endpoint work by changing a URL.
3. **Every request is recorded locally** with compact per-pass states, whoever sent it.
4. **An inspector** that lists sessions and opens any request, showing only what has been tested and saying how far.
5. **A chat window of our own,** minimal, with the inspector beside it.

## Not goals

- The native Mac app. This server is what it would later wrap.
- Contribution to the shared map. The recorder holds what a contribution would be computed from, but nothing is sent.
- A general model server. One model, one machine.
- Competing with existing chat apps on features.

## It has failed if

- Anything listens beyond 127.0.0.1, or any prompt, reply or state leaves the Mac.
- A coding tool pointed at it gets slower or less reliable answers than from the plain MLX server.
- The inspector shows a signal as more than the tests support.
- The recorder fills the disk or keeps prompts the user believes are gone.

## What the inspector may claim, from the tests so far

| View | Status | Shown as |
| --- | --- | --- |
| Thinking and answer, separated | Reliable | The model's own text |
| Where each word lands on the map, and settling across passes | Measured; meaning modest | "Position on a map built from 120 texts" |
| Movement on the last pass against usual | Measured; not known to predict anything | Shown without alarm colours |
| What the model would say after each pass | Works; last pass equals the real prediction | "What it would say next after each pass", never "its reasoning" |
| Thinking text that mentions an instruction the user did not give | Found in 119 of 120 planted-goal replies with thinking on | A highlight on the thinking text, for the user to read |
| "The model is weighing a refusal" | Real for blatant cases; adds nothing over the text | Not shown |
| A goal with no trace in the text | Open: not found, not ruled out | Not shown |
| Steering dials | Not working | Not offered |

Every view carries its status line. New signals are added only after a test with criteria fixed in advance and an independent review.

## Approaches considered

**A. Extend the MLX server that ships with the model runner (recommended, pending a check).** The pinned mlx-lm already contains a 2,000-line server that speaks the standard chat API, with streaming, a prompt cache that reuses work between requests, tool-call handling and a thinking channel. Our capture is installed on the model object itself, so it should keep working whoever calls the model. We would add a recorder and our own routes around it. Least new code, and the hard parts (caching, streaming, tool-call parsing) are someone else's tested work. Risks: it batches requests, and our capture reads one request at a time; it has no tool-call parser written for Inkling; and it is an unmerged fork we already pin.

**B. Grow the prototype server into the API.** Full control and already working for one request at a time. But we would write prompt caching, the streaming protocol and tool calling ourselves, and those are where coding tools break.

**C. Sit in front of another runner (Ollama).** Not possible: other runners do not expose the internal states.

A is the recommendation if the feasibility step passes; B is the fallback, with the API limited to chat without tools at first.

## Step 0: feasibility (before the design is fixed)

1. Start the bundled server with our capture installed and confirm one streamed chat reply comes back with states recorded and aligned to its words. The alignment rule in `spike/latent_capture/gen.py` (skip the prompt's last token) must be re-checked under the server's prompt cache.
2. Confirm what happens with two requests at once. If the server batches them, either turn batching off or make capture batch-aware.
3. Send a request with a tool definition and see whether the model's tool call comes back in the standard form. If not, find out how much parser is needed.
4. Point one real coding tool at it and run a small task.

If 1 fails, fall back to approach B. If 3 fails, the first version is chat-only and tools become their own piece of work.

### Step 0 results (2026-11-24, spike in `server/step0/`)

Approach A is confirmed. The spike wraps the bundled server from outside; no file of the model runner is edited.

| Check | Result |
| --- | --- |
| 1. Capture under the bundled server | **Pass.** States recorded for every reply at about 17 words per second. Compared with a plain pass over the same text, the captured state matched its own word best in 176 of 177, 150 of 150, 61 of 61 and similar. |
| 1b. Alignment under the prompt cache | **Pass, with a new rule.** "Skip the first one-token call" is wrong when the cache leaves one token to read. The rule that holds: forward calls first use up the uncached prompt tokens; each call after that read one generated word. Checked with 1 and 14 uncached tokens. |
| 2. Two requests at once | **Pass.** With batching switched off they are served one after the other, each correct and aligned. The second waits. |
| 3. Tool calls | **Pass after a small adapter.** Out of the box the server knows neither this model's thinking markers nor its tool-call format, and both leak into the reply. A 25-line reply-format description plus a 3-line parser fixes it: thinking arrives in the reasoning field, the answer is clean, and tool calls come back in the standard form, streamed and not. A tool result sent back in the standard form is read correctly. |
| 4. A real coding tool | **Not done.** Codex is the only suitable tool installed, and it no longer speaks the chat API; it needs the newer "responses" route, which the bundled server lacks. |

Found on the way, all now part of the design:

- **The prompt cache never hit.** This model's cache cannot be shortened, and the server only stores the cache as it stands after the reply, which a later prompt never extends (the thinking is not sent back). Saving a copy at the moment the prompt has been read fixes continuing conversations: a 7,283-token follow-up read 14 new tokens and took 6.5 s against 21 s. A new conversation with the same long system prompt still re-reads it (about 500 tokens per second); a second saved copy at the end of the system prompt would fix that.
- **Saved caches are large** (about 0.4GB per 7,000 tokens), so the cache gets a size limit (2GB in the spike).
- **The server contacts huggingface.co when it loads the model**, to check for a newer version. Only the model's name is sent. Running with the offline setting stops it, and the real server will.
- **The server allows web pages from any origin by default.** Ours will allow only its own page.
- **The effort setting** reaches the model only when sent as a template argument; the standard `reasoning_effort` field needs mapping to it.

### Server

- Listens on 127.0.0.1 only, one port.
- **Standard routes:** chat completions with streaming, and the model list. Thinking is returned in the separate reasoning field; the request's reasoning-effort setting maps to the model's.
- **Our routes:** the session list, one request's record, the map bundle, and a readout-on-demand call (below).
- **Capture is always on.** Measured cost is about 6 to 10% of speed.
- **Keep-warm** and the memory-pressure back-off carry over from the prototype.

### Recorder

For each request, written when the reply ends:

- time, client label (from the API key or user-agent the client sends), settings, token counts, speed, wait before the first word;
- the prompt and reply text;
- per generated word: phase (thinking or answer), map cell on each pass for block 2, last-pass movement, loop signal. A few bytes per word;
- per request: the reply's mean state for block 2 and block 3 on each pass, **at full size** (about 55KB). The hidden-goal test could not be re-analysed because only reduced states were kept; this fixes that for the future.

Stored in a single local database file. **Retention is a setting** (default 5 days) with "keep text" able to be switched off, in which case only the numbers are stored and the inspector says the text was not kept. A "forget this session" action deletes its rows.

### Readout on demand

The per-pass readout needs an extra pass over the text, so it is not computed live. Opening a request in the inspector runs it then (a few seconds), and the result is cached with the record. It is unavailable for records stored without text.

### Inspector

A local web page served by the same process.

- **Sessions:** a list of conversations and agent runs by client and time, with size and any highlights.
- **Request:** the prototype's view (words, map, three passes per word), plus the per-pass readout strip, plus the thinking text with instruction-mentions highlighted.
- **Run summary** for an agent session: requests over time, thinking share, and the list of highlights to read. No scores, no verdicts.

### Chat

The prototype's left pane, extended to hold a conversation with history, sharing the inspector's request view. Deliberately plain.

## Constraints

- 32GB Mac, one model copy, about 16 words per second. That speed is fine for chat and slow for agent loops; the design cannot change it.
- The model runner is a pinned commit of an unmerged fork.
- Everything local; no accounts.

## Risks

| Risk | Why it matters | How it is settled |
| --- | --- | --- |
| Capture misaligns under the server's prompt cache | Every per-word view would be wrong | Step 0 check 1, with the existing alignment test adapted |
| Batched requests break capture | Agent tools send overlapping requests | Step 0 check 2 |
| Tool calls are not parsed for this model | Coding tools need them | Step 0 check 3; chat-only first version if needed |
| A coding tool's long prompt overflows memory | The model claims a very long context; 32GB does not | Set and document a context limit after measuring |
| The log quietly becomes a store of sensitive text | It contains everything the user sent | Retention setting, text-off mode, forget action, shown in the inspector |
| Users read the map or readout as the model's reasoning | Three reviews have shown how easy that is | Status line on every view; wording fixed in this document |

## Build order

1. Step 0 feasibility.
2. Server with standard chat route, capture and recorder; the prototype page pointed at it.
3. Inspector: sessions list and request view from the log.
4. Readout on demand.
5. Conversation history in the chat pane.
6. Tool calling, if step 0 showed it needs work.
7. Run summary for agent sessions.

Each step ends with something usable, and step 2 already lets a chat app or coding tool use the model.

### Progress (2026-11-26)

| Step | State |
| --- | --- |
| 1. Feasibility | Done. Approach A confirmed. |
| 2. Server, capture, recorder | Done. |
| 3. Inspector | Done. |
| 4. Readout on demand | Done, including long requests. The readout re-reads the text in the same pieces generation read it in, because the model's output depends slightly on piece size. It takes about 3.5 s per 1,000 prompt tokens, more than the "few seconds" assumed above. |
| 5. Conversation history in the chat pane | Done. Held in the page only; an old conversation cannot be continued after a reload. |
| 6. Tool calling | Done with the small adapter from step 0. Tested with opencode 1.19.6, which settles step 0 check 4. Codex still needs a "responses" route, not built. |
| 7. Run summary | Done. The instruction-mention pattern was widened after seeing the planted-goal texts, so its 119 of 120 is a fitted figure. |

## Left to the builder

- Database layout, route names, page framework.
- How the client label is derived.
- The exact context limit, once measured.

## Decisions (user, 2026-11-24)

- Direction approved.
- Default retention is 5 days ("no need to keep that long, 5 days is enough"). Text is kept by default within that window; the text-off setting stays.
- Tool calling "would be nice": wanted in the first version if step 0 shows it works or needs only a small parser. If it needs substantial work, chat ships first and tools follow as build step 6.
