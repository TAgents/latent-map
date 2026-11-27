# Latent Map server

Runs Inkling 2 Mini on this Mac for any app that speaks the standard chat API, records
what the model's loop did for every request, and shows any request in a local page.

## Start

    ./run.sh

Loading the model takes about a minute. Then:

- **Inspector and chat:** open http://127.0.0.1:8765
- **For a chat app or coding tool:** base URL `http://127.0.0.1:8765/v1`, model `default_model`,
  any text as the API key. Thinking arrives in the `reasoning` field, tool calls in the standard form.
  `reasoning_effort` accepts `none`, `minimal`, `low`, `medium`, `high`.

Only one copy of the model fits in 32GB, so stop the old prototype (`prototype/`) first.
Requests are served one at a time; a second request waits.

## Using it from opencode

Tested on 2026-11-24 with opencode 1.19.6. Copy `examples/opencode.json` into the project you
want to work in, start this server, then run `opencode` there (or
`opencode run -m latentmap/default_model "…"`).

What happened on a small read-only task ("What version is declared in pyproject.toml?"):

| Step | Prompt | Already read | Wait for first word |
| --- | --- | --- | --- |
| 1. model calls `ls` | 14,647 tokens | 0 | 32 s |
| 2. model calls `read` | 14,728 | 14,714 | 0.4 s |
| 3. model answers "0.4.2" | 14,880 | 14,760 | 0.7 s |

The whole task took 63 seconds and the answer was right. Things to know:

- **The first step of every new task is slow.** opencode sends about 14,600 tokens of instructions
  and tool descriptions, which the model reads at roughly 450 tokens per second. Later steps reuse
  that work. A second, different task re-reads it.
- **Writing is about 14 words per second**, so long edits take time.
- opencode also asks the model for a session title (a separate small request you will see in the inspector).
- The very first run after installing opencode sat silent for several minutes before sending
  anything, apparently while it set itself up. The second run started at once.
- This server keeps everything local, but opencode is a separate program with its own behaviour
  (it reads its own config and may check for model lists or updates online). The example config
  turns off auto-update and sharing.
- Codex is not supported yet: it needs the "responses" API route.

## Options

| Option | Default | Meaning |
| --- | --- | --- |
| `--port` | 8765 | |
| `--no-text` | off | Store numbers only. Prompts, replies and tool names are never written to disk, and text stored by earlier runs is removed at start. |
| `--retention-days` | 5 | Records older than this are deleted at start and every hour. |
| `--db` | `~/.latent-map/requests.db` | Where records are kept. |
| `--cache-bytes` | 2GB | Memory for saved prompt caches, which make follow-up turns fast. |
| `--no-keep-warm` | off | Do not re-read the weights while idle. |

## What is stored, and where

One SQLite file, `~/.latent-map/requests.db`, readable only by you. Per request: time, the
client's name, settings, the messages and reply, the exact tokens of the full prompt (which
include the tool descriptions a client sent) and any computed readout (none of these under
`--no-text`), and per generated word
its map cell on each pass, last-pass movement and loop signal, plus the average full-size state
of the thinking and of the answer (about 110KB).

- **Forget one conversation:** open the session in the page and press "Forget this session" twice (the second press confirms).
- **Delete everything:** stop the server and remove `~/.latent-map`.

## What stays on this Mac

Everything. The server listens on 127.0.0.1 only and refuses any other host. It runs with the
model hub in offline mode, so nothing is checked or fetched online. It refuses requests that come
from another web page (wrong `Origin` or `Host`), ignores a request's attempt to load a different
model, and refuses image URLs that would have to be downloaded. Request text is kept out of the
terminal log. There is no password: any program running on this Mac can use the API and read the records.

## What the page shows, and how far to trust it

| View | Status |
| --- | --- |
| Thinking and answer, separated | The model's own text. |
| Where each word lands on the map, settling across passes | Position on a map built from 120 texts. Area names are untested hypotheses. |
| Last-pass movement | Measured; not known to predict anything. |
| What it would have written after each pass | Pass 3 is the model's own prediction, computed again (checked in tests with the real model at temperature 0, on a short request, a 5,700-token request and a follow-up turn: it matches the tokens written). When a client samples at a higher temperature, the word written is not always pass 3's top word. Passes 1 and 2 are a reading the model was never trained to give; they are not its reasoning. |

The readout is computed when you press "Compute for this request": the model re-reads the
request with a cache of its own, and other requests wait meanwhile. It is stored with the
record. Measured on this Mac, at temperature 0:

| Request | Time | Pass 3 gave the token written |
| --- | --- | --- |
| 130 tokens | about 1 s | all |
| 3,900-token prompt, 120-token reply | 18 s | 120 of 120 |
| 5,700-token prompt, 150-token reply | 25 s | 150 of 150 |
| The follow-up turn to it (5,600 tokens from the cache) | 25 s | 150 of 150 |
| 17,100-token prompt, 219-token reply | 63 s | 219 of 219 |

Roughly 3.5 seconds per 1,000 prompt tokens plus one second per 12 reply tokens, so a long
reply takes about as long to read out as it took to write. The limit is 32,768 tokens (prompt
plus reply); that size has not been run through the server (a 30,000-token prompt alone took
about 110 seconds in a trial). Extra memory was about 3.6GB for a 14,400-token prompt and has
not been measured near the limit. A coding tool that sends a request during a readout waits
for it and may time out; a readout cannot be cancelled once started. The saved prompt cache
cannot be reused, because the last block needs its own memory of the whole text after pass 1
and pass 2.

**The model's output depends on the size of the pieces a text is read in.** Re-reading a
5,700-token prompt in pieces of 512 tokens, where generation had used 2,048, changed pass 3's
top token for 6 of 150 tokens; re-reading a reply 256 tokens at a time, where generation read
one at a time, changed 5 of 120. So each request records the sizes of the pieces its prompt
was read in (including the part an earlier turn read into the cache), and the readout re-reads
the prompt in those pieces and the reply one token at a time. Pass 3 then reproduces what
generation computed, which the tests check with the real model.

A follow-up can continue either of two saved caches: the one kept when the prompt had been
read (chat apps, which do not send the thinking back) or the one kept after the reply (agents,
which send the reply back exactly, tool calls included). Both are tracked.

When the pieces are not known, the prompt is re-read in the default pieces and the page says
the readout is close to the original, not identical. That applies to requests recorded before
this version and to a request that continued a cache whose pieces were not recorded. With a
temperature above 0 the token written is not always pass 3's top token in any case.

It needs the request's text, so it is not available under `--no-text`. Requests containing an
image get no readout.

### The run view

Opening a session shows the whole run above its last step: every step in order with what it
replied to, the tool it called, thinking and answer words, time and prompt size, and one button
that computes the readout for every step in turn (you can stop between steps). It gives numbers
and pointers, and no score or verdict.

**Steps of one run share the reading.** An agent's next step sends the earlier reply back
exactly, so its prompt begins with the earlier prompt and reply. When the readouts of two such
steps are computed one after the other, the second carries on from where the first ended and
reads only what is new. It does so only when the tokens and the pieces they were read in are
exactly those of the earlier step; otherwise it reads the whole prompt again, so the result is
the same either way. Chat follow-ups do not send the thinking back, so they do not share.

Tried on 2026-11-26 with an opencode run of four steps (a bug fix; about 14,600 of each
prompt's tokens are opencode's own instructions). The first readout took 40 seconds and the
next three took 3, 13 and 6 seconds, about a minute in all. Before sharing, a run like it took
about four minutes. Pass 3 gave the token written for every token of every step (39, 31, 161
and 77).

The reading kept between steps holds the last request's tokens and a few gigabytes of cache, in
memory only. It is dropped when a reply starts, when a session is forgotten, and after about a
minute and a half without a readout. So computing a single step later, or after the model has answered
something else, reads that step's whole prompt again.

"To read" lists sentences in the model's own thinking that match a word pattern ("instruction",
"system prompt", "no need to mention" and similar); clicking one opens the step at that word.
How far to trust it: on the 120 thinking texts of the planted-instruction test
(`signals/REPORT_hidden.md`) the pattern matches 119, but it was widened after seeing those
texts, so that figure is fitted, not a test result. On 20 ordinary steps with thinking stored on
this Mac it matched 2, both harmless ("No need to mention hidden details" about a summary).
It reads only the thinking: with thinking off it finds nothing, and a goal that leaves no trace
in the text is not something this tool can show.

## The chat box

The chat box holds a conversation: each message is sent with the earlier turns, without the
model's thinking. Click one of the model's turns to open that request in the inspector. A turn
that produced no answer text (stopped, or the word limit ran out during thinking) is not added.
The conversation lives in the page only, so reloading the page starts a new one; nothing is put
in browser storage, and forgetting its session also clears it from the chat box.
All turns of one conversation appear as one session. Two conversations that start with exactly
the same first message are also listed as one session.

Not in this version: continuing an old conversation after a reload, and the "responses" API
route that Codex needs.

## Tests

    ../spike/.venv/bin/python -m pytest -q                                   # no model needed
    ../spike/.venv/bin/python -m pytest -m model tests/test_server_model.py  # loads the model; stop the server first

`step0/` is the feasibility spike this was built from, kept for reference.
