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

- **Forget one conversation:** "Forget this session" in the page.
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
| What it would have written after each pass | Pass 3 is the model's real prediction (checked in a test with the real model on a short request at temperature 0: it matches the words written). When a client samples at a higher temperature, the word written is not always pass 3's top word. Passes 1 and 2 are a reading the model was never trained to give; they are not its reasoning. |

The readout is computed when you press "Compute for this request": the model re-reads the whole
request without the cache and makes other requests wait meanwhile (under a second for a
short request; 10 seconds for a 3,900-token request with a 1,000-word reply, where pass 3
matched the word written for 98.9% of words). It is stored
with the record. It needs the request's text, so it is not available under `--no-text`, and in
this version only for requests of up to 4,096 tokens (prompt plus reply), which leaves out
opencode's 14,600-token requests. Requests containing an image get no readout.

Not in this version: conversation history in the
chat box, a summary for agent runs, and the "responses" API route that Codex needs.

## Tests

    ../spike/.venv/bin/python -m pytest -q                                   # no model needed
    ../spike/.venv/bin/python -m pytest -m model tests/test_server_model.py  # loads the model; stop the server first

`step0/` is the feasibility spike this was built from, kept for reference.
