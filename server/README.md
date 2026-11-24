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

## Options

| Option | Default | Meaning |
| --- | --- | --- |
| `--port` | 8765 | |
| `--no-text` | off | Store numbers only. Prompts, replies and tool names are never written to disk. |
| `--retention-days` | 5 | Records older than this are deleted at start and every hour. |
| `--db` | `~/.latent-map/requests.db` | Where records are kept. |
| `--cache-bytes` | 2GB | Memory for saved prompt caches, which make follow-up turns fast. |
| `--no-keep-warm` | off | Do not re-read the weights while idle. |

## What is stored, and where

One SQLite file, `~/.latent-map/requests.db`, readable only by you. Per request: time, the
client's name, settings, the messages and reply (unless `--no-text`), and per generated word
its map cell on each pass, last-pass movement and loop signal, plus the average full-size state
of the thinking and of the answer (about 110KB).

- **Forget one conversation:** "Forget this session" in the page.
- **Delete everything:** stop the server and remove `~/.latent-map`.

## What stays on this Mac

Everything. The server listens on 127.0.0.1 only and refuses any other host. It runs with the
model hub in offline mode, so nothing is checked or fetched online. It refuses requests that come
from another web page (wrong `Origin` or `Host`). Request text is kept out of the terminal log.

## What the page shows, and how far to trust it

| View | Status |
| --- | --- |
| Thinking and answer, separated | The model's own text. |
| Where each word lands on the map, settling across passes | Position on a map built from 120 texts. Area names are untested hypotheses. |
| Last-pass movement | Measured; not known to predict anything. |

Not in this version: what the model would say after each pass, conversation history in the
chat box, a summary for agent runs, and the "responses" API route that Codex needs.

## Tests

    ../spike/.venv/bin/python -m pytest -q                                   # no model needed
    ../spike/.venv/bin/python -m pytest -m model tests/test_server_model.py  # loads the model; stop the server first

`step0/` is the feasibility spike this was built from, kept for reference.
