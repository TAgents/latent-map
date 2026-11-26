# Latent Map

Run Thinking Machines' Inkling 2 Mini on a Mac and see what happened inside its loop.

Inkling 2 Mini runs its middle layers three times for every word. Latent Map records the
model's state after each of those passes and shows it next to the text, for two purposes:

- **Guide:** use the model better.
- **Guard:** see knowledge and goals inside the loop passes that the written text does not show.

Everything runs locally. Prompts, replies and states never leave the machine, and the server
listens on 127.0.0.1 only.

## What works today

`server/` is the usable part: a local server that serves the model to any chat app or coding
tool through the standard chat API, records per-pass states for every request, and shows any
request afterwards in a local page. See [server/README.md](server/README.md) to run it.

It needs an Apple silicon Mac with 32GB of memory and about 20GB of disk for the weights.

## What is in this repository

| Folder | What it is |
| --- | --- |
| `server/` | The local server, recorder and inspector page |
| `spike/` | Capture of per-pass states with MLX, and the per-pass readout ([report](spike/REPORT.md)) |
| `basemap/` | The map of the model's internal space, built from 120 model-written texts ([report](basemap/REPORT.md)) |
| `signals/` | Tests of what can be read from the loop ([signal tests](signals/REPORT.md), [hidden knowledge and goals](signals/REPORT_hidden.md)) |
| `prototype/` | The first throwaway page, replaced by `server/` |
| `docs/` | Designs and plans, with test criteria fixed before each run |
| `research/` | Notes on the model and on related work |

## What the tests support, and what they do not

- The inspector shows what the model would have written after each pass. On the last pass this
  matched the word actually written for every token in the requests tested. It is not the
  model's reasoning and is not labelled as such.
- A fact the model passes through without writing it could be read from the first two passes
  in most test questions. Whether the model uses what is read there was not shown.
- With thinking on, the model wrote a planted instruction into its thinking text in 120 of 120
  replies. For a goal that leaves no trace in the text, the tests found no clear signal in the
  loop and could not rule one out.
- Steering the model by changing its state did not work in the one test run.

The reports give the numbers, the criteria and the corrections made after independent review.

## Not built

The native Mac app, the shared map and the public atlas described in the early designs.

## Licence

MIT. The model has its own licence; see its model card.
