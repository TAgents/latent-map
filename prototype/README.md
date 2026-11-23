# Latent Map prototype (throwaway)

A local web page: type a prompt, Inkling 2 Mini answers on this Mac, and every
word is placed on the block 2 map from `basemap/bundle_v0` with its three loop
passes. Built to learn what is worth showing, not as the product.

## Run

```
./run.sh          # loads the model (about a minute), then serves http://127.0.0.1:8765
```

It listens on 127.0.0.1 only. One generation at a time, one turn per run, no history.

## What you see

- **Left:** the model's thinking and its answer. Each word is underlined in the colour
  of the area it ends in. An amber background means it moved more than usual on the
  last pass (above the 90th or 99th percentile seen for that area in the map's corpus).
- **Right:** the map's 256 cells in 12 areas. The thin line is the answer's path.
  Click a word to see where it sat after pass 1, 2 and 3, its settling, its last-pass
  movement and the loop signal (coordinate 2027 on block 3's output) per pass.

## Known limits

- The map is v0: built from 120 model-written texts, mostly thinking text. See
  `basemap/REPORT.md`. Area names are hypotheses; several are known to be wrong
  ("Sorting algorithms" is really code and technical text in general).
- "Moved more than usual" is not known to predict mistakes.
- Speed: about 16 words per second once it is writing, close to the model with capture alone. The wait before the first word is under a second when the model is in memory. On a Mac short of memory, macOS compresses and swaps the model out while it sits idle, and the first word can then take 5 to 45 seconds. The server reads every weight every 10 seconds while idle to keep the model in memory, and backs off (up to 2 minutes) when that pass is slow, so it does not make the machine swap continuously. Raising MLX's wired limit was tried and does not help. The real fix is free memory: the model needs about 20GB.
- The first one-token forward call is treated as the prompt's last token. That holds
  for the pinned mlx-lm commit and would need rechecking on another version.
- `norms.json` comes from `build_norms.py` and must be rebuilt if the map changes.
