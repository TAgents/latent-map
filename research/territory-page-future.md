# Territory page (regions view) — deferred to a later version

Noted 2026-11-22. Not part of the first version. The first version is six screens: Mac Live, Loop inspector, Sharing; Web Atlas, Walk-through, Contribute.

## What it is

A page per territory of the map, reached from the Atlas, showing the structure inside that territory. An earlier dot-cloud draft ("Region detail", with months on a ring and years on a curve) was removed from the design canvas because it was hard to read and its shapes were unmeasured in this model.

## Why it is worth building later

- It is where the internal structure becomes visible, which the Atlas side panel only summarises.
- On the Mac it could show where the current word sits inside a territory, for example which month on a ring.

## The cheap parts (reuse data the map already has)

- Zoom into the territory: its tiles grouped into sub-territories, each with its own words.
- Words and public-text examples per sub-territory.
- How states settle there: usual movement between passes (the same norm the live view's flag uses).
- Neighbours and crossings: which territories answers commonly arrive from and leave to.
- Real use against public text inside the territory.

## The hard part: shapes

- The global 2D map cannot show local shapes, so each territory needs its own local projection in the map bundle.
- Known candidates (months, weekdays, years, numbers) can be tested with templated prompts. Supervised Multi-Dimensional Scaling has open code: https://github.com/UKPLab/tmlr2026-manifold-analysis
- Discovering shapes nobody predicted is open research.
- A shape may not exist in this model; the panel must then stay empty and say so.

## Proposed design when it is picked up

- Territory page drawn in the tile style of the Atlas, built first from the cheap parts.
- A shape panel that appears only where a tested shape fits, shows how well it fits, and starts with a handful of known candidates.
- Example snippets come from public text only; contributed usage is counts.

## Related research

See `latent-space-visualisation-and-geometry.md`, section 5, for the papers on rings, curves and where they come from (arXiv 2602.15029, 2605.23821, 2604.28119, 2605.18537, 2510.01025).
