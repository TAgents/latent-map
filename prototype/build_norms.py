"""How far block 2 states usually move between pass 2 and pass 3, per fine area,
measured on the corpus the v0 map was built from. Writes norms.json.

Run: PYTHONPATH=../basemap ../spike/.venv/bin/python build_norms.py
"""

import json
from pathlib import Path

import numpy as np

from mapbuild.fit import _prepare

HERE = Path(__file__).parent
BASE = HERE.parent / "basemap"


def main() -> None:
    states = np.load(BASE / "data" / "states.npz")
    reply = states["is_reply"]
    b = np.load(BASE / "bundle_v0" / "block2.npz")
    x = states["block2"][reply]
    move = np.concatenate([
        np.linalg.norm(u[:, 2] - u[:, 1], axis=-1)
        for u in (_prepare(x[i : i + 4000], int(b["drop"])) for i in range(0, len(x), 4000))
    ])
    area = b["fine"][b["cell"][reply, -1]]
    bands = lambda v: [float(p) for p in np.percentile(v, [50, 90, 99])]
    norms = {"all": bands(move)}
    for a in np.unique(area):
        if (area == a).sum() >= 200:
            norms[str(int(a))] = bands(move[area == a])
    (HERE / "norms.json").write_text(json.dumps(norms, indent=1))
    print(json.dumps(norms))


if __name__ == "__main__":
    main()
