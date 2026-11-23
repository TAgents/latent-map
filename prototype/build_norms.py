"""How far block 2 states usually move between pass 2 and pass 3, per fine area,
measured on the corpus that map version was built from. Writes norms_<tag>.json.

Run: PYTHONPATH=../basemap ../spike/.venv/bin/python build_norms.py [tag]
"""

import json
import sys
from pathlib import Path

import numpy as np

from mapbuild.fit import _prepare
from mapbuild.paths import paths

HERE = Path(__file__).parent


def main() -> None:
    tag = sys.argv[1] if len(sys.argv) > 1 else "v1"
    where = paths(tag)
    states = np.load(where["states"])
    b = np.load(where["bundle"] / "block2.npz")
    reply = b["used"] if "used" in b else states["is_reply"]
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
    (HERE / f"norms_{tag}.json").write_text(json.dumps(norms, indent=1))
    print(json.dumps(norms))


if __name__ == "__main__":
    main()
