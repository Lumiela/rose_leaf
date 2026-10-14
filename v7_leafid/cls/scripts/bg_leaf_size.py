# Measure the leaf size of every background photo used by render_data_bbox.py, so pasted leaves can be rendered at
# the same scale as the real leaves around them.  Runs v7 stage-1 leaf search on the background folders (once; reused
# if leaves.json exists) and writes cls/bg_leaf_size.json: {photo name: {"L": median leaf long side px, "n": leaves used}}.
# Median over judged leaves (solidity >= 0.8, not touching the border); falls back to all leaves, then the global median.
# Usage: CLS_VARIANT=bbox python v7_leafid/cls/scripts/bg_leaf_size.py
import os, sys  # noqa: E401
os.environ.setdefault("CLS_VARIANT", "bbox")
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from cls_common import CLS, Progress, paths  # noqa: E402

import json, subprocess  # noqa: E401,E402
import numpy as np  # noqa: E402

SRC = {"bg_cube": paths.REAL_IMAGES_CUBE, "bg_real": paths.REAL_IMAGES}
OUT = CLS / "bg_leaf_size.json"

if __name__ == "__main__":
    Progress.log("bg_size", "leaf search on background photos", "running")
    rows = []
    for name, src in SRC.items():
        run = paths.V7 / "runs" / name
        if not (run / "leaves.json").exists():
            subprocess.run([sys.executable, "-u", str(paths.V7 / "scripts" / "leaf_search.py"), str(src), "--out", str(run)], check=True)
        rows += json.loads((run / "leaves.json").read_text())
        Progress.log("bg_size", f"{name}: {len(rows)} leaves so far")
    side = lambda r: max(r["x1"] - r["x0"], r["y1"] - r["y0"])  # noqa: E731
    judged = [r for r in rows if r["solidity"] >= 0.8 and not r["touches_border"]]
    g = float(np.median([side(r) for r in judged])) if judged else 100.0
    res = {}
    for ph in sorted({r["photo"] for r in rows}):
        j = [side(r) for r in judged if r["photo"] == ph]
        a = [side(r) for r in rows if r["photo"] == ph]
        L, n = (np.median(j), len(j)) if len(j) >= 5 else (np.median(a), len(a)) if len(a) >= 3 else (g, 0)
        res[ph] = dict(L=round(float(L), 1), n=int(n))
        Progress.log("bg_size", f"{ph}: leaf long side median {L:.0f}px (n={n})")
    res["_global"] = dict(L=round(g, 1), n=len(judged))
    OUT.write_text(json.dumps(res, indent=1, ensure_ascii=False))
    Progress.log("bg_size", f"global median {g:.0f}px -> {OUT.name}", "done")
