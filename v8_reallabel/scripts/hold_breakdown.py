# why are leaves held (not judged)? counts per reason on all photos, and on test photos vs the reviewed GT
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from sam_common import load, V8
import numpy as np
from PIL import Image
run = V8 / "pipeline" / "v8_stage1_nobg"
rows = json.loads((run / "leaves.json").read_text())
def reason(r):
    s, b = r["solidity"] < 0.8, bool(r["touches_border"])
    return "both" if s and b else "solidity" if s else "border" if b else None
tot = {}
for r in rows:
    k = reason(r); tot[k or "judged"] = tot.get(k or "judged", 0) + 1
print("ALL", len(rows), tot)
test = json.loads((V8 / "scenes.json").read_text())["split"]["test"]
res = {}
for s in test:
    inst, meta = load(s); area = {int(i): int((inst == i).sum()) for i in np.unique(inst) if i}
    for r in [r for r in rows if Path(r["photo"]).stem == s]:
        k = reason(r)
        if not k: continue
        m = np.asarray(Image.open(run / "crops" / s / f"{r['id']}.png").getchannel("A")) > 127
        sub = inst[r["y0"]:r["y0"] + m.shape[0], r["x0"]:r["x0"] + m.shape[1]]; m = m[:sub.shape[0], :sub.shape[1]]
        ids, cnt = np.unique(sub[m], return_counts=True)
        full = [int(i) for i, n in zip(ids, cnt) if i and n >= 0.5 * area[int(i)]]
        best = max([(n / (m.sum() + area[int(i)] - n), int(i)) for i, n in zip(ids, cnt) if i] or [(0, None)])
        kind = "merged(2+ GT leaves)" if len(full) >= 2 else "single leaf (IoU>=0.5)" if best[0] >= 0.5 else "partial/other"
        res.setdefault(k, {}).setdefault(kind, 0); res[k][kind] += 1
print("TEST held by reason x what it really is:", json.dumps(res, ensure_ascii=False))
