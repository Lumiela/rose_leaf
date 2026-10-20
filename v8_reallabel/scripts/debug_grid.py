import sys
from sam_common import load, load_rgb, gray_world, leaf_color, window, run_sam, paste, solidity_of
import numpy as np, cv2
stem = sys.argv[1]
inst, meta = load(stem); img = load_rgb(meta["photo"]); H, W = inst.shape; L, T = meta["L"], meta["T"]
leaf = leaf_color(gray_world(img))
unc = cv2.erode((leaf & (inst == 0)).astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
g = max(4, int(0.6 * L))
pts = [(x, y) for y in range(g // 2, H, g) for x in range(g // 2, W, g) if unc[y, x]]
c = dict(empty=0, pts=len(pts), small=0, big=0, ok_area=0, fail_new=0, fail_leaf=0, fail_solid=0, ok=0)
areas = []
taken = inst > 0
for p in pts:
    w = window(p[0], p[1], T, W, H)
    res = run_sam(img, w, points=[[p]], labels=[[1]])
    if not res: c["empty"] += 1; continue
    m, s = res[0]
    full = paste((H, W), w, m); ar = full.sum(); areas.append(ar / (L * L))
    if ar < 0.1 * L * L: c["small"] += 1; continue
    if ar > 3 * L * L: c["big"] += 1; continue
    c["ok_area"] += 1
    free = full & ~taken
    if free.sum() < 0.5 * ar: c["fail_new"] += 1; continue
    if leaf[free].mean() < 0.55: c["fail_leaf"] += 1; continue
    if solidity_of(free) < 0.75: c["fail_solid"] += 1; continue
    c["ok"] += 1
print(c); print("area/L2 quantiles", np.percentile(areas, [10, 25, 50, 75, 90]).round(2))
