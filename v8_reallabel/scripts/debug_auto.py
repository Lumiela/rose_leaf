# debug: why does 'auto' add so few leaves? count proposals per filter stage on one photo
import sys
from sam_common import load, load_rgb, gray_world, leaf_color, window, auto_masks, solidity_of
import numpy as np
stem = sys.argv[1]
inst, meta = load(stem); img = load_rgb(meta["photo"]); H, W = inst.shape; L, T = meta["L"], meta["T"]
leaf = leaf_color(gray_world(img))
st = max(1, int(0.7 * T))
wins = sorted({window(cx, cy, T, W, H) for cy in range(T // 2, H + st, st) for cx in range(T // 2, W + st, st)})
c = dict(total=0, area_ok=0, small=0, big=0, free_ok=0, leaf_ok=0, solid_ok=0)
taken = inst > 0
for w in wins:
    for m in auto_masks(img, w):
        c["total"] += 1; ar = m.sum()
        if ar < 0.1 * L * L: c["small"] += 1; continue
        if ar > 4 * L * L: c["big"] += 1; continue
        c["area_ok"] += 1
        free = m & ~taken
        if free.sum() < 0.6 * ar: continue
        c["free_ok"] += 1
        if leaf[free].mean() < 0.6: continue
        c["leaf_ok"] += 1
        if solidity_of(free) < 0.8: continue
        c["solid_ok"] += 1
print("windows", len(wins), "L", L, "T", T, c)
print("uncovered leaf px", int((leaf & ~taken).sum()), "of leaf px", int(leaf.sum()))
