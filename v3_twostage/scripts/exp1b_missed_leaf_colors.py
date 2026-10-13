# Experiment 1b: why does the color mask miss some leaves entirely? Color stats + originals of the missed leaves.
# For leaves with recall < 0.5 (close+fill+dilate, same sample/seed as exp1) prints the median H/S/V of the leaf
# pixels and which threshold failed, and saves a sheet of the original cutouts per class (worst first).
# Usage: python v3_twostage/scripts/exp1b_missed_leaf_colors.py > v3_twostage/logs/exp1b.log 2>&1
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
import paths  # noqa: E402  shared paths + opencv workaround

import random  # noqa: E402
from collections import Counter  # noqa: E402
import numpy as np  # noqa: E402
import cv2  # noqa: E402
from PIL import Image  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stage1 import hsv_leaf_mask, refine_mask, H_RANGE, S_MIN, V_MIN  # noqa: E402

OUT = paths.V3 / "results" / "exp1_hsv_leaf_recall"
LEAF_SIDE, MAX_PER_CLASS = 160, 400
random.seed(0)                                   # same sampling as exp1
cell, per_row = 170, 8
rows = []
for c in paths.CLASS_NAMES:
    files = sorted(f for f in (paths.DATASET / c).glob("*.png") if "_aug_" not in f.name)
    files = random.sample(files, min(MAX_PER_CLASS, len(files)))
    missed, why = [], Counter()
    for f in files:
        im = Image.open(f).convert("RGBA")
        im = im.crop(im.getchannel("A").point(lambda v: 255 if v > 16 else 0).getbbox())
        im.thumbnail((LEAF_SIDE, LEAF_SIDE), Image.LANCZOS)
        a = np.asarray(im)
        gt = cv2.erode((a[..., 3] > 128).astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
        rgb = a[..., :3].copy(); rgb[a[..., 3] <= 128] = 0
        m = refine_mask(hsv_leaf_mask(rgb), close_px=4, dilate_px=4, fill=True).astype(bool)
        r = (m & gt).sum() / max(1, gt.sum())
        if r < 0.5:
            hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)[gt]
            h, s_, v = np.median(hsv[:, 0]), np.median(hsv[:, 1]), np.median(hsv[:, 2])
            fail = "hue" if not (H_RANGE[0] <= h <= H_RANGE[1]) else ("sat" if s_ < S_MIN else ("val" if v < V_MIN else "mixed"))
            why[fail] += 1
            missed.append((r, f.name, int(h), int(s_), int(v), im))
    missed.sort(key=lambda t: t[0])
    print(f"MISSED {c:18s} {len(missed)}/{len(files)} failed_threshold={dict(why)}", flush=True)
    for r, name, h, s_, v, _ in missed[:10]:
        print(f"   {name:40s} recall={r:.2f} median H={h} S={s_} V={v}", flush=True)
    rows.append([t[5] for t in missed[:per_row]])
sheet = Image.new("RGB", (per_row * cell, len(rows) * cell), (128, 128, 128))
for i, row in enumerate(rows):
    for j, im in enumerate(row):
        sheet.paste(im, (j * cell + (cell - im.width) // 2, i * cell + (cell - im.height) // 2), im)
sheet.save(OUT / "missed_leaves_original_colors.jpg", quality=90)
print("SAVED", OUT / "missed_leaves_original_colors.jpg", flush=True)
