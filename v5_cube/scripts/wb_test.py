# Quick test: does white-balance correction remove the pink grow-light effect on class predictions?
# Gray-world white balance (per-channel gain so the mean of low-saturation pixels becomes neutral), then v4 predict.
# Output: v5_cube/eval/wb_test/summary.csv (+ corrected images for 3 photos) and per-photo counts before/after.
# Usage: python v5_cube/scripts/wb_test.py
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
import paths  # noqa: E402

import csv  # noqa: E402
from collections import Counter  # noqa: E402
import numpy as np  # noqa: E402
import cv2  # noqa: E402
from PIL import Image  # noqa: E402
from ultralytics import YOLO  # noqa: E402

OUT = paths.V5 / "eval" / "wb_test"
OUT.mkdir(parents=True, exist_ok=True)
SHORT = [n.split()[0] for n in paths.CLASS_NAMES]


def gray_world(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    ref = (hsv[..., 1] < 60) & (hsv[..., 2] > 60) & (hsv[..., 2] < 240)     # near-neutral pixels (tray, walls)
    if ref.sum() < 1000:
        ref = np.ones(ref.shape, bool)
    mean = img[ref].reshape(-1, 3).mean(0)
    gain = mean.mean() / mean
    return np.clip(img.astype(np.float32) * gain, 0, 255).astype(np.uint8), gain


model = YOLO(str(paths.V4_BEST))
rows = []
for p in sorted(paths.REAL_IMAGES_CUBE.glob("*.jpg")):
    img = np.asarray(Image.open(p).convert("RGB"))
    wb, gain = gray_world(img)
    res = {}
    for tag, im in (("orig", img), ("wb", wb)):
        r = model.predict(Image.fromarray(im), imgsz=640, conf=0.25, device=0, verbose=False)[0]
        res[tag] = Counter(int(c) for c in r.boxes.cls)
    row = dict(photo=p.stem, gain_r=round(gain[0], 2), gain_g=round(gain[1], 2), gain_b=round(gain[2], 2))
    for tag in ("orig", "wb"):
        for c in range(4):
            row[f"{tag}_{SHORT[c]}"] = res[tag].get(c, 0)
    rows.append(row)
    if p.stem in ("cube_13", "cube_06", "cube_05"):
        both = np.concatenate([img, wb], 1)
        im = Image.fromarray(both); im.thumbnail((1600, 1600)); im.save(OUT / f"{p.stem}_orig_vs_wb.jpg", quality=88)
    print("WB", row, flush=True)
with open(OUT / "summary.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
tot = {k: sum(r[k] for r in rows) for k in rows[0] if k.startswith(("orig_", "wb_"))}
print("TOTAL", tot, flush=True)
