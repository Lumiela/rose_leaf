# Experiment 1: how much of each leaf does the color-based stage-1 mask keep, per class?
# Ground truth = alpha channel of the source cutouts (common/Dataset). Leaves are resized to a real-photo
# leaf size first (max side 160 px) so morphology sizes match the photos.
# Variants: raw HSV / + closing & hole filling / + dilation. Reported per class:
#   mean & median pixel recall, share of leaves with recall < 0.8 and < 0.5.
# Usage: python v3_twostage/scripts/exp1_hsv_leaf_recall.py > v3_twostage/logs/exp1.log 2>&1
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
import paths  # noqa: E402  shared paths + opencv workaround (must come before cv2/ultralytics)

import json  # noqa: E402
import random  # noqa: E402
from concurrent.futures import ThreadPoolExecutor  # noqa: E402
import numpy as np  # noqa: E402
import cv2  # noqa: E402
from PIL import Image  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stage1 import hsv_leaf_mask, refine_mask  # noqa: E402

OUT = paths.V3 / "results" / "exp1_hsv_leaf_recall"
LEAF_SIDE = 160
MAX_PER_CLASS = 400
VARIANTS = {"raw": dict(close_px=0, dilate_px=0, fill=False),
            "close+fill": dict(close_px=4, dilate_px=0, fill=True),
            "close+fill+dilate": dict(close_px=4, dilate_px=4, fill=True)}
random.seed(0)


def load(path):
    im = Image.open(path).convert("RGBA")
    im = im.crop(im.getchannel("A").point(lambda v: 255 if v > 16 else 0).getbbox())
    im.thumbnail((LEAF_SIDE, LEAF_SIDE), Image.LANCZOS)
    a = np.asarray(im)
    gt = a[..., 3] > 128
    gt = cv2.erode(gt.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)   # ignore 1px cutout halo
    rgb = a[..., :3].copy()
    rgb[a[..., 3] <= 128] = 0                                                       # transparent -> black (not green)
    return path, rgb, gt


def recall(mask, gt):
    return float((mask.astype(bool) & gt).sum() / max(1, gt.sum()))


OUT.mkdir(parents=True, exist_ok=True)
summary, samples = {}, {}
for c in paths.CLASS_NAMES:
    files = sorted(f for f in (paths.DATASET / c).glob("*.png") if "_aug_" not in f.name)
    files = random.sample(files, min(MAX_PER_CLASS, len(files)))
    with ThreadPoolExecutor(4) as ex:
        leaves = list(ex.map(load, files))
    per = {v: [] for v in VARIANTS}
    recs = []
    for path, rgb, gt in leaves:
        raw = hsv_leaf_mask(rgb)
        row = {}
        for v, kw in VARIANTS.items():
            row[v] = recall(refine_mask(raw, **kw), gt)
            per[v].append(row[v])
        recs.append((row["close+fill+dilate"], path, rgb, gt, raw))
    summary[c] = {"n": len(leaves)}
    for v, r in per.items():
        r = np.array(r)
        summary[c][v] = dict(mean=round(float(r.mean()), 3), median=round(float(np.median(r)), 3),
                             lt80=round(float((r < 0.8).mean()), 3), lt50=round(float((r < 0.5).mean()), 3))
        print(f"RECALL {c:18s} {v:18s} n={len(r):4d} mean={r.mean():.3f} median={np.median(r):.3f} "
              f"<0.8={(r < 0.8).mean():.1%} <0.5={(r < 0.5).mean():.1%}", flush=True)
    recs.sort(key=lambda t: t[0])
    samples[c] = recs[:4] + random.sample(recs[4:], min(4, len(recs) - 4))   # 4 worst + 4 random

# visualization: kept leaf pixels in original color, missed leaf pixels red, background dark gray
cell = 170
sheet = Image.new("RGB", (8 * cell, len(paths.CLASS_NAMES) * cell), (25, 25, 25))
for i, c in enumerate(paths.CLASS_NAMES):
    for j, (r, path, rgb, gt, raw) in enumerate(samples[c]):
        m = refine_mask(raw, **VARIANTS["close+fill+dilate"]).astype(bool)
        vis = np.full(rgb.shape, 60, np.uint8)
        vis[gt & m] = rgb[gt & m]
        vis[gt & ~m] = (255, 0, 0)
        t = Image.fromarray(vis)
        sheet.paste(t, (j * cell + (cell - t.width) // 2, i * cell + (cell - t.height) // 2))
sheet.save(OUT / "worst4_random4_by_class.jpg", quality=90)
(OUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
print("SAVED", OUT, flush=True)
