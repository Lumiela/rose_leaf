# Run the v6 pipeline (v4 model + white balance + Hole filter) on your own photos.
# Usage (from /helpai-storage/model_git):
#   python v6_holefilter/scripts/run_v6.py <image file or folder> [--out <folder>] [--no-wb] [--hole-min-leaf 0.5] [--conf 0.25]
#   default output: v6_holefilter/runs/<input name>_<YYYYmmdd_HHMMSS>/
# Outputs per photo: <photo>_v6.jpg (non-Fresh = filled, Fresh = thin green outline, dropped Hole = gray outline)
# and results.csv (per-photo counts, leaf-area share, yellow-pixel share inside leaves).
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pipeline import Pipeline, HOLE, paths  # noqa: E402

import argparse  # noqa: E402
import csv  # noqa: E402
import time  # noqa: E402
from collections import Counter  # noqa: E402
import numpy as np  # noqa: E402
import cv2  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

SHORT = [n.split()[0] for n in paths.CLASS_NAMES]
COLORS = [(255, 40, 40), (40, 220, 40), (30, 120, 255), (255, 215, 0)]
EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")

ap = argparse.ArgumentParser()
ap.add_argument("src")
ap.add_argument("--out")
ap.add_argument("--no-wb", action="store_true")
ap.add_argument("--hole-min-leaf", type=float, default=0.5)
ap.add_argument("--conf", type=float, default=0.25)
a = ap.parse_args()

src = Path(a.src).resolve()
photos = [src] if src.is_file() else sorted(p for p in src.iterdir() if p.suffix.lower() in EXTS)
if not photos:
    sys.exit(f"no images in {src}")
out = Path(a.out).resolve() if a.out else paths.V6 / "runs" / f"{src.stem}_{time.strftime('%Y%m%d_%H%M%S')}"
out.mkdir(parents=True, exist_ok=True)
try:
    F_BIG, F_SMALL = ImageFont.load_default(size=40), ImageFont.load_default(size=22)
except TypeError:
    F_BIG = F_SMALL = ImageFont.load_default()

pl = Pipeline(wb=not a.no_wb, hole_min_leaf=a.hole_min_leaf, conf=a.conf)
rows = []
for p in photos:
    t0 = time.time()
    img = np.asarray(Image.open(p).convert("RGB"))
    res, im = pl.predict(img)
    H, W = im.shape[:2]
    kept = [x for x in res if not x["dropped"]]
    cnt = Counter(x["cls"] for x in kept)
    leaf = np.zeros((H, W), bool)
    for x in kept:
        leaf |= x["mask"]
    hsv = cv2.cvtColor(im, cv2.COLOR_RGB2HSV)
    yellow = (hsv[..., 0] >= 15) & (hsv[..., 0] <= 32) & (hsv[..., 1] >= 70) & (hsv[..., 2] >= 80)
    vis = im.astype(np.float32)
    for x in kept:
        if x["cls"] != 1:
            vis[x["mask"]] = vis[x["mask"]] * 0.5 + np.array(COLORS[x["cls"]], np.float32) * 0.5
    vis = vis.astype(np.uint8)
    th = max(2, round(max(H, W) / 500))
    for x in res:
        cn, _ = cv2.findContours(x["mask"].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        color = (160, 160, 160) if x["dropped"] else COLORS[x["cls"]]
        cv2.drawContours(vis, cn, -1, color, 1 if (x["cls"] == 1 and not x["dropped"]) else th)
    imv = Image.fromarray(vis); d = ImageDraw.Draw(imv, "RGBA")
    d.rectangle((10, 10, 620, 310), fill=(0, 0, 0, 170))
    d.text((24, 18), f"v6 | {p.stem}", font=F_BIG, fill=(255, 255, 255))
    for k in range(4):
        d.rectangle((24, 76 + 50 * k, 60, 112 + 50 * k), outline=COLORS[k], width=4)
        d.text((74, 72 + 50 * k), f"{SHORT[k]}: {cnt.get(k, 0)}", font=F_BIG, fill=(255, 255, 255))
    d.text((24, 278), f"gray = dropped Hole ({sum(x['dropped'] for x in res)})", font=F_SMALL, fill=(200, 200, 200))
    imv.save(out / f"{p.stem}_v6.jpg", quality=90)
    row = dict(photo=p.name, **{SHORT[k]: cnt.get(k, 0) for k in range(4)},
               hole_dropped=sum(x["dropped"] for x in res), leaf_area=round(float(leaf.mean()), 4),
               yellow_in_leaf=round(float((yellow & leaf).sum() / max(1, leaf.sum())), 4), sec=round(time.time() - t0, 2))
    rows.append(row)
    print("RESULT", row, flush=True)
with open(out / "results.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print("SAVED", out, flush=True)
