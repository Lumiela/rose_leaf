# Evaluate the v6 pipeline on all real photo sets (cube 18, fixed-camera frame, original 3).
# For each photo: counts for v4 raw / v4+WB / v4+WB+Hole filter at thresholds 0.3, 0.5, 0.7.
# Visual review: every Hole instance (after WB) cropped with its leaf-pixel share, marked KEEP/DROP at 0.5.
# Outputs: v6_holefilter/eval/summary.csv, hole_review_<k>.jpg, <photo>_v6.jpg
# Usage: python v6_holefilter/scripts/eval_v6.py
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pipeline import Pipeline, HOLE, paths  # noqa: E402

import csv  # noqa: E402
from collections import Counter  # noqa: E402
import numpy as np  # noqa: E402
import cv2  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

OUT = paths.V6 / "eval"
OUT.mkdir(parents=True, exist_ok=True)
SHORT = [n.split()[0] for n in paths.CLASS_NAMES]
COLORS = [(255, 40, 40), (40, 220, 40), (30, 120, 255), (255, 215, 0)]
SETS = [paths.REAL_IMAGES_CUBE, paths.COMMON / "real_images_camfix", paths.REAL_IMAGES]


def font(size):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


F_BIG, F_SMALL = font(40), font(20)
raw = Pipeline(wb=False, hole_min_leaf=None)
v6 = Pipeline(wb=True, hole_min_leaf=0.5)
rows, crops = [], []
for d in SETS:
    for p in sorted(list(d.glob("*.jpg")) + list(d.glob("*.webp"))):
        img = np.asarray(Image.open(p).convert("RGB"))
        r_raw, _ = raw.predict(img)
        r_wb, im = v6.predict(img)
        row = dict(set=d.name, photo=p.name)
        for k in range(4):
            row[f"raw_{SHORT[k]}"] = sum(1 for x in r_raw if x["cls"] == k)
        for k in range(4):
            row[f"wb_{SHORT[k]}"] = sum(1 for x in r_wb if x["cls"] == k)
        for t in (0.3, 0.5, 0.7):
            row[f"hole_kept@{t}"] = sum(1 for x in r_wb if x["cls"] == HOLE and x["leaf_frac"] >= t)
        rows.append(row)
        H, W = im.shape[:2]
        for x in r_wb:
            if x["cls"] != HOLE:
                continue
            x0, y0, x1, y1 = x["box"]
            mg = max(50, int(0.5 * max(x1 - x0, y1 - y0)))
            cx0, cy0, cx1, cy1 = max(0, x0 - mg), max(0, y0 - mg), min(W, x1 + mg), min(H, y1 + mg)
            crop = im[cy0:cy1, cx0:cx1].copy()
            cn, _ = cv2.findContours(x["mask"][cy0:cy1, cx0:cx1].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(crop, cn, -1, (255, 0, 0) if x["dropped"] else (0, 255, 0), 3)
            t = Image.fromarray(crop); t.thumbnail((240, 240))
            cell = Image.new("RGB", (250, 280), (20, 20, 20)); cell.paste(t, ((250 - t.width) // 2, 34))
            ImageDraw.Draw(cell).text((5, 6), f"{p.stem[-7:]} leaf={x['leaf_frac']:.2f} {'DROP' if x['dropped'] else 'KEEP'}",
                                      font=F_SMALL, fill=(255, 90, 90) if x["dropped"] else (90, 255, 90))
            crops.append(cell)
        vis = im.astype(np.float32)
        for x in r_wb:
            if x["cls"] != 1 and not x["dropped"]:
                vis[x["mask"]] = vis[x["mask"]] * 0.5 + np.array(COLORS[x["cls"]], np.float32) * 0.5
        vis = vis.astype(np.uint8)
        th = max(2, round(max(H, W) / 500))
        for x in r_wb:
            cn, _ = cv2.findContours(x["mask"].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if x["dropped"]:
                cv2.drawContours(vis, cn, -1, (160, 160, 160), th)
            else:
                cv2.drawContours(vis, cn, -1, COLORS[x["cls"]], th if x["cls"] != 1 else 1)
        cnt = Counter(x["cls"] for x in r_wb if not x["dropped"])
        imv = Image.fromarray(vis); dr = ImageDraw.Draw(imv, "RGBA")
        dr.rectangle((10, 10, 600, 300), fill=(0, 0, 0, 170))
        dr.text((24, 18), f"v6 | {p.stem}", font=F_BIG, fill=(255, 255, 255))
        for k in range(4):
            dr.rectangle((24, 76 + 50 * k, 60, 112 + 50 * k), outline=COLORS[k], width=4)
            dr.text((74, 72 + 50 * k), f"{SHORT[k]}: {cnt.get(k, 0)}", font=F_BIG, fill=(255, 255, 255))
        dr.text((24, 272), f"gray outline = dropped Hole ({sum(x['dropped'] for x in r_wb)})", font=F_SMALL, fill=(200, 200, 200))
        imv.thumbnail((1400, 1400)); imv.save(OUT / f"{p.stem}_v6.jpg", quality=88)
        print("ROW", row, flush=True)
for k in range(0, len(crops), 30):
    part = crops[k:k + 30]
    sheet = Image.new("RGB", (6 * 250, -(-len(part) // 6) * 280), (20, 20, 20))
    for i, c in enumerate(part):
        sheet.paste(c, ((i % 6) * 250, (i // 6) * 280))
    sheet.save(OUT / f"hole_review_{k // 30 + 1}.jpg", quality=88)
with open(OUT / "summary.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
for s in [d.name for d in SETS]:
    sub = [r for r in rows if r["set"] == s]
    tot = {k: sum(r[k] for r in sub) for k in rows[0] if k not in ("set", "photo")}
    print("TOTAL", s, len(sub), tot, flush=True)
print("ALL DONE", len(crops), "hole crops", flush=True)
