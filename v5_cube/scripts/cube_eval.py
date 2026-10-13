# Run models on the cube photos (common/real_images_cube) and collect review material.
# Outputs (v5_cube/eval/<name>/):
#   <photo>_disease.jpg      : photo with non-Fresh instances (mask + outline in class color), Fresh outlined thin green
#   disease_crops_<k>.jpg    : every non-Fresh instance cropped with context (for visual review)
#   summary.csv              : per photo: leaf instances, per-class counts, leaf-area share, yellow-pixel share in leaves
# Usage: python v5_cube/scripts/cube_eval.py [--src <photo dir>] <name>=<weights> [...]
#   default src: common/real_images_cube; outputs go to v5_cube/eval/<name> (or <name>_<src dir name> with --src)
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
import paths  # noqa: E402  shared paths + opencv workaround

import csv  # noqa: E402
from collections import Counter  # noqa: E402
import numpy as np  # noqa: E402
import cv2  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402
from ultralytics import YOLO  # noqa: E402

NAMES = paths.CLASS_NAMES
SHORT = [n.split()[0] for n in NAMES]
COLORS = [(255, 40, 40), (40, 220, 40), (30, 120, 255), (255, 215, 0)]
FRESH = 1
CONF = 0.25


def font(size):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


F_BIG, F_SMALL = font(40), font(20)
args = sys.argv[1:]
SRC, suffix = paths.REAL_IMAGES_CUBE, ""
if "--src" in args:
    i = args.index("--src"); SRC = Path(args[i + 1]).resolve(); suffix = "_" + SRC.name; del args[i:i + 2]
photos = sorted(SRC.glob("*.jpg"))
for arg in args:
    name, wp = arg.split("=", 1)
    out = paths.V5 / "eval" / (name + suffix)
    out.mkdir(parents=True, exist_ok=True)
    model = YOLO(wp)
    rows, crops = [], []
    for p in photos:
        img = np.asarray(Image.open(p).convert("RGB"))
        H, W = img.shape[:2]
        r = model.predict(Image.fromarray(img), imgsz=640, conf=CONF, device=0, retina_masks=True, verbose=False)[0]
        masks = (r.masks.data.cpu().numpy() > 0.5) if r.masks is not None else np.zeros((0, H, W), bool)
        cls = r.boxes.cls.cpu().numpy().astype(int).tolist()
        confs = r.boxes.conf.cpu().numpy().tolist()
        boxes = r.boxes.xyxy.cpu().numpy().astype(int).tolist()
        leaf = masks.any(0) if len(masks) else np.zeros((H, W), bool)
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        yellow = (hsv[..., 0] >= 15) & (hsv[..., 0] <= 32) & (hsv[..., 1] >= 70) & (hsv[..., 2] >= 80)
        cnt = Counter(cls)
        rows.append(dict(photo=p.name, instances=len(cls), **{SHORT[c]: cnt.get(c, 0) for c in range(4)},
                         leaf_area=round(float(leaf.mean()), 3),
                         yellow_in_leaf=round(float((yellow & leaf).sum() / max(1, leaf.sum())), 4)))
        vis = img.astype(np.float32)
        for m, c in zip(masks, cls):
            if c != FRESH:
                vis[m] = vis[m] * 0.5 + np.array(COLORS[c], np.float32) * 0.5
        vis = vis.astype(np.uint8)
        th = max(2, round(max(H, W) / 500))
        for m, c in zip(masks, cls):
            cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(vis, cnts, -1, COLORS[c], th if c != FRESH else 1)
        im = Image.fromarray(vis); d = ImageDraw.Draw(im, "RGBA")
        d.rectangle((10, 10, 560, 70 + 50 * 4), fill=(0, 0, 0, 170))
        d.text((24, 18), f"{name} | {p.stem} | n={len(cls)}", font=F_BIG, fill=(255, 255, 255))
        for k in range(4):
            d.rectangle((24, 76 + 50 * k, 60, 112 + 50 * k), outline=COLORS[k], width=4)
            d.text((74, 72 + 50 * k), f"{SHORT[k]}: {cnt.get(k, 0)}" + ("  (thin outline)" if k == FRESH else ""), font=F_BIG, fill=(255, 255, 255))
        im.thumbnail((1400, 1400)); im.save(out / f"{p.stem}_disease.jpg", quality=88)
        for m, c, s, b in zip(masks, cls, confs, boxes):
            if c == FRESH:
                continue
            x0, y0, x1, y1 = b
            mg = max(60, int(0.6 * max(x1 - x0, y1 - y0)))
            cx0, cy0, cx1, cy1 = max(0, x0 - mg), max(0, y0 - mg), min(W, x1 + mg), min(H, y1 + mg)
            crop = img[cy0:cy1, cx0:cx1].copy()
            cn, _ = cv2.findContours(m[cy0:cy1, cx0:cx1].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(crop, cn, -1, (255, 0, 255), 2)
            t = Image.fromarray(crop); t.thumbnail((260, 260))
            cell = Image.new("RGB", (270, 300), (20, 20, 20)); cell.paste(t, ((270 - t.width) // 2, 34))
            ImageDraw.Draw(cell).text((5, 6), f"#{len(crops)} {p.stem[-2:]} {SHORT[c]} {s:.2f}", font=F_SMALL, fill=COLORS[c])
            crops.append(cell)
        print("PHOTO", name, rows[-1], flush=True)
    for k in range(0, len(crops), 24):
        part = crops[k:k + 24]
        sheet = Image.new("RGB", (6 * 270, -(-len(part) // 6) * 300), (20, 20, 20))
        for i, c in enumerate(part):
            sheet.paste(c, ((i % 6) * 270, (i // 6) * 300))
        sheet.save(out / f"disease_crops_{k // 24 + 1}.jpg", quality=88)
    with open(out / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    tot = Counter(); [tot.update({k: r[k] for k in SHORT}) for r in rows]
    print("TOTAL", name, dict(tot), "non-Fresh crops", len(crops), flush=True)
print("ALL DONE", flush=True)
