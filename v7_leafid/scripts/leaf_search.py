# v7 stage 1: find every leaf, give it an ID, and export per-leaf data for the stage-2 classifier.
# Leaf search = v6 pipeline (white balance + v4 model + Hole filter) with class-agnostic NMS, then mask-IoU
# de-duplication. The v4 class is kept only as a reference ("v4_cls"); the final state comes from stage 2.
# Non-leaf filter (all classes): instances whose leaf-colored pixel share is below --min-leaf-color are dropped
# (empty pots, tray, shadows, pot rims). Note: this also drops brown/dead leaves, which are not a target class.
# Background-object filter (after de-duplication), from per-instance color statistics on the white-balanced image:
#   exg = median of (2G-R-B)/(R+G+B), sat = median HSV saturation, dark = share of pixels with max(RGB) < 0.2
#   drop if exg < 0.12                      (no green at all)
#        or dark > 0.25 and exg < 0.25      (empty pot / water / pot rim: dark and weakly green)
#        or sat < 0.26 and exg < 0.15       (gray tray between leaves)
#   Thresholds were set on 642 detections of 5 fixed-camera photos (2026-10-14): 25 of 26 background objects
#   flagged by eye were removed, 0 of 6 leaf look-alikes. Dropped instances are saved for review (<photo>_bg_dropped.jpg).
#
# Usage (from /helpai-storage/model):
#   python v7_leafid/scripts/leaf_search.py <image file or folder> [--out <folder>] [--conf 0.25] [--min-area 400]
#   default output: v7_leafid/runs/<input name>_<YYYYmmdd_HHMMSS>/
# Outputs:
#   <photo>_ids.jpg              every leaf outlined (own color) with its ID number
#   <photo>_cutouts.jpg          contact sheet of all cutouts with IDs (review)
#   crops/<photo>/<ID>.png       RGBA cutout (transparent background, same format as common/Dataset) -> classifier input
#   context/<photo>/<ID>.jpg     crop with surrounding context and the mask outline
#   leaves.csv / leaves.json     one row per leaf: id, box, centroid, area, v4 class/conf, leaf-color share,
#                                sharpness, border contact, solidity (low = possibly several leaves merged)
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "v6_holefilter" / "scripts"))
from pipeline import Pipeline, paths  # noqa: E402

import argparse  # noqa: E402
import csv  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
import colorsys  # noqa: E402
import numpy as np  # noqa: E402
import cv2  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")
SHORT = [n.split()[0] for n in paths.CLASS_NAMES]

ap = argparse.ArgumentParser()
ap.add_argument("src")
ap.add_argument("--out")
ap.add_argument("--conf", type=float, default=0.25)
ap.add_argument("--min-area", type=int, default=400, help="minimum mask area in pixels")
ap.add_argument("--dedup-iou", type=float, default=0.5)
ap.add_argument("--min-leaf-color", type=float, default=0.3, help="drop instances with less leaf-colored pixels (0 = off)")
ap.add_argument("--no-bg-filter", action="store_true", help="disable the background-object color filter")
a = ap.parse_args()

src = Path(a.src).resolve()
photos = [src] if src.is_file() else sorted(p for p in src.iterdir() if p.suffix.lower() in EXTS)
if not photos:
    sys.exit(f"no images in {src}")
out = Path(a.out).resolve() if a.out else paths.V7 / "runs" / f"{src.stem}_{time.strftime('%Y%m%d_%H%M%S')}"
out.mkdir(parents=True, exist_ok=True)
try:
    F_ID, F_BIG = ImageFont.load_default(size=22), ImageFont.load_default(size=36)
except TypeError:
    F_ID = F_BIG = ImageFont.load_default()


def color(i):
    r, g, b = colorsys.hsv_to_rgb((i * 0.618034) % 1.0, 0.85, 1.0)
    return int(r * 255), int(g * 255), int(b * 255)


def color_stats(im, m):
    rgb = im[m].astype(np.float32) / 255.0
    mx, mn = rgb.max(1), rgb.min(1)
    sat = np.where(mx > 0, (mx - mn) / np.maximum(mx, 1e-6), 0)
    exg = (2 * rgb[:, 1] - rgb[:, 0] - rgb[:, 2]) / np.maximum(rgb.sum(1), 1e-6)
    return float(np.median(exg)), float(np.median(sat)), float((mx < 0.2).mean())


def is_background(exg, sat, dark):
    return exg < 0.12 or (dark > 0.25 and exg < 0.25) or (sat < 0.26 and exg < 0.15)


def mask_iou(a_, b_):
    """IoU of two instance dicts, computed only inside the union of their boxes (fast on large photos)."""
    ba, bb = a_["box"], b_["box"]
    if ba[2] <= bb[0] or bb[2] <= ba[0] or ba[3] <= bb[1] or bb[3] <= ba[1]:
        return 0.0
    x0, y0, x1, y1 = min(ba[0], bb[0]), min(ba[1], bb[1]), max(ba[2], bb[2]) + 1, max(ba[3], bb[3]) + 1
    ma, mb = a_["mask"][y0:y1, x0:x1], b_["mask"][y0:y1, x0:x1]
    inter = (ma & mb).sum()
    return inter / max(1, (ma | mb).sum())


pl = Pipeline(wb=True, hole_min_leaf=0.5, conf=a.conf, agnostic=True)
rows = []
for p in photos:
    t0 = time.time()
    img = np.asarray(Image.open(p).convert("RGB"))
    res, im = pl.predict(img)
    H, W = im.shape[:2]
    n_raw, n_hole_drop = len(res), sum(x["dropped"] for x in res)
    n_nonleaf = sum(1 for x in res if not x["dropped"] and x["leaf_frac"] < a.min_leaf_color)
    cand = sorted([x for x in res if not x["dropped"] and x["leaf_frac"] >= a.min_leaf_color
                   and x["mask"].sum() >= a.min_area], key=lambda x: -x["conf"])
    leaves = []
    for x in cand:                                   # mask-IoU de-duplication (keep higher confidence)
        if all(mask_iou(x, k) <= a.dedup_iou for k in leaves):
            leaves.append(x)
    bg_dropped = []
    for x in leaves:
        x["exg"], x["sat"], x["dark"] = color_stats(im, x["mask"])
    if not a.no_bg_filter:
        bg_dropped = [x for x in leaves if is_background(x["exg"], x["sat"], x["dark"])]
        leaves = [x for x in leaves if not is_background(x["exg"], x["sat"], x["dark"])]
    leaves.sort(key=lambda x: ((x["box"][1] + x["box"][3]) // 2 // 150, (x["box"][0] + x["box"][2]) // 2))  # reading order
    gray = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_32F)
    (out / "crops" / p.stem).mkdir(parents=True, exist_ok=True)
    (out / "context" / p.stem).mkdir(parents=True, exist_ok=True)
    vis = (im.astype(np.float32) * 0.55).astype(np.uint8)
    thumbs = []
    for i, x in enumerate(leaves, 1):
        lid = f"{p.stem}_L{i:04d}"
        m = x["mask"]
        ys, xs = np.nonzero(m)
        x0, y0, x1, y1 = int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
        area = int(m.sum())
        cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        c = max(cnts, key=cv2.contourArea)
        hull = cv2.contourArea(cv2.convexHull(c))
        solidity = float(area / max(1.0, hull))
        sharp = float(lap[m].var())
        border = bool(x0 <= 2 or y0 <= 2 or x1 >= W - 2 or y1 >= H - 2)
        # RGBA cutout (tight box) and context crop
        rgba = np.dstack([im[y0:y1, x0:x1], (m[y0:y1, x0:x1] * 255).astype(np.uint8)])
        Image.fromarray(rgba, "RGBA").save(out / "crops" / p.stem / f"{lid}.png")
        mg = max(40, int(0.4 * max(x1 - x0, y1 - y0)))
        cx0, cy0, cx1, cy1 = max(0, x0 - mg), max(0, y0 - mg), min(W, x1 + mg), min(H, y1 + mg)
        ctx = im[cy0:cy1, cx0:cx1].copy()
        cv2.drawContours(ctx, [c - [cx0, cy0]], -1, (255, 0, 255), 2)
        Image.fromarray(ctx).save(out / "context" / p.stem / f"{lid}.jpg", quality=90)
        # overlay
        col = color(i)
        vis[m] = (im[m].astype(np.float32) * 0.6 + np.array(col, np.float32) * 0.4).astype(np.uint8)
        cv2.drawContours(vis, [c], -1, col, 2)
        t = Image.fromarray(rgba, "RGBA"); t.thumbnail((150, 150))
        thumbs.append((i, t))
        rows.append(dict(id=lid, photo=p.name, idx=i, x0=x0, y0=y0, x1=x1, y1=y1,
                         cx=int(xs.mean()), cy=int(ys.mean()), area_px=area, area_frac=round(area / (H * W), 5),
                         v4_cls=SHORT[x["cls"]], v4_conf=round(x["conf"], 3), leaf_color_frac=round(x["leaf_frac"], 3),
                         sharpness=round(sharp, 1), touches_border=border, solidity=round(solidity, 3),
                         exg=round(x["exg"], 3), sat=round(x["sat"], 3), dark=round(x["dark"], 3)))
    ov = Image.fromarray(vis); d = ImageDraw.Draw(ov)
    for r in rows[-len(leaves):] if leaves else []:
        d.text((r["cx"], r["cy"]), str(r["idx"]), font=F_ID, fill=(255, 255, 255), anchor="mm", stroke_width=3, stroke_fill=(0, 0, 0))
    d.rectangle((8, 8, 760, 60), fill=(0, 0, 0))
    d.text((18, 14), f"v7 leaf search | {p.stem} | leaves {len(leaves)}", font=F_BIG, fill=(255, 255, 255))
    ov.save(out / f"{p.stem}_ids.jpg", quality=90)
    cols = 10; cell = 160
    sheet = Image.new("RGB", (cols * cell, max(1, -(-len(thumbs) // cols)) * (cell + 22)), (128, 128, 128))
    dd = ImageDraw.Draw(sheet)
    for k, (i, t) in enumerate(thumbs):
        cx, cy = (k % cols) * cell, (k // cols) * (cell + 22)
        sheet.paste(t, (cx + (cell - t.width) // 2, cy + 22 + (cell - t.height) // 2), t)
        dd.text((cx + 4, cy + 2), f"{i}", font=F_ID, fill=(255, 255, 0))
    sheet.save(out / f"{p.stem}_cutouts.jpg", quality=88)
    if bg_dropped:                                   # review sheet of background-filter drops
        cells = []
        for x in bg_dropped:
            ys, xs = np.nonzero(x["mask"]); x0, y0, x1, y1 = xs.min(), ys.min(), xs.max() + 1, ys.max() + 1
            mg = max(40, int(0.4 * max(x1 - x0, y1 - y0)))
            cx0, cy0, cx1, cy1 = max(0, x0 - mg), max(0, y0 - mg), min(W, x1 + mg), min(H, y1 + mg)
            ctx = im[cy0:cy1, cx0:cx1].copy()
            cn, _ = cv2.findContours(x["mask"][cy0:cy1, cx0:cx1].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(ctx, cn, -1, (255, 0, 0), 2)
            t = Image.fromarray(ctx); t.thumbnail((200, 200))
            c_ = Image.new("RGB", (210, 236), (20, 20, 20)); c_.paste(t, ((210 - t.width) // 2, 32))
            ImageDraw.Draw(c_).text((4, 4), f"exg{x['exg']:.2f} s{x['sat']:.2f} d{x['dark']:.2f}", font=F_ID, fill=(255, 120, 120))
            cells.append(c_)
        sh = Image.new("RGB", (6 * 210, -(-len(cells) // 6) * 236), (20, 20, 20))
        for k, c_ in enumerate(cells):
            sh.paste(c_, ((k % 6) * 210, (k // 6) * 236))
        sh.save(out / f"{p.stem}_bg_dropped.jpg", quality=88)
    print(f"PHOTO {p.name} {W}x{H} raw={n_raw} hole_dropped={n_hole_drop} nonleaf_dropped={n_nonleaf} bg_dropped={len(bg_dropped)} leaves={len(leaves)} "
          f"merged_suspect(solidity<0.8)={sum(1 for r in rows if r['photo'] == p.name and r['solidity'] < 0.8)} "
          f"border={sum(1 for r in rows if r['photo'] == p.name and r['touches_border'])} {time.time() - t0:.1f}s", flush=True)
if rows:
    with open(out / "leaves.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    (out / "leaves.json").write_text(json.dumps(rows, indent=1, ensure_ascii=False))
print("SAVED", out, "total leaves", len(rows), flush=True)
