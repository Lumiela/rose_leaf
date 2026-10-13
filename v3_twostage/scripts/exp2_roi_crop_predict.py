# Experiment 2: does stage 1 (plant ROIs from the color mask) + full-resolution crops reduce the errors of the
# existing v2 model (ft_from_v1) on the real photos?  No retraining: ft_from_v1 is used as the stage-2 model.
# Modes per photo:
#   baseline       : whole photo resized to 640 (current single-stage use), scale ~0.32
#   tiles800       : plant ROIs split into 800 px tiles (25% overlap), each predicted at 640 (scale 0.8)
#   tiles640       : same with 640 px tiles (scale 1.0)
#   tiles800_gray  : tiles800, pixels outside the (dilated) leaf mask filled with gray 114
# Tiles with < 1% leaf pixels are skipped. Overlap duplicates are removed by keeping an instance only in the tile
# whose core region (its share of the overlap) contains the instance's box center.
# A first run cropped whole plant ROIs; on these photos the ROIs merged into ~the full frame (no resolution gain),
# so tiling was added.
# Output: one comparison sheet per photo (stage-1 view + 3 modes), all classes and disease-only, plus summary.json.
# Usage: python v3_twostage/scripts/exp2_roi_crop_predict.py > v3_twostage/logs/exp2.log 2>&1
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
import paths  # noqa: E402  shared paths + opencv workaround (must come before cv2/ultralytics)

import json  # noqa: E402
from collections import Counter  # noqa: E402
import numpy as np  # noqa: E402
import cv2  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402
from ultralytics import YOLO  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stage1 import hsv_leaf_mask, refine_mask, plant_rois  # noqa: E402

OUT = paths.V3 / "results" / "exp2_roi_crop"
WEIGHTS = paths.V2_FT_BEST
CONF, IMGSZ, GRAY, TILE = 0.25, 640, 114, 620
NAMES = paths.CLASS_NAMES
SHORT = [n.split()[0] for n in NAMES]
COLORS = [(255, 40, 40), (40, 220, 40), (30, 120, 255), (255, 215, 0)]
FRESH = 1


def font(size):
    for f in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        if Path(f).exists():
            return ImageFont.truetype(f, size)
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


F_TITLE, F_ROW = font(30), font(26)


def tiles(roi, T, overlap):
    """Overlapping T x T windows covering roi, each with its core region (window shrunk to the midpoints of overlaps)."""
    x0, y0, x1, y1 = roi
    step = int(T * (1 - overlap))
    def starts(a, b):
        if b - a <= T:
            return [a]
        s = list(range(a, b - T, step)) + [b - T]
        return sorted(set(s))
    xs, ys = starts(x0, x1), starts(y0, y1)
    out = []
    for i, ty in enumerate(ys):
        for j, tx in enumerate(xs):
            cx0 = x0 if j == 0 else (tx + xs[j - 1] + T) // 2
            cx1 = x1 if j == len(xs) - 1 else (xs[j + 1] + tx + T) // 2
            cy0 = y0 if i == 0 else (ty + ys[i - 1] + T) // 2
            cy1 = y1 if i == len(ys) - 1 else (ys[i + 1] + ty + T) // 2
            out.append(((tx, ty, min(tx + T, x1), min(ty + T, y1)), (cx0, cy0, cx1, cy1)))
    return out


def predict(model, rgb):
    """-> list of (mask bool HxW of rgb, cls, conf)"""
    r = model.predict(Image.fromarray(rgb), imgsz=IMGSZ, conf=CONF, device=0, retina_masks=True, verbose=False)[0]
    if r.masks is None:
        return []
    ms = r.masks.data.cpu().numpy() > 0.5
    return list(zip(ms, r.boxes.cls.cpu().numpy().astype(int).tolist(), r.boxes.conf.cpu().numpy().tolist()))


def render(img, insts, hide_fresh, leaf=None, rois=()):
    im = img.astype(np.float32)
    H, W = im.shape[:2]
    thick = max(2, round(max(H, W) / 600))
    if leaf is not None:                         # stage-1 view: non-leaf area darkened, ROI boxes in white
        im[leaf == 0] *= 0.35
    order = sorted(range(len(insts)), key=lambda i: insts[i][3] != FRESH)
    for i in order:
        y0, x0, m, c, _ = insts[i]
        if hide_fresh and c == FRESH:
            continue
        sub = im[y0:y0 + m.shape[0], x0:x0 + m.shape[1]]
        sub[m] = sub[m] * 0.6 + np.array(COLORS[c], np.float32) * 0.4
    out = im.clip(0, 255).astype(np.uint8)
    for i in order:
        y0, x0, m, c, _ = insts[i]
        if hide_fresh and c == FRESH:
            continue
        cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE, offset=(x0, y0))
        cv2.drawContours(out, cnts, -1, COLORS[c], thick)
    for (a, b, c2, d) in rois:
        cv2.rectangle(out, (a, b), (c2, d), (255, 255, 255), thick * 2)
    return out


def legend(tile, title, lines):
    d = ImageDraw.Draw(tile, "RGBA")
    lh = 34
    d.rectangle((8, 8, 8 + 400, 8 + 48 + lh * len(lines) + 8), fill=(0, 0, 0, 175))
    d.text((18, 14), title, font=F_TITLE, fill=(255, 255, 255))
    for k, (color, txt, dim) in enumerate(lines):
        y = 8 + 52 + lh * k
        if color is not None:
            d.rectangle((18, y + 4, 42, y + 28), outline=color, width=3, fill=None if dim else color + (110,))
        d.text((52, y + 2), txt, font=F_ROW, fill=(160, 160, 160) if dim else (255, 255, 255))


model = YOLO(str(WEIGHTS))
OUT.mkdir(parents=True, exist_ok=True)
summary = {}
for p in sorted(paths.REAL_IMAGES.glob("*.webp")):
    img = np.asarray(Image.open(p).convert("RGB"))
    H, W = img.shape[:2]
    S = max(H, W)
    raw = hsv_leaf_mask(img)
    leaf = refine_mask(raw, close_px=0.004 * S, dilate_px=0.0075 * S, fill=True)
    rois = plant_rois(leaf, merge_px=0.02 * S, min_area_frac=0.01, margin=0.15)
    modes = {"baseline": [(0, 0, m, c, s) for m, c, s in predict(model, img)]}
    n_tiles = {}
    for mode, T, gray in (("tiles800", 800, False), ("tiles640", 640, False), ("tiles800_gray", 800, True)):
        insts, used = [], 0
        for roi in rois:
            for (x0, y0, x1, y1), core in tiles(roi, T, 0.25):
                if leaf[y0:y1, x0:x1].mean() < 0.01:
                    continue
                used += 1
                crop = img[y0:y1, x0:x1].copy()
                if gray:
                    crop[leaf[y0:y1, x0:x1] == 0] = GRAY
                for m, c, s in predict(model, np.ascontiguousarray(crop)):
                    ys, xs = np.nonzero(m)
                    cy, cx = y0 + (ys.min() + ys.max()) / 2, x0 + (xs.min() + xs.max()) / 2
                    if core[0] <= cx < core[2] and core[1] <= cy < core[3]:
                        insts.append((y0, x0, m, c, s))
        modes[mode] = insts
        n_tiles[mode] = used
    stats = {"rois": [list(map(int, r)) for r in rois],
             "tiles_used": n_tiles,
             "baseline_scale": round(IMGSZ / S, 2), "leaf_mask_frac": round(float(leaf.mean()), 3)}
    for mode, insts in modes.items():
        cnt = Counter(SHORT[c] for *_, c, _ in insts)
        outside = 0
        for y0, x0, m, c, _ in insts:            # instances lying mostly outside the stage-1 leaf mask
            lm = leaf[y0:y0 + m.shape[0], x0:x0 + m.shape[1]].astype(bool)
            outside += int((m & lm).sum() < 0.5 * max(1, m.sum()))
        area = [int(m.sum() * 1.0) for _, _, m, _, _ in insts]
        stats[mode] = dict(n=len(insts), counts=dict(cnt), outside_leaf_mask=outside,
                           median_area_px=int(np.median(area)) if area else 0)
        print("RES", p.stem, mode, stats[mode], flush=True)
    print("ROI", p.stem, stats["rois"], "tiles", n_tiles, "baseline_scale", stats["baseline_scale"], flush=True)
    summary[p.stem] = stats

    for hide_fresh, tag in ((False, "all"), (True, "disease")):
        panels = []
        t = Image.fromarray(render(img, [], False, leaf=leaf, rois=rois)); t.thumbnail((TILE, TILE), Image.LANCZOS)
        legend(t, "stage1: leaf mask + ROI", [(None, f"ROIs: {len(rois)}", False),
                                               (None, f"leaf area: {leaf.mean():.0%}", False)])
        panels.append(t)
        for mode, insts in modes.items():
            cnt = Counter(c for *_, c, _ in insts)
            t = Image.fromarray(render(img, insts, hide_fresh)); t.thumbnail((TILE, TILE), Image.LANCZOS)
            legend(t, mode, [(COLORS[c], f"{SHORT[c]}: {cnt.get(c, 0)}" + ("  (hidden)" if hide_fresh and c == FRESH else ""),
                              hide_fresh and c == FRESH) for c in range(len(NAMES))])
            panels.append(t)
        Wt = sum(t.width for t in panels) + 8 * (len(panels) - 1)
        sheet = Image.new("RGB", (Wt, max(t.height for t in panels)), (17, 17, 17))
        x = 0
        for t in panels:
            sheet.paste(t, (x, 0)); x += t.width + 8
        sheet.save(OUT / f"{p.stem}_{tag}.jpg", quality=90)

(OUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
print("SAVED", OUT, flush=True)
