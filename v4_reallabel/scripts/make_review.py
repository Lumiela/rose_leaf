# Step 2: review material for the proposals.
#   review/<photo>_sheet<k>.jpg : 30 numbered crops per sheet (outside the proposal mask dimmed, mask outline cyan,
#                                 header = #id, model class, confidence)
#   review/<photo>_overview.jpg : all proposals outlined with their id on the photo
#   review/<photo>_coverage.jpg : leaf pixels (color mask) not covered by any proposal shown in magenta
# Usage: python v4_reallabel/scripts/make_review.py > v4_reallabel/logs/make_review.log 2>&1
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
import paths  # noqa: E402

import json  # noqa: E402
import numpy as np  # noqa: E402
import cv2  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

sys.path.insert(0, str(paths.V3 / "scripts"))
from stage1 import hsv_leaf_mask, refine_mask  # noqa: E402

PRE, OUT = paths.V4 / "prelabel", paths.V4 / "review"
CELL, COLS, ROWS = 250, 6, 5
SHORT = {n: n.split()[0] for n in paths.CLASS_NAMES}


def font(size):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


F_ID, F_BIG = font(20), font(34)
OUT.mkdir(parents=True, exist_ok=True)
for meta_f in sorted(PRE.glob("*_proposals.json")):
    meta = json.loads(meta_f.read_text())
    stem = meta_f.name.replace("_proposals.json", "")
    img = np.asarray(Image.open(paths.REAL_IMAGES / meta["photo"]).convert("RGB"))
    H, W = img.shape[:2]
    masks = np.load(PRE / f"{stem}.npz")
    props = meta["proposals"]
    full = lambda d: np.pad(masks[f"m{d['id']}"], ((d["box"][1], H - d["box"][3]), (d["box"][0], W - d["box"][2])))

    # crop sheets
    per = COLS * ROWS
    for k in range(0, len(props), per):
        sheet = Image.new("RGB", (COLS * CELL, ROWS * CELL), (20, 20, 20))
        for n, d in enumerate(props[k:k + per]):
            x0, y0, x1, y1 = d["box"]
            mg = max(30, int(0.3 * max(x1 - x0, y1 - y0)))
            cx0, cy0, cx1, cy1 = max(0, x0 - mg), max(0, y0 - mg), min(W, x1 + mg), min(H, y1 + mg)
            crop = img[cy0:cy1, cx0:cx1].astype(np.float32)
            m = np.zeros(crop.shape[:2], bool)
            m[y0 - cy0:y1 - cy0, x0 - cx0:x1 - cx0] = masks[f"m{d['id']}"]
            crop[~m] *= 0.4
            crop = crop.astype(np.uint8)
            cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(crop, cnts, -1, (0, 255, 255), max(1, round(max(crop.shape) / 150)))
            t = Image.fromarray(crop); t.thumbnail((CELL - 10, CELL - 34))
            cx, cy = (n % COLS) * CELL, (n // COLS) * CELL
            sheet.paste(t, (cx + (CELL - t.width) // 2, cy + 30 + (CELL - 34 - t.height) // 2))
            dr = ImageDraw.Draw(sheet)
            dr.text((cx + 6, cy + 4), f"#{d['id']} {SHORT[d['model_cls']]} {d['conf']:.2f}", font=F_ID, fill=(255, 255, 0))
        sheet.save(OUT / f"{stem}_sheet{k // per + 1}.jpg", quality=90)

    # overview with ids
    ov = img.copy()
    for d in props:
        cnts, _ = cv2.findContours(full(d).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(ov, cnts, -1, (0, 255, 255), 3)
    ovi = Image.fromarray(ov); dr = ImageDraw.Draw(ovi)
    for d in props:
        x0, y0, x1, y1 = d["box"]
        dr.text(((x0 + x1) // 2, (y0 + y1) // 2), str(d["id"]), font=F_BIG, fill=(255, 0, 255), anchor="mm",
                stroke_width=3, stroke_fill=(0, 0, 0))
    ovi.save(OUT / f"{stem}_overview.jpg", quality=88)

    # coverage: color-mask leaf pixels not covered by proposals
    leaf = refine_mask(hsv_leaf_mask(img), close_px=0.004 * max(H, W), dilate_px=0, fill=True).astype(bool)
    union = np.zeros((H, W), bool)
    for d in props:
        union[d["box"][1]:d["box"][3], d["box"][0]:d["box"][2]] |= masks[f"m{d['id']}"]
    miss = leaf & ~union
    cv_ = img.astype(np.float32); cv_[union] *= 0.5; cv_[miss] = (255, 0, 255)
    Image.fromarray(cv_.astype(np.uint8)).save(OUT / f"{stem}_coverage.jpg", quality=88)
    print("REVIEW", stem, "proposals", len(props), "sheets", -(-len(props) // per),
          "leaf_px_uncovered %.1f%%" % (100 * miss.sum() / max(1, leaf.sum())), flush=True)
