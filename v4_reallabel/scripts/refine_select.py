# Step 3: refine proposal masks and score them, so only clean single-leaf masks go to review as cutout candidates.
# Refinement: mask AND (color leaf mask dilated 2 px) -> largest component -> fill holes -> light opening.
# Scores: keep (refined/original area), solidity (area / convex hull), straight (share of contour on long
# axis-aligned straight runs = blocky mask), side (sqrt area, px).
# Candidate rule: keep >= 0.80, solidity >= 0.85, straight <= 0.25, side >= 40.
# (first try solidity 0.90 / straight 0.12 gave only 22 candidates: model masks are often clipped by their box,
#  which adds straight edges even to single leaves; the reviewer rejects truncated leaves)
# Output: prelabel/<photo>_refined.npz, prelabel/<photo>_scores.json, review/<photo>_cand<k>.jpg (35 per sheet)
# Usage: python v4_reallabel/scripts/refine_select.py > v4_reallabel/logs/refine_select.log 2>&1
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
import paths  # noqa: E402

import json  # noqa: E402
import numpy as np  # noqa: E402
import cv2  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

sys.path.insert(0, str(paths.V3 / "scripts"))
from stage1 import hsv_leaf_mask, fill_holes  # noqa: E402

PRE, OUT = paths.V4 / "prelabel", paths.V4 / "review"
RULE = dict(keep=0.80, solidity=0.85, straight=0.25, side=40)
CELL, COLS, ROWS = 250, 7, 5


def font(size):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


F_ID = font(19)


def straight_share(m):
    cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return 1.0
    c = max(cnts, key=len)[:, 0, :]
    run, best, total = 0, 0, 0
    d = np.diff(np.vstack([c, c[:1]]), axis=0)
    axis = (d[:, 0] == 0) | (d[:, 1] == 0)
    # count contour points that belong to axis-aligned runs of >= 12 px in the same direction
    same = np.r_[False, (d[1:] == d[:-1]).all(1)] & axis
    straight = 0
    for v in same:
        run = run + 1 if v else 0
        if run == 12:
            straight += 12
        elif run > 12:
            straight += 1
    return straight / max(1, len(c))


for meta_f in sorted(PRE.glob("*_proposals.json")):
    meta = json.loads(meta_f.read_text())
    stem = meta_f.name.replace("_proposals.json", "")
    img = np.asarray(Image.open(paths.REAL_IMAGES / meta["photo"]).convert("RGB"))
    leaf = cv2.dilate(hsv_leaf_mask(img), np.ones((5, 5), np.uint8)).astype(bool)
    masks = np.load(PRE / f"{stem}.npz")
    refined, scores = {}, []
    for d in meta["proposals"]:
        x0, y0, x1, y1 = d["box"]
        m0 = masks[f"m{d['id']}"]
        m = m0 & leaf[y0:y1, x0:x1]
        n, lab, st, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), 8)
        if n > 1:
            m = lab == (1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA])))
        m = fill_holes(m.astype(np.uint8))
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)).astype(bool)
        area = int(m.sum())
        hull = cv2.convexHull(cv2.findNonZero(m.astype(np.uint8))) if area else None
        sol = area / max(1.0, cv2.contourArea(hull)) if area else 0.0
        sc = dict(id=d["id"], keep=round(area / max(1, int(m0.sum())), 3), solidity=round(sol, 3),
                  straight=round(straight_share(m), 3), side=round(float(np.sqrt(area)), 1),
                  model_cls=d["model_cls"], conf=d["conf"])
        sc["candidate"] = bool(sc["keep"] >= RULE["keep"] and sc["solidity"] >= RULE["solidity"]
                               and sc["straight"] <= RULE["straight"] and sc["side"] >= RULE["side"])
        refined[f"m{d['id']}"] = m
        scores.append(sc)
    np.savez_compressed(PRE / f"{stem}_refined.npz", **refined)
    (PRE / f"{stem}_scores.json").write_text(json.dumps(dict(rule=RULE, scores=scores), indent=1))
    cands = [s for s in scores if s["candidate"]]
    per = COLS * ROWS
    for k in range(0, len(cands), per):
        sheet = Image.new("RGB", (COLS * CELL, ROWS * CELL), (20, 20, 20))
        for n_, s in enumerate(cands[k:k + per]):
            d = meta["proposals"][s["id"]]
            x0, y0, x1, y1 = d["box"]
            m = refined[f"m{s['id']}"]
            crop = img[y0:y1, x0:x1].astype(np.float32)
            crop[~m] = crop[~m] * 0.25 + 40
            crop = crop.astype(np.uint8)
            cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(crop, cnts, -1, (0, 255, 255), max(1, round(max(crop.shape) / 120)))
            t = Image.fromarray(crop); t.thumbnail((CELL - 10, CELL - 32))
            if max(t.size) < 120:                       # upscale small leaves for readability
                f = 120 / max(t.size); t = t.resize((round(t.width * f), round(t.height * f)), Image.NEAREST)
            cx, cy = (n_ % COLS) * CELL, (n_ // COLS) * CELL
            sheet.paste(t, (cx + (CELL - t.width) // 2, cy + 28 + (CELL - 30 - t.height) // 2))
            ImageDraw.Draw(sheet).text((cx + 5, cy + 4), f"#{s['id']} {s['model_cls'].split()[0]} {int(s['side'])}px",
                                       font=F_ID, fill=(255, 255, 0))
        sheet.save(OUT / f"{stem}_cand{k // per + 1}.jpg", quality=90)
    print("REFINE", stem, "proposals", len(scores), "candidates", len(cands),
          "model_cls_of_candidates", {c: sum(1 for s in cands if s["model_cls"] == c) for c in paths.CLASS_NAMES}, flush=True)
