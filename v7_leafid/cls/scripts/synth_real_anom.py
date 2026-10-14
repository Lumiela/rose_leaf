# Shortcut countermeasure for bbox_real: synthetic anomalies painted on REAL healthy cube leaves, so "looks like a real
# cube photo" no longer means "Fresh". Each reviewed healthy leaf (cls/real/real_labels.json, tag F) gets ONE anomaly,
# cycling Hole -> Yellow -> Black, painted at native resolution inside its stage-1 mask (runs/bg_*/crops/<photo>/<id>.png,
# alpha = mask, tight box at x0,y0), then the usual bbox square (1.25 x long side) -> 224:
#   Hole   1-3 ellipses (r 6-15% of leaf) inside the eroded mask, filled with background texture from the same square
#          (50%), wall-like light gray (25%) or soil-like dark (25%); brown rim 60%
#   Yellow hue pushed to yellow on the whole leaf (40%) or a soft band from one side covering ~30-80% (60%)
#   Black  2-8 dark brown/black blobs (r 3-8%), soft edges, yellow halo 50%
# Outputs (CLS_VARIANT=bbox_real_aug -> data_bbox_real_aug, hard-linked copy of data_bbox_real):
#   train photos -> train/<class>/ranom_<id>.jpg (50% also degraded like the other real crops)
#   val photos   -> val/<class>/ranom_<id>.jpg
#   test photos  -> test_real_anom/<class>/ranom_<id>.jpg  (B4: does the model still see anomalies on real leaves?)
#   check sheet  -> cls/real/anom_check.jpg
# Usage: CLS_VARIANT=bbox_real_aug python v7_leafid/cls/scripts/synth_real_anom.py
import os, sys  # noqa: E401
os.environ.setdefault("CLS_VARIANT", "bbox_real_aug")
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from cls_common import CLS, DATA, NAMES, BOX_SCALE, Progress, gray_world, to_input, paths  # noqa: E402
from build_real_data import degrade  # noqa: E402

import json, os as _os, shutil  # noqa: E401,E402
import cv2  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

SRC_DATA = CLS / "data_bbox_real"
RUNS_SRC = {"bg_cube": paths.V7 / "runs" / "bg_cube", "bg_real": paths.V7 / "runs" / "bg_real"}
SRC_IMG = {"bg_cube": paths.REAL_IMAGES_CUBE, "bg_real": paths.REAL_IMAGES}
TYPES = [(2, "Hole"), (3, "Yellow"), (0, "Black")]


def sq(arr, cx, cy, side, fill):
    H, W = arr.shape[:2]
    side = max(4, int(round(side)))
    x0, y0 = int(round(cx - side / 2)), int(round(cy - side / 2))
    out = np.empty((side, side) + arr.shape[2:], arr.dtype); out[:] = fill
    sx0, sy0, sx1, sy1 = max(0, x0), max(0, y0), min(W, x0 + side), min(H, y0 + side)
    if sx1 > sx0 and sy1 > sy0:
        out[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = arr[sy0:sy1, sx0:sx1]
    return out


def ellipse(shape, c, r, rng):
    m = np.zeros(shape, np.uint8)
    cv2.ellipse(m, (int(c[1]), int(c[0])), (max(1, int(r * rng.uniform(0.6, 1.4))), max(1, int(r * rng.uniform(0.6, 1.4)))),
                rng.uniform(0, 180), 0, 360, 1, -1)
    return m.astype(bool)


def pick(mask, rng):
    ys, xs = np.nonzero(mask)
    k = rng.integers(len(ys)); return ys[k], xs[k]


def hole(img, M, L, rng):
    out = img.astype(np.float32)
    inner = cv2.erode(M.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=max(1, int(0.12 * L))).astype(bool)
    if inner.sum() < 10:
        inner = M
    bg = ~M
    for _ in range(rng.integers(1, 4)):
        h = ellipse(M.shape, pick(inner, rng), rng.uniform(0.06, 0.15) * L, rng) & M
        if not h.any():
            continue
        mode = rng.random()
        if mode < 0.5 and bg.sum() > 50:
            by, bx = pick(bg, rng); cy, cx = np.argwhere(h).mean(0)
            sh = np.roll(np.roll(img, int(cy - by), 0), int(cx - bx), 1).astype(np.float32)
            out[h] = sh[h]
        elif mode < 0.75:
            out[h] = rng.uniform(195, 240) + rng.normal(0, 6, (h.sum(), 1))
        else:
            out[h] = np.array([rng.uniform(35, 70), rng.uniform(28, 55), rng.uniform(18, 40)]) + rng.normal(0, 5, (h.sum(), 1))
        if rng.random() < 0.6:
            ring = cv2.dilate(h.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=max(1, int(0.02 * L))).astype(bool) & ~h & M
            out[ring] = out[ring] * 0.4 + np.array([115, 85, 45]) * 0.6
    return np.clip(out, 0, 255).astype(np.uint8)


def yellow(img, M, L, rng):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)
    if rng.random() < 0.4:
        w = np.ones(M.shape, np.float32)
    else:
        yy, xx = np.mgrid[:M.shape[0], :M.shape[1]].astype(np.float32)
        a = rng.uniform(0, 2 * np.pi); proj = xx * np.cos(a) + yy * np.sin(a)
        v = proj[M]; t = np.quantile(v, 1 - rng.uniform(0.3, 0.8)); soft = max(1.0, 0.15 * L)
        w = np.clip((proj - t) / soft, 0, 1)
    w = cv2.GaussianBlur(w, (0, 0), max(0.5, 0.03 * L)) * M * rng.uniform(0.75, 1.0)
    hsv[..., 0] = hsv[..., 0] * (1 - w) + rng.uniform(22, 30) * w
    hsv[..., 1] = np.clip(hsv[..., 1] * (1 + 0.25 * w), 0, 255)
    hsv[..., 2] = np.clip(hsv[..., 2] * (1 + 0.15 * w), 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)


def black(img, M, L, rng):
    out = img.astype(np.float32)
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)
    halo = rng.random() < 0.5
    for _ in range(rng.integers(2, 9)):
        r = rng.uniform(0.03, 0.08) * L; c = pick(M, rng)
        b = cv2.GaussianBlur(ellipse(M.shape, c, r, rng).astype(np.float32), (0, 0), max(0.5, r / 3)) * M
        if halo:
            hb = cv2.GaussianBlur(ellipse(M.shape, c, 1.7 * r, rng).astype(np.float32), (0, 0), max(0.5, r / 2)) * M * 0.6
            hsv[..., 0] = hsv[..., 0] * (1 - hb) + 26 * hb
            out = out * (1 - hb[..., None]) + cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32) * hb[..., None]
        col = np.array([rng.uniform(20, 50), rng.uniform(15, 40), rng.uniform(10, 30)])
        out = out * (1 - b[..., None]) + col * b[..., None]
    return np.clip(out, 0, 255).astype(np.uint8)


FN = {"Hole": hole, "Yellow": yellow, "Black": black}

if __name__ == "__main__":
    Progress.log("synth_anom", "start", "running")
    if DATA.exists():
        shutil.rmtree(DATA)
    for p in SRC_DATA.rglob("*.jpg"):
        q = DATA / p.relative_to(SRC_DATA); q.parent.mkdir(parents=True, exist_ok=True); _os.link(p, q)
    for c in NAMES:
        (DATA / "test_real_anom" / c).mkdir(parents=True, exist_ok=True)
    lab = json.loads((CLS / "real" / "real_labels.json").read_text())["labels"]
    cand = [r for r in json.loads((CLS / "real" / "candidates.json").read_text()) if lab.get(r["id"]) == "F"]
    cnt, sheet, k = {}, [], 0
    for photo in sorted({r["photo"] for r in cand}):
        rs = [r for r in cand if r["photo"] == photo]
        img = gray_world(np.asarray(Image.open(SRC_IMG[rs[0]["src"]] / photo).convert("RGB")))
        H, W = img.shape[:2]
        for r in rs:
            ci, cname = TYPES[k % 3]; k += 1
            rng = np.random.default_rng([23, k])
            a = np.asarray(Image.open(RUNS_SRC[r["src"]] / "crops" / Path(photo).stem / f"{r['id']}.png").getchannel("A")) > 127
            full = np.zeros((H, W), bool); full[r["y0"]:r["y1"], r["x0"]:r["x1"]] = a
            L = max(r["x1"] - r["x0"], r["y1"] - r["y0"]); side = BOX_SCALE * L
            cx, cy = (r["x0"] + r["x1"]) / 2, (r["y0"] + r["y1"]) / 2
            S, M = sq(img, cx, cy, side, 128), sq(full, cx, cy, side, False)
            if M.sum() < 30:
                continue
            im = to_input(Image.fromarray(FN[cname](S, M, L, rng)))
            split = {"train": "train", "val": "val", "test": "test_real_anom"}[r["split"]]
            if split == "train" and rng.random() < 0.5:
                im = degrade(im, side, rng)
            im.save(DATA / split / NAMES[ci] / f"ranom_{r['id']}.jpg", quality=95)
            cnt[(split, cname)] = cnt.get((split, cname), 0) + 1
            if len(sheet) < 48 and k % 7 == 0:
                sheet.append((Image.open(CLS / r["crop"]).convert("RGB"), im, cname))
        Progress.log("synth_anom", f"{photo}: {len(rs)} leaves")
    t = 150; sh = Image.new("RGB", (8 * t, ((len(sheet) + 3) // 4) * t), (30, 30, 30))
    for i, (a, b, _) in enumerate(sheet):
        sh.paste(a.resize((t, t)), ((i % 4) * 2 * t, (i // 4) * t)); sh.paste(b.resize((t, t)), ((i % 4) * 2 * t + t, (i // 4) * t))
    sh.save(CLS / "real" / "anom_check.jpg", quality=90)
    Progress.log("synth_anom", "done: " + ", ".join(f"{s}/{c} {n}" for (s, c), n in sorted(cnt.items())) + " (check: real/anom_check.jpg)", "done")
