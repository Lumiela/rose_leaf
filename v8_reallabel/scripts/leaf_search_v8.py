# v8 stage 1 for the v7 pipeline: find every leaf with the 1-class v8 leaf model and export v7-compatible results
# (leaves.json/csv, crops/<photo>/<ID>.png RGBA, context/, <photo>_ids.jpg) so v7_leafid/scripts/classify_leaves.py runs unchanged.
#   leaf length L: estimated per photo from a first pass on the whole photo (long side 1280) = median box long side
#   tiled inference: tile = tile_mult x L, 25% overlap; masks cut by inner tile edges dropped; duplicates (mask IoU > 0.5) merged
#   then the same post-processing as v7: min area, background colour filter (exg/sat/dark), reading-order IDs
# Usage: python v8_reallabel/scripts/leaf_search_v8.py <image or folder> --out <dir> [--weights W] [--conf 0.35]
#        [--tile-mult 8] [--L px (skip estimation)] [--min-area 400] [--no-bg-filter]
import argparse, csv, json, time, colorsys  # noqa: E401
from pathlib import Path
from sam_common import V8, load_rgb
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".jfif")
ap = argparse.ArgumentParser(); ap.add_argument("src"); ap.add_argument("--out", required=True)
ap.add_argument("--weights", default=str(V8 / "runs" / "r2_fromv4" / "weights" / "best.pt"))
ap.add_argument("--conf", type=float, default=0.35); ap.add_argument("--tile-mult", type=float, default=8.0)
ap.add_argument("--L", type=float); ap.add_argument("--min-area", type=int, default=400)
ap.add_argument("--no-bg-filter", action="store_true")
a = ap.parse_args()
from ultralytics import YOLO  # noqa: E402
model = YOLO(a.weights)
src = Path(a.src).resolve()
photos = [src] if src.is_file() else sorted(p for p in src.iterdir() if p.suffix.lower() in EXTS)
out = Path(a.out).resolve(); out.mkdir(parents=True, exist_ok=True)
try:
    F_ID, F_BIG = ImageFont.load_default(size=22), ImageFont.load_default(size=36)
except TypeError:
    F_ID = F_BIG = ImageFont.load_default()


def color(i):
    r, g, b = colorsys.hsv_to_rgb((i * 0.618034) % 1.0, 0.85, 1.0)
    return int(r * 255), int(g * 255), int(b * 255)


def color_stats(im, m):  # same as v7 leaf_search (on the gray-world image)
    rgb = im[m].astype(np.float32) / 255.0
    mx, mn = rgb.max(1), rgb.min(1)
    sat = np.where(mx > 0, (mx - mn) / np.maximum(mx, 1e-6), 0)
    exg = (2 * rgb[:, 1] - rgb[:, 0] - rgb[:, 2]) / np.maximum(rgb.sum(1), 1e-6)
    return float(np.median(exg)), float(np.median(sat)), float((mx < 0.2).mean())


def is_background(exg, sat, dark):
    return exg < 0.12 or (dark > 0.25 and exg < 0.25) or (sat < 0.26 and exg < 0.15)


def gray_world(img):
    f = img.astype(np.float32); m = f.reshape(-1, 3).mean(0); f *= m.mean() / np.maximum(m, 1)
    return np.clip(f, 0, 255).astype(np.uint8)


def estimate_L(img):
    H, W = img.shape[:2]
    r = model.predict(np.ascontiguousarray(img[:, :, ::-1]), imgsz=1280, conf=a.conf, verbose=False)[0]
    if r.boxes is None or len(r.boxes) < 5:
        return 90.0
    b = r.boxes.xyxy.cpu().numpy()
    return float(np.median(np.maximum(b[:, 2] - b[:, 0], b[:, 3] - b[:, 1])))


def predict(img, L):
    H, W = img.shape[:2]
    R = int(min(max(H, W), max(320, a.tile_mult * L))); st = max(1, int(R * 0.75))
    xs = sorted(set(list(range(0, max(1, W - R), st)) + [max(0, W - R)]))
    ys = sorted(set(list(range(0, max(1, H - R), st)) + [max(0, H - R)]))
    preds = []
    for ty in ys:
        for tx in xs:
            x1, y1 = min(W, tx + R), min(H, ty + R)
            r = model.predict(np.ascontiguousarray(img[ty:y1, tx:x1, ::-1]), imgsz=640, conf=a.conf, retina_masks=True, verbose=False)[0]
            if r.masks is None:
                continue
            for m, c in zip(r.masks.data.cpu().numpy() > 0.5, r.boxes.conf.cpu().numpy()):
                if m.shape != (y1 - ty, x1 - tx):
                    m = cv2.resize(m.astype(np.uint8), (x1 - tx, y1 - ty), interpolation=cv2.INTER_NEAREST).astype(bool)
                ys_, xs_ = np.nonzero(m)
                if len(xs_) < 20:
                    continue
                e = 2
                if (xs_.min() <= e and tx > 0) or (ys_.min() <= e and ty > 0) or (xs_.max() >= x1 - tx - 1 - e and x1 < W) or (ys_.max() >= y1 - ty - 1 - e and y1 < H):
                    continue
                b = (tx + xs_.min(), ty + ys_.min(), tx + xs_.max() + 1, ty + ys_.max() + 1)
                preds.append((float(c), b, m[b[1] - ty:b[3] - ty, b[0] - tx:b[2] - tx].copy()))
    preds.sort(key=lambda t: -t[0])
    keep = []
    for c, b, m in preds:
        dup = False
        for c2, b2, m2 in keep:
            ix0, iy0, ix1, iy1 = max(b[0], b2[0]), max(b[1], b2[1]), min(b[2], b2[2]), min(b[3], b2[3])
            if ix0 >= ix1 or iy0 >= iy1:
                continue
            inter = (m[iy0 - b[1]:iy1 - b[1], ix0 - b[0]:ix1 - b[0]] & m2[iy0 - b2[1]:iy1 - b2[1], ix0 - b2[0]:ix1 - b2[0]]).sum()
            if inter and inter / (m.sum() + m2.sum() - inter) > 0.5:
                dup = True; break
        if not dup:
            keep.append((c, b, m))
    return keep


rows = []
for p in photos:
    t0 = time.time()
    raw = load_rgb(p); im = gray_world(raw); H, W = raw.shape[:2]
    L = a.L or estimate_L(raw)
    P = predict(raw, L)
    n_raw = len(P)
    P = [x for x in P if x[2].sum() >= a.min_area]
    leaves = []
    for c, b, m in P:
        sub = im[b[1]:b[3], b[0]:b[2]]
        exg, sat, dark = color_stats(sub, m)
        leaves.append(dict(conf=c, box=b, m=m, exg=exg, sat=sat, dark=dark))
    bg = [x for x in leaves if is_background(x["exg"], x["sat"], x["dark"])] if not a.no_bg_filter else []
    if bg:
        leaves = [x for x in leaves if not is_background(x["exg"], x["sat"], x["dark"])]
    leaves.sort(key=lambda x: ((x["box"][1] + x["box"][3]) // 2 // 150, (x["box"][0] + x["box"][2]) // 2))
    lap = cv2.Laplacian(cv2.cvtColor(im, cv2.COLOR_RGB2GRAY), cv2.CV_32F)
    (out / "crops" / p.stem).mkdir(parents=True, exist_ok=True); (out / "context" / p.stem).mkdir(parents=True, exist_ok=True)
    vis = (im.astype(np.float32) * 0.55).astype(np.uint8)
    n0 = len(rows)
    for i, x in enumerate(leaves, 1):
        lid = f"{p.stem}_L{i:04d}"
        x0, y0, x1, y1 = [int(v) for v in x["box"]]; m = x["m"]
        area = int(m.sum())
        cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        c = max(cnts, key=cv2.contourArea)
        solidity = float(area / max(1.0, cv2.contourArea(cv2.convexHull(c))))
        sharp = float(lap[y0:y1, x0:x1][m].var())
        border = bool(x0 <= 2 or y0 <= 2 or x1 >= W - 2 or y1 >= H - 2)
        rgba = np.dstack([im[y0:y1, x0:x1], (m * 255).astype(np.uint8)])
        Image.fromarray(rgba, "RGBA").save(out / "crops" / p.stem / f"{lid}.png")
        mg = max(40, int(0.4 * max(x1 - x0, y1 - y0)))
        cx0, cy0, cx1, cy1 = max(0, x0 - mg), max(0, y0 - mg), min(W, x1 + mg), min(H, y1 + mg)
        ctx = im[cy0:cy1, cx0:cx1].copy()
        cv2.drawContours(ctx, [c + [x0 - cx0, y0 - cy0]], -1, (255, 0, 255), 2)
        Image.fromarray(ctx).save(out / "context" / p.stem / f"{lid}.jpg", quality=90)
        col = color(i)
        cv2.drawContours(vis, [c + [x0, y0]], -1, col, 2)
        ys, xs = np.nonzero(m)
        rows.append(dict(id=lid, photo=p.name, idx=i, x0=x0, y0=y0, x1=x1, y1=y1, cx=int(xs.mean()) + x0, cy=int(ys.mean()) + y0,
                         area_px=area, area_frac=round(area / (H * W), 5), v4_cls="leaf", v4_conf=round(x["conf"], 3), leaf_color_frac=None,
                         sharpness=round(sharp, 1), touches_border=border, solidity=round(solidity, 3),
                         exg=round(x["exg"], 3), sat=round(x["sat"], 3), dark=round(x["dark"], 3), L=round(L, 1)))
    ov = Image.fromarray(vis); d = ImageDraw.Draw(ov)
    for r in rows[n0:]:
        d.text((r["cx"], r["cy"]), str(r["idx"]), font=F_ID, fill=(255, 255, 255), anchor="mm", stroke_width=3, stroke_fill=(0, 0, 0))
    d.rectangle((8, 8, 900, 60), fill=(0, 0, 0)); d.text((18, 14), f"v8 leaf search | {p.stem} | leaves {len(leaves)} | L {L:.0f}", font=F_BIG, fill=(255, 255, 255))
    ov.save(out / f"{p.stem}_ids.jpg", quality=90)
    print(f"PHOTO {p.name} {W}x{H} L={L:.0f} raw={n_raw} bg_dropped={len(bg)} leaves={len(leaves)} "
          f"merged_suspect(solidity<0.8)={sum(1 for r in rows[n0:] if r['solidity'] < 0.8)} border={sum(1 for r in rows[n0:] if r['touches_border'])} {time.time() - t0:.1f}s", flush=True)
if rows:
    with open(out / "leaves.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    (out / "leaves.json").write_text(json.dumps(rows, indent=1, ensure_ascii=False))
print("SAVED", out, "total leaves", len(rows), flush=True)
