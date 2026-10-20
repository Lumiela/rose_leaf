# v8 review renderer: overview + zoomed tiles with instance outlines, IDs, coordinate ticks and uncovered leaf pixels.
#   magenta tint = leaf-coloured pixels not covered by any instance (possible missed leaf)
#   red outline/label = flagged instance (solidity<0.85, big, small, colour, multi-part); hatched grey = ignore box
# Output: v8_reallabel/review/<stem>/overview.jpg, t<NN>.jpg, tiles.json, props.json
# Usage: python v8_reallabel/scripts/review.py <stem> [--tile-mult 7]
import argparse, json, shutil  # noqa: E401
from sam_common import load, load_rgb, gray_world, leaf_color, props, REVIEW
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

ap = argparse.ArgumentParser(); ap.add_argument("stem"); ap.add_argument("--tile-mult", type=float, default=7.0)
a = ap.parse_args()
inst, meta = load(a.stem)
img = load_rgb(meta["photo"]); H, W = img.shape[:2]; L = meta["L"]
leaf = leaf_color(gray_world(img))
P = props(inst, leaf, L)
unc = leaf & (inst == 0)
unc = cv2.morphologyEx(unc.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)).astype(bool)
out = REVIEW / a.stem
shutil.rmtree(out, ignore_errors=True); out.mkdir(parents=True)
rng = np.random.default_rng(7)
PAL = [tuple(int(c) for c in rng.integers(60, 256, 3)) for _ in range(4096)]


def font(s):
    try:
        return ImageFont.load_default(size=s)
    except TypeError:
        return ImageFont.load_default()


def render(x0, y0, x1, y1, S, label_ids=True, margin=26):
    crop = img[y0:y1, x0:x1].copy()
    u = unc[y0:y1, x0:x1]
    crop[u] = (0.5 * crop[u] + 0.5 * np.array([255, 0, 255])).astype(np.uint8)
    sc = S / max(x1 - x0, y1 - y0)
    w, h = int(round((x1 - x0) * sc)), int(round((y1 - y0) * sc))
    big = cv2.resize(crop, (w, h), interpolation=cv2.INTER_CUBIC if sc > 1 else cv2.INTER_AREA)
    sub = inst[y0:y1, x0:x1]
    subr = cv2.resize(sub, (w, h), interpolation=cv2.INTER_NEAREST)
    labels = []
    for i in np.unique(subr):
        if not i:
            continue
        p = P[int(i)]
        bad = bool(p["flags"])
        cs, _ = cv2.findContours((subr == i).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(big, cs, -1, (255, 0, 0) if bad else PAL[int(i) % 4096], 3 if bad else 2)
        if label_ids and x0 <= p["cx"] < x1 and y0 <= p["cy"] < y1:
            labels.append(((p["cx"] - x0) * sc, (p["cy"] - y0) * sc, str(int(i)), bad))
    for b in meta.get("ignore", []):
        bx0, by0, bx1, by1 = [int(round((v - o) * sc)) for v, o in zip(b, (x0, y0, x0, y0))]
        ov = big.copy(); cv2.rectangle(ov, (bx0, by0), (bx1, by1), (128, 128, 128), -1)
        big = cv2.addWeighted(ov, 0.5, big, 0.5, 0)
    canvas = Image.new("RGB", (w + margin, h + margin), (0, 0, 0))
    canvas.paste(Image.fromarray(big), (margin, margin))
    d = ImageDraw.Draw(canvas); f = font(12); fl = font(15)
    step = 50 if (x1 - x0) <= 600 else 100
    for v in range((x0 // step + 1) * step, x1, step):
        X = margin + int((v - x0) * sc); d.line([(X, margin - 6), (X, margin)], fill=(255, 255, 255)); d.text((X - 12, 2), str(v), font=f, fill=(255, 255, 0))
    for v in range((y0 // step + 1) * step, y1, step):
        Y = margin + int((v - y0) * sc); d.line([(margin - 6, Y), (margin, Y)], fill=(255, 255, 255)); d.text((0, Y - 6), str(v)[-4:], font=f, fill=(255, 255, 0))
    for x, y, t, bad in labels:
        d.text((margin + x - 8, margin + y - 8), t, font=fl, fill=(255, 60, 60) if bad else (255, 255, 255), stroke_width=3, stroke_fill=(0, 0, 0))
    return canvas


# overview with tile grid
R = int(np.clip(a.tile_mult * L, 240, 900)); stp = max(1, int(R * 0.85))
xs = list(range(0, max(1, W - R), stp)) + [max(0, W - R)]
ys = list(range(0, max(1, H - R), stp)) + [max(0, H - R)]
xs, ys = sorted(set(xs)), sorted(set(ys))
tiles = []
for r, ty in enumerate(ys):
    for c, tx in enumerate(xs):
        x1, y1 = min(W, tx + R), min(H, ty + R)
        sub = inst[ty:y1, tx:x1]
        ids = [int(i) for i in np.unique(sub) if i]
        fl = [i for i in ids if P[i]["flags"] and tx <= P[i]["cx"] < x1 and ty <= P[i]["cy"] < y1]
        tiles.append(dict(t=len(tiles), x0=tx, y0=ty, x1=x1, y1=y1, n=len(ids), flagged=fl, uncovered_px=int(unc[ty:y1, tx:x1].sum())))
ov = render(0, 0, W, H, 1300, label_ids=False)
d = ImageDraw.Draw(ov); sc = 1300 / max(W, H)
for t in tiles:
    X0, Y0 = 26 + t["x0"] * sc, 26 + t["y0"] * sc
    d.rectangle([X0, Y0, 26 + t["x1"] * sc, 26 + t["y1"] * sc], outline=(0, 255, 255), width=1)
    d.text((X0 + 4, Y0 + 2), "t%d" % t["t"], font=font(18), fill=(0, 255, 255), stroke_width=2, stroke_fill=(0, 0, 0))
ov.save(out / "overview.jpg", quality=88)
for t in tiles:
    im = render(t["x0"], t["y0"], t["x1"], t["y1"], 640)
    ImageDraw.Draw(im).text((30, 30), "%s t%d n=%d flag=%d" % (a.stem, t["t"], t["n"], len(t["flagged"])), font=font(14), fill=(0, 255, 255), stroke_width=2, stroke_fill=(0, 0, 0))
    im.save(out / ("t%02d.jpg" % t["t"]), quality=90)
(out / "tiles.json").write_text(json.dumps(tiles, indent=1))
(out / "props.json").write_text(json.dumps({str(k): v for k, v in P.items()}, indent=1))
nf = sum(1 for p in P.values() if p["flags"])
print("REVIEW %s: instances=%d flagged=%d tiles=%d R=%d uncovered_px=%d" % (a.stem, len(P), nf, len(tiles), R, int(unc.sum())))
print("FLAGGED", {k: v["flags"] for k, v in P.items() if v["flags"]})
