# Step 1: leaf mask proposals on the real photos with ft_from_v1 (draft labels to be reviewed).
# Per photo: full-image prediction (conf >= 0.10). If leaves look small (median proposal side < 70 px at full res)
# 640 px tiles are added (conf >= 0.15; a tile keeps only instances whose box center is in its core region).
# All proposals are merged greedily by confidence (mask IoU > 0.5 -> duplicate, dropped).
# Output: prelabel/<photo>.npz (masks cropped to their boxes) + prelabel/<photo>_proposals.json
# Usage: python v4_reallabel/scripts/prelabel.py > v4_reallabel/logs/prelabel.log 2>&1
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
import paths  # noqa: E402  shared paths + opencv workaround

import json  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402
from ultralytics import YOLO  # noqa: E402

OUT = paths.V4 / "prelabel"
CONF_FULL, CONF_TILE, TILE, OVERLAP, SMALL_SIDE = 0.10, 0.15, 640, 0.25, 70


def tiles(W, H, T, overlap):
    """Overlapping T x T windows over the image, each with its core region."""
    step = int(T * (1 - overlap))
    starts = lambda n: [0] if n <= T else sorted(set(list(range(0, n - T, step)) + [n - T]))
    xs, ys = starts(W), starts(H)
    out = []
    for i, ty in enumerate(ys):
        for j, tx in enumerate(xs):
            cx0 = 0 if j == 0 else (tx + xs[j - 1] + T) // 2
            cx1 = W if j == len(xs) - 1 else (xs[j + 1] + tx + T) // 2
            cy0 = 0 if i == 0 else (ty + ys[i - 1] + T) // 2
            cy1 = H if i == len(ys) - 1 else (ys[i + 1] + ty + T) // 2
            out.append(((tx, ty, min(tx + T, W), min(ty + T, H)), (cx0, cy0, cx1, cy1)))
    return out


def predict(model, rgb, conf, x0=0, y0=0, core=None):
    r = model.predict(Image.fromarray(rgb), imgsz=640, conf=conf, device=0, retina_masks=True, verbose=False)[0]
    out = []
    if r.masks is None:
        return out
    for m, c, s in zip(r.masks.data.cpu().numpy() > 0.5, r.boxes.cls.cpu().numpy().astype(int), r.boxes.conf.cpu().numpy()):
        ys, xs = np.nonzero(m)
        if len(xs) < 50:
            continue
        bx0, by0, bx1, by1 = xs.min(), ys.min(), xs.max() + 1, ys.max() + 1
        if core is not None:
            cx, cy = x0 + (bx0 + bx1) / 2, y0 + (by0 + by1) / 2
            if not (core[0] <= cx < core[2] and core[1] <= cy < core[3]):
                continue
        out.append(dict(box=[int(x0 + bx0), int(y0 + by0), int(x0 + bx1), int(y0 + by1)],
                        mask=m[by0:by1, bx0:bx1].copy(), cls=int(c), conf=float(s)))
    return out


def iou(a, b):
    x0, y0 = max(a["box"][0], b["box"][0]), max(a["box"][1], b["box"][1])
    x1, y1 = min(a["box"][2], b["box"][2]), min(a["box"][3], b["box"][3])
    if x1 <= x0 or y1 <= y0:
        return 0.0
    ma = a["mask"][y0 - a["box"][1]:y1 - a["box"][1], x0 - a["box"][0]:x1 - a["box"][0]]
    mb = b["mask"][y0 - b["box"][1]:y1 - b["box"][1], x0 - b["box"][0]:x1 - b["box"][0]]
    inter = (ma & mb).sum()
    return inter / (a["mask"].sum() + b["mask"].sum() - inter)


model = YOLO(str(paths.V2_FT_BEST))
OUT.mkdir(parents=True, exist_ok=True)
for p in sorted(paths.REAL_IMAGES.glob("*.webp")):
    img = np.asarray(Image.open(p).convert("RGB"))
    H, W = img.shape[:2]
    props = [dict(d, src="full") for d in predict(model, img, CONF_FULL)]
    med_side = float(np.median([np.sqrt(d["mask"].sum()) for d in props])) if props else 0
    use_tiles = med_side < SMALL_SIDE
    if use_tiles:
        for (x0, y0, x1, y1), core in tiles(W, H, TILE, OVERLAP):
            props += [dict(d, src="tile") for d in predict(model, np.ascontiguousarray(img[y0:y1, x0:x1]), CONF_TILE, x0, y0, core)]
    props.sort(key=lambda d: -d["conf"])
    kept = []
    for d in props:
        if all(iou(d, k) <= 0.5 for k in kept):
            kept.append(d)
    kept.sort(key=lambda d: (d["box"][1] // 200, d["box"][0]))       # reading order for review sheets
    np.savez_compressed(OUT / f"{p.stem}.npz", **{f"m{i}": d["mask"] for i, d in enumerate(kept)})
    meta = [dict(id=i, box=d["box"], model_cls=paths.CLASS_NAMES[d["cls"]], conf=round(d["conf"], 3), src=d["src"],
                 area=int(d["mask"].sum())) for i, d in enumerate(kept)]
    (OUT / f"{p.stem}_proposals.json").write_text(json.dumps(dict(photo=p.name, size=[W, H], median_side=round(med_side, 1),
                                                                  tiles=use_tiles, proposals=meta), indent=1))
    print("PRELABEL", p.stem, "raw", len(props), "kept", len(kept), "median_side", round(med_side, 1), "tiles", use_tiles, flush=True)
