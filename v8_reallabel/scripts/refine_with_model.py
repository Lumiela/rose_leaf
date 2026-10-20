# v8 round 2: improve auto labels with a trained leaf model (model-in-the-loop). Reviewed photos are skipped.
#   replace: label instance that contains >= 50% of two or more model leaves which together cover >= 70% of it (merged)
#            -> replaced by those model leaves (clipped to the old instance)
#   add    : model leaf whose pixels are >= 60% unlabeled (and >= 0.1 L^2) -> new instance on the unlabeled part
# Predictions: tiled (tile = tile_mult x L), masks cut by inner tile edges dropped, duplicates (mask IoU > 0.5) merged.
# Previous state goes to work/<stem>/history (undo: apply_edits.py <stem> undo).
# Usage: python v8_reallabel/scripts/refine_with_model.py <weights> [stems ...] [--conf 0.4] [--dry]
import argparse, json, time  # noqa: E401
from sam_common import load, save, load_rgb, V8
import numpy as np
import cv2

ap = argparse.ArgumentParser(); ap.add_argument("weights"); ap.add_argument("stems", nargs="*")
ap.add_argument("--conf", type=float, default=0.4); ap.add_argument("--tile-mult", type=float, default=8.0)
ap.add_argument("--dry", action="store_true")
a = ap.parse_args()
from ultralytics import YOLO  # noqa: E402
model = YOLO(a.weights)


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
    return [(b, m) for _, b, m in keep]


sc = json.loads((V8 / "scenes.json").read_text())
stems = a.stems or (sc["split"]["train"] + sc["split"]["val"])
tot = dict(added=0, replaced=0)
for s in stems:
    inst, meta = load(s)
    if meta.get("reviewed"):
        print("SKIP reviewed", s, flush=True); continue
    t0 = time.time()
    img = load_rgb(meta["photo"]); L = meta["L"]
    P = predict(img, L)
    area = {int(i): int(n) for i, n in zip(*np.unique(inst, return_counts=True)) if i}
    by_label = {}
    for k, (b, m) in enumerate(P):
        ids, cnt = np.unique(inst[b[1]:b[3], b[0]:b[2]][m], return_counts=True)
        for i, n in zip(ids, cnt):
            if i and n >= 0.5 * m.sum():
                by_label.setdefault(int(i), []).append((k, int(n)))
    n_rep = 0
    nid = int(inst.max())
    for i, lst in by_label.items():
        if len(lst) >= 2 and sum(n for _, n in lst) >= 0.7 * area[i]:
            ys, xs = np.nonzero(inst == i)
            region = np.zeros(inst.shape, bool); region[ys, xs] = True
            inst[region] = 0; meta["ids"].pop(str(i), None)
            for k, _ in lst:
                b, m = P[k]
                sub = np.zeros(inst.shape, bool); sub[b[1]:b[3], b[0]:b[2]] = m
                part = sub & region & (inst == 0)
                if part.sum() < 0.05 * L * L:
                    continue
                nid += 1; inst[part] = nid; meta["ids"][str(nid)] = dict(src="model_split")
            n_rep += 1
    n_add = 0
    for b, m in P:
        sub = inst[b[1]:b[3], b[0]:b[2]]
        free = m & (sub == 0)
        if free.sum() >= 0.6 * m.sum() and free.sum() >= 0.1 * L * L:
            nid += 1; sub[free] = nid; meta["ids"][str(nid)] = dict(src="model_add"); n_add += 1
    meta["log"].append(dict(t=time.strftime("%Y-%m-%d %H:%M:%S"), ops=[dict(op="refine_with_model", weights=a.weights, added=n_add, replaced=n_rep)]))
    if not a.dry:
        save(s, inst, meta, tag="refine_" + time.strftime("%H%M%S"))
    tot["added"] += n_add; tot["replaced"] += n_rep
    print(f"{s}: model={len(P)} added={n_add} replaced_merged={n_rep} instances={len(np.unique(inst)) - 1} {time.time() - t0:.0f}s", flush=True)
print("REFINE_DONE", tot)
