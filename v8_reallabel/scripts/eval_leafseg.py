# v8: evaluate a leaf instance-segmentation model on full-resolution real photos (tiled inference).
#   GT = work/<stem>/inst.npz (+ meta ignore boxes). Tiles: side = tile_mult x L (L = leaf length of the photo).
#   Predictions from all YOLO classes are treated as "leaf" (so the 4-class v4 model can be compared).
#   Per tile, masks touching an inner tile edge are dropped; overlapping duplicates across tiles merged (mask IoU > 0.5).
#   Matching: greedy by confidence, mask IoU >= 0.5. A prediction is IGNORED (not FP) when >= 50% of it lies on
#   unlabeled pixels inside an ignore box. merged = prediction that contains >= 50% of two or more GT leaves.
# Usage: python v8_reallabel/scripts/eval_leafseg.py <weights> <name> [--split test] [--tile-mult 8] [--imgsz 640] [--conf 0.25]
import argparse, json, time  # noqa: E401
from sam_common import load, load_rgb, V8, WORK
import numpy as np
import cv2
from PIL import Image, ImageDraw

ap = argparse.ArgumentParser(); ap.add_argument("weights"); ap.add_argument("name")
ap.add_argument("--split", default="test"); ap.add_argument("--tile-mult", type=float, default=8.0)
ap.add_argument("--imgsz", type=int, default=640); ap.add_argument("--conf", type=float, default=0.25)
a = ap.parse_args()
from ultralytics import YOLO  # noqa: E402
model = YOLO(a.weights)
sc = json.loads((V8 / "scenes.json").read_text())
out = V8 / "eval" / a.name; out.mkdir(parents=True, exist_ok=True)


def predict(img, L):
    H, W = img.shape[:2]
    R = int(min(max(H, W), max(320, a.tile_mult * L))); st = max(1, int(R * 0.75))
    xs = sorted(set(list(range(0, max(1, W - R), st)) + [max(0, W - R)]))
    ys = sorted(set(list(range(0, max(1, H - R), st)) + [max(0, H - R)]))
    preds = []
    for ty in ys:
        for tx in xs:
            x1, y1 = min(W, tx + R), min(H, ty + R)
            r = model.predict(np.ascontiguousarray(img[ty:y1, tx:x1, ::-1]), imgsz=a.imgsz, conf=a.conf, retina_masks=True, verbose=False, agnostic_nms=True)[0]
            if r.masks is None:
                continue
            M = r.masks.data.cpu().numpy() > 0.5; C = r.boxes.conf.cpu().numpy()
            for m, c in zip(M, C):
                if m.shape != (y1 - ty, x1 - tx):
                    m = cv2.resize(m.astype(np.uint8), (x1 - tx, y1 - ty), interpolation=cv2.INTER_NEAREST).astype(bool)
                ys_, xs_ = np.nonzero(m)
                if len(xs_) < 20:
                    continue
                e = 2
                if (xs_.min() <= e and tx > 0) or (ys_.min() <= e and ty > 0) or (xs_.max() >= x1 - tx - 1 - e and x1 < W) or (ys_.max() >= y1 - ty - 1 - e and y1 < H):
                    continue  # cut by an inner tile edge -> another tile sees it whole
                bx = (tx + xs_.min(), ty + ys_.min(), tx + xs_.max() + 1, ty + ys_.max() + 1)
                crop = m[bx[1] - ty:bx[3] - ty, bx[0] - tx:bx[2] - tx]
                preds.append((float(c), bx, crop))
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


tot = dict(gt=0, pred=0, tp=0, fp=0, ignored=0, merged=0)
per = {}
t0 = time.time()
for s in sc["split"][a.split]:
    inst, meta = load(s); img = load_rgb(meta["photo"]); H, W = inst.shape; L = meta["L"]
    ign = np.zeros((H, W), bool)
    for b in meta.get("ignore", []):
        x0, y0, x1, y1 = [int(v) for v in b]; ign[y0:y1, x0:x1] = True
    ign &= inst == 0
    gt_ids = [int(i) for i in np.unique(inst) if i]
    gt_area = {i: int((inst == i).sum()) for i in gt_ids}
    P = predict(img, L)
    matched_gt = set(); tp = fp = ig = mg = 0
    vis = (img * 0.6).astype(np.uint8)
    for c, b, mc in P:
        sub_inst = inst[b[1]:b[3], b[0]:b[2]]
        ids, cnt = np.unique(sub_inst[mc], return_counts=True)
        msum = int(mc.sum())
        best, biou = None, 0.0
        for i, n in zip(ids, cnt):
            if i == 0 or int(i) in matched_gt:
                continue
            iou = n / (msum + gt_area[int(i)] - n)
            if iou > biou:
                best, biou = int(i), iou
        covers = sum(1 for i, n in zip(ids, cnt) if i and n >= 0.5 * gt_area[int(i)])
        if covers >= 2:
            mg += 1
        if best is not None and biou >= 0.5:
            matched_gt.add(best); tp += 1; col = (0, 255, 0)
        elif ign[b[1]:b[3], b[0]:b[2]][mc].mean() >= 0.5:
            ig += 1; col = (160, 160, 160)
        else:
            fp += 1; col = (0, 140, 255)
        cs, _ = cv2.findContours(mc.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE, offset=(int(b[0]), int(b[1])))
        cv2.drawContours(vis, cs, -1, col, 2)
    for i in gt_ids:
        if i not in matched_gt:
            cs, _ = cv2.findContours((inst == i).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(vis, cs, -1, (255, 0, 0), 2)
    n_gt = len(gt_ids)
    per[s] = dict(gt=n_gt, pred=len(P), tp=tp, fp=fp, ignored=ig, merged=mg,
                  recall=round(tp / max(1, n_gt), 3), precision=round(tp / max(1, tp + fp), 3))
    for k in ("gt", "tp", "fp", "ignored", "merged"):
        tot[k] += per[s][k]
    tot["pred"] += len(P)
    im = Image.fromarray(vis); d = ImageDraw.Draw(im)
    d.text((10, 10), f"{a.name} {s}: GT {n_gt} TP {tp} FP {fp} ign {ig} merged {mg} | green=TP blue=FP red=missed grey=ignored", fill=(255, 255, 255))
    im.save(out / f"{s}.jpg", quality=88)
    print(s, per[s], flush=True)
tot["recall"] = round(tot["tp"] / max(1, tot["gt"]), 3); tot["precision"] = round(tot["tp"] / max(1, tot["tp"] + tot["fp"]), 3)
tot["f1"] = round(2 * tot["recall"] * tot["precision"] / max(1e-9, tot["recall"] + tot["precision"]), 3)
(out / "summary.json").write_text(json.dumps(dict(weights=a.weights, split=a.split, tile_mult=a.tile_mult, imgsz=a.imgsz, conf=a.conf, total=tot, per_photo=per), indent=1))
print("TOTAL", tot, f"{time.time() - t0:.0f}s")
