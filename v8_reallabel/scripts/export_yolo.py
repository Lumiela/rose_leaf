# v8: export reviewed instance maps to a 1-class YOLO-seg dataset ("leaf").
#   tiles: square, side = TILE_MULT x leaf length L of that photo (so leaves have a similar size in the network input),
#          25% overlap; instances clipped by a tile are kept if the clipped part >= 0.05 L^2 (largest part only)
#   ignore boxes: inside them, leaf-coloured pixels that belong to no instance are painted grey (114) -> no false negatives
#   splits from scenes.json; val/test also get full images + masks (work/<stem>/inst.npz) for tiled evaluation
# Usage: python v8_reallabel/scripts/export_yolo.py [--tile-mult 8] [--only-reviewed]
import argparse, json, shutil  # noqa: E401
from sam_common import load, load_rgb, gray_world, leaf_color, V8, WORK
import numpy as np
import cv2
from PIL import Image

ap = argparse.ArgumentParser(); ap.add_argument("--tile-mult", type=float, default=8.0)
ap.add_argument("--only-reviewed", action="store_true"); ap.add_argument("--out", default=str(V8 / "data" / "leaf_seg"))
a = ap.parse_args()
sc = json.loads((V8 / "scenes.json").read_text())
out = __import__("pathlib").Path(a.out)
shutil.rmtree(out, ignore_errors=True)
stats = {}
for split, stems in sc["split"].items():
    for s in stems:
        if not (WORK / s / "inst.npz").exists():
            print("MISSING", s); continue
        inst, meta = load(s)
        if a.only_reviewed and not meta.get("reviewed"):
            print("SKIP (not reviewed)", s); continue
        img = load_rgb(meta["photo"]).copy(); H, W = inst.shape; L = meta["L"]
        if meta.get("ignore"):
            leaf = leaf_color(gray_world(img))
            for b in meta["ignore"]:
                x0, y0, x1, y1 = [int(v) for v in b]
                sub = (inst[y0:y1, x0:x1] == 0) & leaf[y0:y1, x0:x1]
                sub = cv2.dilate(sub.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool) & (inst[y0:y1, x0:x1] == 0)
                img[y0:y1, x0:x1][sub] = 114
        R = int(min(max(H, W), max(320, a.tile_mult * L))); st = max(1, int(R * 0.75))
        xs = sorted(set(list(range(0, max(1, W - R), st)) + [max(0, W - R)]))
        ys = sorted(set(list(range(0, max(1, H - R), st)) + [max(0, H - R)]))
        ids = [int(i) for i in np.unique(inst) if i]
        n_t = n_i = 0
        for ty in ys:
            for tx in xs:
                x1, y1 = min(W, tx + R), min(H, ty + R)
                sub = inst[ty:y1, tx:x1]
                lines = []
                for i in np.unique(sub):
                    if not i:
                        continue
                    m = (sub == i).astype(np.uint8)
                    cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    c = max(cs, key=cv2.contourArea)
                    if cv2.contourArea(c) < 0.05 * L * L or len(c) < 3:
                        continue
                    c = cv2.approxPolyDP(c, 1.0, True).reshape(-1, 2)
                    if len(c) < 3:
                        continue
                    w, h = x1 - tx, y1 - ty
                    lines.append("0 " + " ".join(f"{x / w:.5f} {y / h:.5f}" for x, y in c))
                name = f"{s}_x{tx}_y{ty}"
                (out / "images" / split).mkdir(parents=True, exist_ok=True); (out / "labels" / split).mkdir(parents=True, exist_ok=True)
                Image.fromarray(img[ty:y1, tx:x1]).save(out / "images" / split / f"{name}.jpg", quality=95)
                (out / "labels" / split / f"{name}.txt").write_text("\n".join(lines))
                n_t += 1; n_i += len(lines)
        if split in ("val", "test"):
            fd = out / "full" / split; fd.mkdir(parents=True, exist_ok=True)
            Image.fromarray(img).save(fd / f"{s}.jpg", quality=95)
            np.savez_compressed(fd / f"{s}.npz", inst=inst, L=L)
        stats[s] = dict(split=split, leaves=len(ids), tiles=n_t, tile_instances=n_i, R=R, L=round(L, 1), reviewed=bool(meta.get("reviewed")))
        print(s, stats[s], flush=True)
(out / "leaf_seg.yaml").write_text(f"path: {out}\ntrain: images/train\nval: images/val\nnames:\n  0: leaf\n")
(out / "stats.json").write_text(json.dumps(stats, indent=1))
print("EXPORTED", out)
