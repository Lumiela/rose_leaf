# v8 stage-2 data: v7 recipe (bbox_real_aug) with the REAL part rebuilt from v8 stage-1 crops.
#   synthetic part : hard links to v7_leafid/cls/data_bbox/{train,val,test,test_deg} (Dataset cutouts pasted on cube patches)
#   real healthy   : every judged leaf (solidity >= 0.8, not touching the frame) of the v8 stage-1 run on the v8 split photos
#                    train -> train/Fresh (clean + degraded copy), val -> val/Fresh, test -> test_real/Fresh
#                    (user, 2026-10-15: plants are healthy; the 13 'Yellow' calls on test photos were all healthy)
#   real anomalies : one synthetic anomaly per real leaf (Hole -> Yellow -> Black cycling, v7 synth_real_anom painters,
#                    painted inside the v8 mask): train -> train/<class> (50% degraded), val -> val/<class>, test -> test_real_anom/<class>
# Crops = v7 classifier input: gray world (cls_common), square 1.25 x leaf long side around the box centre, 224 px.
# Usage: python v8_reallabel/cls/scripts/build_cls_v8.py [--run v8_reallabel/pipeline/v8_stage1_nobg]
import os, sys, json, shutil, argparse  # noqa: E401
from pathlib import Path
MODEL = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(MODEL / "v7_leafid" / "cls" / "scripts"))
from cls_common import NAMES, BOX_SCALE, gray_world, square_crop, to_input, paths  # noqa: E402  (opencv workaround first)
from build_real_data import degrade  # noqa: E402
from synth_real_anom import FN, sq  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

ap = argparse.ArgumentParser(); ap.add_argument("--run", default=str(MODEL / "v8_reallabel" / "pipeline" / "v8_stage1_nobg"))
a = ap.parse_args()
RUN = Path(a.run); V8 = MODEL / "v8_reallabel"; OUT = V8 / "cls" / "data"
SRC_SYN = MODEL / "v7_leafid" / "cls" / "data_bbox"
FRESH = NAMES[1]
TYPES = [(2, "Hole"), (3, "Yellow"), (0, "Black")]
if OUT.exists():
    shutil.rmtree(OUT)
n_syn = 0
for p in SRC_SYN.rglob("*.jpg"):
    q = OUT / p.relative_to(SRC_SYN); q.parent.mkdir(parents=True, exist_ok=True); os.link(p, q); n_syn += 1
for s in ("train", "val", "test_real", "test_real_anom"):
    for c in NAMES:
        (OUT / s / c).mkdir(parents=True, exist_ok=True)
split_of = {}
for sp, stems in json.loads((V8 / "scenes.json").read_text())["split"].items():
    for s in stems:
        split_of[s] = sp
rows = json.loads((RUN / "leaves.json").read_text())
cnt = {}
meta = []
k = 0
for photo in sorted({r["photo"] for r in rows}):
    stem = Path(photo).stem; sp = split_of.get(stem)
    if sp is None:
        continue
    raw = np.asarray(Image.open(V8 / "inputs" / photo).convert("RGB")); img = gray_world(raw); H, W = img.shape[:2]
    for r in [r for r in rows if r["photo"] == photo]:
        if r["solidity"] < 0.8 or r["touches_border"]:
            continue
        L = max(r["x1"] - r["x0"], r["y1"] - r["y0"]); side = BOX_SCALE * L
        cx, cy = (r["x0"] + r["x1"]) / 2, (r["y0"] + r["y1"]) / 2
        clean = to_input(square_crop(img, cx, cy, side))
        rng = np.random.default_rng([31, k]); k += 1
        if sp == "train":
            clean.save(OUT / "train" / FRESH / f"v8real_{r['id']}_c.jpg", quality=95)
            degrade(clean, side, rng).save(OUT / "train" / FRESH / f"v8real_{r['id']}_d.jpg", quality=95)
        elif sp == "val":
            clean.save(OUT / "val" / FRESH / f"v8real_{r['id']}_c.jpg", quality=95)
        else:
            clean.save(OUT / "test_real" / FRESH / f"v8real_{r['id']}.jpg", quality=95)
        cnt[(sp, "Fresh")] = cnt.get((sp, "Fresh"), 0) + 1
        a_ = np.asarray(Image.open(RUN / "crops" / stem / f"{r['id']}.png").getchannel("A")) > 127
        full = np.zeros((H, W), bool); full[r["y0"]:r["y0"] + a_.shape[0], r["x0"]:r["x0"] + a_.shape[1]] = a_[:H - r["y0"], :W - r["x0"]]
        S, M = sq(img, cx, cy, side, 128), sq(full, cx, cy, side, False)
        if M.sum() < 30:
            continue
        ci, cname = TYPES[k % 3]
        im = to_input(Image.fromarray(FN[cname](S, M, L, rng)))
        dst = {"train": "train", "val": "val", "test": "test_real_anom"}[sp]
        if dst == "train" and rng.random() < 0.5:
            im = degrade(im, side, rng)
        im.save(OUT / dst / NAMES[ci] / f"v8anom_{r['id']}.jpg", quality=95)
        cnt[(dst, cname)] = cnt.get((dst, cname), 0) + 1
        meta.append(dict(id=r["id"], photo=photo, split=sp, L=L, sharpness=r["sharpness"], solidity=r["solidity"], anomaly=cname))
    print(photo, sp, flush=True)
(V8 / "cls" / "real_leaves.json").write_text(json.dumps(meta, indent=1))
print("synthetic linked", n_syn)
print("real:", {f"{s}/{c}": n for (s, c), n in sorted(cnt.items())})
for s in ("train", "val", "test", "test_deg", "test_real", "test_real_anom"):
    print(s, {c.split()[0]: len(list((OUT / s / c).glob("*.jpg"))) for c in NAMES})
print("BUILD_DONE")
