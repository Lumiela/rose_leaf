# Step 1 of "add real healthy leaves": candidate real leaves for the bbox_real variant.
# Sources: stage-1 leaf search on the phone photos (runs/bg_cube, 18 photos) + rose_real, rose_real2 (runs/bg_real;
# rose_real3 stays held out). The camfix photos are never used (they are the ground-truth check, cls/gt/).
#   - judged leaves only (solidity >= 0.8, not on the border), blurry and partial leaves are KEPT (45% of camfix leaves
#     are blurry and they carry most false alarms)
#   - photo split (seed 42): phone photos -> test 2 / val 2 / train 14; rose_real 1,2 -> train
#   - bbox crop exactly like export_bbox.py (gray-world, 1.25 x long side square, 224) -> cls/real/crops/<photo>/<id>.jpg
#   - current best model (runs_bbox/effnetv2s) predicts each crop: only used to ORDER the review, never shown on sheets
#   - review sheets (context crop with the leaf outline, numbered, no predictions):
#       A = train-photo leaves the model calls non-Fresh (hard cases, all of them)
#       T = every leaf of the val/test photos (needed for the held-out real metric A4)
#       B = random sample of train-photo leaves the model calls Fresh, so that real Fresh <= synthetic Fresh / 2 leaves
# Output: cls/real/candidates.json, cls/real/sheets/{A,T,B}_XX.jpg, cls/real/sheet_index.json
# Usage: CLS_VARIANT=bbox python v7_leafid/cls/scripts/real_prep.py
import os, sys  # noqa: E401
os.environ.setdefault("CLS_VARIANT", "bbox")
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from cls_common import CLS, NAMES, SHORT, BOX_SCALE, gray_world, square_crop, to_input, paths  # noqa: E402

import json  # noqa: E402
import numpy as np  # noqa: E402
import torch, torch.nn as nn  # noqa: E401,E402
from torchvision import transforms as T  # noqa: E402
from torchvision.models import efficientnet_v2_s  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

OUT = CLS / "real"
RUNS_SRC = {"bg_cube": paths.V7 / "runs" / "bg_cube", "bg_real": paths.V7 / "runs" / "bg_real"}
SRC_IMG = {"bg_cube": paths.REAL_IMAGES_CUBE, "bg_real": paths.REAL_IMAGES}
SKIP = {"rose_real3.webp"}
COLS, ROWS, TILE, HDR = 6, 5, 210, 22


def sheets(rows, prefix, src_of):
    try:
        F = ImageFont.load_default(size=18)
    except TypeError:
        F = ImageFont.load_default()
    per, index = COLS * ROWS, []
    for s in range(0, len(rows), per):
        sh = Image.new("RGB", (COLS * TILE, ROWS * (TILE + HDR)), (30, 30, 30)); d = ImageDraw.Draw(sh)
        for k, r in enumerate(rows[s:s + per]):
            n = s + k + 1
            im = Image.open(src_of(r)).convert("RGB"); im.thumbnail((TILE - 4, TILE - 4))
            x, y = k % COLS * TILE, k // COLS * (TILE + HDR)
            sh.paste(im, (x + (TILE - im.width) // 2, y + HDR + (TILE - im.height) // 2))
            d.text((x + 4, y + 2), f"{prefix}{n}  {r['id'][-14:]}", font=F, fill=(255, 255, 0))
            index.append(dict(key=f"{prefix}{n}", id=r["id"]))
        sh.save(OUT / "sheets" / f"{prefix}_{s // per + 1:02d}.jpg", quality=92)
    return index


if __name__ == "__main__":
    (OUT / "sheets").mkdir(parents=True, exist_ok=True)
    rows = []
    for src, run in RUNS_SRC.items():
        for r in json.loads((run / "leaves.json").read_text()):
            if r["photo"] in SKIP or r["solidity"] < 0.8 or r["touches_border"]:
                continue
            r["src"] = src; rows.append(r)
    cube = sorted({r["photo"] for r in rows if r["src"] == "bg_cube"})
    perm = list(np.random.default_rng(42).permutation(cube))
    split = {p: "test" for p in perm[:2]} | {p: "val" for p in perm[2:4]} | {p: "train" for p in perm[4:]}
    split |= {p: "train" for p in {r["photo"] for r in rows if r["src"] == "bg_real"}}
    print("split:", {s: sorted(p for p in split if split[p] == s) for s in ("train", "val", "test")}, flush=True)

    # bbox crops (same recipe as export_bbox.py)
    for photo in sorted({r["photo"] for r in rows}):
        rs = [r for r in rows if r["photo"] == photo]
        img = gray_world(np.asarray(Image.open(SRC_IMG[rs[0]["src"]] / photo).convert("RGB")))
        od = OUT / "crops" / Path(photo).stem; od.mkdir(parents=True, exist_ok=True)
        for r in rs:
            side = BOX_SCALE * max(r["x1"] - r["x0"], r["y1"] - r["y0"])
            to_input(square_crop(img, (r["x0"] + r["x1"]) / 2, (r["y0"] + r["y1"]) / 2, side)).save(od / f"{r['id']}.jpg", quality=95)
            r["crop"] = str((od / f"{r['id']}.jpg").relative_to(CLS)); r["split"] = split[photo]
            r["long_side"] = max(r["x1"] - r["x0"], r["y1"] - r["y0"])
    print(f"crops: {len(rows)} judged leaves from {len(split)} photos", flush=True)

    # current best model -> review order only
    model = efficientnet_v2_s(); model.classifier[1] = nn.Linear(model.classifier[1].in_features, 4)
    model.load_state_dict(torch.load(CLS / "runs_bbox" / "effnetv2s" / "best.pt", map_location="cpu")); model = model.cuda().eval()
    tf = T.Compose([T.ToTensor(), T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    with torch.no_grad():
        for i in range(0, len(rows), 128):
            x = torch.stack([tf(Image.open(CLS / r["crop"]).convert("RGB")) for r in rows[i:i + 128]]).cuda()
            pr = model(x).float().softmax(1).cpu()
            for r, p in zip(rows[i:i + 128], pr):
                r["pred"], r["conf"] = SHORT[int(p.argmax())], round(float(p.max()), 3)

    synth_fresh = len(list((CLS / "data_bbox" / "train" / NAMES[1]).glob("*.jpg")))
    cap = synth_fresh // 2
    A = [r for r in rows if r["split"] == "train" and r["pred"] != "Fresh"]
    Tset = [r for r in rows if r["split"] in ("val", "test")]
    pool = [r for r in rows if r["split"] == "train" and r["pred"] == "Fresh"]
    nb = max(0, min(len(pool), cap - len(A)))
    B = [pool[i] for i in sorted(np.random.default_rng(0).choice(len(pool), nb, replace=False))]
    ctx = lambda r: RUNS_SRC[r["src"]] / "context" / Path(r["photo"]).stem / f"{r['id']}.jpg"  # noqa: E731
    rng = np.random.default_rng(1)
    shuf = lambda xs: [xs[i] for i in rng.permutation(len(xs))]  # noqa: E731  (mix photos on a sheet)
    index = sheets(shuf(A), "A", ctx) + sheets(sorted(Tset, key=lambda r: r["id"]), "T", ctx) + sheets(shuf(B), "B", ctx)
    for r in rows:
        r.pop("src_path", None)
    (OUT / "candidates.json").write_text(json.dumps(rows, indent=0))
    (OUT / "sheet_index.json").write_text(json.dumps(index, indent=0))
    print(f"synthetic Fresh train images {synth_fresh} -> real leaf cap {cap}", flush=True)
    print(f"review sheets: A {len(A)} (train, model non-Fresh) | T {len(Tset)} (val/test photos) | B {len(B)} of {len(pool)} (train, model Fresh)", flush=True)
    print("pred counts:", {s: {c: sum(1 for r in rows if r['split'] == s and r['pred'] == c) for c in SHORT} for s in ('train', 'val', 'test')}, flush=True)
