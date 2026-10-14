# v7 stage 2: classify every stage-1 leaf ID with the stage-2 classifier and draw the result on the photo.
#   input : a leaf_search run (leaves.json + crops/<photo>/<ID>.png masks) and the folder of its source photos
#   crop  : same recipe as training (gray-world WB, square 1.25 x leaf long side with background, 224)
#   hold  : solidity < 0.8 (merged leaves) or touching the frame -> not judged (drawn gray, prediction kept in the csv)
#   model : EfficientNetV2-S, default = bbox_real_aug (cls/runs_bbox_real_aug/effnetv2s/best.pt)
# Output (<run>/stage2/): <photo>_stage2.jpg (outlines by class + IDs of flagged leaves + legend/counts),
#   flagged_sheet.jpg (judged non-Fresh leaves, context crops), leaves_stage2.csv / summary.json
# Usage: python v7_leafid/scripts/classify_leaves.py <leaf_search run dir> <photo dir> [--weights W] [--min-conf 0.0]
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
import paths  # noqa: E402,F401  (opencv workaround first)
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "cls" / "scripts"))
from cls_common import BOX_SCALE, gray_world, square_crop, to_input  # noqa: E402

import argparse, csv, json  # noqa: E401,E402
import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch, torch.nn as nn  # noqa: E401,E402
from torchvision import transforms as T  # noqa: E402
from torchvision.models import efficientnet_v2_s  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

CLS = ["Black Spot", "Fresh", "Hole", "Yellow"]
COL = {0: (255, 40, 40), 1: (60, 230, 60), 2: (40, 140, 255), 3: (255, 220, 0), "hold": (150, 150, 150)}
ap = argparse.ArgumentParser()
ap.add_argument("run"); ap.add_argument("photos")
ap.add_argument("--weights", default=str(paths.V7 / "cls" / "runs_bbox_real_aug" / "effnetv2s" / "best.pt"))
ap.add_argument("--min-conf", type=float, default=0.0, help="non-Fresh below this confidence is reported as Fresh")
a = ap.parse_args()
run, pdir = Path(a.run).resolve(), Path(a.photos).resolve()
out = run / "stage2"; out.mkdir(exist_ok=True)
rows = json.loads((run / "leaves.json").read_text())
m = efficientnet_v2_s(); m.classifier[1] = nn.Linear(m.classifier[1].in_features, 4)
m.load_state_dict(torch.load(a.weights, map_location="cpu")); m = m.cuda().eval()
TF = T.Compose([T.ToTensor(), T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
try:
    F, FB = ImageFont.load_default(size=22), ImageFont.load_default(size=30)
except TypeError:
    F = FB = ImageFont.load_default()
summary, flagged, out_rows = {}, [], []
for photo in sorted({r["photo"] for r in rows}):
    rs = [r for r in rows if r["photo"] == photo]
    img = gray_world(np.asarray(Image.open(pdir / photo).convert("RGB")))
    H, W = img.shape[:2]
    crops = [to_input(square_crop(img, (r["x0"] + r["x1"]) / 2, (r["y0"] + r["y1"]) / 2, BOX_SCALE * max(r["x1"] - r["x0"], r["y1"] - r["y0"]))) for r in rs]
    with torch.no_grad():
        pr = torch.cat([m(torch.stack([TF(c) for c in crops[i:i + 128]]).cuda()).float().softmax(1).cpu() for i in range(0, len(crops), 128)])
    vis = (img.astype(np.float32) * 0.8).astype(np.uint8)
    cnt = {k: 0 for k in CLS + ["Hold"]}
    labels = []
    for r, p, crop in zip(rs, pr, crops):
        k, conf = int(p.argmax()), float(p.max())
        if k != 1 and conf < a.min_conf:
            k = 1
        hold = bool(r["solidity"] < 0.8 or r["touches_border"])
        al = np.asarray(Image.open(run / "crops" / Path(photo).stem / f"{r['id']}.png").getchannel("A")) > 127
        cs, _ = cv2.findContours(al.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cs = [c + [r["x0"], r["y0"]] for c in cs]
        col = COL["hold"] if hold else COL[k]
        th = 2 if (hold or k == 1) else 5
        cv2.drawContours(vis, cs, -1, col, th)
        if hold:
            cnt["Hold"] += 1
        else:
            cnt[CLS[k]] += 1
            if k != 1:
                labels.append((r, k, conf))
                flagged.append((r, k, conf, crop))
        out_rows.append(dict(id=r["id"], photo=photo, pred=CLS[k], conf=round(conf, 3), hold=hold,
                             **{f"p_{c.split()[0]}": round(float(v), 3) for c, v in zip(CLS, p)}))
    pil = Image.fromarray(vis); d = ImageDraw.Draw(pil)
    for r, k, conf in labels:
        d.text((r["x0"], max(0, r["y0"] - 26)), f"{r['id'][-5:]} {CLS[k]} {conf:.2f}", font=F, fill=COL[k], stroke_width=3, stroke_fill=(0, 0, 0))
    # legend (top-left): outline colour per class + counts
    items = [(COL[1], f"Fresh {cnt['Fresh']}"), (COL[0], f"Black Spot {cnt['Black Spot']}"), (COL[2], f"Hole {cnt['Hole']}"),
             (COL[3], f"Yellow {cnt['Yellow']}"), (COL["hold"], f"Hold (merged/edge, not judged) {cnt['Hold']}")]
    bw = 520 if W > 1500 else 420
    d.rectangle([8, 8, 8 + bw, 8 + 46 + 34 * len(items)], fill=(0, 0, 0))
    d.text((18, 14), f"v7 stage2 | {Path(photo).stem} | leaves {len(rs)}", font=FB if W > 1500 else F, fill=(255, 255, 255))
    for i, (c, t) in enumerate(items):
        y = 56 + 34 * i
        d.rectangle([18, y + 4, 52, y + 24], outline=c, width=4); d.text((64, y), t, font=F, fill=(255, 255, 255))
    pil.save(out / f"{Path(photo).stem}_stage2.jpg", quality=90)
    summary[photo] = cnt
    print(photo, cnt, flush=True)
with open(out / "leaves_stage2.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(out_rows[0])); w.writeheader(); w.writerows(out_rows)
(out / "summary.json").write_text(json.dumps(dict(weights=a.weights, min_conf=a.min_conf, per_photo=summary), indent=1))
if flagged:
    t = 224; sh = Image.new("RGB", (6 * t, ((len(flagged) + 5) // 6) * (t + 26)), (25, 25, 25)); d = ImageDraw.Draw(sh)
    for i, (r, k, conf, crop) in enumerate(sorted(flagged, key=lambda x: -x[2])):
        x, y = i % 6 * t, i // 6 * (t + 26)
        sh.paste(crop, (x, y + 26)); d.text((x + 4, y + 2), f"{r['id'][-14:]} {CLS[k]} {conf:.2f}", font=F, fill=COL[k])
    sh.save(out / "flagged_sheet.jpg", quality=90)
print(f"judged non-Fresh: {len(flagged)} -> {out}", flush=True)
