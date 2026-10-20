# v8 stage 2 for a v7-format stage-1 run: v8 classifier + decision rule (cls/runs/<name>/states.json) -> per-leaf state
#   HOLD_SHAPE  merged / cut by the frame (solidity < 0.8 or touches border)   [v7 rule]
#   HOLD        quality gate (blur / dark / too small)                           [states.json]
#   FRESH / ANOMALY:<class> / UNKNOWN                                            [states.json]
# Output <run>/stage2_v8/: <photo>_stage2.jpg (outline colour = state), leaves_stage2.csv, summary.json, flagged_sheet.jpg (ANOMALY + UNKNOWN)
# Usage: python v8_reallabel/cls/scripts/classify_v8.py <run dir> <photo dir> [--name effnetv2s]
import sys, argparse, csv, json  # noqa: E401
from pathlib import Path
MODEL = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(MODEL / "v7_leafid" / "cls" / "scripts"))
from cls_common import BOX_SCALE, gray_world, square_crop, to_input  # noqa: E402
import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch, torch.nn as nn  # noqa: E401,E402
from torchvision import transforms as T  # noqa: E402
from torchvision.models import efficientnet_v2_s  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

ap = argparse.ArgumentParser(); ap.add_argument("run"); ap.add_argument("photos"); ap.add_argument("--name", default="effnetv2s")
a = ap.parse_args()
R = MODEL / "v8_reallabel" / "cls" / "runs" / a.name
th = json.loads((R / "states.json").read_text())["thresholds"]
run, pdir = Path(a.run).resolve(), Path(a.photos).resolve(); out = run / "stage2_v8"; out.mkdir(exist_ok=True)
rows = json.loads((run / "leaves.json").read_text())
m = efficientnet_v2_s(); m.classifier[1] = nn.Linear(m.classifier[1].in_features, 4)
m.load_state_dict(torch.load(R / "best.pt", map_location="cpu")); m = m.cuda().eval()
TF = T.Compose([T.ToTensor(), T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
CLS = ["Black", "Fresh", "Hole", "Yellow"]
COL = {"FRESH": (60, 230, 60), "Black": (255, 40, 40), "Hole": (40, 140, 255), "Yellow": (255, 220, 0), "UNKNOWN": (255, 0, 255), "HOLD": (150, 150, 150), "HOLD_SHAPE": (90, 90, 90)}
try:
    F, FB = ImageFont.load_default(size=22), ImageFont.load_default(size=30)
except TypeError:
    F = FB = ImageFont.load_default()
summary, flagged, out_rows = {}, [], []
for photo in sorted({r["photo"] for r in rows}):
    rs = [r for r in rows if r["photo"] == photo]
    img = gray_world(np.asarray(Image.open(pdir / photo).convert("RGB")))
    crops = [to_input(square_crop(img, (r["x0"] + r["x1"]) / 2, (r["y0"] + r["y1"]) / 2, BOX_SCALE * max(r["x1"] - r["x0"], r["y1"] - r["y0"]))) for r in rs]
    with torch.no_grad():
        pr = torch.cat([m(torch.stack([TF(c) for c in crops[i:i + 128]]).cuda()).float().softmax(1).cpu() for i in range(0, len(crops), 128)]).numpy()
    vis = (img.astype(np.float32) * 0.8).astype(np.uint8)
    cnt = {}
    for r, p, crop in zip(rs, pr, crops):
        L = max(r["x1"] - r["x0"], r["y1"] - r["y0"])
        pa = p[[0, 2, 3]]
        if r["solidity"] < 0.8 or r["touches_border"]:
            st = "HOLD_SHAPE"
        elif r["sharpness"] < th["s_min"] or r["dark"] > th["d_max"] or L < th["L_min"]:
            st = "HOLD"
        elif pa.max() >= th["t_anom"]:
            st = "ANOMALY:" + ["Black", "Hole", "Yellow"][int(pa.argmax())]
        elif p[1] >= th["t_fresh"]:
            st = "FRESH"
        else:
            st = "UNKNOWN"
        cnt[st] = cnt.get(st, 0) + 1
        key = st.split(":")[-1]
        al = np.asarray(Image.open(run / "crops" / Path(photo).stem / f"{r['id']}.png").getchannel("A")) > 127
        cs, _ = cv2.findContours(al.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, [c + [r["x0"], r["y0"]] for c in cs], -1, COL[key], 5 if (st.startswith("ANOMALY") or st == "UNKNOWN") else 2)
        if st.startswith("ANOMALY") or st == "UNKNOWN":
            flagged.append((r, st, float(max(pa.max(), p[1])), crop))
        out_rows.append(dict(id=r["id"], photo=photo, state=st, **{f"p_{c}": round(float(v), 4) for c, v in zip(CLS, p)}, sharpness=r["sharpness"], dark=r["dark"], L=L))
    pil = Image.fromarray(vis); d = ImageDraw.Draw(pil)
    items = sorted(cnt.items(), key=lambda kv: kv[0])
    d.rectangle([8, 8, 600, 56 + 32 * len(items)], fill=(0, 0, 0)); d.text((18, 14), f"v8 stage2 | {Path(photo).stem} | leaves {len(rs)}", font=FB, fill=(255, 255, 255))
    for i, (k, v) in enumerate(items):
        d.rectangle([18, 58 + 32 * i, 48, 78 + 32 * i], outline=COL[k.split(':')[-1]], width=4); d.text((60, 54 + 32 * i), f"{k} {v}", font=F, fill=(255, 255, 255))
    pil.save(out / f"{Path(photo).stem}_stage2.jpg", quality=90)
    summary[photo] = cnt; print(photo, cnt, flush=True)
with open(out / "leaves_stage2.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(out_rows[0])); w.writeheader(); w.writerows(out_rows)
tot = {}
for c in summary.values():
    for k, v in c.items(): tot[k] = tot.get(k, 0) + v
(out / "summary.json").write_text(json.dumps(dict(model=str(R), thresholds=th, total=tot, per_photo=summary), indent=1))
if flagged:
    t = 224; sh = Image.new("RGB", (6 * t, ((len(flagged) + 5) // 6) * (t + 26)), (25, 25, 25)); d = ImageDraw.Draw(sh)
    for i, (r, st, conf, crop) in enumerate(sorted(flagged, key=lambda x: x[1])):
        x, y = i % 6 * t, i // 6 * (t + 26); sh.paste(crop, (x, y + 26)); d.text((x + 4, y + 2), f"{r['id'][-14:]} {st.split(':')[-1]}", font=F, fill=COL[st.split(':')[-1]])
    sh.save(out / "flagged_sheet.jpg", quality=90)
print("TOTAL", tot)
