# How much does blur cost? (question: would an autofocus fixed camera classify better?)
#  1) detection of synthetic anomalies painted on the camfix ground-truth healthy leaves, split by the hand label
#     F (sharp) / Fp (partial mask) / Fb (blurry). All 3 anomaly types per leaf, test-only images (nothing trained).
#     bbox_real never saw these synthetic anomalies (fair); bbox_real_aug was trained on the same generator (optimistic).
#  2) per-leaf Laplacian sharpness (leaves.json) quartiles vs detection rate
#  3) stage-1 hold share (merged/edge, not judged): camfix (fixed focus) vs phone photos (autofocus)
# Output: cls/gt/blur_effect.md     Usage: python v7_leafid/cls/scripts/eval_blur_effect.py
import os, sys  # noqa: E401
os.environ.setdefault("CLS_VARIANT", "bbox")
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from cls_common import CLS, V7_RUN, BOX_SCALE, gray_world, to_input, paths  # noqa: E402
from synth_real_anom import sq, hole, yellow, black  # noqa: E402
import json  # noqa: E402
import numpy as np  # noqa: E402
import torch, torch.nn as nn  # noqa: E401,E402
from torchvision import transforms as T  # noqa: E402
from torchvision.models import efficientnet_v2_s  # noqa: E402
from PIL import Image  # noqa: E402

GT = json.loads((CLS / "gt" / "camfix_gt.json").read_text())
rows = {r["id"]: r for r in json.loads((V7_RUN / "leaves.json").read_text())}
TYPES = [(2, "Hole", hole), (3, "Yellow", yellow), (0, "Black", black)]
items = []   # (pil, true class, tag, sharpness)
for photo in sorted({v["photo"] for v in GT.values()}):
    img = gray_world(np.asarray(Image.open(paths.COMMON / "real_images_camfix" / photo).convert("RGB")))
    H, W = img.shape[:2]
    for lid, g in GT.items():
        if g["photo"] != photo or g["tag"] not in ("F", "Fp", "Fb"):
            continue
        r = rows[lid]
        a = np.asarray(Image.open(V7_RUN / "crops" / Path(photo).stem / f"{lid}.png").getchannel("A")) > 127
        full = np.zeros((H, W), bool); full[r["y0"]:r["y1"], r["x0"]:r["x1"]] = a
        L = max(r["x1"] - r["x0"], r["y1"] - r["y0"]); cx, cy = (r["x0"] + r["x1"]) / 2, (r["y0"] + r["y1"]) / 2
        S, M = sq(img, cx, cy, BOX_SCALE * L, 128), sq(full, cx, cy, BOX_SCALE * L, False)
        if M.sum() < 30:
            continue
        for j, (ci, name, fn) in enumerate(TYPES):
            rng = np.random.default_rng([31, g["n"], j])
            items.append((to_input(Image.fromarray(fn(S, M, L, rng))), ci, g["tag"], r["sharpness"]))
print(f"{len(items)} anomaly images from camfix healthy leaves", flush=True)
TF = T.Compose([T.ToTensor(), T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
sh = np.array([it[3] for it in items]); qs = np.quantile(sh, [0.25, 0.5, 0.75])
qbin = np.digitize(sh, qs)
md = ["# 흐림이 판정에 주는 영향 (고정 카메라 건강 잎 + 합성 이상, 잎마다 구멍·황변·반점 3종)", "",
      "감지율 = 이상으로라도 판정 / 정확 = 해당 클래스로 판정", "",
      "| 모델 | 선명 F | 일부 Fp | 흐림 Fb | 선명도 하위 25% | 25~50% | 50~75% | 상위 25% |", "|---|---|---|---|---|---|---|---|"]
for run in ("runs_bbox_real", "runs_bbox_real_aug"):
    m = efficientnet_v2_s(); m.classifier[1] = nn.Linear(m.classifier[1].in_features, 4)
    m.load_state_dict(torch.load(CLS / run / "effnetv2s" / "best.pt", map_location="cpu")); m = m.cuda().eval()
    P = []
    with torch.no_grad():
        for i in range(0, len(items), 128):
            P += m(torch.stack([TF(it[0]) for it in items[i:i + 128]]).cuda()).float().argmax(1).tolist()
    P = np.array(P); Y = np.array([it[1] for it in items]); tags = np.array([it[2] for it in items])
    cell = lambda s: f"{(P[s] != 1).mean():.0%} / {(P[s] == Y[s]).mean():.0%} (n={s.sum()})"  # noqa: E731
    md.append(f"| {run.replace('runs_', '')} | " + " | ".join(cell(tags == t) for t in ("F", "Fp", "Fb")) + " | "
              + " | ".join(cell(qbin == b) for b in range(4)) + " |")
def hold_share(run, skip=()):
    rs = [r for r in json.loads((run / "leaves.json").read_text()) if r["photo"] not in skip]
    return sum(r["solidity"] < 0.8 or r["touches_border"] for r in rs) / len(rs), len(rs), float(np.median([r["sharpness"] for r in rs]))
cf = hold_share(V7_RUN); ph = hold_share(paths.V7 / "runs" / "bg_cube")
md += ["", "## 1단계 판정 보류 비율과 선명도", "", "| 사진 | 잎 수 | 보류 비율 | 선명도 중앙값 (Laplacian 분산) |", "|---|---|---|---|",
       f"| 고정 카메라 5장 (고정 초점) | {cf[1]} | {cf[0]:.1%} | {cf[2]:.0f} |", f"| 스마트폰 18장 (오토포커스) | {ph[1]} | {ph[0]:.1%} | {ph[2]:.0f} |",
       "", f"선명도 사분위 경계 (고정 카메라 잎): {', '.join(f'{q:.0f}' for q in qs)}"]
(CLS / "gt" / "blur_effect.md").write_text("\n".join(md))
print("\n".join(md), flush=True)
