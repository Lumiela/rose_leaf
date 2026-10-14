# Blur sweep: same leaf, same lesion, increasing blur (answers "would autofocus help?" without the flaw of painting
# crisp lesions on already-blurry leaves). Source = held-out autofocus phone photos (cube_21, cube_22; reviewed healthy
# leaves, never trained on). Each leaf gets Hole / Yellow / Black painted at native resolution, then the WHOLE square is
# Gaussian-blurred (sigma in native px) before the usual 224 resize. Sharpness = Laplacian variance inside the leaf on the
# blurred native patch (same measure as leaves.json; camfix median 34, phone median ~400).
# Also: false alarm on the same healthy leaves blurred the same way.
# Output: cls/gt/blur_sweep.md     Usage: python v7_leafid/cls/scripts/eval_blur_sweep.py
import os, sys  # noqa: E401
os.environ.setdefault("CLS_VARIANT", "bbox")
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from cls_common import CLS, BOX_SCALE, gray_world, to_input, paths  # noqa: E402
from synth_real_anom import sq, hole, yellow, black  # noqa: E402
import json  # noqa: E402
import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch, torch.nn as nn  # noqa: E401,E402
from torchvision import transforms as T  # noqa: E402
from torchvision.models import efficientnet_v2_s  # noqa: E402
from PIL import Image  # noqa: E402

SIG = [0, 0.8, 1.5, 2.5, 4.0]
lab = json.loads((CLS / "real" / "real_labels.json").read_text())["labels"]
cand = [r for r in json.loads((CLS / "real" / "candidates.json").read_text()) if r["split"] == "test" and lab.get(r["id"]) == "F"]
TYPES = [(2, hole), (3, yellow), (0, black)]
data = {s: [] for s in SIG}   # sigma -> list of (pil, true class, sharpness)
for photo in sorted({r["photo"] for r in cand}):
    img = gray_world(np.asarray(Image.open(paths.REAL_IMAGES_CUBE / photo).convert("RGB")))
    H, W = img.shape[:2]
    for r in [x for x in cand if x["photo"] == photo]:
        a = np.asarray(Image.open(paths.V7 / "runs" / "bg_cube" / "crops" / Path(photo).stem / f"{r['id']}.png").getchannel("A")) > 127
        full = np.zeros((H, W), bool); full[r["y0"]:r["y1"], r["x0"]:r["x1"]] = a
        L = max(r["x1"] - r["x0"], r["y1"] - r["y0"]); cx, cy = (r["x0"] + r["x1"]) / 2, (r["y0"] + r["y1"]) / 2
        S, M = sq(img, cx, cy, BOX_SCALE * L, 128), sq(full, cx, cy, BOX_SCALE * L, False)
        if M.sum() < 30:
            continue
        variants = [(1, S)] + [(ci, fn(S, M, L, np.random.default_rng([41, hash(r["id"]) % 10**6, ci]))) for ci, fn in TYPES]
        for s in SIG:
            for ci, v in variants:
                b = cv2.GaussianBlur(v, (0, 0), s) if s > 0 else v
                g = cv2.cvtColor(b, cv2.COLOR_RGB2GRAY).astype(np.float32)
                sharp = float(cv2.Laplacian(g, cv2.CV_32F)[M].var())
                data[s].append((to_input(Image.fromarray(b)), ci, sharp))
TF = T.Compose([T.ToTensor(), T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
md = ["# 흐림 정도에 따른 판정 변화 (학습에 안 쓴 오토포커스 스마트폰 사진의 건강 잎, 같은 잎·같은 이상에 흐림만 증가)", "",
      f"잎 {len(data[0]) // 4}개 x (건강 원본 + 구멍·황변·반점). 칸 = 이상 감지율 / 정확 분류율, 오판 = 건강 원본을 이상으로 판정한 비율", "",
      "| 흐림 σ (원본 px) | 선명도 중앙값 | 모델 | 이상 감지율 / 정확 | 구멍 | 황변 | 반점 | 건강 잎 오판 |", "|---|---|---|---|---|---|---|---|"]
for run in ("runs_bbox_real", "runs_bbox_real_aug"):
    m = efficientnet_v2_s(); m.classifier[1] = nn.Linear(m.classifier[1].in_features, 4)
    m.load_state_dict(torch.load(CLS / run / "effnetv2s" / "best.pt", map_location="cpu")); m = m.cuda().eval()
    for s in SIG:
        it = data[s]; P = []
        with torch.no_grad():
            for i in range(0, len(it), 128):
                P += m(torch.stack([TF(x[0]) for x in it[i:i + 128]]).cuda()).float().argmax(1).tolist()
        P, Y = np.array(P), np.array([x[1] for x in it]); sh = np.median([x[2] for x in it if x[1] == 1])
        an = Y != 1
        per = lambda c: f"{(P[Y == c] == c).mean():.0%}"  # noqa: E731
        md.append(f"| {s} | {sh:.0f} | {run.replace('runs_', '')} | {(P[an] != 1).mean():.0%} / {(P[an] == Y[an]).mean():.0%} | "
                  f"{per(2)} | {per(3)} | {per(0)} | {(P[Y == 1] != 1).mean():.1%} |")
md += ["", "참고: 고정 카메라 잎 선명도 중앙값 34 (사분위 16/34/67), 스마트폰(오토포커스) 잎 중앙값 약 400"]
(CLS / "gt" / "blur_sweep.md").write_text("\n".join(md))
print("\n".join(md), flush=True)
