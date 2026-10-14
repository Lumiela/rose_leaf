# A4 + B4 on held-out real phone photos (cube_21, cube_22; never trained on) for the EfficientNet runs:
#   A4 false alarm rate on the reviewed healthy leaves (data_bbox_real/test_real)
#   B4 recall on the same leaves with ONE synthetic anomaly painted on each (data_bbox_real_aug/test_real_anom):
#      a model that learned "real-looking = Fresh" (shortcut) scores low here.
# Output: cls/gt/eval_real_test.md / .json     Usage: python v7_leafid/cls/scripts/eval_real_test.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from cls_common import CLS, SHORT, NAMES  # noqa: E402
import json  # noqa: E402
from collections import Counter  # noqa: E402
import torch, torch.nn as nn  # noqa: E401,E402
from torchvision import transforms as T  # noqa: E402
from torchvision.models import efficientnet_v2_s  # noqa: E402
from PIL import Image  # noqa: E402

TF = T.Compose([T.ToTensor(), T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
healthy = sorted((CLS / "data_bbox_real" / "test_real" / NAMES[1]).glob("*.jpg"))
anom = [(p, ci) for ci in (0, 2, 3) for p in sorted((CLS / "data_bbox_real_aug" / "test_real_anom" / NAMES[ci]).glob("*.jpg"))]


def predict(m, files):
    P = []
    with torch.no_grad():
        for i in range(0, len(files), 128):
            x = torch.stack([TF(Image.open(f).convert("RGB")) for f in files[i:i + 128]]).cuda()
            P += m(x).float().argmax(1).tolist()
    return P


res = []
for run in ("runs_bbox", "runs_bbox_real", "runs_bbox_real_aug"):
    w = CLS / run / "effnetv2s" / "best.pt"
    if not w.exists():
        continue
    m = efficientnet_v2_s(); m.classifier[1] = nn.Linear(m.classifier[1].in_features, 4)
    m.load_state_dict(torch.load(w, map_location="cpu")); m = m.cuda().eval()
    P = predict(m, healthy); fa = [SHORT[p] for p in P if p != 1]
    r = dict(run=run, n=len(healthy), false_alarm=len(fa), rate=round(len(fa) / max(1, len(healthy)), 3), by_class=dict(Counter(fa)))
    if anom:
        Pa = predict(m, [p for p, _ in anom])
        r["anom_recall"] = {SHORT[ci]: round(sum(1 for (_, c), q in zip(anom, Pa) if c == ci and q == ci) / max(1, sum(1 for _, c in anom if c == ci)), 3) for ci in (0, 2, 3)}
        r["anom_flagged"] = round(sum(1 for q in Pa if q != 1) / len(Pa), 3)   # any non-Fresh (anomaly noticed at all)
        r["anom_n"] = {SHORT[ci]: sum(1 for _, c in anom if c == ci) for ci in (0, 2, 3)}
    res.append(r)
md = ["# 학습에 안 쓴 실제 사진(스마트폰 cube_21, cube_22) 평가", "",
      "A4 = 건강 잎 오판율, B4 = 같은 잎에 합성 이상을 넣었을 때 클래스별 재현율 / 이상으로라도 잡은 비율", "",
      "| 실행 | A4 오판율 | A4 클래스별 (Black/Hole/Yellow) | B4 재현율 (Black/Hole/Yellow) | B4 이상 감지율 |", "|---|---|---|---|---|"]
for r in res:
    c, a = r["by_class"], r.get("anom_recall", {})
    md.append(f"| {r['run']} | **{r['rate']:.1%}** ({r['false_alarm']}/{r['n']}) | {c.get('Black', 0)}/{c.get('Hole', 0)}/{c.get('Yellow', 0)} | "
              + ("/".join(str(a.get(k, '-')) for k in ('Black', 'Hole', 'Yellow')) + f" | {r.get('anom_flagged', '-')} |" if a else "- | - |"))
if res and "anom_n" in res[0]:
    md += ["", f"B4 표본 수: {res[0]['anom_n']}"]
(CLS / "gt").mkdir(exist_ok=True)
(CLS / "gt" / "eval_real_test.md").write_text("\n".join(md)); (CLS / "gt" / "eval_real_test.json").write_text(json.dumps(res, indent=1))
print("\n".join(md), flush=True)
