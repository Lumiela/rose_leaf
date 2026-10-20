# v8: compare eval summaries (eval/<name>/summary.json) -> eval/compare.md
# Usage: python v8_reallabel/scripts/compare_eval.py baseline_v4 r1_fromv4 r1_coco11s ...
import sys, json
from pathlib import Path
V8 = Path(__file__).resolve().parents[1]
names = sys.argv[1:]
S = {n: json.loads((V8 / "eval" / n / "summary.json").read_text()) for n in names if (V8 / "eval" / n / "summary.json").exists()}
photos = list(next(iter(S.values()))["per_photo"]) if S else []
L = ["# test 평가 비교 (잎 탐색, mask IoU >= 0.5)", "",
     "| 모델 | 재현율 | 정밀도 | F1 | TP | FP | 합쳐진 예측 | 제외 영역 예측 |", "|---|---|---|---|---|---|---|---|"]
for n, s in S.items():
    t = s["total"]
    L.append(f"| {n} | {t['recall']:.3f} | {t['precision']:.3f} | {t['f1']:.3f} | {t['tp']}/{t['gt']} | {t['fp']} | {t['merged']} | {t['ignored']} |")
L += ["", "## 사진별 재현율 / 정밀도", "", "| 사진 | " + " | ".join(S) + " |", "|---|" + "---|" * len(S)]
for p in photos:
    L.append(f"| {p} | " + " | ".join(f"{s['per_photo'][p]['recall']:.2f} / {s['per_photo'][p]['precision']:.2f}" for s in S.values()) + " |")
(V8 / "eval" / "compare.md").write_text("\n".join(L) + "\n")
print("\n".join(L))
