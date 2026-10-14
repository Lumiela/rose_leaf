# Stage-0 metrics: score every stage-2 run (cutout runs/*, bbox runs_bbox/*, later variants) against the hand-labelled
# camfix ground truth (cls/gt/camfix_gt.json, 462 judged leaves, reviewed by eye on cls/gt/sheets without predictions).
# Tags: F clear healthy | Fp healthy, partial/merged mask | Fb healthy, blurry | Fn pale/reddish new leaf | X not judgeable
#       Black/Hole/Yellow real anomaly (none in this set).
# A1 false alarm rate on healthy leaves (F+Fp+Fb+Fn), A2 false alarms per photo, A3 false alarms per class, plus the
# rate per tag (what kind of healthy leaf gets flagged).  Output: cls/gt/eval_gt.md / .json
# Usage: python v7_leafid/cls/scripts/eval_gt.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from cls_common import CLS  # noqa: E402
import csv, json  # noqa: E401,E402
from collections import Counter  # noqa: E402

GT = json.loads((CLS / "gt" / "camfix_gt.json").read_text())
HEALTHY = ("F", "Fp", "Fb", "Fn")
TAGS = ("F", "Fp", "Fb", "Fn")

if __name__ == "__main__":
    healthy = {k: v for k, v in GT.items() if v["tag"] in HEALTHY}
    photos = sorted({v["photo"] for v in healthy.values()})
    res = []
    for p in sorted(CLS.glob("runs*/*/v7_crop_preds.csv")):
        if "aborted" in p.parent.parent.name:
            continue
        pred = {r["id"]: r["pred"] for r in csv.DictReader(open(p))}
        fa = [k for k in healthy if pred.get(k, "Fresh") != "Fresh"]
        by_tag = {t: (sum(1 for k in fa if healthy[k]["tag"] == t), sum(1 for v in healthy.values() if v["tag"] == t)) for t in TAGS}
        res.append(dict(variant=p.parent.parent.name.replace("runs", "").strip("_") or "cutout", model=p.parent.name,
                        n=len(healthy), false_alarm=len(fa), rate=round(len(fa) / len(healthy), 3),
                        per_photo=round(len(fa) / len(photos), 1), by_class=dict(Counter(pred[k] for k in fa)),
                        by_tag={t: dict(fa=a, n=n, rate=round(a / max(1, n), 3)) for t, (a, n) in by_tag.items()}))
    tc = Counter(v["tag"] for v in GT.values())
    md = ["# 실사 정답 기준 평가 (고정 카메라 5장, 판정 대상 잎 462개)", "",
          f"정답 분포: " + ", ".join(f"{t} {tc.get(t, 0)}" for t in ("F", "Fp", "Fb", "Fn", "X", "Black", "Hole", "Yellow")),
          "", "F 선명한 건강 잎 / Fp 마스크가 일부·합쳐진 건강 잎 / Fb 흐린 건강 잎 / Fn 연한 새잎 / X 판정 불가(제외)", "",
          "| 입력 | 모델 | A1 건강 잎 오판율 | A2 사진당 오판 | A3 클래스별 오판 (Black/Hole/Yellow) | F | Fp | Fb | Fn |",
          "|---|---|---|---|---|---|---|---|---|"]
    for r in sorted(res, key=lambda r: r["rate"]):
        c, t = r["by_class"], r["by_tag"]
        md.append(f"| {r['variant']} | {r['model']} | **{r['rate']:.1%}** ({r['false_alarm']}/{r['n']}) | {r['per_photo']} | "
                  f"{c.get('Black', 0)}/{c.get('Hole', 0)}/{c.get('Yellow', 0)} | "
                  + " | ".join(f"{t[k]['rate']:.0%} ({t[k]['fa']}/{t[k]['n']})" for k in TAGS) + " |")
    (CLS / "gt" / "eval_gt.md").write_text("\n".join(md)); (CLS / "gt" / "eval_gt.json").write_text(json.dumps(res, indent=1))
    print("\n".join(md), flush=True)
