# v8: evaluate a v7-format pipeline run (stage 1 + optional stage 2) against the reviewed test labels.
#   run dir: leaves.json + crops/<photo>/<ID>.png (alpha = mask in the box), optional stage2/leaves_stage2.csv
#   stage 1 : recall / precision / merged vs GT (mask IoU >= 0.5; predictions mostly inside ignore regions not counted as FP)
#   judged  : leaves not held by the stage-2 rule (solidity >= 0.8 and not touching the frame) -> share of GT leaves that reach
#             stage 2 as a clean single-leaf mask ("judged TP"), and judged predictions that are FP or merged
#   stage 2 : class counts of judged leaves (all test leaves are healthy except where noted in the review)
# Usage: python v8_reallabel/scripts/eval_pipeline.py <name>=<run dir> [...] [--split test]
import argparse, csv, json  # noqa: E401
from pathlib import Path
from sam_common import load, V8
import numpy as np
from PIL import Image

ap = argparse.ArgumentParser(); ap.add_argument("runs", nargs="+"); ap.add_argument("--split", default="test")
a = ap.parse_args()
sc = json.loads((V8 / "scenes.json").read_text())
stems = sc["split"][a.split]
res = {}
for spec in a.runs:
    name, rd = spec.split("=", 1); rd = Path(rd)
    rows = json.loads((rd / "leaves.json").read_text())
    st2 = {}
    if (rd / "stage2" / "leaves_stage2.csv").exists():
        for r in csv.DictReader(open(rd / "stage2" / "leaves_stage2.csv")):
            st2[r["id"]] = r
    T = dict(gt=0, pred=0, tp=0, fp=0, ign=0, merged=0, judged=0, judged_tp=0, judged_fp=0, judged_merged=0, cls={})
    per = {}
    for s in stems:
        inst, meta = load(s); H, W = inst.shape
        ign = np.zeros((H, W), bool)
        for b in meta.get("ignore", []):
            x0, y0, x1, y1 = [int(v) for v in b]; ign[y0:y1, x0:x1] = True
        ign &= inst == 0
        gt_ids = [int(i) for i in np.unique(inst) if i]; area = {i: int((inst == i).sum()) for i in gt_ids}
        rs = sorted([r for r in rows if Path(r["photo"]).stem == s], key=lambda r: -float(r.get("v4_conf") or 0))
        matched = set(); p = dict(gt=len(gt_ids), pred=len(rs), tp=0, fp=0, ign=0, merged=0, judged=0, judged_tp=0, judged_fp=0, judged_merged=0)
        for r in rs:
            m = np.asarray(Image.open(rd / "crops" / s / f"{r['id']}.png").getchannel("A")) > 127
            x0, y0 = int(r["x0"]), int(r["y0"]); sub = inst[y0:y0 + m.shape[0], x0:x0 + m.shape[1]]
            m = m[:sub.shape[0], :sub.shape[1]]
            ids, cnt = np.unique(sub[m], return_counts=True)
            best, biou = None, 0.0
            for i, n in zip(ids, cnt):
                if i and int(i) not in matched:
                    iou = n / (m.sum() + area[int(i)] - n)
                    if iou > biou: best, biou = int(i), iou
            merged = sum(1 for i, n in zip(ids, cnt) if i and n >= 0.5 * area[int(i)]) >= 2
            hold = float(r["solidity"]) < 0.8 or str(r["touches_border"]) in ("True", "true", "1")
            if best is not None and biou >= 0.5:
                matched.add(best); kind = "tp"
            elif ign[y0:y0 + m.shape[0], x0:x0 + m.shape[1]][m].mean() >= 0.5:
                kind = "ign"
            else:
                kind = "fp"
            p[kind] += 1; p["merged"] += merged
            if not hold:
                p["judged"] += 1
                if kind == "tp": p["judged_tp"] += 1
                if kind == "fp": p["judged_fp"] += 1
                if merged: p["judged_merged"] += 1
                c = st2.get(r["id"], {}).get("pred")
                if c: T["cls"][c] = T["cls"].get(c, 0) + 1
        per[s] = p
        for k in p: T[k] += p[k]
    T["recall"] = round(T["tp"] / max(1, T["gt"]), 3); T["precision"] = round(T["tp"] / max(1, T["tp"] + T["fp"]), 3)
    T["judged_share_of_gt"] = round(T["judged_tp"] / max(1, T["gt"]), 3); T["hold_rate"] = round(1 - T["judged"] / max(1, T["pred"]), 3)
    res[name] = dict(total=T, per_photo=per)
out = V8 / "eval" / "pipeline"; out.mkdir(parents=True, exist_ok=True)
(out / "summary.json").write_text(json.dumps(res, indent=1))
print("| 실행 | GT | 예측 | 재현율 | 정밀도 | 합쳐진 | 보류율 | 판정 대상 | 판정 대상 중 정답 잎 1:1 | 정답 잎 중 판정까지 간 비율 | 판정 대상 오탐 | 판정 대상 합쳐짐 | 2단계 분포 |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
for n, r in res.items():
    t = r["total"]
    print(f"| {n} | {t['gt']} | {t['pred']} | {t['recall']} | {t['precision']} | {t['merged']} | {t['hold_rate']} | {t['judged']} | {t['judged_tp']} | {t['judged_share_of_gt']} | {t['judged_fp']} | {t['judged_merged']} | {t['cls']} |")
