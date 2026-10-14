# Collect metrics.json of every model of the current variant into compare<SUF>.md / .json.
# Real-crop columns: non-Fresh share over all stage-1 leaves and over judged leaves only (hold = merged or border).
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from cls_common import RUNS, CLS, SUF, VARIANT, Progress, v7_hold  # noqa: E402
import csv, json  # noqa: E401,E402

Progress.log("compare", "start", "running")
hold = v7_hold()
rows = []
for p in sorted(RUNS.glob("*/metrics.json")):
    r = json.loads(p.read_text())
    preds = list(csv.DictReader(open(p.parent / "v7_crop_preds.csv")))
    judged = [x for x in preds if not hold.get(x["id"], False)]
    nf = lambda xs: round(sum(x["pred"] != "Fresh" for x in xs) / max(1, len(xs)), 3)
    jc = {k: sum(x["pred"] == k for x in judged) for k in ("Black", "Fresh", "Hole", "Yellow")}
    r["real"] = dict(n=len(preds), nonfresh_all=nf(preds), n_judged=len(judged), nonfresh_judged=nf(judged), judged_counts=jc)
    rows.append(r)
md = [f"## 비교 결과 ({VARIANT or 'cutout'})", "",
      "| 모델 | test acc / macro-F1 | 변형 test acc / macro-F1 | 변형 test 재현율 (Black/Fresh/Hole/Yellow) | ms/장 | 실사 이상판정 비율 (전체) | 실사 이상판정 비율 (판정 대상만) | 판정 대상 분포 (Black/Fresh/Hole/Yellow) |",
      "|---|---|---|---|---|---|---|---|"]
for r in rows:
    t, d, q = r["test"], r["test_deg"], r["real"]
    md.append(f"| {r['model']} | {t['acc']} / {t['macro_f1']} | {d['acc']} / {d['macro_f1']} | "
              f"{'/'.join(str(d['recall'][k]) for k in ('Black', 'Fresh', 'Hole', 'Yellow'))} | {d['ms_per_img']} | "
              f"{q['nonfresh_all']} ({q['n']}) | {q['nonfresh_judged']} ({q['n_judged']}) | "
              f"{'/'.join(str(q['judged_counts'][k]) for k in ('Black', 'Fresh', 'Hole', 'Yellow'))} |")
(CLS / f"compare{SUF}.md").write_text("\n".join(md)); (CLS / f"compare{SUF}.json").write_text(json.dumps(rows, indent=1))
Progress.log("compare", f"{len(rows)} models compared -> compare{SUF}.md", "done")
