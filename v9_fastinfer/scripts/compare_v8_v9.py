"""v9 vs v8 deploy package: identical output? how much faster?

    cd /helpai-storage/model_git && python v9_fastinfer/scripts/compare_v8_v9.py [--reps 2] [--device cuda:0]

Photos: v8_reallabel/inputs (27 unique scenes) + v8_reallabel/inputs_20261019 (9 newest pot photos, 12MP).
Each package runs in its own process (dump_run.py). Writes v9_fastinfer/eval/:
  compare_v8_v9.json  per photo: same leaf count / L / counts, leaf-level mismatches, timings of both versions
  bench.csv           one row per photo: size, leaves, v8 and v9 mean wall / stage1 / stage2 times, speed-up
"""
import argparse, csv, json, statistics, subprocess, sys, tempfile  # noqa: E401
from pathlib import Path

HERE = Path(__file__).resolve().parent; V9 = HERE.parent; M = V9.parent
DEPLOY = {"v8": M / "v8_reallabel" / "deploy", "v9": V9 / "deploy"}
ap = argparse.ArgumentParser(); ap.add_argument("--reps", type=int, default=2); ap.add_argument("--device")
a = ap.parse_args()
photos = sorted(str(p) for p in (M / "v8_reallabel" / "inputs").glob("*.jpg")) + \
         sorted(str(p) for p in (M / "v8_reallabel" / "inputs_20261019").glob("*.jpg"))
res = {}
with tempfile.TemporaryDirectory() as tmp:
    for v, d in DEPLOY.items():
        out = Path(tmp) / f"{v}.json"
        cmd = [sys.executable, str(HERE / "dump_run.py"), str(d), str(out), "--reps", str(a.reps)] + (["--device", a.device] if a.device else []) + photos
        print("RUN", v, flush=True)
        subprocess.run(cmd, check=True)
        res[v] = json.loads(out.read_text())

FIELDS = ["id", "box", "polygon", "area", "conf", "solidity", "touches_border", "sharpness", "dark", "L", "state", "anomaly_class", "probs", "mask_md5"]
mean = lambda rs, k: statistics.mean(r[k] for r in rs)  # noqa: E731
rows, report = [], []
for p8, p9 in zip(res["v8"]["photos"], res["v9"]["photos"]):
    assert p8["photo"] == p9["photo"]
    bad = {}
    for l8, l9 in zip(p8["leaves"], p9["leaves"]):
        for f in FIELDS:
            if l8.get(f) != l9.get(f):
                bad[f] = bad.get(f, 0) + 1
    same = len(p8["leaves"]) == len(p9["leaves"]) and p8["L"] == p9["L"] and p8["counts"] == p9["counts"] and not bad
    W, H = p8["size"]
    row = dict(photo=Path(p8["photo"]).name, size=f"{max(W, H)}x{min(W, H)}", L=p8["L"], leaves=len(p8["leaves"]), identical=same,
               v8_wall_s=round(mean(p8["runs"], "wall_s"), 3), v8_stage1_s=round(mean(p8["runs"], "stage1_s"), 2), v8_stage2_s=round(mean(p8["runs"], "stage2_s"), 2),
               v9_wall_s=round(mean(p9["runs"], "wall_s"), 3), v9_stage1_s=round(mean(p9["runs"], "stage1_s"), 2), v9_stage2_s=round(mean(p9["runs"], "stage2_s"), 2))
    row["speedup"] = round(row["v8_wall_s"] / row["v9_wall_s"], 2)
    rows.append(row)
    report.append(dict(row, leaves_v9=len(p9["leaves"]), L_v9=p9["L"], counts_v8=p8["counts"], counts_v9=p9["counts"], field_mismatches=bad))

E = V9 / "eval"; E.mkdir(exist_ok=True)
by = {}
for r in rows:
    by.setdefault(r["size"], []).append(r)
summary = {s: dict(photos=len(rs), v8_wall_s=round(statistics.mean(r["v8_wall_s"] for r in rs), 2), v9_wall_s=round(statistics.mean(r["v9_wall_s"] for r in rs), 2),
                   speedup=round(statistics.mean(r["v8_wall_s"] for r in rs) / statistics.mean(r["v9_wall_s"] for r in rs), 2)) for s, rs in by.items()}
(E / "compare_v8_v9.json").write_text(json.dumps(dict(
    all_identical=all(r["identical"] for r in rows), photos=len(rows), leaves=sum(r["leaves"] for r in rows), reps=a.reps,
    device=res["v9"]["device"], load_s={v: round(res[v]["load_s"], 2) for v in res}, by_size=summary, per_photo=report), ensure_ascii=False, indent=1))
with open(E / "bench.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print("ALL_IDENTICAL", all(r["identical"] for r in rows), "photos", len(rows), "leaves", sum(r["leaves"] for r in rows))
for s, d in summary.items():
    print(s, d)
