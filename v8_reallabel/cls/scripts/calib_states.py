# v8 stage 2 decision rule, chosen on VAL, applied once to TEST.
# States per judged leaf:
#   HOLD     quality gate: sharpness < s_min OR dark share > d_max OR leaf long side < L_min  (not judged, look again later)
#   ANOMALY  max(p_Black, p_Hole, p_Yellow) >= t_anom  (class = argmax of the anomaly probabilities)
#   FRESH    p_Fresh >= t_fresh
#   UNKNOWN  otherwise ("abnormal-looking but unsure" -> watch list, not an alarm)
# Objective on val: maximise synthetic-anomaly detection (ANOMALY) on real leaves subject to
#   real healthy -> ANOMALY <= 0.5 %, real healthy -> UNKNOWN <= 3 %, real healthy -> HOLD <= 15 %.
# Inputs: cls/runs/<name>/probs_{val,test_real,test_real_anom}.csv + v8 stage-1 leaves.json (features by leaf id)
# Usage: python v8_reallabel/cls/scripts/calib_states.py [--name effnetv2s] [--run pipeline/v8_stage1_nobg]
import argparse, csv, json, itertools  # noqa: E401
from pathlib import Path
import numpy as np
V8 = Path(__file__).resolve().parents[2]
ap = argparse.ArgumentParser(); ap.add_argument("--name", default="effnetv2s"); ap.add_argument("--run", default="pipeline/v8_stage1_nobg")
a = ap.parse_args()
R = V8 / "cls" / "runs" / a.name
feat = {r["id"]: r for r in json.loads((V8 / a.run / "leaves.json").read_text())}


def load(split, prefix):
    out = []
    for r in csv.DictReader(open(R / f"probs_{split}.csv")):
        name = Path(r["path"]).stem
        if not name.startswith(prefix):
            continue
        lid = name[len(prefix):]
        for suf in ("_c", "_d"):
            if lid.endswith(suf): lid = lid[:-2]
        f = feat.get(lid)
        if f is None:
            continue
        p = np.array([float(r["p_Black"]), float(r["p_Fresh"]), float(r["p_Hole"]), float(r["p_Yellow"])])
        out.append(dict(id=lid, label=r["label"], p=p, sharp=float(f["sharpness"]), dark=float(f["dark"]), L=max(f["x1"] - f["x0"], f["y1"] - f["y0"])))
    return out


def state(x, th):
    s_min, d_max, L_min, t_anom, t_fresh = th
    if x["sharp"] < s_min or x["dark"] > d_max or x["L"] < L_min:
        return "HOLD"
    pa = x["p"][[0, 2, 3]]
    if pa.max() >= t_anom:
        return "ANOMALY"
    if x["p"][1] >= t_fresh:
        return "FRESH"
    return "UNKNOWN"


def rates(xs, th):
    st = [state(x, th) for x in xs]; n = max(1, len(st))
    return {k: round(st.count(k) / n, 4) for k in ("FRESH", "ANOMALY", "UNKNOWN", "HOLD")}


val_h, val_a = load("val", "v8real_"), load("val", "v8anom_")
sh = np.array([x["sharp"] for x in val_h]); dk = np.array([x["dark"] for x in val_h]); Ls = np.array([x["L"] for x in val_h])
grid_s = [0] + [float(np.quantile(sh, q)) for q in (0.03, 0.06, 0.1)]
grid_d = [1.01] + [float(np.quantile(dk, q)) for q in (0.97, 0.94, 0.9)]
grid_L = [0, 30, 40]
best = None
for th in itertools.product(grid_s, grid_d, grid_L, [0.5, 0.6, 0.7, 0.8, 0.9, 0.95], [0.3, 0.4, 0.5, 0.6, 0.7]):
    rh, ra = rates(val_h, th), rates(val_a, th)
    if rh["ANOMALY"] > 0.005 or rh["UNKNOWN"] > 0.03 or rh["HOLD"] > 0.15:
        continue
    score = (ra["ANOMALY"], ra["ANOMALY"] + ra["UNKNOWN"], -rh["HOLD"])
    if best is None or score > best[0]:
        best = (score, th, rh, ra)
assert best, "no threshold set meets the val constraints"
_, th, rh, ra = best
names = ["s_min", "d_max", "L_min", "t_anom", "t_fresh"]
res = dict(thresholds=dict(zip(names, [round(v, 4) for v in th])), val=dict(real_healthy=rh, real_synthetic_anomaly=ra, n_healthy=len(val_h), n_anom=len(val_a)))
te_h, te_a = load("test_real", "v8real_"), load("test_real_anom", "v8anom_")
res["test"] = dict(real_healthy=rates(te_h, th), real_synthetic_anomaly=rates(te_a, th), n_healthy=len(te_h), n_anom=len(te_a))
for cname in ("Black", "Hole", "Yellow"):
    xs = [x for x in te_a if x["label"] == cname]
    st = [state(x, th) for x in xs]
    ok = sum(1 for x, s in zip(xs, st) if s == "ANOMALY" and ["Black", "Hole", "Yellow"][int(np.argmax(x["p"][[0, 2, 3]]))] == cname)
    res["test"][f"synthetic_{cname}"] = dict(n=len(xs), anomaly=round(st.count("ANOMALY") / max(1, len(xs)), 3), correct_class=round(ok / max(1, len(xs)), 3), unknown=round(st.count("UNKNOWN") / max(1, len(xs)), 3))
# default argmax rule (no gate, no unknown) for comparison
argm = lambda xs: round(sum(1 for x in xs if int(np.argmax(x["p"])) != 1) / max(1, len(xs)), 4)
res["test"]["argmax_rule"] = dict(real_healthy_flagged=argm(te_h), synthetic_detected=argm(te_a))
(R / "states.json").write_text(json.dumps(res, indent=1)); print(json.dumps(res, indent=1))
