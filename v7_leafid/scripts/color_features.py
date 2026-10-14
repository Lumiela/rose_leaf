# Color features of each detected leaf (from its RGBA cutout) to separate background objects from leaves.
# Usage: python v7_leafid/scripts/color_features.py <run dir>
import sys, json
from pathlib import Path
import numpy as np
from PIL import Image
import colorsys

run = Path(sys.argv[1]); rows = json.loads((run / "leaves.json").read_text())
bg = set("01_15 01_18 01_28 01_35 01_82 02_14 02_19 02_37 02_59 02_74 02_84 03_13 03_23 03_30 03_32 03_34 03_77 04_70 04_85 04_86 04_107 04_116 04_165 05_12 05_44 05_56".split())
leaf = set("01_38 01_71 02_43 03_20 03_66 05_33".split())
out = []
for r in rows:
    a = np.asarray(Image.open(run / "crops" / Path(r["photo"]).stem / f"{r['id']}.png")).astype(np.float32)
    m = a[..., 3] > 128
    rgb = a[..., :3][m] / 255.0
    R, G, B = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    mx, mn = rgb.max(1), rgb.min(1)
    S = np.where(mx > 0, (mx - mn) / np.maximum(mx, 1e-6), 0)
    exg = (2 * G - R - B) / np.maximum(R + G + B, 1e-6)
    key = r["photo"][7:9] + "_" + str(r["idx"])
    f = dict(id=r["id"], key=key, tag="bg" if key in bg else ("leaf*" if key in leaf else ""),
             sat_med=round(float(np.median(S)), 3), exg_med=round(float(np.median(exg)), 3),
             green_frac=round(float(((exg > 0.08) & (S > 0.2)).mean()), 3), dark=round(float((mx < 0.2).mean()), 3),
             lc=r["leaf_color_frac"])
    out.append(f)
(run / "color_features.json").write_text(json.dumps(out, indent=1))
def q(v): v = sorted(v); return [round(v[int(p * (len(v) - 1))], 3) for p in (0, 0.1, 0.5, 0.9, 1)] if v else []
for tag in ("bg", "leaf*", ""):
    sub = [f for f in out if f["tag"] == tag]
    print("TAG", tag or "other", len(sub), "exg", q([f["exg_med"] for f in sub]), "sat", q([f["sat_med"] for f in sub]), "green_frac", q([f["green_frac"] for f in sub]))
for f in out:
    if f["tag"]: print("ROW", f["key"], f["tag"], "exg", f["exg_med"], "sat", f["sat_med"], "gf", f["green_frac"], "dark", f["dark"], "lc", f["lc"])
for th in (0.3, 0.4, 0.5, 0.6):
    drop = [f for f in out if f["green_frac"] < th]
    print("GF<", th, "drop", len(drop), "bg_hit", sum(f["tag"] == "bg" for f in drop), "/", len([f for f in out if f["tag"] == "bg"]),
          "leaf*_hit", sum(f["tag"] == "leaf*" for f in drop), "other_dropped", sum(f["tag"] == "" for f in drop))
