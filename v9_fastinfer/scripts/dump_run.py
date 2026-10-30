"""Run one deploy package on a list of photos and write timings + a per-leaf digest (JSON).

    python v9_fastinfer/scripts/dump_run.py <deploy dir> <out.json> [--reps N] [--device cuda:0|cpu] <photo> ...

Runs in its own process so v8 and v9 (both packages are named leafdx) never share an interpreter.
Digest per leaf = every field of the output JSON + md5 of the boolean mask, so two runs can be compared exactly.
"""
import argparse, hashlib, json, sys, time  # noqa: E401
from pathlib import Path

ap = argparse.ArgumentParser(); ap.add_argument("deploy"); ap.add_argument("out"); ap.add_argument("photos", nargs="+")
ap.add_argument("--reps", type=int, default=2); ap.add_argument("--device"); ap.add_argument("--warmup", type=int, default=2)
a = ap.parse_args()
sys.path.insert(0, str(Path(a.deploy).resolve()))
import leafdx  # noqa: E402  (applies the opencv workaround before cv2 / ultralytics)
from leafdx import LeafPipeline, load_rgb  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def digest(res):
    out = []
    for l in res["leaves"]:
        d = {k: v for k, v in l.items() if not k.startswith("_")}
        m = l["_mask"]
        d["mask_md5"] = hashlib.md5(np.packbits(m).tobytes() + repr(m.shape).encode()).hexdigest()
        out.append(d)
    return out


t = time.perf_counter()
pl = LeafPipeline(Path(a.deploy) / "models", config=({"device": a.device} if a.device else None)); sync()
load_s = time.perf_counter() - t
first = load_rgb(a.photos[0])
for _ in range(a.warmup):
    pl.run(first)
rows = []
for p in a.photos:
    t = time.perf_counter(); rgb = load_rgb(p); dec = time.perf_counter() - t
    runs = []
    for _ in range(a.reps):
        sync(); t = time.perf_counter(); res = pl.run(rgb); sync()
        runs.append(dict(wall_s=time.perf_counter() - t, **res["timing"]))
    rows.append(dict(photo=p, size=res["size"], L=res["L"], counts=res["counts"], decode_s=dec, runs=runs, leaves=digest(res)))
    print(Path(p).name, len(res["leaves"]), res["counts"], [round(r["wall_s"], 2) for r in runs], flush=True)
Path(a.out).write_text(json.dumps(dict(deploy=str(Path(a.deploy).resolve()), version=leafdx.__version__, load_s=load_s,
                                       device=(torch.cuda.get_device_name(0) if torch.cuda.is_available() and a.device != "cpu" else "cpu"),
                                       photos=rows)))
