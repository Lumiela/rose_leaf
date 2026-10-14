# Render the classifier dataset from split.json (RGBA cutouts -> gray-letterboxed ${SIZE} px JPEGs).
#   train/<class>/      1 clean (random rotation/flip) + N degraded variants per image (N per class balances classes)
#   val/<class>/        clean + 1 degraded per image (model selection on both conditions)
#   test/<class>/       clean originals          test_deg/<class>/   1 degraded copy (fixed seed) = cube-crop-like
# Usage: python v7_leafid/cls/scripts/render_data.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from cls_common import DATA, CLS, NAMES, SHORT, compose, degrade, load_rgba, Progress, paths  # noqa: E402

import json  # noqa: E402
import shutil  # noqa: E402
from multiprocessing import Pool  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

N_DEG = {"Black": 2, "Fresh": 1, "Hole": 2, "Yellow": 10}


def rot(im, rng):
    im = im.rotate(rng.uniform(0, 360), resample=Image.BICUBIC, expand=True)
    return im.transpose(Image.FLIP_LEFT_RIGHT) if rng.random() < 0.5 else im


def job(args):
    split, rel, ci, idx = args
    rng = np.random.default_rng([42, {"train": 0, "val": 1, "test": 2}[split], idx])
    rgba = load_rgba(paths.DATASET / rel)
    stem, c = Path(rel).stem, NAMES[ci]
    outs = []
    if split == "train":
        outs.append((f"train/{c}/{stem}_c.jpg", compose(rot(rgba, rng))))
        for k in range(N_DEG[SHORT[ci]]):
            outs.append((f"train/{c}/{stem}_d{k}.jpg", compose(degrade(rot(rgba, rng), rng))))
    elif split == "val":
        outs += [(f"val/{c}/{stem}_c.jpg", compose(rgba)), (f"val/{c}/{stem}_d.jpg", compose(degrade(rgba, rng)))]
    else:
        outs += [(f"test/{c}/{stem}.jpg", compose(rgba)), (f"test_deg/{c}/{stem}.jpg", compose(degrade(rgba, rng)))]
    for rp, im in outs:
        im.save(DATA / rp, quality=95)
    return len(outs)


if __name__ == "__main__":
    Progress.log("render", "start")
    split = json.loads((CLS / "split.json").read_text())
    if DATA.exists():
        shutil.rmtree(DATA)
    for s in ("train", "val", "test", "test_deg"):
        for c in NAMES:
            (DATA / s / c).mkdir(parents=True, exist_ok=True)
    for s in ("train", "val", "test"):
        jobs = [(s, rel, ci, i) for i, (rel, ci) in enumerate(split[s])]
        with Pool(4) as pool:
            n = 0
            for k, r in enumerate(pool.imap_unordered(job, jobs, chunksize=16), 1):
                n += r
                if k % 500 == 0:
                    Progress.log("render", f"{s} {k}/{len(jobs)} originals")
        counts = {sd: {SHORT[i]: len(list((DATA / sd / c).glob('*.jpg'))) for i, c in enumerate(NAMES)}
                  for sd in ((s,) if s != "test" else ("test", "test_deg"))}
        Progress.log("render", f"{s}: {n} images {counts}")
    Progress.log("render", "dataset ready -> " + str(DATA), "done")
