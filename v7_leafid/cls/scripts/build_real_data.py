# Step 3-4 data for the bbox_real variant: data_bbox (scale-matched synthetic) + reviewed real cube leaves as Fresh.
#   real labels: cls/real/real_labels.json (F healthy -> used; U uncertain / X not a leaf -> excluded)
#   train photos (14 phone + rose_real 1,2): each healthy leaf -> train/Fresh clean crop + 1 degraded copy
#   val photos (cube_11, cube_17): clean crop -> val/Fresh (model selection sees real leaves too)
#   test photos (cube_21, cube_22): clean crop -> test_real/Fresh (held-out real metric A4, not mixed into test/test_deg
#   so the synthetic B metrics stay comparable with the bbox run)
# Degraded copy = the native-resolution square shrunk x U(0.4, 1.0), blur (70%), brightness/contrast/colour jitter,
# same recipe as render_data_bbox.py.  Synthetic files are hard links to data_bbox (no extra disk).
# Usage: CLS_VARIANT=bbox_real python v7_leafid/cls/scripts/build_real_data.py
import os, sys  # noqa: E401
os.environ.setdefault("CLS_VARIANT", "bbox_real")
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from cls_common import CLS, DATA, NAMES, BOX_SCALE, Progress, to_input  # noqa: E402

import json, os as _os, shutil  # noqa: E401,E402
import numpy as np  # noqa: E402
from PIL import Image, ImageFilter, ImageEnhance  # noqa: E402

SRC = CLS / "data_bbox"
FRESH = NAMES[1]


def degrade(im, native, rng):
    n = max(16, round(native * rng.uniform(0.4, 1.0)))
    im = im.resize((n, n), Image.BILINEAR)
    if rng.random() < 0.7:
        im = im.filter(ImageFilter.GaussianBlur(rng.uniform(0.3, 1.2)))
    im = ImageEnhance.Brightness(im).enhance(rng.uniform(0.8, 1.2))
    im = ImageEnhance.Contrast(im).enhance(rng.uniform(0.85, 1.15))
    im = ImageEnhance.Color(im).enhance(rng.uniform(0.85, 1.15))
    return to_input(im)


if __name__ == "__main__":
    Progress.log("build_real", "start", "running")
    if DATA.exists():
        shutil.rmtree(DATA)
    n_syn = 0
    for p in SRC.rglob("*.jpg"):
        q = DATA / p.relative_to(SRC); q.parent.mkdir(parents=True, exist_ok=True)
        _os.link(p, q); n_syn += 1
    for c in NAMES:
        (DATA / "test_real" / c).mkdir(parents=True, exist_ok=True)
    lab = json.loads((CLS / "real" / "real_labels.json").read_text())["labels"]
    cand = {r["id"]: r for r in json.loads((CLS / "real" / "candidates.json").read_text())}
    cnt = {"train": 0, "val": 0, "test_real": 0, "excluded": 0}
    for i, (lid, tag) in enumerate(sorted(lab.items())):
        r = cand[lid]
        if tag != "F":
            cnt["excluded"] += 1; continue
        im = Image.open(CLS / r["crop"]).convert("RGB")
        if r["split"] == "train":
            rng = np.random.default_rng([11, i])
            im.save(DATA / "train" / FRESH / f"real_{lid}_c.jpg", quality=95)
            degrade(im, BOX_SCALE * r["long_side"], rng).save(DATA / "train" / FRESH / f"real_{lid}_d.jpg", quality=95)
            cnt["train"] += 1
        elif r["split"] == "val":
            im.save(DATA / "val" / FRESH / f"real_{lid}_c.jpg", quality=95); cnt["val"] += 1
        else:
            im.save(DATA / "test_real" / FRESH / f"real_{lid}.jpg", quality=95); cnt["test_real"] += 1
    syn_f = len([p for p in (DATA / "train" / FRESH).glob("*.jpg") if not p.name.startswith("real_")])
    Progress.log("build_real", f"synthetic {n_syn} linked; real healthy leaves: train {cnt['train']} (x2 images, "
                 f"synthetic Fresh {syn_f}), val {cnt['val']}, test_real {cnt['test_real']}, excluded U/X {cnt['excluded']}", "done")
