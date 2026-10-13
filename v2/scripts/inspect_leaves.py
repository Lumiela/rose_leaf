# Inspect source leaf cutouts: per-class sample sheet + size/shape stats. Read-only on common/Dataset.
# Usage: python v2/scripts/inspect_leaves.py   -> v2/debug/leaf_samples.jpg
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
import paths  # noqa: E402  shared paths + opencv workaround (must come before cv2/ultralytics)

import random
from pathlib import Path
import numpy as np
from PIL import Image

SRC = paths.DATASET
CLASSES = sorted(d.name for d in SRC.iterdir() if d.is_dir())
random.seed(0)
cell, per = 150, 8
sheet = Image.new("RGB", (per * cell, len(CLASSES) * cell), (128, 128, 128))
for ci, c in enumerate(CLASSES):
    files = sorted(f for f in (SRC / c).glob("*.png") if "_aug_" not in f.name)
    sizes, fills, aspects = [], [], []
    for f in random.sample(files, min(60, len(files))):
        im = Image.open(f).convert("RGBA")
        a = np.asarray(im.getchannel("A")) > 16
        ys, xs = np.nonzero(a)
        bw, bh = xs.max() - xs.min() + 1, ys.max() - ys.min() + 1
        sizes.append(max(im.size)); fills.append(a.sum() / (bw * bh)); aspects.append(max(bw, bh) / min(bw, bh))
    print("CLASS", c, "orig_files", len(files), "img_max_side median", int(np.median(sizes)),
          "fill_in_bbox %.2f" % np.median(fills), "aspect %.2f" % np.median(aspects))
    for j, f in enumerate(random.sample(files, per)):
        im = Image.open(f).convert("RGBA")
        im = im.crop(im.getchannel("A").point(lambda v: 255 if v > 16 else 0).getbbox())
        im.thumbnail((cell - 6, cell - 6))
        bg = Image.new("RGB", (cell, cell), (128, 128, 128))
        bg.paste(im, ((cell - im.width) // 2, (cell - im.height) // 2), im)
        sheet.paste(bg, (j * cell, ci * cell))
sheet.save(paths.V2 / "debug" / "leaf_samples.jpg", quality=88)
print("ORDER", CLASSES)
