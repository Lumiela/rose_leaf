# Visualize synthetic scenes: raw image (left) + label polygons (right).
# Usage: python common/tools/viz_labels.py <dataset_dir> <split> <n> <out.jpg>   e.g. v2/data/rose_seg_v2 train 6 v2/debug/check.jpg
import sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw

ds, split, n, out = Path(sys.argv[1]), sys.argv[2], int(sys.argv[3]), sys.argv[4]
COLORS = [(255, 60, 60), (60, 230, 60), (60, 140, 255), (255, 210, 0)]   # BlackSpot, Fresh, Hole, Yellow
imgs = sorted((ds / "images" / split).glob("*.jpg"))
rng = np.random.default_rng(1)
pick = [imgs[i] for i in sorted(rng.choice(len(imgs), min(n, len(imgs)), replace=False))]
T = 400
sheet = Image.new("RGB", (2 * T + 10, len(pick) * (T + 6)), (17, 17, 17))
for r, p in enumerate(pick):
    im = Image.open(p).convert("RGB")
    lab = im.copy(); d = ImageDraw.Draw(lab, "RGBA")
    lines = (ds / "labels" / split / (p.stem + ".txt")).read_text().splitlines()
    for line in lines:
        v = line.split(); ci = int(v[0])
        pts = (np.array(v[1:], np.float32).reshape(-1, 2) * im.width).tolist()
        d.polygon([tuple(q) for q in pts], fill=COLORS[ci] + (90,), outline=COLORS[ci] + (255,))
    sheet.paste(im.resize((T, T)), (0, r * (T + 6)))
    sheet.paste(lab.resize((T, T)), (T + 10, r * (T + 6)))
    print("VIZ", p.name, "instances", len(lines))
sheet.save(out, quality=88)
