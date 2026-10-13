# Step 1: extract leaf-free background crops from real greenhouse photos.
# Reads common/real_images; writes only v2/backgrounds and v2/debug.
# Usage: python v2/scripts/make_backgrounds.py
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
import paths  # noqa: E402  shared paths + opencv workaround (must come before cv2/ultralytics)

from pathlib import Path
import numpy as np
import cv2
from PIL import Image

SRC = paths.REAL_IMAGES
SOURCES = ["rose_real.webp", "rose_real2.webp"]   # rose_real3.webp is held out for evaluation
OUT = paths.V2 / "backgrounds"
DBG = paths.V2 / "debug"
PER_IMAGE = 60
MIN_SIDE, MAX_FRAC = 256, 0.75      # crop short-side range (px, fraction of short side)
ASPECT = (0.6, 1.6)                 # crop width / height
MAX_LEAF = 0.005                    # max leaf-pixel ratio inside a crop
MAX_IOU = 0.6                       # diversity: max IoU with already accepted crops
SEED = 42

def leaf_mask(rgb):
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    m = ((h >= 22) & (h <= 95) & (s >= 45) & (v >= 40)).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return cv2.dilate(m, np.ones((31, 31), np.uint8))   # cover leaf edges and thin stems

def iou(a, b):
    x0, y0, x1, y1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x1 - x0) * max(0, y1 - y0)
    area = lambda r: (r[2] - r[0]) * (r[3] - r[1])
    return inter / (area(a) + area(b) - inter)

for old in OUT.glob("*.jpg") if OUT.exists() else []:
    old.unlink()   # regenerate from scratch (files created by this script only)
OUT.mkdir(parents=True, exist_ok=True); DBG.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(SEED)
thumbs = []
for name in SOURCES:
    rgb = np.asarray(Image.open(SRC / name).convert("RGB"))
    H, W = rgb.shape[:2]
    m = leaf_mask(rgb)
    ii = cv2.integral(m)
    boxes = []
    for _ in range(20000):
        if len(boxes) >= PER_IMAGE:
            break
        side = rng.uniform(MIN_SIDE, MAX_FRAC * min(H, W))
        ar = rng.uniform(*ASPECT)
        cw, ch = min(W, int(side * max(ar, 1))), min(H, int(side / min(ar, 1)))
        x, y = int(rng.integers(0, W - cw + 1)), int(rng.integers(0, H - ch + 1))
        leaf = ii[y + ch, x + cw] - ii[y, x + cw] - ii[y + ch, x] + ii[y, x]
        if leaf > MAX_LEAF * cw * ch:
            continue
        b = (x, y, x + cw, y + ch)
        if any(iou(b, o) > MAX_IOU for o in boxes):
            continue
        boxes.append(b)
    stem = Path(name).stem
    for i, (x0, y0, x1, y1) in enumerate(boxes):
        crop = Image.fromarray(rgb[y0:y1, x0:x1])
        crop.save(OUT / f"{stem}_bg{i:03d}.jpg", quality=95)
        t = crop.copy(); t.thumbnail((160, 160)); thumbs.append(t)
    # debug overlay: leaf mask in red, accepted crops in yellow
    vis = rgb.copy()
    vis[m > 0] = (0.55 * vis[m > 0] + 0.45 * np.array([255, 0, 0])).astype(np.uint8)
    for (x0, y0, x1, y1) in boxes:
        cv2.rectangle(vis, (x0, y0), (x1, y1), (255, 230, 0), 4)
    Image.fromarray(vis).save(DBG / f"{stem}_mask_crops.jpg", quality=85)
    print("SRC", name, f"{W}x{H}", "leaf_ratio=%.3f" % (m.mean()), "crops=", len(boxes),
          "short_side_range=", (min(min(b[2]-b[0], b[3]-b[1]) for b in boxes), max(min(b[2]-b[0], b[3]-b[1]) for b in boxes)) if boxes else None)

# contact sheet
cols = 10
rows = (len(thumbs) + cols - 1) // cols
sheet = Image.new("RGB", (cols * 164, rows * 164), (30, 30, 30))
for i, t in enumerate(thumbs):
    sheet.paste(t, ((i % cols) * 164 + 2, (i // cols) * 164 + 2))
sheet.save(DBG / "backgrounds_sheet.jpg", quality=85)
print("TOTAL", len(thumbs))
