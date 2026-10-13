# Side-by-side real-photo comparison (masks only, class colors) + per-class counts.
# Usage: python compare_real.py <out.jpg> <name>=<weights> [<name>=<weights> ...]
import importlib.abc, sys

# Shared env has mixed opencv 4.7/4.11 files; skip the broken cv2.typing submodule.
class _BlockCv2Typing(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name == "cv2.typing":
            raise ImportError("cv2.typing skipped")
        return None

sys.meta_path.insert(0, _BlockCv2Typing())

from collections import Counter
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
REAL_DIR = ROOT.parent / "runs/rose_seg/eval_test_20261012_0647/real_images"
COLORS = np.array([(255, 40, 40), (40, 230, 40), (40, 120, 255), (255, 210, 0)], np.float32)  # BlackSpot, Fresh, Hole, Yellow
out_path = sys.argv[1]
models = [a.split("=", 1) for a in sys.argv[2:]]
photos = sorted(REAL_DIR.glob("*.webp"))
W = 520
cols = []
for name, wp in models:
    model = YOLO(wp)
    tiles = []
    for p in photos:
        r = model.predict(str(p), imgsz=640, conf=0.25, device=0, retina_masks=True, verbose=False)[0]
        im = np.asarray(Image.open(p).convert("RGB"), np.float32)
        if r.masks is not None:
            for m, c in zip(r.masks.data.cpu().numpy() > 0.5, r.boxes.cls.cpu().numpy().astype(int)):
                im[m] = im[m] * 0.45 + COLORS[c] * 0.55
        cnt = Counter(r.names[int(c)].split()[0] for c in r.boxes.cls)
        print("CMP", name, p.name, "n=", len(r.boxes), dict(cnt), flush=True)
        t = Image.fromarray(im.clip(0, 255).astype(np.uint8)); t.thumbnail((W, W))
        d = ImageDraw.Draw(t); d.rectangle((0, 0, W, 22), fill=(0, 0, 0))
        d.text((6, 5), f"{name} | {p.stem} | n={len(r.boxes)} " + " ".join(f"{k}:{v}" for k, v in sorted(cnt.items())), fill=(255, 255, 255))
        tiles.append(t)
    cols.append(tiles)
H = sum(max(c[i].height for c in cols) + 6 for i in range(len(photos)))
sheet = Image.new("RGB", (len(cols) * (W + 6), H), (17, 17, 17))
y = 0
for i in range(len(photos)):
    for j, c in enumerate(cols):
        sheet.paste(c[i], (j * (W + 6), y))
    y += max(c[i].height for c in cols) + 6
sheet.save(out_path, quality=88)
print("SAVED", out_path, sheet.size)
