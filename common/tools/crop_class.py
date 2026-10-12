# Crop every instance of one class predicted on the real photos, for visual checking.
# Usage: python common/tools/crop_class.py <weights> <class name> <out.jpg> [conf]
#   e.g. python common/tools/crop_class.py v4_reallabel/runs/ft_from_v2ft/weights/best.pt "Yellow Leaves" /tmp/yellow.jpg
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import paths  # noqa: E402  shared paths + opencv workaround

import numpy as np  # noqa: E402
import cv2  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402
from ultralytics import YOLO  # noqa: E402

weights, cname, out = sys.argv[1], sys.argv[2], sys.argv[3]
conf = float(sys.argv[4]) if len(sys.argv) > 4 else 0.25
model = YOLO(weights)
cid = paths.CLASS_NAMES.index(cname)
try:
    F = ImageFont.load_default(size=18)
except TypeError:
    F = ImageFont.load_default()
cells = []
for p in sorted(paths.REAL_IMAGES.glob("*.webp")):
    img = np.asarray(Image.open(p).convert("RGB"))
    H, W = img.shape[:2]
    r = model.predict(Image.fromarray(img), imgsz=640, conf=conf, device=0, retina_masks=True, verbose=False)[0]
    if r.masks is None:
        continue
    for m, c, s, b in zip(r.masks.data.cpu().numpy() > 0.5, r.boxes.cls.cpu().numpy().astype(int),
                          r.boxes.conf.cpu().numpy(), r.boxes.xyxy.cpu().numpy().astype(int)):
        if c != cid:
            continue
        x0, y0, x1, y1 = b
        mg = max(60, int(0.6 * max(x1 - x0, y1 - y0)))
        cx0, cy0, cx1, cy1 = max(0, x0 - mg), max(0, y0 - mg), min(W, x1 + mg), min(H, y1 + mg)
        crop = img[cy0:cy1, cx0:cx1].copy()
        cnts, _ = cv2.findContours(m[cy0:cy1, cx0:cx1].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(crop, cnts, -1, (255, 0, 255), 2)
        t = Image.fromarray(crop); t.thumbnail((300, 300))
        cell = Image.new("RGB", (310, 340), (20, 20, 20)); cell.paste(t, ((310 - t.width) // 2, 34))
        ImageDraw.Draw(cell).text((6, 6), f"{p.stem} ({x0},{y0}) {s:.2f}", font=F, fill=(255, 255, 0))
        cells.append(cell)
        print("INST", p.stem, [int(v) for v in b], round(float(s), 3), flush=True)
cols = 4
rows = max(1, -(-len(cells) // cols))
sheet = Image.new("RGB", (cols * 310, rows * 340), (20, 20, 20))
for i, c in enumerate(cells):
    sheet.paste(c, ((i % cols) * 310, (i // cols) * 340))
sheet.save(out, quality=90)
print("SAVED", out, len(cells), flush=True)
