# Real-photo comparison of several models, one image per photo (models side by side).
# Each instance: translucent fill + outline in its class color, no per-instance text.
# Top-left legend per tile: model name, class colors, per-class counts.
# Two variants per photo: all classes, and disease-only (Fresh hidden, still counted in the legend).
# Usage: python common/tools/compare_real.py <out_dir> <name>=<weights> [<name>=<weights> ...]
#   e.g. ... v2/eval/compare_3models v1=v1/runs/yolo26s-seg/weights/best.pt ft=v2/runs/ft_from_v1/weights/best.pt
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import paths  # noqa: E402  shared paths + opencv workaround (must come before cv2/ultralytics)

from collections import Counter
from pathlib import Path
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

REAL_DIR = paths.REAL_IMAGES
CLASS_NAMES = paths.CLASS_NAMES
COLORS = [(255, 40, 40), (40, 220, 40), (30, 120, 255), (255, 215, 0)]
FRESH = 1
CONF = 0.25
TILE = 900

def font(size):
    for f in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        if Path(f).exists():
            return ImageFont.truetype(f, size)
    return ImageFont.load_default()

F_TITLE, F_ROW = font(26), font(22)

def render(img_rgb, masks, classes, hide_fresh):
    im = img_rgb.astype(np.float32)
    H, W = im.shape[:2]
    thick = max(2, round(max(H, W) / 600))
    order = [i for i in range(len(classes)) if classes[i] == FRESH] + [i for i in range(len(classes)) if classes[i] != FRESH]
    for i in order:                              # Fresh first so disease outlines stay on top
        c = classes[i]
        if hide_fresh and c == FRESH:
            continue
        m = masks[i]
        im[m] = im[m] * 0.6 + np.array(COLORS[c], np.float32) * 0.4
    out = im.clip(0, 255).astype(np.uint8)
    for i in order:
        c = classes[i]
        if hide_fresh and c == FRESH:
            continue
        cnts, _ = cv2.findContours(masks[i].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, cnts, -1, COLORS[c], thick)
    return out

def legend(tile, title, counts, hide_fresh):
    d = ImageDraw.Draw(tile, "RGBA")
    rows = [(c, CLASS_NAMES[c], counts.get(c, 0)) for c in range(len(CLASS_NAMES))]
    w, lh = 330, 30
    d.rectangle((8, 8, 8 + w, 8 + 40 + lh * len(rows) + 8), fill=(0, 0, 0, 170))
    d.text((18, 14), title, font=F_TITLE, fill=(255, 255, 255))
    for k, (c, name, n) in enumerate(rows):
        y = 8 + 44 + lh * k
        hidden = hide_fresh and c == FRESH
        d.rectangle((18, y + 4, 40, y + 24), outline=COLORS[c], width=3,
                    fill=None if hidden else COLORS[c] + (110,))
        txt = f"{name.replace(' Leaves', '')}: {n}" + ("  (hidden)" if hidden else "")
        d.text((50, y + 2), txt, font=F_ROW, fill=(160, 160, 160) if hidden else (255, 255, 255))

out_dir = Path(sys.argv[1]); out_dir.mkdir(parents=True, exist_ok=True)
models = [a.split("=", 1) for a in sys.argv[2:]]
photos = sorted(REAL_DIR.glob("*.webp"))
preds = {}
for name, wp in models:
    model = YOLO(wp)
    for p in photos:
        r = model.predict(str(p), imgsz=640, conf=CONF, device=0, retina_masks=True, verbose=False)[0]
        masks = (r.masks.data.cpu().numpy() > 0.5) if r.masks is not None else np.zeros((0, 1, 1), bool)
        cls = r.boxes.cls.cpu().numpy().astype(int).tolist()
        preds[(name, p.name)] = (masks, cls)
        print("CNT", name, p.name, {CLASS_NAMES[c].split()[0]: n for c, n in sorted(Counter(cls).items())}, flush=True)

for p in photos:
    img = np.asarray(Image.open(p).convert("RGB"))
    for hide_fresh, tag in ((False, "all"), (True, "disease")):
        tiles = []
        for name, _ in models:
            masks, cls = preds[(name, p.name)]
            t = Image.fromarray(render(img, masks, cls, hide_fresh))
            t.thumbnail((TILE, TILE), Image.LANCZOS)
            legend(t, name, Counter(cls), hide_fresh)
            tiles.append(t)
        Wt = sum(t.width for t in tiles) + 8 * (len(tiles) - 1)
        sheet = Image.new("RGB", (Wt, max(t.height for t in tiles)), (17, 17, 17))
        x = 0
        for t in tiles:
            sheet.paste(t, (x, 0)); x += t.width + 8
        fp = out_dir / f"{p.stem}_{tag}.jpg"
        sheet.save(fp, quality=90)
        print("SAVED", fp, sheet.size, flush=True)
