# sheet of the stage-2 "Yellow" judgments on test photos for human confirmation: numbered, larger context, probability shown
import sys, csv, json
from pathlib import Path
from sam_common import load_rgb
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2
run = Path(sys.argv[1]); outp = sys.argv[2]; photos = sys.argv[3:]
rows = [r for r in csv.DictReader(open(run / "stage2" / "leaves_stage2.csv")) if r["hold"] == "False" and r["pred"] == "Yellow" and Path(r["photo"]).stem in photos]
rows.sort(key=lambda r: (r["photo"], -float(r["conf"])))
L = {r["id"]: r for r in json.loads((run / "leaves.json").read_text())}
try:
    F, FB = ImageFont.load_default(size=18), ImageFont.load_default(size=28)
except TypeError:
    F = FB = ImageFont.load_default()
t = 300; cols = 4
sh = Image.new("RGB", (cols * t, -(-len(rows) // cols) * (t + 48)), (25, 25, 25)); d = ImageDraw.Draw(sh)
imgs = {}
info = []
for k, r in enumerate(rows, 1):
    p = r["photo"]
    if p not in imgs: imgs[p] = load_rgb(run.parent.parent / "inputs" / p)
    im = imgs[p]; H, W = im.shape[:2]; q = L[r["id"]]
    x0, y0, x1, y1 = q["x0"], q["y0"], q["x1"], q["y1"]; s = max(x1 - x0, y1 - y0)
    cx, cy = (x0 + x1) // 2, (y0 + y1) // 2; h = int(1.6 * s)
    a0, b0, a1, b1 = max(0, cx - h), max(0, cy - h), min(W, cx + h), min(H, cy + h)
    crop = im[b0:b1, a0:a1].copy()
    m = np.asarray(Image.open(run / "crops" / Path(p).stem / f"{r['id']}.png").getchannel("A")) > 127
    cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(crop, [c + [x0 - a0, y0 - b0] for c in cs], -1, (255, 0, 255), 2)
    c = Image.fromarray(crop); c.thumbnail((t - 6, t - 6))
    x, y = ((k - 1) % cols) * t, ((k - 1) // cols) * (t + 48)
    sh.paste(c, (x + 3, y + 46))
    d.text((x + 6, y + 2), f"#{k}", font=FB, fill=(255, 220, 0))
    d.text((x + 60, y + 4), f"{Path(p).stem} {r['id'][-5:]}", font=F, fill=(255, 255, 255))
    d.text((x + 60, y + 24), f"Yellow p={float(r['p_Yellow']):.2f} Fresh p={float(r['p_Fresh']):.2f}", font=F, fill=(180, 220, 255))
    info.append(dict(n=k, id=r["id"], photo=p, p_yellow=float(r["p_Yellow"]), p_fresh=float(r["p_Fresh"]), box=[x0, y0, x1, y1]))
sh.save(outp, quality=92)
Path(outp).with_suffix(".json").write_text(json.dumps(info, indent=1))
print(len(rows), "->", outp)
