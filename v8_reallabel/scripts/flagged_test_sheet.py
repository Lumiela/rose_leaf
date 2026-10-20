# sheet of stage-2 non-Fresh judged leaves for the given photos (context crops), sorted by class then confidence
import sys, csv, json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
run, outp, photos = Path(sys.argv[1]), sys.argv[2], set(sys.argv[3:])
rows = [r for r in csv.DictReader(open(run / "stage2" / "leaves_stage2.csv")) if r["hold"] == "False" and r["pred"] != "Fresh" and Path(r["photo"]).stem in photos]
rows.sort(key=lambda r: (r["pred"], -float(r["conf"])))
try:
    F = ImageFont.load_default(size=16)
except TypeError:
    F = ImageFont.load_default()
t = 200; cols = 5
sh = Image.new("RGB", (cols * t, max(1, -(-len(rows) // cols)) * (t + 22)), (25, 25, 25)); d = ImageDraw.Draw(sh)
for k, r in enumerate(rows):
    im = Image.open(run / "context" / Path(r["photo"]).stem / f"{r['id']}.jpg"); im.thumbnail((t - 4, t - 4))
    x, y = (k % cols) * t, (k // cols) * (t + 22)
    sh.paste(im, (x + 2, y + 22)); d.text((x + 3, y + 2), f"{r['pred'][:6]} {float(r['conf']):.2f} {r['id'][-9:]}", font=F, fill=(255, 220, 0))
sh.save(outp, quality=90); print(len(rows), "flagged ->", outp)
