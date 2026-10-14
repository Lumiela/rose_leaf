# Review sheet of suspicious leaf detections (likely background) from a leaf_search run.
# Suspicious = leaf_color_frac < --lc or area_frac > --area or solidity < --sol. Context crops labeled with stats.
# Usage: python v7_leafid/scripts/review_suspects.py <run dir> [--lc 0.8] [--area 0.01] [--sol 0.6]
import sys, json, argparse
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

ap = argparse.ArgumentParser(); ap.add_argument("run"); ap.add_argument("--lc", type=float, default=0.8)
ap.add_argument("--area", type=float, default=0.01); ap.add_argument("--sol", type=float, default=0.6)
a = ap.parse_args()
run = Path(a.run)
rows = json.loads((run / "leaves.json").read_text())
sus = [r for r in rows if r["leaf_color_frac"] < a.lc or r["area_frac"] > a.area or r["solidity"] < a.sol]
try:
    F = ImageFont.load_default(size=17)
except TypeError:
    F = ImageFont.load_default()
cell, cols = 230, 7
sheet = Image.new("RGB", (cols * cell, max(1, -(-len(sus) // cols)) * (cell + 40)), (20, 20, 20))
d = ImageDraw.Draw(sheet)
for k, r in enumerate(sus):
    im = Image.open(run / "context" / Path(r["photo"]).stem / f"{r['id']}.jpg"); im.thumbnail((cell - 6, cell - 6))
    x, y = (k % cols) * cell, (k // cols) * (cell + 40)
    sheet.paste(im, (x + (cell - im.width) // 2, y + 40 + (cell - im.height) // 2))
    d.text((x + 4, y + 2), f"{r['photo'][7:9]}#{r['idx']} {r['v4_cls'][:5]}", font=F, fill=(255, 255, 0))
    d.text((x + 4, y + 20), f"lc{r['leaf_color_frac']:.2f} a{r['area_frac']*100:.1f}% s{r['solidity']:.2f} sh{int(r['sharpness'])}", font=F, fill=(200, 200, 200))
sheet.save(run / "suspects.jpg", quality=88)
print("SUSPECTS", len(sus), "of", len(rows), run / "suspects.jpg")
import collections
for r in sus: print("S", r["id"], r["v4_cls"], r["leaf_color_frac"], r["area_frac"], r["solidity"], int(r["sharpness"]))
