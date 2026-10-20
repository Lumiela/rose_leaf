# example sheet for the report: real test leaf (left) and the same leaf with its synthetic anomaly (right), 4 pairs per anomaly type
import random
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
V8 = Path(__file__).resolve().parents[1]; D = V8 / "cls" / "data"
rows = [("Hole", "Hole Leaves"), ("Yellow", "Yellow Leaves"), ("Black Spot", "Black Spot Leaves")]
try:
    F = ImageFont.load_default(size=20); FB = ImageFont.load_default(size=24)
except TypeError:
    F = FB = ImageFont.load_default()
t, gap, head, lab = 224, 14, 34, 150
W = lab + 4 * (2 * t + gap); H = len(rows) * (t + head)
sh = Image.new("RGB", (W, H), (245, 246, 241)); d = ImageDraw.Draw(sh)
random.seed(5)
for r, (name, folder) in enumerate(rows):
    anom = sorted((D / "test_real_anom" / folder).glob("v8anom_*.jpg"))
    pick = random.sample(anom, 4)
    y = r * (t + head)
    d.text((10, y + head + t // 2 - 14), name, font=FB, fill=(47, 107, 60))
    for k, a in enumerate(pick):
        lid = a.stem[len("v8anom_"):]
        orig = D / "test_real" / "Fresh Leaves" / f"v8real_{lid}.jpg"
        x = lab + k * (2 * t + gap)
        d.text((x + 4, y + 6), "original", font=F, fill=(93, 104, 94)); d.text((x + t + 4, y + 6), "synthetic " + name.lower(), font=F, fill=(163, 50, 43))
        sh.paste(Image.open(orig).convert("RGB").resize((t, t)), (x, y + head)); sh.paste(Image.open(a).convert("RGB").resize((t, t)), (x + t, y + head))
out = V8 / "share_assets" / "09_synthetic_anomaly_examples.jpg"; sh.save(out, quality=88)
print(out, sh.size, round(out.stat().st_size / 1e3), "KB")
