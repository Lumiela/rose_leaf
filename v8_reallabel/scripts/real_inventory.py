# inventory of all real photos on the server: md5, size, near-duplicates (32x24 grayscale correlation)
import sys, hashlib, glob
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
import paths  # noqa: F401,E402
from PIL import Image, ImageOps  # noqa: E402
import numpy as np  # noqa: E402

fs = sorted(glob.glob("common/real_images*/*") + glob.glob("v6_holefilter/inputs/*"))
rows = []
for f in fs:
    b = open(f, "rb").read()
    im = ImageOps.exif_transpose(Image.open(f)).convert("L"); w, h = im.size
    t = np.asarray(im.resize((32, 24)), np.float32); t = (t - t.mean()) / (t.std() + 1e-6)
    rows.append((f, hashlib.md5(b).hexdigest()[:10], w, h, t.ravel()))
for f, m, w, h, _ in rows:
    print(f"{m} {w}x{h} {f}")
print("--- near-duplicate pairs (corr>0.9)")
for i in range(len(rows)):
    for j in range(i + 1, len(rows)):
        c = float(np.dot(rows[i][4], rows[j][4]) / rows[i][4].size)
        if c > 0.9:
            print(f"{c:.3f} {rows[i][0]} <> {rows[j][0]}")
