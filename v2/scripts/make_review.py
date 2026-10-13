# Build small review images from v2/debug outputs of make_backgrounds.py.
# Usage: python v2/scripts/make_review.py
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
import paths  # noqa: E402  shared paths + opencv workaround (must come before cv2/ultralytics)
from pathlib import Path
from PIL import Image
D = paths.V2 / "debug"
a = Image.open(D / "rose_real_mask_crops.jpg"); b = Image.open(D / "rose_real2_mask_crops.jpg")
a.thumbnail((780, 780)); b.thumbnail((780, 780))
top = Image.new("RGB", (a.width + b.width + 10, max(a.height, b.height)), (17, 17, 17))
top.paste(a, (0, 0)); top.paste(b, (a.width + 10, 0)); top.save(D / "review_masks.jpg", quality=88)
s = Image.open(D / "backgrounds_sheet.jpg"); s.thumbnail((1200, 1200)); s.save(D / "review_sheet.jpg", quality=88)
print("REVIEW", top.size, s.size)
