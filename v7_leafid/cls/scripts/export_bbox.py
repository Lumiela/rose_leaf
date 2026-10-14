# Stage-1 -> stage-2 input, bbox variant: square crop (side = BOX_SCALE x leaf long side) around every stage-1 leaf,
# taken from the white-balanced photo WITH its real background, resized to SIZE. Also writes judge/hold flags:
# hold = several leaves merged (solidity < 0.8) or cut by the image border.
# Usage: python v7_leafid/cls/scripts/export_bbox.py   (uses V7_RUN from cls_common)
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from cls_common import V7_RUN, BOX_SCALE, Progress, gray_world, square_crop, to_input, v7_hold, paths  # noqa: E402

import json  # noqa: E402
from collections import defaultdict  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

Progress.log("export_bbox", "start", "running")
rows = json.loads((V7_RUN / "leaves.json").read_text())
hold = v7_hold()
by_photo = defaultdict(list)
for r in rows:
    by_photo[r["photo"]].append(r)
src_dir = paths.COMMON / "real_images_camfix"
for photo, rs in by_photo.items():
    img = gray_world(np.asarray(Image.open(src_dir / photo).convert("RGB")))
    od = V7_RUN / "bbox" / Path(photo).stem
    od.mkdir(parents=True, exist_ok=True)
    for r in rs:
        side = BOX_SCALE * max(r["x1"] - r["x0"], r["y1"] - r["y0"])
        crop = square_crop(img, (r["x0"] + r["x1"]) / 2, (r["y0"] + r["y1"]) / 2, side)
        to_input(crop).save(od / f"{r['id']}.jpg", quality=95)
    Progress.log("export_bbox", f"{photo}: {len(rs)} crops, hold {sum(hold[r['id']] for r in rs)}")
(V7_RUN / "bbox_flags.json").write_text(json.dumps({k: {"hold": v} for k, v in hold.items()}, indent=0))
Progress.log("export_bbox", f"done: {len(rows)} crops, hold {sum(hold.values())} -> {V7_RUN / 'bbox'}", "done")
