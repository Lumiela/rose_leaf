# Step 4: export reviewed results.
#   real_cutouts/<class>/<photo>_<id>.png : RGBA cutouts of accepted leaves (same format as common/Dataset)
#   labels_draft/{images,labels}/          : DRAFT YOLO-seg labels for all 3 photos (refined proposals, class Fresh,
#                                            proposals overlapping dead-leaf boxes dropped) -> import into CVAT / Label Studio
#                                            and correct merged / missing leaves before using them as ground truth
#   review/final_cutouts_sheet.jpg, review/<photo>_draft_labels.jpg : visual checks
# Usage: python v4_reallabel/scripts/finalize.py > v4_reallabel/logs/finalize.log 2>&1
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
import paths  # noqa: E402

import json  # noqa: E402
import numpy as np  # noqa: E402
import cv2  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

V4 = paths.V4
PRE, REV = V4 / "prelabel", V4 / "review"
CUT, LAB = V4 / "real_cutouts", V4 / "labels_draft"
DEC = json.loads((REV / "decisions.json").read_text())
CLS = {n.split()[0]: i for i, n in enumerate(paths.CLASS_NAMES)}
MIN_AREA = 400


def overlaps(box, boxes):
    return any(box[0] < b[2] and b[0] < box[2] and box[1] < b[3] and b[1] < box[3] for b in boxes)


for d in (CUT, LAB / "images", LAB / "labels"):
    d.mkdir(parents=True, exist_ok=True)
thumbs = []
for meta_f in sorted(PRE.glob("*_proposals.json")):
    meta = json.loads(meta_f.read_text())
    stem = meta_f.name.replace("_proposals.json", "")
    img = np.asarray(Image.open(paths.REAL_IMAGES / meta["photo"]).convert("RGB"))
    H, W = img.shape[:2]
    ref = np.load(PRE / f"{stem}_refined.npz")
    props = {p["id"]: p for p in meta["proposals"]}

    # cutouts (training photos only)
    acc = DEC.get(stem, {}).get("accept", {})
    if stem != "rose_real3":
        for sid, cname in acc.items():
            p = props[int(sid)]
            x0, y0, x1, y1 = p["box"]
            m = ref[f"m{sid}"]
            a = (cv2.GaussianBlur(m.astype(np.float32), (3, 3), 0) * 255).astype(np.uint8)   # 1 px soft edge
            rgba = np.dstack([img[y0:y1, x0:x1], a])
            out = CUT / f"{cname} Leaves" / f"{stem}_{sid}.png"
            out.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(rgba, "RGBA").save(out)
            thumbs.append(Image.fromarray(rgba, "RGBA"))

    # draft labels: every refined proposal, class Fresh, minus dead-leaf boxes
    excl = DEC["exclude_boxes"].get(stem, [])
    lines, kept_masks, dropped = [], [], 0
    for p in meta["proposals"]:
        m = ref[f"m{p['id']}"]
        if m.sum() < MIN_AREA or overlaps(p["box"], excl):
            dropped += 1
            continue
        cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        c = max(cnts, key=cv2.contourArea)
        c = cv2.approxPolyDP(c, 0.003 * cv2.arcLength(c, True), True).reshape(-1, 2) + [p["box"][0], p["box"][1]]
        if len(c) < 3:
            dropped += 1
            continue
        cls = CLS[acc.get(str(p["id"]), "Fresh")]
        lines.append(f"{cls} " + " ".join(f"{x / W:.5f} {y / H:.5f}" for x, y in c))
        kept_masks.append(c)
    Image.fromarray(img).save(LAB / "images" / f"{stem}.jpg", quality=95)
    (LAB / "labels" / f"{stem}.txt").write_text("\n".join(lines))
    vis = img.copy()
    cv2.polylines(vis, [c.astype(np.int32) for c in kept_masks], True, (0, 255, 255), 3)
    for b in excl:
        cv2.rectangle(vis, (b[0], b[1]), (b[2], b[3]), (255, 0, 255), 4)
    Image.fromarray(vis).save(REV / f"{stem}_draft_labels.jpg", quality=88)
    print("FINAL", stem, "cutouts", 0 if stem == "rose_real3" else len(acc), "draft_instances", len(lines), "dropped", dropped, flush=True)

(LAB / "data.yaml").write_text(f"path: {LAB.as_posix()}\ntrain: images\nval: images\nnames:\n" +
                               "".join(f"  {i}: {n}\n" for i, n in enumerate(paths.CLASS_NAMES)))
cell, cols = 130, 9
rows = -(-len(thumbs) // cols)
sheet = Image.new("RGB", (cols * cell, rows * cell), (128, 128, 128))
for i, t in enumerate(thumbs):
    t = t.copy(); t.thumbnail((cell - 8, cell - 8))
    sheet.paste(t, ((i % cols) * cell + (cell - t.width) // 2, (i // cols) * cell + (cell - t.height) // 2), t)
sheet.save(REV / "final_cutouts_sheet.jpg", quality=92)
print("CUTOUTS", len(thumbs), "->", CUT, flush=True)
