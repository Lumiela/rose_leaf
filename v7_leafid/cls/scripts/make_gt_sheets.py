# Stage-0 ground truth for the real-crop check: contact sheets of the judged camfix leaves (solidity >= 0.8, not on the
# border) built from stage-1 context crops (mask outline = the leaf to label). Model predictions are NOT shown, so the
# labels are not anchored on them.  Output: cls/gt/sheets/gt_XX.jpg (30 leaves each, numbered) + cls/gt/index.json
# Usage: python v7_leafid/cls/scripts/make_gt_sheets.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from cls_common import CLS, V7_RUN, v7_hold  # noqa: E402

import json  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

OUT = CLS / "gt"
COLS, ROWS, T, HDR = 6, 5, 210, 22

if __name__ == "__main__":
    (OUT / "sheets").mkdir(parents=True, exist_ok=True)
    rows = json.loads((V7_RUN / "leaves.json").read_text())
    hold = v7_hold()
    judged = sorted((r for r in rows if not hold[r["id"]]), key=lambda r: r["id"])
    try:
        F = ImageFont.load_default(size=18)
    except TypeError:
        F = ImageFont.load_default()
    index, per = [], COLS * ROWS
    for s in range(0, len(judged), per):
        sheet = Image.new("RGB", (COLS * T, ROWS * (T + HDR)), (30, 30, 30))
        d = ImageDraw.Draw(sheet)
        for k, r in enumerate(judged[s:s + per]):
            n = s + k + 1
            im = Image.open(V7_RUN / "context" / Path(r["photo"]).stem / f"{r['id']}.jpg").convert("RGB")
            im.thumbnail((T - 4, T - 4))
            x, y = k % COLS * T, k // COLS * (T + HDR)
            sheet.paste(im, (x + (T - im.width) // 2, y + HDR + (T - im.height) // 2))
            d.text((x + 4, y + 2), f"#{n}  {r['id'][-9:]}", font=F, fill=(255, 255, 0))
            index.append(dict(n=n, id=r["id"], photo=r["photo"], sheet=s // per + 1))
        sheet.save(OUT / "sheets" / f"gt_{s // per + 1:02d}.jpg", quality=92)
    (OUT / "index.json").write_text(json.dumps(index, indent=0))
    print(f"{len(judged)} judged leaves -> {(len(judged) + per - 1) // per} sheets in {OUT / 'sheets'}", flush=True)
