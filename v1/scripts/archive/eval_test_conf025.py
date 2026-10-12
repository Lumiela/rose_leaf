# Standalone test-split evaluation. Writes only inside this directory.
import importlib.abc, sys

# Shared env has mixed opencv 4.7/4.11 files; cv2.typing fails to import (AttributeError).
# Turn it into ImportError so cv2's bootstrap skips it. In-process only, no package changes.
class _BlockCv2Typing(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name == "cv2.typing":
            raise ImportError("cv2.typing skipped")
        return None

sys.meta_path.insert(0, _BlockCv2Typing())

from pathlib import Path
from ultralytics import YOLO

here = Path(__file__).resolve().parent
model = YOLO(str(here / "best_snapshot.pt"))
m = model.val(
    data="/helpai-storage/model/data/rose_seg/rose_seg.yaml",
    split="test",
    batch=8,
    workers=0,
    imgsz=640,
    device=0,
    project=str(here),
    name="test_conf025", conf=0.25,
    exist_ok=True,
    plots=True,
)
print("BOX  mp/mr/map50/map:", m.box.mp, m.box.mr, m.box.map50, m.box.map)
print("MASK mp/mr/map50/map:", m.seg.mp, m.seg.mr, m.seg.map50, m.seg.map)
for i, c in enumerate(m.ap_class_index):
    print("CLASS", m.names[c], "box_map50", m.box.ap50[i], "box_map", m.box.ap[i], "mask_map50", m.seg.ap50[i], "mask_map", m.seg.ap[i])
