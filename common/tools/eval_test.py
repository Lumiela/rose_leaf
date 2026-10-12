# Evaluate weights on a dataset split: box/mask mAP, per-class AP, confusion matrix, PR curves.
# Usage: python common/tools/eval_test.py <weights> <data.yaml> <out_dir> [conf] [split]   (default conf 0.001, split test)
#   e.g. python common/tools/eval_test.py v1/runs/yolo26s-seg/weights/best.pt v1/data/rose_seg/rose_seg.yaml v1/eval/test
#   conf 0.25 gives a confusion matrix that ignores very low-confidence predictions.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import paths  # noqa: E402  shared paths + opencv workaround (must come before cv2/ultralytics)

from ultralytics import YOLO  # noqa: E402

weights, data, out_dir = sys.argv[1], sys.argv[2], Path(sys.argv[3]).resolve()
conf = float(sys.argv[4]) if len(sys.argv) > 4 else 0.001
split = sys.argv[5] if len(sys.argv) > 5 else "test"
m = YOLO(weights).val(data=data, split=split, batch=8, workers=0, imgsz=640, device=0, conf=conf,
                      project=str(out_dir.parent), name=out_dir.name, exist_ok=True, plots=True)
print("BOX  mp/mr/map50/map:", m.box.mp, m.box.mr, m.box.map50, m.box.map)
print("MASK mp/mr/map50/map:", m.seg.mp, m.seg.mr, m.seg.map50, m.seg.map)
for i, c in enumerate(m.ap_class_index):
    print("CLASS", m.names[c], "box_map50", m.box.ap50[i], "box_map", m.box.ap[i],
          "mask_map50", m.seg.ap50[i], "mask_map", m.seg.ap[i])
