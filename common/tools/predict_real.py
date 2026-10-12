# Predict the real photos (common/real_images) with any weights and save Ultralytics-style images
# (mask + box + class/confidence label per instance), one folder per confidence threshold.
# Usage: python common/tools/predict_real.py <weights> <out_dir> [conf ...]      (default conf: 0.25 0.10)
#   e.g. python common/tools/predict_real.py v2/runs/ft_from_v1/weights/best.pt v2/eval/predict_ft_from_v1
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import paths  # noqa: E402  shared paths + opencv workaround (must come before cv2/ultralytics)

from collections import Counter  # noqa: E402
from ultralytics import YOLO  # noqa: E402

weights, out_dir = sys.argv[1], Path(sys.argv[2]).resolve()
confs = [float(c) for c in sys.argv[3:]] or [0.25, 0.10]
model = YOLO(weights)
for conf in confs:
    res = model.predict(source=str(paths.REAL_IMAGES), imgsz=640, conf=conf, device=0, retina_masks=True, save=True,
                        project=str(out_dir), name=f"conf{int(conf * 100):03d}", exist_ok=True, verbose=False)
    for r in res:
        print("PRED", conf, Path(r.path).name, "n=", len(r.boxes),
              dict(Counter(r.names[int(c)] for c in r.boxes.cls)), flush=True)
