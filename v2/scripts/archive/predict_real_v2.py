# Quick real-photo prediction with a given weights file. Usage: python predict_real_v2.py <weights> <run_name>
import importlib.abc, sys

# Shared env has mixed opencv 4.7/4.11 files; skip the broken cv2.typing submodule.
class _BlockCv2Typing(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name == "cv2.typing":
            raise ImportError("cv2.typing skipped")
        return None

sys.meta_path.insert(0, _BlockCv2Typing())

from collections import Counter
from pathlib import Path
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
REAL_DIR = ROOT.parent / "runs/rose_seg/eval_test_20261012_0647/real_images"
weights, name = sys.argv[1], sys.argv[2]
model = YOLO(weights)
for conf in (0.25, 0.10):
    res = model.predict(source=str(REAL_DIR), imgsz=640, conf=conf, device=0, retina_masks=True, save=True,
                        project=str(ROOT / "eval_v2_early"), name=f"{name}_conf{int(conf * 100):03d}", exist_ok=True, verbose=False)
    for r in res:
        print("PRED", name, conf, Path(r.path).name, "n=", len(r.boxes), dict(Counter(r.names[int(c)] for c in r.boxes.cls)),
              "max_conf=%.2f" % (float(r.boxes.conf.max()) if len(r.boxes) else 0))
