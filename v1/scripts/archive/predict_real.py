# Predict on real photos with the snapshot model. Writes only inside this directory.
import importlib.abc, sys

# Same cv2.typing workaround as eval_test.py (mixed opencv install in shared env).
class _BlockCv2Typing(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name == "cv2.typing":
            raise ImportError("cv2.typing skipped")
        return None

sys.meta_path.insert(0, _BlockCv2Typing())

from collections import Counter
from pathlib import Path
from ultralytics import YOLO

here = Path(__file__).resolve().parent
model = YOLO(str(here / "best_snapshot.pt"))
for conf in (0.25, 0.10):
    results = model.predict(
        source=str(here / "real_images"),
        imgsz=640,
        conf=conf,
        device=0,
        retina_masks=True,
        save=True,
        project=str(here / "real_images"),
        name=f"pred_conf{int(conf * 100):03d}",
        exist_ok=True,
    )
    for r in results:
        names = [r.names[int(c)] for c in r.boxes.cls]
        print("CONF", conf, Path(r.path).name, "n=", len(names), dict(Counter(names)))
        for c, s, b in zip(names, r.boxes.conf.tolist(), r.boxes.xyxy.tolist()):
            print("  DET", c, round(s, 3), [round(v) for v in b])
