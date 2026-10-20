# v8: train a 1-class ("leaf") YOLO-seg on the real-photo tiles exported by export_yolo.py.
# Usage: python v8_reallabel/scripts/train_v8.py <run name> <init weights> [--epochs 100] [--imgsz 640] [--batch 16] [--data yaml]
#   init: common/weights/yolo11s-seg.pt (COCO) or a previous seg model (head is re-initialised for 1 class)
import argparse
from sam_common import V8, paths  # noqa: F401  (opencv workaround first)
from ultralytics import YOLO

ap = argparse.ArgumentParser(); ap.add_argument("name"); ap.add_argument("init")
ap.add_argument("--epochs", type=int, default=100); ap.add_argument("--imgsz", type=int, default=640)
ap.add_argument("--batch", type=int, default=16); ap.add_argument("--data", default=str(V8 / "data" / "leaf_seg" / "leaf_seg.yaml"))
a = ap.parse_args()
m = YOLO(a.init)
m.train(data=a.data, epochs=a.epochs, imgsz=a.imgsz, batch=a.batch, patience=30, project=str(V8 / "runs"), name=a.name,
        exist_ok=True, degrees=15, flipud=0.5, fliplr=0.5, scale=0.5, mosaic=1.0, close_mosaic=10, overlap_mask=False,
        single_cls=True, workers=4, seed=0, plots=True, verbose=False)
