"""v6 inference pipeline = v4 model + gray-world white balance + Hole post-filter (no retraining).

Hole post-filter: a predicted Hole instance is kept only if the share of leaf-colored (green / yellow-green) pixels
inside its mask is >= HOLE_MIN_LEAF. Empty pot cells, tray holes and dark gaps between leaves contain almost no
leaf-colored pixels and are dropped; a real holed leaf is still mostly leaf pixels.

Usage (library):
    from pipeline import Pipeline
    pl = Pipeline()                       # defaults: v4 weights, wb=True, hole_min_leaf=0.5
    res = pl.predict(rgb_uint8)           # list of dict(cls, conf, mask, box, leaf_frac, dropped)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
import paths  # noqa: E402  shared paths + opencv workaround

import numpy as np  # noqa: E402
import cv2  # noqa: E402
from PIL import Image  # noqa: E402
from ultralytics import YOLO  # noqa: E402

HOLE = paths.CLASS_NAMES.index("Hole Leaves")


def gray_world(img):
    """White balance: scale channels so near-neutral pixels (tray, walls) become gray."""
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    ref = (hsv[..., 1] < 60) & (hsv[..., 2] > 60) & (hsv[..., 2] < 240)
    if ref.sum() < 1000:
        ref = np.ones(ref.shape, bool)
    mean = img[ref].reshape(-1, 3).mean(0)
    return np.clip(img.astype(np.float32) * (mean.mean() / mean), 0, 255).astype(np.uint8)


def leaf_color_mask(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    return (h >= 20) & (h <= 95) & (s >= 40) & (v >= 40)


class Pipeline:
    def __init__(self, weights=paths.V4_BEST, wb=True, hole_min_leaf=0.5, conf=0.25, imgsz=640, agnostic=False):
        self.model = YOLO(str(weights))
        self.wb, self.hole_min_leaf, self.conf, self.imgsz = wb, hole_min_leaf, conf, imgsz
        self.agnostic = agnostic      # True: NMS across classes (one instance per leaf regardless of class)

    def predict(self, img):
        im = gray_world(img) if self.wb else img
        r = self.model.predict(Image.fromarray(im), imgsz=self.imgsz, conf=self.conf, device=0,
                               retina_masks=True, agnostic_nms=self.agnostic, verbose=False)[0]
        if r.masks is None:
            return [], im
        leafc = leaf_color_mask(im)
        out = []
        for m, c, s, b in zip(r.masks.data.cpu().numpy() > 0.5, r.boxes.cls.cpu().numpy().astype(int),
                              r.boxes.conf.cpu().numpy(), r.boxes.xyxy.cpu().numpy().astype(int)):
            frac = float((leafc & m).sum() / max(1, m.sum()))
            dropped = bool(c == HOLE and self.hole_min_leaf is not None and frac < self.hole_min_leaf)
            out.append(dict(cls=int(c), conf=float(s), mask=m, box=[int(v) for v in b], leaf_frac=frac, dropped=dropped))
        return out, im
