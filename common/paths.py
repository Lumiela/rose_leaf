"""Shared paths for /helpai-storage/model + opencv workaround.

Import this before cv2 / ultralytics in every script and notebook:
    sys.path.insert(0, "<model>/common"); import paths
The server has a mixed opencv 4.7/4.11 install: importing cv2.typing raises AttributeError, so
`import ultralytics` fails in new processes. Importing this module makes cv2 skip that submodule.
"""
import importlib.abc
import sys
from pathlib import Path


class _BlockCv2Typing(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name == "cv2.typing":
            raise ImportError("cv2.typing skipped")
        return None


if not any(type(f).__name__ == "_BlockCv2Typing" for f in sys.meta_path):
    sys.meta_path.insert(0, _BlockCv2Typing())

MODEL = Path(__file__).resolve().parents[1]           # /helpai-storage/model
COMMON = MODEL / "common"
DATASET = COMMON / "Dataset"                          # class folders of RGBA leaf cutouts
WEIGHTS = COMMON / "weights"                          # pretrained weights (yolo26s-seg, yolo11s-seg, yolo26n)
REAL_IMAGES = COMMON / "real_images"
REAL_IMAGES_CUBE = COMMON / "real_images_cube"          # 18 smartphone photos of the cube (2026-10-13), not used for training                  # real greenhouse photos (rose_real3 never used for training)
TOOLS = COMMON / "tools"

V1 = MODEL / "v1"
V1_DATA_YAML = V1 / "data/rose_seg/rose_seg.yaml"
V1_BEST = V1 / "runs/yolo26s-seg/weights/best.pt"

V2 = MODEL / "v2"
V2_DATA_YAML = V2 / "data/rose_seg_v2/rose_seg_v2.yaml"
V2_FT_BEST = V2 / "runs/ft_from_v1/weights/best.pt"
V2_COCO_BEST = V2 / "runs/coco_init/weights/best.pt"

V3 = MODEL / "v3_twostage"                          # two-stage architecture experiments
V4 = MODEL / "v4_reallabel"                          # real-photo labeling (model proposals + review)
V4_BEST = V4 / "runs/ft_from_v2ft/weights/best.pt"
V5 = MODEL / "v5_cube"                                # cube-specific work
V6 = MODEL / "v6_holefilter"                         # v4 model + white balance + Hole post-filter
V7 = MODEL / "v7_leafid"                             # two-stage: leaf search + per-leaf ID -> classifier

CLASS_NAMES = ["Black Spot Leaves", "Fresh Leaves", "Hole Leaves", "Yellow Leaves"]
