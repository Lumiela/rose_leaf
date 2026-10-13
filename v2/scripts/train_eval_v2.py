# Step 3+4: train on v2 synthetic data (two inits) and evaluate.
#   A) ft_from_v1 : start from v1 best.pt (yolo26s-seg trained on v1 synthetic data)
#   B) coco_init  : start from COCO-pretrained yolo26s-seg.pt
# Waits for make_synth_v2.py to finish (v2/logs/synth_v2.log). Writes only v2/runs and v2/eval/final.
# Runs that already have weights/best.pt are skipped (no overwrite) unless --retrain is given.
# Usage: python v2/scripts/train_eval_v2.py [--retrain] > v2/logs/train_eval_v2.log 2>&1
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
import paths  # noqa: E402  shared paths + opencv workaround (must come before cv2/ultralytics)

import json, os, time
from collections import Counter
from pathlib import Path
import torch
from ultralytics import YOLO

V1_BEST = paths.V1_BEST
COCO = paths.WEIGHTS / "yolo26s-seg.pt"
V1_YAML = paths.V1_DATA_YAML
V2_YAML = paths.V2_DATA_YAML
REAL_DIR = paths.REAL_IMAGES
PROJECT = paths.V2 / "runs"
EVAL = paths.V2 / "eval" / "final"
RETRAIN = "--retrain" in sys.argv

RUNS = [
    ("ft_from_v1", V1_BEST, 50, 15),
    ("coco_init",  COCO,   100, 20),
]
# same training hyper-parameters as the v1 notebook (cell 14)
COMMON = dict(
    imgsz=640, batch=32, device=0, workers=4, seed=42, exist_ok=True, cos_lr=True,
    hsv_h=0.005, hsv_s=0.3, hsv_v=0.35, degrees=15, scale=0.4, translate=0.1,
    fliplr=0.5, flipud=0.5, mosaic=1.0, close_mosaic=10, mixup=0.0, copy_paste=0.0, plots=True,
)

def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)

# wait for synthetic data
synth_log = paths.V2 / "logs" / "synth_v2.log"
while not (synth_log.exists() and "DONE" in synth_log.read_text() and V2_YAML.exists()):
    time.sleep(30)
log("synthetic data ready:", V2_YAML)

# Ultralytics resolves its AMP-check model (yolo26n.pt) from the relative 'weights' dir -> common/weights
os.chdir(paths.COMMON)

weights = {}
for name, init, epochs, patience in RUNS:
    if (PROJECT / name / "weights" / "best.pt").exists() and not RETRAIN:
        weights[name] = PROJECT / name / "weights" / "best.pt"
        log("SKIP train (already has best.pt)", name)
        continue
    log("TRAIN start", name, "init=", init.name, "epochs=", epochs)
    for batch in (COMMON["batch"], 16):
        try:
            YOLO(str(init)).train(data=str(V2_YAML), project=str(PROJECT), name=name,
                                  epochs=epochs, patience=patience, **{**COMMON, "batch": batch})
            break
        except torch.cuda.OutOfMemoryError:
            log("OOM at batch", batch, "-> retry with 16")
            torch.cuda.empty_cache()
    weights[name] = PROJECT / name / "weights" / "best.pt"
    log("TRAIN done", name, weights[name])

# evaluation: v1 baseline vs the two v2 models
weights = {"v1_baseline": V1_BEST, **weights}
EVAL.mkdir(parents=True, exist_ok=True)
summary = {}
for name, wp in weights.items():
    model = YOLO(str(wp))
    summary[name] = {}
    for tag, yaml in (("v1_test", V1_YAML), ("v2_test", V2_YAML)):
        for conf in (0.001, 0.25):
            m = model.val(data=str(yaml), split="test", imgsz=640, batch=16, device=0, conf=conf,
                          project=str(EVAL), name=f"{name}_{tag}_conf{conf}", exist_ok=True, plots=True, verbose=False)
            summary[name][f"{tag}_conf{conf}"] = dict(
                box_map50=m.box.map50, box_map=m.box.map, mask_map50=m.seg.map50, mask_map=m.seg.map,
                mask_p=m.seg.mp, mask_r=m.seg.mr,
                per_class_mask_map50={m.names[int(c)]: float(a) for c, a in zip(m.ap_class_index, m.seg.ap50)})
            log("EVAL", name, tag, "conf", conf, "mask_map50=%.3f mask_map=%.3f" % (m.seg.map50, m.seg.map))
    res = model.predict(source=str(REAL_DIR), imgsz=640, conf=0.25, device=0, retina_masks=True, save=True,
                        project=str(EVAL), name=f"{name}_real_conf025", exist_ok=True, verbose=False)
    summary[name]["real_conf025"] = {Path(r.path).name: dict(Counter(r.names[int(c)] for c in r.boxes.cls))
                                     for r in res}
    log("REAL", name, summary[name]["real_conf025"])

(EVAL / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
log("ALL DONE", EVAL / "summary.json")
