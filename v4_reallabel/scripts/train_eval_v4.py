# v4: fine-tune ft_from_v1 (best v2 model) on v4 synthetic data (v2 recipe + 27 real Fresh cutouts) and evaluate.
# Waits for make_synth_v4.py (v4_reallabel/logs/synth_v4.log contains DONE). Skips training if best.pt exists
# unless --retrain. Writes only v4_reallabel/runs and v4_reallabel/eval.
# Evaluation (all on data never used for training):
#   - v2 synthetic test (800 scenes, Dataset leaves only) and v1 synthetic test, conf 0.001, for v1 / ft_from_v1 / v4
#   - real photos: common/tools/compare_real.py sheets (rose_real3 is the held-out photo)
# Usage: python v4_reallabel/scripts/train_eval_v4.py [--retrain] > v4_reallabel/logs/train_eval_v4.log 2>&1
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
import paths  # noqa: E402  shared paths + opencv workaround

import json  # noqa: E402
import os  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402
from ultralytics import YOLO  # noqa: E402

DATA = paths.V4 / "data/rose_seg_v4/rose_seg_v4.yaml"
PROJECT, NAME = paths.V4 / "runs", "ft_from_v2ft"
EVAL = paths.V4 / "eval"
EPOCHS, PATIENCE = 25, 8          # v2 fine-tune peaked at epoch 30 of 45 (~14.5 min/epoch); shorter run from a better start
COMMON = dict(
    imgsz=640, batch=32, device=0, workers=4, seed=42, exist_ok=True, cos_lr=True,
    hsv_h=0.005, hsv_s=0.3, hsv_v=0.35, degrees=15, scale=0.4, translate=0.1,
    fliplr=0.5, flipud=0.5, mosaic=1.0, close_mosaic=5, mixup=0.0, copy_paste=0.0, plots=True,
)


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


synth_log = paths.V4 / "logs" / "synth_v4.log"
while not (synth_log.exists() and "DONE" in synth_log.read_text() and DATA.exists()):
    time.sleep(30)
log("data ready", DATA)

best = PROJECT / NAME / "weights" / "best.pt"
if best.exists() and "--retrain" not in sys.argv:
    log("SKIP train (best.pt exists)")
else:
    os.chdir(paths.COMMON)            # Ultralytics AMP check resolves weights/yolo26n.pt relative to cwd
    log("TRAIN start", NAME, "init", paths.V2_FT_BEST)
    YOLO(str(paths.V2_FT_BEST)).train(data=str(DATA), project=str(PROJECT), name=NAME,
                                      epochs=EPOCHS, patience=PATIENCE, **COMMON)
    log("TRAIN done", best)

models = {"v1": paths.V1_BEST, "ft_from_v1": paths.V2_FT_BEST, "v4": best}
EVAL.mkdir(parents=True, exist_ok=True)
summary = {}
for name, wp in models.items():
    summary[name] = {}
    for tag, yaml in (("v2_test", paths.V2_DATA_YAML), ("v1_test", paths.V1_DATA_YAML)):
        m = YOLO(str(wp)).val(data=str(yaml), split="test", imgsz=640, batch=16, device=0, conf=0.001,
                              project=str(EVAL), name=f"{name}_{tag}", exist_ok=True, plots=True, verbose=False)
        summary[name][tag] = dict(mask_map50=round(m.seg.map50, 4), mask_map=round(m.seg.map, 4),
                                  box_map50=round(m.box.map50, 4), box_map=round(m.box.map, 4),
                                  per_class_mask_map50={m.names[int(c)]: round(float(a), 4) for c, a in zip(m.ap_class_index, m.seg.ap50)})
        log("EVAL", name, tag, summary[name][tag]["mask_map50"], summary[name][tag]["mask_map"])
(EVAL / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
subprocess.run([sys.executable, str(paths.TOOLS / "compare_real.py"), str(EVAL / "compare_real")] +
               [f"{n}={w}" for n, w in models.items()], check=False)
log("ALL DONE", EVAL)
