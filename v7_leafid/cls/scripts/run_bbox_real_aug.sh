#!/bin/bash
# bbox_real_aug: bbox_real + synthetic anomalies painted on real healthy leaves (shortcut countermeasure) -> EfficientNetV2-S
# then all metrics (A1-A3 eval_gt, A4 + B4 eval_real_test, B synthetic test/test_deg via compare).
# Progress: v7_leafid/cls/progress_bbox_real_aug.md and this terminal.  Log: v7_leafid/cls/logs/run_bbox_real_aug.log
set -e
cd /helpai-storage/model
S=v7_leafid/cls/scripts
export CLS_VARIANT=bbox_real_aug
rm -f v7_leafid/cls/progress_bbox_real_aug.json v7_leafid/cls/compare_bbox_real_aug.md
python -u $S/synth_real_anom.py
python -u $S/eval_real_test.py          # B4 of the existing bbox / bbox_real models (before training)
python -u $S/train_effnet.py
python -u $S/compare.py
python -u $S/eval_gt.py
python -u $S/eval_real_test.py
echo "=== BBOX_REAL_AUG RUN DONE $(date -u +%H:%M:%S) UTC ==="
