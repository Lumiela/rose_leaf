#!/bin/bash
# bbox_real variant: bbox data + reviewed real cube leaves as Fresh -> EfficientNetV2-S -> metrics
#   A1-A3 camfix ground truth (eval_gt.py), A4 held-out phone photos (eval_real_test.py), B synthetic test/test_deg.
# Progress: v7_leafid/cls/progress_bbox_real.md and this terminal.  Log: v7_leafid/cls/logs/run_bbox_real.log
set -e
cd /helpai-storage/model
S=v7_leafid/cls/scripts
export CLS_VARIANT=bbox_real
rm -f v7_leafid/cls/progress_bbox_real.json v7_leafid/cls/compare_bbox_real.md
python -u $S/build_real_data.py
python -u $S/train_effnet.py
python -u $S/compare.py
python -u $S/eval_gt.py
python -u $S/eval_real_test.py
echo "=== BBOX_REAL RUN DONE $(date -u +%H:%M:%S) UTC ==="
