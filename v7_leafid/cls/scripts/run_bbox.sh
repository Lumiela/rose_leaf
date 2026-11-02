#!/bin/bash
# bbox variant: export stage-1 box crops -> background leaf size -> render bbox dataset (scale-matched) -> 3 models
# -> compare (+ refresh cutout compare).
# Progress: v7_leafid/cls/progress_bbox.md and this terminal.  Log: v7_leafid/cls/logs/run_bbox.log
set -e
cd /helpai-storage/model_git
S=v7_leafid/cls/scripts
export CLS_VARIANT=bbox
rm -f v7_leafid/cls/progress_bbox.json v7_leafid/cls/compare_bbox.md
[ -d v7_leafid/cls/runs_bbox ] && mv v7_leafid/cls/runs_bbox v7_leafid/cls/runs_bbox_aborted_$(date -u +%H%M%S)   # keep the stopped run
python -u $S/export_bbox.py
python -u $S/bg_leaf_size.py
python -u $S/render_data_bbox.py
python -u $S/train_yolo.py
python -u $S/train_effnet.py
python -u $S/train_dino.py
python -u $S/compare.py
CLS_VARIANT= python -u $S/compare.py      # recompute cutout comparison with the judged-only columns
echo "=== BBOX RUN DONE $(date -u +%H:%M:%S) UTC ==="
