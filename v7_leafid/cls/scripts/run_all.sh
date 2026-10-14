#!/bin/bash
# Full comparison: download -> render -> 3 models -> compare. Each step continues even if a previous model fails.
# Progress: v7_leafid/cls/progress.md (Markdown preview) and this terminal.  Log: v7_leafid/cls/logs/run_all.log
cd /helpai-storage/model
S=v7_leafid/cls/scripts
rm -f v7_leafid/cls/progress.json v7_leafid/cls/compare.md
bash $S/download_weights.sh
python -u $S/render_data.py
python -u $S/train_yolo.py
python -u $S/train_effnet.py
python -u $S/train_dino.py
python -u $S/compare.py
echo "=== ALL DONE $(date -u +%H:%M:%S) UTC ==="
