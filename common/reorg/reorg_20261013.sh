#!/bin/bash
# 2026-10-13: reorganize /helpai-storage/model into common / phase1_cls / v1 / v2.
# Files are only moved (never overwritten). Removed: one identical duplicate of yolo26n.pt and a __pycache__,
# both auto-created by v2 runs. Empty leftover directories are removed with rmdir.
set -euo pipefail
cd /helpai-storage/model

mvv() {  # mvv SRC DST: move without overwriting; abort the whole run on conflict
  if [ -e "$2" ]; then echo "CONFLICT exists: $2"; return 1; fi
  mkdir -p "$(dirname "$2")"; mv "$1" "$2"; echo "MV $1 -> $2"
}

E=runs/rose_seg/eval_test_20261012_0647
R=v2_realbg

# common: pretrained weights, source dataset, real photos, tools
mvv yolo26s-seg.pt common/weights/yolo26s-seg.pt
mvv yolo11s-seg.pt common/weights/yolo11s-seg.pt
mvv weights/yolo26n.pt common/weights/yolo26n.pt
mvv Dataset common/Dataset
mvv Dataset.rar common/Dataset.rar
for f in rose_real.webp rose_real2.webp rose_real3.webp; do mvv $E/real_images/$f common/real_images/$f; done
mvv $R/compare_real_v2.py common/tools/compare_real.py
mvv $R/ckpt_info.py common/tools/ckpt_info.py
mvv $R/viz_labels.py common/tools/viz_labels.py

# phase1_cls
mvv 01_phase1_cls_baseline.ipynb phase1_cls/01_phase1_cls_baseline.ipynb
mvv .ipynb_checkpoints/01_phase1_cls_baseline-checkpoint.ipynb phase1_cls/.ipynb_checkpoints/01_phase1_cls_baseline-checkpoint.ipynb

# v1
mvv 02_phase2_seg_synthetic.ipynb v1/02_phase2_seg_synthetic.ipynb
mvv .ipynb_checkpoints/02_phase2_seg_synthetic-checkpoint.ipynb v1/.ipynb_checkpoints/02_phase2_seg_synthetic-checkpoint.ipynb
mvv data/rose_seg v1/data/rose_seg
mvv runs/rose_seg/yolo26s-seg v1/runs/yolo26s-seg
mvv runs/rose_seg/yolo11s-seg v1/runs/yolo11s-seg
mvv $E/test v1/eval/test
mvv $E/test_conf025 v1/eval/test_conf025
mvv $E/real_images/pred_conf025 v1/eval/real_pred_conf025
mvv $E/real_images/pred_conf010 v1/eval/real_pred_conf010
mvv $E/best_snapshot.pt v1/eval/best_snapshot.pt
for f in eval.log eval_conf025.log predict_real.log predict_real2.log; do mvv $E/$f v1/eval/logs/$f; done
for f in eval_test.py eval_test_conf025.py predict_real.py; do mvv $E/$f v1/scripts/archive/$f; done

# v2
for f in make_backgrounds.py inspect_leaves.py make_synth_v2.py train_eval_v2.py make_review.py; do mvv $R/$f v2/scripts/$f; done
mvv $R/compare_real.py v2/scripts/archive/compare_real_simple.py
mvv $R/predict_real_v2.py v2/scripts/archive/predict_real_v2.py
mvv $R/.ipynb_checkpoints v2/scripts/.ipynb_checkpoints
for f in $R/*.log; do mvv "$f" v2/logs/$(basename "$f"); done
mvv $R/backgrounds v2/backgrounds
mvv $R/data v2/data
mvv $R/debug v2/debug
mvv runs/rose_seg_v2/ft_from_v1 v2/runs/ft_from_v1
mvv runs/rose_seg_v2/coco_init v2/runs/coco_init
for d in $R/eval_v2_early/*; do mvv "$d" v2/eval/$(basename "$d"); done

# auto-created artifacts of v2 runs
if cmp -s $R/weights/yolo26n.pt common/weights/yolo26n.pt; then rm $R/weights/yolo26n.pt; echo "RM identical duplicate $R/weights/yolo26n.pt"; fi
rm -rf $R/__pycache__ && echo "RM $R/__pycache__"

# leftover directories (only if empty)
for d in $R/weights $R/eval_v2_early $E/real_images $E runs/rose_seg runs/rose_seg_v2 runs data weights $R .ipynb_checkpoints; do
  if rmdir "$d" 2>/dev/null; then echo "RMDIR $d"; else echo "KEEP (not empty or missing) $d"; fi
done
echo REORG_DONE
