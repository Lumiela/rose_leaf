#!/bin/bash
# round 2: export refined labels -> data/leaf_seg_r2, train from v4 (same recipe as r1_fromv4), evaluate on test, compare
cd /helpai-storage/model
V8=v8_reallabel
python $V8/scripts/export_yolo.py --out $V8/data/leaf_seg_r2 2>&1 | grep -v -i warning | tail -3
python $V8/scripts/train_v8.py r2_fromv4 v4_reallabel/runs/ft_from_v2ft/weights/best.pt --data $V8/data/leaf_seg_r2/leaf_seg.yaml > $V8/logs/train_r2_fromv4.log 2>&1
python $V8/scripts/eval_leafseg.py $V8/runs/r2_fromv4/weights/best.pt r2_fromv4 > $V8/logs/eval_r2_fromv4.log 2>&1
python $V8/scripts/compare_eval.py baseline_v4 r1_fromv4 r1_coco11s r2_fromv4
echo R2_DONE
