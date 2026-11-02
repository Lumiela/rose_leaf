#!/bin/bash
# wait for r1_fromv4 training to finish, then (re)start r1_coco11s (GPU is shared; two runs at once ran out of memory)
cd /helpai-storage/model_git
while pgrep -f "train_v8.py r1_fromv4" > /dev/null; do sleep 60; done
rm -rf v8_reallabel/runs/r1_coco11s
python v8_reallabel/scripts/train_v8.py r1_coco11s common/weights/yolo11s-seg.pt > v8_reallabel/logs/train_r1_coco11s.log 2>&1
