#!/bin/bash
# after train_cls_v8.py: calibrate states on val -> test numbers; run v8 stage 2 on the v8 stage-1 run (all photos)
cd /helpai-storage/model_git
while pgrep -f train_cls_v8.py > /dev/null; do sleep 20; done
python v8_reallabel/cls/scripts/calib_states.py > v8_reallabel/cls/logs/calib.log 2>&1
python v8_reallabel/cls/scripts/classify_v8.py v8_reallabel/pipeline/v8_stage1_nobg v8_reallabel/inputs 2>&1 | grep -v -i warning > v8_reallabel/cls/logs/classify.log
echo AFTER_DONE >> v8_reallabel/cls/logs/classify.log
