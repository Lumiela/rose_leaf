#!/bin/bash
# v7 pipeline with v7 stage 1 vs v8 stage 1 (+/- background filter): run the unchanged v7 stage-2 classifier, then evaluate on test
cd /helpai-storage/model
V8=v8_reallabel
while pgrep -f leaf_search_v8.py > /dev/null; do sleep 10; done
python v7_leafid/scripts/classify_leaves.py $V8/v7boxes $V8/v7in 2>&1 | grep -v -i warning | tail -2
python v7_leafid/scripts/classify_leaves.py $V8/pipeline/v8_stage1 $V8/inputs 2>&1 | grep -v -i warning | tail -2
python v7_leafid/scripts/classify_leaves.py $V8/pipeline/v8_stage1_nobg $V8/inputs 2>&1 | grep -v -i warning | tail -2
python $V8/scripts/eval_pipeline.py v7=$V8/v7boxes v8=$V8/pipeline/v8_stage1 v8_nobg=$V8/pipeline/v8_stage1_nobg
echo PIPE_DONE
