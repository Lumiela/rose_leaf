#!/bin/bash
# after each round-1 training finishes, evaluate its best.pt on the reviewed test set (same protocol as baseline_v4)
cd /helpai-storage/model_git
for r in r1_fromv4 r1_coco11s; do
  # wait until the run exists, then until no training process for it remains
  while [ ! -f v8_reallabel/runs/$r/results.csv ] || pgrep -f "train_v8.py $r " > /dev/null; do sleep 60; done
  if [ -f v8_reallabel/runs/$r/weights/best.pt ]; then
    python v8_reallabel/scripts/eval_leafseg.py v8_reallabel/runs/$r/weights/best.pt $r > v8_reallabel/logs/eval_$r.log 2>&1
    echo "$(date) evaluated $r" 
  else
    echo "$(date) no best.pt for $r"
  fi
done
echo AUTO_EVAL_DONE
