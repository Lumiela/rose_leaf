#!/bin/bash
# v9 deploy pipeline (unchanged) on the pasted composites
cd /helpai-storage/model_git || exit 1
B=v10_farm_infer
for c in asis transp; do
  python v9_fastinfer/deploy/run.py $B/inputs_paste/$c --out $B/runs_paste/$c > $B/logs/run_paste_$c.log 2>&1
  echo "exit $c $?" >> $B/logs/run_paste_$c.log
done
echo PASTE_DONE >> $B/logs/run_paste_transp.log
