#!/bin/bash
# v10_farm_infer: 농장 사진에 v9 배포 패키지(v8과 같은 출력) 추론만 실행. 모델·설정 변경 없음.
cd /helpai-storage/model_git || exit 1
B=v10_farm_infer
mkdir -p $B/logs
{ date; which python; nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv; } > $B/logs/env.log 2>&1
for d in minirose blackspot; do
  python v9_fastinfer/deploy/run.py $B/inputs/$d --out $B/runs/$d > $B/logs/run_$d.log 2>&1
  echo "exit $d $?" >> $B/logs/run_$d.log
done
echo ALL_DONE >> $B/logs/env.log
