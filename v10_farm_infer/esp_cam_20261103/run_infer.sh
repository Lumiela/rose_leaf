#!/bin/bash
# esp_cam_20261103: 교체한 ESP 카메라 사진에 v9 배포 패키지 추론만 실행. 모델·설정 변경 없음.
# 비교용으로 이전 고정 카메라 사진(common)도 같은 패키지로 실행.
cd /helpai-storage/model_git || exit 1
B=v10_farm_infer/esp_cam_20261103
mkdir -p $B/logs
{ date; which python; nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv; } > $B/logs/env.log 2>&1
python v9_fastinfer/deploy/run.py $B/inputs --out $B/runs/new_cam > $B/logs/run_new_cam.log 2>&1
echo "exit new_cam $?" >> $B/logs/run_new_cam.log
for d in real_images_camfix real_images_local; do
  python v9_fastinfer/deploy/run.py common/$d --out $B/runs/$d > $B/logs/run_$d.log 2>&1
  echo "exit $d $?" >> $B/logs/run_$d.log
done
echo ALL_DONE >> $B/logs/env.log
