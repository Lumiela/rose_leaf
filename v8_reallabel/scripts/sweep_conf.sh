#!/bin/bash
# pick the confidence threshold on VAL (not test), per model: writes eval/<model>_val_c<conf>/summary.json
cd /helpai-storage/model_git
for m in r1_fromv4 r2_fromv4; do
  for c in 0.25 0.35 0.45 0.55; do
    python v8_reallabel/scripts/eval_leafseg.py v8_reallabel/runs/$m/weights/best.pt ${m}_val_c$c --split val --conf $c > /dev/null 2>&1
    python -c "import json;t=json.load(open('v8_reallabel/eval/${m}_val_c$c/summary.json'))['total'];print('$m', $c, t['recall'], t['precision'], t['f1'], t['merged'])"
  done
done
echo SWEEP_DONE
