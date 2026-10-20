#!/bin/bash
# v8: fill missed leaves with leaf-sized box prompts on a grid (2 passes) for every drafted photo, then re-render review
# Usage (from /helpai-storage/model): bash v8_reallabel/scripts/run_grid.sh [stem ...]   (default: all except camfix_01/02)
V8=v8_reallabel
STEMS="$@"
[ -z "$STEMS" ] && STEMS=$(ls $V8/work | grep -v -E '^camfix_0[12]$')
for s in $STEMS; do
  python $V8/scripts/apply_edits.py $s '[{"op":"grid","spacing":0.5},{"op":"grid","spacing":0.35}]' 2>&1 | grep -v -i warning | tail -3
  python $V8/scripts/review.py $s 2>&1 | grep REVIEW
done
echo ALL_GRID_DONE
