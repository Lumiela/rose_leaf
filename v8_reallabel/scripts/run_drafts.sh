#!/bin/bash
# v8: build inputs/ (symlinks, unique scenes only), run v7 leaf search for boxes, then SAM drafts + review sheets.
# Usage (from /helpai-storage/model): bash v8_reallabel/scripts/run_drafts.sh
V8=v8_reallabel; IN=$V8/inputs; mkdir -p $IN $V8/v7boxes
ln -sf ../../common/real_images_camfix/camfix_01.jpg $IN/camfix_01.jpg
ln -sf ../../common/real_images_camfix/camfix_02.jpg $IN/camfix_02.jpg
for f in 44be89e5-ba26-4fd0-837d-a5a36a1fca8f 84496bcf-808a-4b82-a597-1453e2815c54 92ccc604-a9d7-4b68-84eb-d5d21d3c2545 ce7a654f-dc9c-4187-aed9-231aa7e998d1 eafda190-5f86-4d2e-b98f-f41ed7cdcb46; do
  ln -sf ../../common/real_images_local/$f.jfif $IN/fix_${f%%-*}.jpg; done
ln -sf ../../v6_holefilter/inputs/c6485924-1968-4a9e-8136-7aad5121e655.jpg $IN/fix_c6485924.jpg
ln -sf ../../v6_holefilter/inputs/8b90aa7e-5d4a-42d8-a09a-8d0c302d526b.jpg $IN/fix_8b90aa7e.jpg
for i in 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20 21 22; do ln -sf ../../common/real_images_cube/cube_$i.jpg $IN/cube_$i.jpg; done
ls $IN | wc -l
# v7 boxes for everything except camfix_01 (already drafted + reviewed)
mkdir -p $V8/v7in; for f in $IN/*.jpg; do b=$(basename $f); [ "$b" = camfix_01.jpg ] || ln -sf ../inputs/$b $V8/v7in/$b; done
[ -f $V8/v7boxes/leaves.json ] || python v7_leafid/scripts/leaf_search.py $V8/v7in --out $V8/v7boxes 2>&1 | grep -v -i warning | tail -40
for f in $V8/v7in/*.jpg; do
  s=$(basename $f .jpg)
  python $V8/scripts/sam_draft.py $IN/$s.jpg --v7 $V8/v7boxes/leaves.json 2>&1 | grep -v -i warning | tail -8
  python $V8/scripts/review.py $s 2>&1 | grep -v -i warning | grep -v FLAGGED | tail -3
done
echo ALL_DRAFTS_DONE
