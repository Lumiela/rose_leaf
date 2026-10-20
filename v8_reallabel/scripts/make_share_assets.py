# collect + downscale result images for the progress report -> share_assets/ and share_assets.zip
import zipfile
from pathlib import Path
from PIL import Image
V8 = Path(__file__).resolve().parents[1]; OUT = V8 / "share_assets"; OUT.mkdir(exist_ok=True)
items = {
    "01_v8_stage2_fix_eafda190.jpg": V8 / "pipeline/v8_stage1_nobg/stage2_v8/fix_eafda190_stage2.jpg",
    "02_v7_stage2_fix_eafda190.jpg": V8 / "v7boxes/stage2/fix_eafda190_stage2.jpg",
    "03_label_review_tile.jpg": V8 / "review/fix_84496bcf/t04.jpg",
    "04_eval_baseline_v4_cube_07.jpg": V8 / "eval/baseline_v4/cube_07.jpg",
    "05_eval_v8r2_cube_07.jpg": V8 / "eval/r2_fromv4_c035/cube_07.jpg",
    "06_v7_false_alarms_test.jpg": V8 / "eval/pipeline/flagged_v7_test.jpg",
    "07_yellow_calls_user_checked_healthy.jpg": V8 / "eval/pipeline/yellow_check.jpg",
    "08_v8_stage1_ids_fix_84496bcf.jpg": V8 / "pipeline/v8_stage1_nobg/fix_84496bcf_ids.jpg",
}
with zipfile.ZipFile(V8 / "share_assets.zip", "w", zipfile.ZIP_DEFLATED) as z:
    for name, src in items.items():
        im = Image.open(src).convert("RGB"); im.thumbnail((1600, 1600))
        im.save(OUT / name, quality=85); z.write(OUT / name, name)
        print(name, im.size, round((OUT / name).stat().st_size / 1e3), "KB")
print("ZIP", round((V8 / "share_assets.zip").stat().st_size / 1e6, 2), "MB")
