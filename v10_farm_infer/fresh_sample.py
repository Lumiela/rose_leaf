"""흑점병 사진의 FRESH 판정 잎 무작위 100개 시트 (놓친 병든 잎 확인용)"""
import json, glob, random, sys
sys.path.insert(0, 'v9_fastinfer/deploy')
from leafdx import load_rgb
from PIL import Image, ImageDraw
B = 'v10_farm_infer'; d = 'blackspot'
fresh = []
for jf in sorted(glob.glob(f'{B}/runs/{d}/*.json')):
    r = json.load(open(jf)); fresh += [(r['photo'], l) for l in r['leaves'] if l['state'] == 'FRESH']
random.seed(0); s = random.sample(fresh, 100); s.sort(key=lambda t: t[0])
cell, C = 160, 10; sh = Image.new('RGB', (C * cell, 10 * cell), 'white'); dr = ImageDraw.Draw(sh); cache = {}
for i, (ph, l) in enumerate(s):
    if ph not in cache: cache = {ph: Image.fromarray(load_rgb(f'{B}/inputs/{d}/{ph}'))}
    x0, y0, x1, y1 = l['box']; z = max(x1 - x0, y1 - y0) * 1.6; cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    b = (int(cx - z / 2), int(cy - z / 2), int(cx + z / 2), int(cy + z / 2)); cr = cache[ph].crop(b)
    ImageDraw.Draw(cr).line([(p[0] - b[0], p[1] - b[1]) for p in l['polygon']] + [(l['polygon'][0][0] - b[0], l['polygon'][0][1] - b[1])], fill=(60, 230, 60), width=max(2, int(z / 80)))
    sh.paste(cr.resize((cell - 2, cell - 2)), ((i % C) * cell + 1, (i // C) * cell + 1)); dr.text(((i % C) * cell + 3, (i // C) * cell + 3), str(i), fill=(255, 255, 0))
sh.save(f'{B}/review/{d}/fresh_sample.jpg', quality=88)
print('fresh total', len(fresh))
