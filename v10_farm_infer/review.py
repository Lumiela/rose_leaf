"""v10_farm_infer 결과 검토: 폴더별 상태 집계(summary.json), 이상·의심 잎 모음 시트, 사진별 축소 결과 이미지.
python v10_farm_infer/review.py   (model/에서)"""
import json, glob, os, sys, collections
sys.path.insert(0, 'v9_fastinfer/deploy')
from leafdx import load_rgb
from PIL import Image, ImageDraw, ImageFont
import numpy as np
B = 'v10_farm_infer'
F = ImageFont.load_default()
COL = {'Black Spot': (255, 40, 40), 'Hole': (40, 140, 255), 'Yellow': (255, 220, 0), None: (255, 0, 255)}
summ = {}
for d in ('minirose', 'blackspot'):
    run = f'{B}/runs/{d}'; os.makedirs(f'{B}/review/{d}', exist_ok=True)
    tot = collections.Counter(); flagged = []
    for jf in sorted(glob.glob(f'{run}/*.json')):
        r = json.load(open(jf)); tot.update(r['counts']); tot['leaves'] += len(r['leaves'])
        for l in r['leaves']:
            if l['state'] in ('ANOMALY', 'UNKNOWN'): flagged.append((r['photo'], l))
        vis = Image.open(jf[:-5] + '_vis.jpg'); vis.thumbnail((1600, 1600)); vis.save(f'{B}/review/{d}/{os.path.basename(jf)[:-5]}_vis_small.jpg', quality=85)
    summ[d] = dict(photos=len(glob.glob(f'{run}/*.json')), **tot, flagged=len(flagged))
    cell, C = 160, 10; cache = {}
    flagged.sort(key=lambda t: (t[1]['anomaly_class'] or 'zz', -max(t[1]['probs'].values())))
    for k in range(0, len(flagged), 100):
        ch = flagged[k:k + 100]
        sh = Image.new('RGB', (C * cell, ((len(ch) + C - 1) // C) * (cell + 14)), 'white'); dr = ImageDraw.Draw(sh)
        for i, (ph, l) in enumerate(ch):
            if ph not in cache: cache = {ph: Image.fromarray(load_rgb(f'{B}/inputs/{d}/{ph}'))}
            im = cache[ph]; x0, y0, x1, y1 = l['box']; s = max(x1 - x0, y1 - y0) * 1.6; cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            b = (int(cx - s / 2), int(cy - s / 2), int(cx + s / 2), int(cy + s / 2)); cr = im.crop(b).convert('RGB')
            ImageDraw.Draw(cr).line([(p[0] - b[0], p[1] - b[1]) for p in l['polygon']] + [(l['polygon'][0][0] - b[0], l['polygon'][0][1] - b[1])], fill=COL[l['anomaly_class']], width=max(2, int(s / 80)))
            cr = cr.resize((cell - 2, cell - 2)); X, Y = (i % C) * cell, (i // C) * (cell + 14); sh.paste(cr, (X + 1, Y + 1))
            tag = (l['anomaly_class'] or 'UNK')[:5] + ' %.2f' % max(v for kk, v in l['probs'].items() if kk != 'Fresh')
            dr.text((X + 2, Y + cell), f'{k + i} {tag}', fill=(0, 0, 0), font=F)
        sh.save(f'{B}/review/{d}/flagged_{k // 100:02d}.jpg', quality=88)
    with open(f'{B}/review/{d}/flagged.json', 'w') as f:
        json.dump([dict(idx=i, photo=ph, id=l['id'], state=l['state'], cls=l['anomaly_class'], probs=l['probs'], box=l['box'], L=l['L']) for i, (ph, l) in enumerate(flagged)], f)
json.dump(summ, open(f'{B}/review/summary.json', 'w'), indent=1, ensure_ascii=False)
print(json.dumps(summ, indent=1, ensure_ascii=False))
