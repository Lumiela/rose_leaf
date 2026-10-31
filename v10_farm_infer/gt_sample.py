"""정답 대조용 블라인드 표본: 판정 상태별 층화 추출 → 섞어서 시트(판정 숨김, 흰 윤곽) + manifest.
python v10_farm_infer/gt_sample.py   (model/에서) → v10_farm_infer/gt/<폴더>/sheet_NN.jpg, manifest.json"""
import json, glob, random, os, sys
sys.path.insert(0, 'v9_fastinfer/deploy')
from leafdx import load_rgb
from PIL import Image, ImageDraw, ImageFont
B = 'v10_farm_infer'; FONT = ImageFont.load_default(size=22)
PLAN = {'blackspot': {'ANOMALY:Black Spot': 50, 'ANOMALY:Hole': 50, 'ANOMALY:Yellow': 50, 'UNKNOWN': 30, 'FRESH': 150},
        'minirose': {'ANOMALY:Black Spot': 15, 'ANOMALY:Hole': 6, 'ANOMALY:Yellow': 50, 'UNKNOWN': 20, 'FRESH': 60}}
for d, plan in PLAN.items():
    pool = {}
    for jf in sorted(glob.glob(f'{B}/runs/{d}/*.json')):
        r = json.load(open(jf))
        for l in r['leaves']:
            k = l['state'] + (':' + l['anomaly_class'] if l['state'] == 'ANOMALY' else '')
            pool.setdefault(k, []).append((r['photo'], l))
    rng = random.Random(42); items = []
    for k, n in plan.items():
        for ph, l in rng.sample(pool.get(k, []), min(n, len(pool.get(k, [])))):
            items.append(dict(photo=ph, id=l['id'], stratum=k, pool=len(pool[k]), box=l['box'], polygon=l['polygon'], probs=l['probs']))
    rng.shuffle(items)
    os.makedirs(f'{B}/gt/{d}', exist_ok=True)
    cell, C, R = 240, 4, 4; cache = {}
    for k in range(0, len(items), C * R):
        ch = items[k:k + C * R]; sh = Image.new('RGB', (C * cell, ((len(ch) + C - 1) // C) * cell), 'white'); dr = ImageDraw.Draw(sh)
        for i, it in enumerate(ch):
            ph = it['photo']
            if ph not in cache: cache[ph] = Image.fromarray(load_rgb(f'{B}/inputs/{d}/{ph}'))
            x0, y0, x1, y1 = it['box']; z = max(x1 - x0, y1 - y0) * 1.8; cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            b = (int(cx - z / 2), int(cy - z / 2), int(cx + z / 2), int(cy + z / 2)); cr = cache[ph].crop(b)
            pts = [(p[0] - b[0], p[1] - b[1]) for p in it['polygon']]
            ImageDraw.Draw(cr).line(pts + [pts[0]], fill=(255, 255, 255), width=max(2, int(z / 120)))
            X, Y = (i % C) * cell, (i // C) * cell; sh.paste(cr.resize((cell - 4, cell - 4), Image.LANCZOS), (X + 2, Y + 2))
            dr.rectangle([X + 2, Y + 2, X + 44, Y + 26], fill=(0, 0, 0)); dr.text((X + 5, Y + 3), str(k + i), fill=(255, 255, 0), font=FONT)
        sh.save(f'{B}/gt/{d}/sheet_{k // (C * R):02d}.jpg', quality=90)
    for i, it in enumerate(items): it['idx'] = i
    json.dump(items, open(f'{B}/gt/{d}/manifest.json', 'w'))
    print(d, len(items), {k: len(v) for k, v in pool.items()}, flush=True)
