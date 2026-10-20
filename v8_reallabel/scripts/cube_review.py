"""Review sheets for a deploy run.py output folder (no ground truth).
python cube_review.py <run_out_dir> <img_dir>
Writes <run_out_dir>/review/{flagged.jpg, yellowest.jpg, stats.json}."""
import sys, json, glob, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'common'))
import paths  # noqa: F401  (must precede cv2)
import cv2, numpy as np
from PIL import Image, ImageDraw

out, imgd = sys.argv[1], sys.argv[2]
rd = os.path.join(out, 'review'); os.makedirs(rd, exist_ok=True)
C = 170

def crop(img, l):
    x1, y1, x2, y2 = l['box']; cx, cy = (x1 + x2) / 2, (y1 + y2) / 2; s = max(x2 - x1, y2 - y1) * 1.25 / 2
    a, b, c, d = int(max(cx - s, 0)), int(max(cy - s, 0)), int(min(cx + s, img.shape[1])), int(min(cy + s, img.shape[0]))
    im = Image.fromarray(img[b:d, a:c]).resize((C, C))
    poly = np.array(l['polygon'], float); poly[:, 0] = (poly[:, 0] - a) * C / max(c - a, 1); poly[:, 1] = (poly[:, 1] - b) * C / max(d - b, 1)
    ImageDraw.Draw(im).polygon([tuple(p) for p in poly], outline=(0, 255, 255))
    return im

def sheet(items, path, cols=8):
    if not items: return
    rows = (len(items) + cols - 1) // cols; S = Image.new('RGB', (cols * C, rows * (C + 30)), 'white'); dr = ImageDraw.Draw(S)
    for i, (im, t1, t2) in enumerate(items):
        x, y = (i % cols) * C, (i // cols) * (C + 30); S.paste(im, (x, y)); dr.text((x + 2, y + C + 1), t1, fill='black'); dr.text((x + 2, y + C + 15), t2, fill=(150, 0, 0))
    S.save(path, quality=88)

flag, allleaves, stats = [], [], {}
for jp in sorted(glob.glob(os.path.join(out, '*.json'))):
    d = json.load(open(jp)); stem = d['photo'].rsplit('.', 1)[0]
    img = np.asarray(Image.open(os.path.join(imgd, d['photo'])).convert('RGB'))
    # gray-world WB on the whole photo, then per-leaf yellow fraction (hue 15-32 in OpenCV units, sat>70)
    g = img.reshape(-1, 3).mean(0); wb = np.clip(img * (g.mean() / g), 0, 255).astype(np.uint8)
    hsv = cv2.cvtColor(wb, cv2.COLOR_RGB2HSV)
    for l in d['leaves']:
        m = np.zeros(img.shape[:2], np.uint8); cv2.fillPoly(m, [np.array(l['polygon'], np.int32)], 1)
        h, s = hsv[..., 0][m > 0], hsv[..., 1][m > 0]
        l['yfrac'] = float(((h >= 15) & (h < 32) & (s > 70)).mean()) if len(h) else 0.0
        l['photo'] = stem; allleaves.append((l, img))
        if l['state'] not in ('FRESH', 'HOLD_SHAPE'):
            p = l['probs']; flag.append((crop(img, l), f"{stem[-6:]} {l['id']} {l['state'][:14]}", f"F{p['Fresh']:.2f} Y{p['Yellow']:.2f} B{p['Black Spot']:.2f} H{p['Hole']:.2f}"))
    ys = [l['yfrac'] for l in d['leaves']]
    stats[stem] = dict(n=len(d['leaves']), L=d['L'], counts=d['counts'], timing=d['timing'],
                       yfrac_gt30=int(sum(y > 0.3 for y in ys)), pY_max=max(l['probs']['Yellow'] for l in d['leaves']),
                       pFresh_lt09=int(sum(l['probs']['Fresh'] < 0.9 for l in d['leaves'])),
                       hold_border=int(sum(l['touches_border'] for l in d['leaves'] if l['state'] == 'HOLD_SHAPE')),
                       hold_solid=int(sum(l['solidity'] < 0.8 for l in d['leaves'] if l['state'] == 'HOLD_SHAPE')))
sheet(flag, os.path.join(rd, 'flagged.jpg'))
yl = sorted(allleaves, key=lambda t: -t[0]['yfrac'])[:48]
sheet([(crop(im, l), f"{l['photo'][-6:]} {l['id']} {l['state'][:10]}", f"yf{l['yfrac']:.2f} pY{l['probs']['Yellow']:.2f} pF{l['probs']['Fresh']:.2f}") for l, im in yl], os.path.join(rd, 'yellowest.jpg'))
fr = [t for t in allleaves if t[0]['state'] == 'FRESH']
stats['_all'] = dict(n=len(allleaves), fresh_yfrac_gt30=sum(t[0]['yfrac'] > 0.3 for t in fr), fresh_yfrac_gt50=sum(t[0]['yfrac'] > 0.5 for t in fr),
                     yellowest48_states={s: sum(t[0]['state'] == s for t in yl) for s in set(t[0]['state'] for t in yl)})
json.dump(stats, open(os.path.join(rd, 'stats.json'), 'w'), indent=1)
print(json.dumps(stats['_all'])); print('flagged', len(flag))
