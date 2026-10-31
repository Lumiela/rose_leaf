"""Match pasted cutouts to v9 detections (mask IoU >= 0.5 with the pasted leaf extent) and tabulate states.
Run from model/: python v10_farm_infer/paste/paste_eval.py  -> paste/results.json, paste/sheet_<cond>_<cls>_NN.jpg"""
import json, os
from collections import Counter, defaultdict
import numpy as np
from PIL import Image, ImageDraw, ImageFont

B = 'v10_farm_infer'
meta = json.load(open(f'{B}/paste/meta.json'))
CONDS = ('asis', 'transp')
rows = []
for photo in sorted({m['photo'] for m in meta}):
    idmap = np.load(f'{B}/paste/masks/{photo[:-4]}.npz')['idmap']
    for c in CONDS:
        det = json.load(open(f'{B}/runs_paste/{c}/{photo[:-4]}.json'))['leaves']
        for m in [m for m in meta if m['photo'] == photo]:
            x0, y0, x1, y1 = m['box']
            pad = int(m['L'])
            X0, Y0, X1, Y1 = max(0, x0 - pad), max(0, y0 - pad), x1 + pad, y1 + pad
            gt = idmap[Y0:Y1, X0:X1] == m['id']
            best = (0.0, None)
            for d in det:
                b = d['box']
                if b[2] < x0 or b[0] > x1 or b[3] < y0 or b[1] > y1:
                    continue
                pm = Image.new('L', (X1 - X0, Y1 - Y0), 0)
                ImageDraw.Draw(pm).polygon([(x - X0, y - Y0) for x, y in d['polygon']], fill=1)
                pm = np.array(pm, bool)[:gt.shape[0], :gt.shape[1]]
                inter = (pm & gt).sum()
                # detections spilling outside the window still count against IoU via their full area
                iou = inter / (gt.sum() + d['area'] - inter)
                if iou > best[0]:
                    best = (float(iou), d)
            iou, d = best
            found = iou >= 0.5
            rows.append({**m, 'cond': c, 'iou': round(iou, 3), 'found': found,
                         'state': (d['state'] + (':' + d['anomaly_class'] if d.get('anomaly_class') else '')) if found else 'MISSED',
                         'probs': d['probs'] if found else None})
json.dump(rows, open(f'{B}/paste/results.json', 'w'), indent=0)

print('cond   cls    n  found  states (of pasted)')
for c in CONDS:
    for cls in ('Hole', 'Fresh'):
        R = [r for r in rows if r['cond'] == c and r['cls'] == cls]
        st = Counter(r['state'] for r in R)
        n = len(R)
        print(f"{c:6} {cls:5} {n:3} {sum(r['found'] for r in R)/n:5.2f}  " + ', '.join(f'{k} {v} ({v/n:.0%})' for k, v in st.most_common()))
for c in CONDS:
    R = [r for r in rows if r['cond'] == c and r['cls'] == 'Hole' and r['found']]
    ph = np.array([r['probs']['Hole'] for r in R])
    print(c, 'Hole pasted & found: mean p_Hole', ph.mean().round(3), ' p_Hole>=0.5', (ph >= 0.5).mean().round(2))
# hole_frac bins for transp vs asis
for lo, hi in ((0, 1e-9), (1e-9, 0.03), (0.03, 0.08), (0.08, 1)):
    for c in CONDS:
        R = [r for r in rows if r['cond'] == c and r['cls'] == 'Hole' and lo <= r['hole_frac'] < hi + (hi == 1e-9) * 0]
        if lo == 0:
            R = [r for r in rows if r['cond'] == c and r['cls'] == 'Hole' and r['hole_frac'] == 0]
        if R:
            print(f'hole_frac [{lo:.2f},{hi:.2f}) {c:6} n={len(R):2} ANOMALY:Hole {sum(r["state"]=="ANOMALY:Hole" for r in R)/len(R):.0%}')

# sheets: per pasted Hole/Fresh, crop 2.4x box from both conditions side by side + state
font = ImageFont.load_default()
for cls in ('Hole', 'Fresh'):
    ids = [m for m in meta if m['cls'] == cls]
    C = 220; cols = 4; per = 16
    for s in range(0, len(ids), per):
        chunk = ids[s:s + per]
        sheet = Image.new('RGB', (cols * 2 * C, ((len(chunk) + cols - 1) // cols) * (C + 16)), (0, 0, 0))
        for i, m in enumerate(chunk):
            x0, y0, x1, y1 = m['box']; cx, cy = (x0 + x1) / 2, (y0 + y1) / 2; side = 1.6 * max(x1 - x0, y1 - y0)
            for j, c in enumerate(CONDS):
                im = Image.open(f'{B}/inputs_paste/{c}/{m["photo"]}')
                cr = im.crop((int(cx - side / 2), int(cy - side / 2), int(cx + side / 2), int(cy + side / 2))).resize((C, C))
                r = next(r for r in rows if r['id'] == m['id'] and r['cond'] == c)
                X, Y = ((i % cols) * 2 + j) * C, (i // cols) * (C + 16)
                sheet.paste(cr, (X, Y))
                col = (255, 80, 80) if r['state'] == 'ANOMALY:Hole' else (80, 255, 80) if r['state'] == 'FRESH' else (255, 255, 0)
                ImageDraw.Draw(sheet).text((X + 3, Y + C + 2), f"{m['id']} {c} {r['state'].replace('ANOMALY:','A:')} {r['iou']:.2f}", fill=col, font=font)
        sheet.save(f'{B}/paste/sheet_{cls}_{s // per:02d}.jpg', quality=85)
print('sheets done')
