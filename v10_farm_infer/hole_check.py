"""Hole 기하 검증 시험 (재학습 없음).
잎 박스로 SAM2.1 → 잎 조직 마스크 → (1) 윤곽 안에 완전히 둘러싸인 비조직 영역(안쪽 구멍) (2) 가장자리 볼록 결함 깊이(홈)
두 조건 중 하나가 맞으면 '구멍 확인'. 평가 대상:
  real  : gt/<폴더>/gt_labels.json 표본 전체 (정답 있음, 실제 구멍 0개)
  synth : 정답 '건강' 잎에 구멍/홈을 디지털로 뚫음 (같은 사진의 다른 위치 픽셀 = 실제 배경)
python v10_farm_infer/hole_check.py   (model/에서) -> v10_farm_infer/holecheck/results.json, sheet_*.jpg"""
import sys, json, random, os
sys.path.insert(0, 'v8_reallabel/scripts')
from sam_common import load_rgb, run_sam  # paths(opencv 우회) 먼저 import됨
import numpy as np, cv2
from scipy import ndimage as ndi
from PIL import Image, ImageDraw, ImageFont
B = 'v10_farm_infer'; OUT = f'{B}/holecheck'; os.makedirs(OUT, exist_ok=True)
HOLE_MIN_FRAC, HOLE_MIN_PX, NOTCH_DEPTH = 0.01, 30, 0.12
FONT = ImageFont.load_default(size=18)

def poly_mask(poly, shape, off):
    m = np.zeros(shape, np.uint8)
    cv2.fillPoly(m, [np.round(np.array(poly) - off).astype(np.int32)], 1)
    return m.astype(bool)

def analyze(crop, box, pipe_mask):
    """crop RGB, box in crop coords -> dict + sam mask + hole mask"""
    res = run_sam(crop, (0, 0, crop.shape[1], crop.shape[0]), boxes=[box])
    if not res:
        return dict(ok=False), None, None
    m = res[0][0]
    n, lab = cv2.connectedComponents(m.astype(np.uint8))
    if n > 2:
        sizes = [(lab == k).sum() for k in range(1, n)]; m = lab == (1 + int(np.argmax(sizes)))
    filled = ndi.binary_fill_holes(m)
    inner = filled & ~m
    hl, hn = ndi.label(inner); hole = np.zeros_like(m); hole_px = 0
    for k in range(1, hn + 1):
        c = hl == k
        if c.sum() >= HOLE_MIN_PX: hole |= c; hole_px += int(c.sum())
    fa = int(filled.sum()); L = max(box[2] - box[0], box[3] - box[1])
    cs, _ = cv2.findContours(filled.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    depth = 0.0
    if cs:
        c = max(cs, key=cv2.contourArea)
        hull = cv2.convexHull(c, returnPoints=False)
        if hull is not None and len(hull) > 3 and len(c) > 3:
            try:
                d = cv2.convexityDefects(c, hull)
                if d is not None: depth = float(d[:, 0, 3].max() / 256.0)
            except cv2.error:
                pass
    iou = float((filled & pipe_mask).sum() / max(1, (filled | pipe_mask).sum()))
    hf = hole_px / max(1, fa); nd = depth / max(1, L)
    return dict(ok=True, hole_frac=round(hf, 4), hole_px=hole_px, notch=round(nd, 3), iou=round(iou, 3),
                interior=bool(hf >= HOLE_MIN_FRAC), edge=bool(nd >= NOTCH_DEPTH),
                verified=bool(hf >= HOLE_MIN_FRAC or nd >= NOTCH_DEPTH)), m, hole

def blob(shape, cx, cy, r, rng):
    """irregular blob mask (bite-like) around (cx,cy) radius r"""
    ang = np.linspace(0, 2 * np.pi, 24, endpoint=False)
    rad = r * (0.7 + 0.6 * rng.random(24)); rad = np.convolve(np.r_[rad, rad[:2]], np.ones(3) / 3, 'valid')
    pts = np.stack([cx + rad * np.cos(ang), cy + rad * np.sin(ang)], 1).astype(np.int32)
    mm = np.zeros(shape, np.uint8); cv2.fillPoly(mm, [pts], 1); return mm.astype(bool)

rows, vis = [], []
rng = random.Random(7); nrng = np.random.default_rng(7)
for d in ('blackspot', 'minirose'):
    gt = {g['idx']: g for g in json.load(open(f'{B}/gt/{d}/gt_labels.json'))}
    man = json.load(open(f'{B}/gt/{d}/manifest.json'))
    cache = {}
    for it in sorted(man, key=lambda t: t['photo']):
        ph = it['photo']
        if ph not in cache: cache = {ph: load_rgb(f'{B}/inputs/{d}/{ph}')}
        img = cache[ph]; Hh, Ww = img.shape[:2]
        x0, y0, x1, y1 = it['box']; L = max(x1 - x0, y1 - y0); cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        S = int(max(256, 2.2 * L)); wx0 = int(np.clip(cx - S / 2, 0, Ww - S)); wy0 = int(np.clip(cy - S / 2, 0, Hh - S))
        crop = img[wy0:wy0 + S, wx0:wx0 + S].copy()
        bx = [x0 - wx0 - 0.03 * L, y0 - wy0 - 0.03 * L, x1 - wx0 + 0.03 * L, y1 - wy0 + 0.03 * L]
        pm = poly_mask(it['polygon'], crop.shape[:2], (wx0, wy0))
        r, m, hole = analyze(crop, bx, pm)
        g = gt[it['idx']]['gt']
        rows.append(dict(kind='real', folder=d, idx=it['idx'], stratum=it['stratum'], gt=g, **r))
        if it['stratum'] == 'ANOMALY:Hole' or (r.get('verified') and len(vis) < 400):
            vis.append((crop, m, hole, f"{d[:2]}{it['idx']} {it['stratum'].split(':')[-1][:4]} gt{g} {'V' if r.get('verified') else '-'}", 'real'))
        # synthetic: healthy leaves only
        if g == 'H' and pm.sum() > 400:
            for typ in ('interior', 'notch'):
                c2 = crop.copy()
                er = cv2.erode(pm.astype(np.uint8), np.ones((max(3, int(0.25 * L)),) * 2, np.uint8)).astype(bool)
                if typ == 'interior':
                    ys, xs = np.nonzero(er)
                    if len(ys) == 0: continue
                    k = rng.randrange(len(ys)); hb = blob(pm.shape, xs[k], ys[k], rng.uniform(0.07, 0.13) * L, nrng) & pm
                else:
                    cs, _ = cv2.findContours(pm.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                    c = max(cs, key=cv2.contourArea)[:, 0]; p = c[rng.randrange(len(c))]
                    hb = blob(pm.shape, p[0], p[1], rng.uniform(0.15, 0.25) * L, nrng) & pm
                if hb.sum() < 30: continue
                # source = same photo, shifted 1.5~3 L (real background: soil, pot, or another leaf)
                for _ in range(10):
                    a = rng.uniform(0, 2 * np.pi); dist = rng.uniform(1.5, 3.0) * L
                    sx, sy = int(wx0 + dist * np.cos(a)), int(wy0 + dist * np.sin(a))
                    if 0 <= sx and 0 <= sy and sx + S <= Ww and sy + S <= Hh: break
                else:
                    continue
                src = img[sy:sy + S, sx:sx + S]
                c2[hb] = src[hb]
                f = src[hb].astype(np.float32) + 1; exg = float(((2 * f[:, 1] - f[:, 0] - f[:, 2]) / f.sum(1)).mean())
                if rng.random() < 0.5:  # brown rim (damage halo) on half of them
                    rim = cv2.dilate(hb.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool) & ~hb & pm
                    c2[rim] = (0.55 * c2[rim] + 0.45 * np.array([120, 80, 40])).astype(np.uint8)
                r2, m2, hole2 = analyze(c2, bx, pm)
                rows.append(dict(kind='synth', folder=d, idx=it['idx'], typ=typ, bg_green=bool(exg > 0.05), bg_exg=round(exg, 3), **r2))
                if len(vis) < 520 or rng.random() < 0.3:
                    vis.append((c2, m2, hole2, f"S{typ[:3]} {'grn' if exg > 0.05 else 'bg'} {'V' if r2.get('verified') else '-'}", 'synth'))
    print(d, 'done', flush=True)
json.dump(rows, open(f'{OUT}/results.json', 'w'))
# sheets: real Hole-flagged first, then synthetic sample
cell, C = 200, 6
vis_real = [v for v in vis if v[4] == 'real']; vis_syn = [v for v in vis if v[4] == 'synth'][:60]
for name, vv in (('real', vis_real[:60]), ('synth', vis_syn)):
    for k in range(0, len(vv), 30):
        ch = vv[k:k + 30]; sh = Image.new('RGB', (C * cell, ((len(ch) + C - 1) // C) * cell), 'white'); dr = ImageDraw.Draw(sh)
        for i, (cr, m, hole, tag, _) in enumerate(ch):
            im = cr.copy()
            if m is not None:
                edge = m ^ cv2.erode(m.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
                im[cv2.dilate(edge.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)] = (255, 255, 255)
                im[hole] = (255, 0, 0)
            p = Image.fromarray(im).resize((cell - 4, cell - 4)); X, Y = (i % C) * cell, (i // C) * cell; sh.paste(p, (X + 2, Y + 2))
            dr.rectangle([X + 2, Y + 2, X + 190, Y + 22], fill=(0, 0, 0)); dr.text((X + 4, Y + 3), tag, fill=(255, 255, 0), font=FONT)
        sh.save(f'{OUT}/sheet_{name}_{k // 30}.jpg', quality=88)
print('rows', len(rows))
