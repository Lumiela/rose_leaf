"""Paste test: held-out Dataset cutouts (v7 cls split 'test', never trained on) pasted onto the 9 healthy farm photos.
Hole cutouts (67) + Fresh cutouts (67, control) on top of existing healthy leaves, long side = photo L x U(1.0,1.4).
Two conditions, same placements:
  asis   cutout as it is in Dataset (hole pixels show the white/black background baked into the original photo)
  transp hole-like pixels inside the leaf (white/gray: S<0.18 & V>0.55, or black: V<0.16, comps >= 0.2% of leaf)
         made transparent (Hole cutouts only; Fresh is pasted unchanged in both), so the farm scene behind shows through the hole
Outputs: inputs_paste/<cond>/*.jpg, paste/masks/<photo>.npz (paste id map, leaf extent incl. holes), paste/meta.json
Run from model/: python v10_farm_infer/paste/paste_make.py"""
import json, glob, os
import numpy as np
from PIL import Image, ImageFilter
from scipy import ndimage as ndi

B = 'v10_farm_infer'
rng = np.random.default_rng(7)
split = json.load(open('v7_leafid/cls/split.json'))
hole_fs = [f for f, c in split['test'] if c == 2]
fresh_fs = [f for f, c in split['test'] if c == 1]
fresh_fs = [fresh_fs[i] for i in rng.choice(len(fresh_fs), len(hole_fs), replace=False)]
items = [('Hole', f) for f in hole_fs] + [('Fresh', f) for f in fresh_fs]
rng.shuffle(items)
photos = sorted(glob.glob(f'{B}/inputs/minirose/*.jpg'))
per = [len(items) // len(photos) + (i < len(items) % len(photos)) for i in range(len(photos))]
for c in ('asis', 'transp'):
    os.makedirs(f'{B}/inputs_paste/{c}', exist_ok=True)
os.makedirs(f'{B}/paste/masks', exist_ok=True)


def hole_mask(rgba):
    a = rgba[..., 3] > 127
    rgb = rgba[..., :3].astype(np.float32) / 255
    v = rgb.max(-1); s = (v - rgb.min(-1)) / np.maximum(v, 1e-3)
    cand = a & (((s < 0.18) & (v > 0.55)) | (v < 0.16))
    cand = ndi.binary_opening(cand, iterations=1)
    lab, n = ndi.label(cand)
    if n == 0:
        return cand
    sizes = ndi.sum(cand, lab, range(1, n + 1))
    keep = np.isin(lab, 1 + np.nonzero(sizes >= 0.002 * a.sum())[0])
    return ndi.binary_dilation(keep, iterations=1) & a


meta = []
k = 0
for pi, p in enumerate(photos):
    name = os.path.basename(p)
    im = Image.open(p); im.load()
    W, H = im.size
    run = json.load(open(f'{B}/runs/minirose/{name[:-4]}.json'))
    L = run['L']
    base = np.array(im.convert('RGB'))
    out = {'asis': base.copy(), 'transp': base.copy()}
    idmap = np.zeros((H, W), np.uint8)
    cands = [l for l in run['leaves'] if l['state'] == 'FRESH']
    order = rng.permutation(len(cands))
    placed = 0
    for ci in order:
        if placed >= per[pi]:
            break
        cls, f = items[k]
        rgba = np.array(Image.open('common/Dataset/' + f).convert('RGBA'))
        ys, xs = np.nonzero(rgba[..., 3] > 127)
        rgba = rgba[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
        side = L * rng.uniform(1.0, 1.4)
        sc = side / max(rgba.shape[:2])
        img = Image.fromarray(rgba).resize((max(4, round(rgba.shape[1] * sc)), max(4, round(rgba.shape[0] * sc))), Image.LANCZOS)
        if rng.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        img = img.rotate(rng.uniform(0, 360), expand=True, resample=Image.BICUBIC)
        r = np.array(img)
        h, w = r.shape[:2]
        bx = cands[ci]['box']; cx, cy = (bx[0] + bx[2]) / 2, (bx[1] + bx[3]) / 2
        x0, y0 = int(cx - w / 2), int(cy - h / 2)
        if x0 < 8 or y0 < 8 or x0 + w > W - 8 or y0 + h > H - 8:
            continue
        ext = ndi.binary_fill_holes(r[..., 3] > 127)
        if (idmap[y0:y0 + h, x0:x0 + w][ext] > 0).any() or (ndi.binary_dilation(idmap[y0:y0+h, x0:x0+w] > 0, iterations=int(0.3 * L)) & ext).any():
            continue
        hm = hole_mask(r) if cls == 'Hole' else np.zeros(r.shape[:2], bool)
        k += 1; placed += 1
        idmap[y0:y0 + h, x0:x0 + w][ext] = k
        for c in ('asis', 'transp'):
            a = r[..., 3].astype(np.float32)
            if c == 'transp':
                a[hm] = 0
            a = np.array(Image.fromarray(a.astype(np.uint8)).filter(ImageFilter.GaussianBlur(0.8))).astype(np.float32)[..., None] / 255
            reg = out[c][y0:y0 + h, x0:x0 + w].astype(np.float32)
            out[c][y0:y0 + h, x0:x0 + w] = (r[..., :3] * a + reg * (1 - a)).round().astype(np.uint8)
        meta.append({'id': k, 'photo': name, 'cls': cls, 'file': f, 'box': [x0, y0, x0 + w, y0 + h], 'L': L,
                     'hole_frac': float(hm.sum() / max(1, ext.sum())), 'ext_area': int(ext.sum())})
    for c in ('asis', 'transp'):
        Image.fromarray(out[c]).save(f'{B}/inputs_paste/{c}/{name}', quality=95)
    np.savez_compressed(f'{B}/paste/masks/{name[:-4]}.npz', idmap=idmap)
    print(name, W, H, 'L', L, 'placed', placed, 'of', per[pi], flush=True)
json.dump(meta, open(f'{B}/paste/meta.json', 'w'), indent=0)
from collections import Counter
print('total', len(meta), Counter(m['cls'] for m in meta), 'unused items', len(items) - k)
hf = np.array([m['hole_frac'] for m in meta if m['cls'] == 'Hole']); ff = np.array([m['hole_frac'] for m in meta if m['cls'] == 'Fresh'])
print('hole_frac Hole: >0', (hf > 0).mean().round(2), 'median', np.median(hf).round(3), '| Fresh >0', (ff > 0).mean().round(2))
