# v8 step 1: SAM2.1 draft instance masks for one real photo.
#   prompts: v7 stage-1 boxes (optional) + seeds on leaf-coloured pixels that no mask covers yet (2 rounds)
#   SAM runs on windows of ~10 leaf lengths so small leaves keep detail.
# Output: v8_reallabel/work/<stem>/inst.npz (uint16 id map) + meta.json
# Usage: python v8_reallabel/scripts/sam_draft.py <photo> [--v7 leaves.json] [--L leaf_long_side_px]
import argparse, json, time  # noqa: E401
from pathlib import Path
from sam_common import load_rgb, gray_world, leaf_color, window, run_sam, save, peaks, split_by_points, is_thin, props  # imports paths (opencv fix) first
import numpy as np
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("photo"); ap.add_argument("--v7"); ap.add_argument("--L", type=float)
a = ap.parse_args()
t0 = time.time()
photo = Path(a.photo); stem = photo.stem
img = load_rgb(photo); H, W = img.shape[:2]
leaf = leaf_color(gray_world(img))
boxes = []
if a.v7:
    rows = [r for r in json.loads(Path(a.v7).read_text()) if Path(r["photo"]).stem == stem]
    boxes = [[r["x0"], r["y0"], r["x1"], r["y1"]] for r in rows]
L = a.L or (float(np.median([max(b[2] - b[0], b[3] - b[1]) for b in boxes])) if boxes else 80.0)
T = int(np.clip(10 * L, 512, 1024))
st = max(1, int(0.6 * T))
wins = sorted({window(cx, cy, T, W, H) for cy in range(T // 2, H + st, st) for cx in range(T // 2, W + st, st)})
print(f"PHOTO {stem} {W}x{H} L={L:.0f} T={T} windows={len(wins)} v7_boxes={len(boxes)}", flush=True)


def assign(cx, cy, bw, bh):
    best = None
    for w in wins:
        if w[0] <= cx - bw / 2 and w[1] <= cy - bh / 2 and w[2] >= cx + bw / 2 and w[3] >= cy + bh / 2:
            d = ((w[0] + w[2]) / 2 - cx) ** 2 + ((w[1] + w[3]) / 2 - cy) ** 2
            if best is None or d < best[0]:
                best = (d, w)
    return best[1] if best else window(cx, cy, max(T, int(1.6 * max(bw, bh))), W, H)


items = []  # (full-res bool mask, src, score)


def run_groups(groups, kind):
    for win, prom in groups.items():
        if kind == "box":
            res = run_sam(img, win, boxes=prom)
        else:
            res = run_sam(img, win, points=[[p] for p in prom], labels=[[1]] * len(prom))
        for m, s in res:
            full = np.zeros((H, W), bool); full[win[1]:win[3], win[0]:win[2]] = m
            items.append((full, kind, float(s)))


g = {}
for b in boxes:
    w = assign((b[0] + b[2]) / 2, (b[1] + b[3]) / 2, b[2] - b[0], b[3] - b[1])
    g.setdefault(w, []).append(b)
run_groups(g, "box")
print(f"  box prompts -> {len(items)} masks {time.time() - t0:.1f}s", flush=True)

k = max(3, int(L / 10) | 1)
for rnd in range(3):
    union = np.zeros((H, W), bool)
    for m, _, _ in items:
        union |= m
    left = cv2.morphologyEx((leaf & ~union).astype(np.uint8), cv2.MORPH_OPEN, np.ones((k, k), np.uint8))
    n, lab, stt, _ = cv2.connectedComponentsWithStats(left, 8)
    seeds = []
    for c in range(1, n):
        if stt[c, cv2.CC_STAT_AREA] < 0.12 * L * L:
            continue
        x, y, w, h = stt[c, :4]
        comp = (lab[y:y + h, x:x + w] == c).astype(np.uint8)
        dt = cv2.distanceTransform(comp, cv2.DIST_L2, 5)
        yy, xx = np.unravel_index(int(dt.argmax()), dt.shape)
        seeds.append((int(x + xx), int(y + yy)))
    g = {}
    for s in seeds:
        g.setdefault(assign(s[0], s[1], L, L), []).append(s)
    before = len(items)
    tmp = items; items = []
    run_groups(g, "seed")
    new = items; items = tmp
    acc = 0
    for m, src, sc in new:
        if m.sum() > 6 * L * L:
            continue
        m2 = m & ~union  # keep only the part no earlier mask covers
        if m2.sum() < 0.1 * L * L or leaf[m2].mean() < 0.5:
            continue
        items.append((m2, src, sc)); union |= m2; acc += 1
    print(f"  seed round {rnd}: seeds={len(seeds)} accepted={acc} {time.time() - t0:.1f}s", flush=True)

# resolve overlaps: paint largest first so smaller leaves stay on top; drop absorbed / tiny masks
order = sorted(range(len(items)), key=lambda i: -items[i][0].sum())
inst = np.zeros((H, W), np.uint16)
src = {}
nid = 0
for i in order:
    m = items[i][0]
    if m.sum() < 0.05 * L * L:
        continue
    nid += 1
    inst[m] = nid
    src[nid] = dict(src=items[i][1], score=round(items[i][2], 3), area0=int(m.sum()))
keep = {}
for i, s in src.items():
    ar = int((inst == i).sum())
    if ar < 0.4 * s["area0"] or ar < 0.05 * L * L:
        inst[inst == i] = 0
    else:
        keep[i] = s
# drop thin objects (stems, wires); auto-split merged masks at distance-transform peaks (leaflet centres)
P = props(inst, leaf, L)
med = float(np.median([p["area"] for p in P.values()])) if P else L * L
n_thin = n_split = 0
for i, p in P.items():
    region = inst == i
    if is_thin(region, L):
        inst[region] = 0; keep.pop(i, None); n_thin += 1
        continue
    if p["solidity"] < 0.85 or p["area"] > 2.5 * med:
        pts = peaks(region, L)
        if len(pts) < 2:
            continue
        ms = split_by_points(img, region, pts, T, L)
        inst[region] = 0; base = keep.pop(i, dict(src="box"))
        for m in sorted(ms, key=lambda m: -m.sum()):
            m = m & region
            if m.sum() < 0.05 * L * L:
                continue
            nid += 1; inst[m] = nid; keep[nid] = dict(src=base["src"] + "+split", score=0.0, area0=int(m.sum()))
        n_split += 1
for i in list(keep):
    if (inst == i).sum() < 0.05 * L * L:
        inst[inst == i] = 0; keep.pop(i)
print(f"  thin dropped={n_thin} auto-split={n_split} {time.time() - t0:.1f}s", flush=True)
remap = None
# renumber top-left -> bottom-right by centroid row bands
cents = {i: np.argwhere(inst == i).mean(0) for i in keep}
ordered = sorted(keep, key=lambda i: (int(cents[i][0] // max(L, 1)), cents[i][1]))
remap = np.zeros(nid + 1, np.uint16)
meta_ids = {}
for new, old in enumerate(ordered, 1):
    remap[old] = new; meta_ids[new] = keep[old]
inst = remap[inst]
meta = dict(photo=str(photo), stem=stem, W=W, H=H, L=L, T=T, ids={str(k): v for k, v in meta_ids.items()}, ignore=[], log=[])
save(stem, inst, meta)
print(f"DONE {stem}: instances={len(meta_ids)} {time.time() - t0:.1f}s", flush=True)
