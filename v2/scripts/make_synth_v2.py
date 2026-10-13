# Step 2: synthetic rose-leaf scenes on real greenhouse backgrounds (v2).
# Based on 02_phase2_seg_synthetic.ipynb (cells 5-10). Changes vs v1:
#   - real background crops from ./backgrounds (tiled 1x1/2x2/3x3 to vary scale)
#   - scene mix: dense small clustered leaflets / original sparse large leaves / empty backgrounds
#   - Fresh-weighted class sampling, heavier overlap allowed in dense scenes
#   - Hole leaves: near-white "hole" pixels inside the cutout made transparent
# Reads common/Dataset and v2/backgrounds; writes only v2/data/rose_seg_v2 (or _smoke with SMOKE=1).
# Usage: python v2/scripts/make_synth_v2.py > v2/logs/synth_v2.log 2>&1
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
import paths  # noqa: E402  shared paths + opencv workaround (must come before cv2/ultralytics)

import os, random, re, shutil, time
from collections import Counter, defaultdict
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path
import numpy as np
import cv2
from PIL import Image, ImageEnhance, ImageFilter

ROOT = paths.V2
SRC_DIR = paths.DATASET
BG_DIR = ROOT / "backgrounds"
SMOKE = bool(os.environ.get("SMOKE"))         # quick preview run
OUT_DIR = ROOT / "data" / ("rose_seg_v2_smoke" if SMOKE else "rose_seg_v2")
SEED = 42
SPLIT = (0.70, 0.15, 0.15)
N_SCENES = {"train": 24, "val": 4, "test": 4} if SMOKE else {"train": 6000, "val": 800, "test": 800}
IMG = 640
SCENE_MIX = {"dense": 0.6, "sparse": 0.3, "empty": 0.1}
SCENE_CFG = {
    "dense":  dict(n=(40, 100), scale=(0.05, 0.14), max_overlap=0.5, min_visible=0.35, cluster=(4, 9), real_bg=0.8),
    "sparse": dict(n=(1, 6),   scale=(0.18, 0.55), max_overlap=0.15, min_visible=0.5,  cluster=(1, 1), real_bg=0.7),
    "empty":  dict(real_bg=1.0),
}
BG_TILES = {1: 0.4, 2: 0.4, 3: 0.2}       # real background: grid of n x n crops
CLASS_P = {"Black Spot Leaves": 0.18, "Fresh Leaves": 0.50, "Hole Leaves": 0.18, "Yellow Leaves": 0.14}
SAME_CLASS_IN_CLUSTER = 0.8
MIN_PIXELS = 60                           # drop instances with fewer visible pixels
CACHE_SIDE = 384                          # leaves are cached at this max side (>= 0.55 * IMG)
HOLE_WHITE_TO_ALPHA = True
SHADOW_P = 0.5
WORKERS = 4                               # server is otherwise idle

CLASSES = sorted(d.name for d in SRC_DIR.iterdir() if d.is_dir())
CLASS_W = np.array([CLASS_P[c] for c in CLASSES]); CLASS_W /= CLASS_W.sum()

def stem_of(name):
    return re.split(r"_aug_", Path(name).stem)[0]

# ---- split: identical procedure and seed as the v1 notebook -> same leaf split ----
random.seed(SEED); np.random.seed(SEED)
pools = {s: defaultdict(list) for s in ("train", "val", "test")}
for ci, c in enumerate(CLASSES):
    groups = defaultdict(list)
    for f in sorted((SRC_DIR / c).glob("*.png")):
        groups[stem_of(f.name)].append(f)
    stems = sorted(groups); random.shuffle(stems)
    n = len(stems); n_tr = int(n * SPLIT[0]); n_va = int(n * SPLIT[1])
    parts = {"train": stems[:n_tr], "val": stems[n_tr:n_tr + n_va], "test": stems[n_tr + n_va:]}
    for s, ss in parts.items():
        for st in ss:
            files = groups[st]
            if s != "train":
                files = [f for f in files if "_aug_" not in f.name] or files[:1]
            pools[s][ci].extend(files)
for s in pools:
    print(f"POOL {s:5s}", {CLASSES[ci]: len(v) for ci, v in pools[s].items()}, flush=True)

# ---- leaf cache ----
hole_stats = []
def load_leaf(path, ci):
    im = Image.open(path).convert("RGBA")
    im = im.crop(im.getchannel("A").point(lambda v: 255 if v > 16 else 0).getbbox())
    im.thumbnail((CACHE_SIDE, CACHE_SIDE), Image.LANCZOS)
    if HOLE_WHITE_TO_ALPHA and CLASSES[ci] == "Hole Leaves":
        a = np.asarray(im).copy()
        rgb = a[..., :3].astype(np.int16)
        inside = a[..., 3] > 16
        white = inside & (rgb.min(-1) > 200) & (rgb.max(-1) - rgb.min(-1) < 35)
        white = cv2.morphologyEx(white.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)).astype(bool)
        hole_stats.append(white.sum() / max(1, inside.sum()))
        a[white, 3] = 0
        im = Image.fromarray(a, "RGBA")
    return im

t0 = time.time()
all_files = {(f, ci) for s in pools for ci, fs in pools[s].items() for f in fs}
with ThreadPoolExecutor(WORKERS) as ex:
    cache = dict(zip(all_files, ex.map(lambda fc: load_leaf(*fc), all_files)))
print("CACHE", len(cache), "leaves in %.0fs" % (time.time() - t0), flush=True)
if hole_stats:
    hs = np.array(hole_stats)
    print("HOLEFIX leaves=%d with_white(>0.5%%)=%d median_white_frac=%.3f p90=%.3f" %
          (len(hs), (hs > 0.005).sum(), np.median(hs), np.percentile(hs, 90)), flush=True)

BG_FILES = sorted(p for p in BG_DIR.glob("*.jpg"))
BG_IMGS = [Image.open(p).convert("RGB") for p in BG_FILES]
print("BACKGROUNDS", len(BG_IMGS), flush=True)

def procedural_bg(w, h, rng):
    kind = rng.integers(0, 5)
    if kind == 0:
        return np.full((h, w, 3), rng.integers(0, 256, 3), np.uint8)
    if kind == 1:
        a, b = rng.integers(0, 256, 3), rng.integers(0, 256, 3)
        t = np.linspace(0, 1, w)[None, :, None]
        return np.broadcast_to(a * (1 - t) + b * t, (h, w, 3)).astype(np.uint8)
    if kind == 2:
        cell = int(rng.integers(8, 64))
        small = rng.integers(0, 256, (max(h // cell, 2), max(w // cell, 2), 3), dtype=np.uint8)
        return np.array(Image.fromarray(small).resize((w, h), Image.BICUBIC))
    if kind == 3:
        base = np.array([rng.integers(60, 140), rng.integers(40, 100), rng.integers(20, 70)])
        noise = rng.normal(0, rng.uniform(5, 30), (h, w, 1))
        return np.clip(base + noise, 0, 255).astype(np.uint8)
    g = rng.integers(0, 60)
    return np.full((h, w, 3), g, np.uint8)

def real_crop(size, rng):
    im = BG_IMGS[rng.integers(len(BG_IMGS))]
    s = size / min(im.size) * rng.uniform(1.0, 1.4)
    im = im.resize((max(size, int(im.width * s)), max(size, int(im.height * s))), Image.BILINEAR)
    x, y = rng.integers(0, im.width - size + 1), rng.integers(0, im.height - size + 1)
    im = im.crop((x, y, x + size, y + size))
    if rng.random() < 0.5: im = im.transpose(Image.FLIP_LEFT_RIGHT)
    if rng.random() < 0.3: im = im.transpose(Image.FLIP_TOP_BOTTOM)
    im = ImageEnhance.Brightness(im).enhance(rng.uniform(0.8, 1.15))
    return im

def get_background(rng, real_p):
    if BG_IMGS and rng.random() < real_p:
        g = int(rng.choice(list(BG_TILES), p=np.array(list(BG_TILES.values())) / sum(BG_TILES.values())))
        cell = IMG // g
        canvas = Image.new("RGB", (IMG, IMG))
        for i in range(g):
            for j in range(g):
                w = IMG - cell * (g - 1) if j == g - 1 else cell
                h = IMG - cell * (g - 1) if i == g - 1 else cell
                canvas.paste(real_crop(max(w, h), rng).crop((0, 0, w, h)), (j * cell, i * cell))
        return np.asarray(canvas, np.float32).copy()
    return procedural_bg(IMG, IMG, rng).astype(np.float32)

def prepare_leaf(base, scale, rng):
    im = base
    target = rng.uniform(*scale) * IMG
    s = target / max(im.size)
    im = im.resize((max(6, round(im.width * s)), max(6, round(im.height * s))),
                   Image.LANCZOS if target > 120 else Image.BILINEAR)
    if rng.random() < 0.5: im = im.transpose(Image.FLIP_LEFT_RIGHT)
    im = im.rotate(rng.uniform(0, 360), resample=Image.BICUBIC, expand=True)
    bb = im.getchannel("A").point(lambda v: 255 if v > 16 else 0).getbbox()
    if bb is None:
        return None
    im = im.crop(bb)
    if max(im.size) > IMG * 0.9:
        im.thumbnail((int(IMG * 0.9), int(IMG * 0.9)), Image.LANCZOS)
    rgb, a = im.convert("RGB"), im.getchannel("A")
    rgb = ImageEnhance.Brightness(rgb).enhance(rng.uniform(0.75, 1.25))
    rgb = ImageEnhance.Contrast(rgb).enhance(rng.uniform(0.8, 1.2))
    rgb = ImageEnhance.Color(rgb).enhance(rng.uniform(0.85, 1.15))
    rgb.putalpha(a)
    return rgb

def mask_to_polygon(mask):
    cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    c = cv2.approxPolyDP(c, 0.002 * cv2.arcLength(c, True), True).reshape(-1, 2)
    if len(c) < 3:
        return None
    return (c.astype(np.float32) / IMG).clip(0, 1).flatten()

def make_scene(pool, rng):
    kind = str(rng.choice(list(SCENE_MIX), p=np.array(list(SCENE_MIX.values()))))
    cfg = SCENE_CFG[kind]
    canvas = get_background(rng, cfg["real_bg"])
    insts = []
    if kind != "empty":
        occ = np.zeros((IMG, IMG), bool)
        n_target = int(rng.integers(cfg["n"][0], cfg["n"][1] + 1))
        placed_tries = 0
        while len(insts) < n_target and placed_tries < n_target * 3:
            k = int(rng.integers(cfg["cluster"][0], cfg["cluster"][1] + 1))
            cls0 = int(rng.choice(len(CLASSES), p=CLASS_W))
            cx, cy = rng.uniform(0, IMG), rng.uniform(0, IMG)
            for _ in range(k):
                placed_tries += 1
                ci = cls0 if rng.random() < SAME_CLASS_IN_CLUSTER else int(rng.choice(len(CLASSES), p=CLASS_W))
                fs = pool[ci]
                leaf = prepare_leaf(cache[(fs[rng.integers(len(fs))], ci)], cfg["scale"], rng)
                if leaf is None:
                    continue
                w, h = leaf.size
                la = np.asarray(leaf.getchannel("A"), np.float32) / 255.0
                lm = la > 0.5
                if lm.sum() < MIN_PIXELS:
                    continue
                spread = 0.0 if cfg["cluster"][1] == 1 else 0.9 * max(w, h)
                ok = False
                for _try in range(10):
                    if spread:
                        x = int(np.clip(cx + rng.normal(0, spread) - w / 2, 0, IMG - w))
                        y = int(np.clip(cy + rng.normal(0, spread) - h / 2, 0, IMG - h))
                    else:
                        x, y = int(rng.integers(0, IMG - w + 1)), int(rng.integers(0, IMG - h + 1))
                    if (occ[y:y + h, x:x + w] & lm).sum() <= cfg["max_overlap"] * lm.sum():
                        ok = True
                        break
                if not ok:
                    continue
                if rng.random() < SHADOW_P:
                    sh = np.asarray(leaf.getchannel("A").filter(ImageFilter.GaussianBlur(rng.uniform(2, 8))), np.float32) / 255.0
                    d = max(2, int(0.06 * max(w, h)))
                    dx, dy = rng.integers(-d, d + 1, 2)
                    x0, y0, x1, y1 = max(0, x + dx), max(0, y + dy), min(IMG, x + dx + w), min(IMG, y + dy + h)
                    if x1 > x0 and y1 > y0:
                        s_ = sh[y0 - y - dy:y1 - y - dy, x0 - x - dx:x1 - x - dx][..., None]
                        canvas[y0:y1, x0:x1] *= 1 - s_ * rng.uniform(0.2, 0.5)
                rgb = np.asarray(leaf.convert("RGB"), np.float32)
                a = la[..., None]
                canvas[y:y + h, x:x + w] = rgb * a + canvas[y:y + h, x:x + w] * (1 - a)
                m = np.zeros((IMG, IMG), bool); m[y:y + h, x:x + w] = lm
                occ |= m
                insts.append((ci, m))
                if len(insts) >= n_target:
                    break

    labels = []
    # visibility: later leaves are on top (accumulate occluders from the end)
    above = np.zeros((IMG, IMG), bool)
    vis_list = []
    for ci, m in reversed(insts):
        vis = m & ~above
        above |= m
        vis_list.append((ci, m, vis))
    for ci, m, vis in reversed(vis_list):
        if vis.sum() < max(MIN_PIXELS, cfg.get("min_visible", 0) * m.sum()):
            continue
        poly = mask_to_polygon(vis)
        if poly is not None:
            labels.append(f"{ci} " + " ".join(f"{v:.5f}" for v in poly))
    if rng.random() < 0.3:
        canvas += rng.normal(0, rng.uniform(2, 6), canvas.shape)
    img = Image.fromarray(canvas.clip(0, 255).astype(np.uint8))
    if rng.random() < 0.2:
        img = img.filter(ImageFilter.GaussianBlur(rng.uniform(0.3, 1.0)))
    return kind, img, labels

def scene_job(args):
    split, i, n, t0 = args
    sid = ("train", "val", "test").index(split)
    rng = np.random.default_rng([SEED, 100 + sid, i])
    kind, img, labels = make_scene(pools[split], rng)
    img.save(OUT_DIR / "images" / split / f"{split}_{i:05d}.jpg", quality=92)
    (OUT_DIR / "labels" / split / f"{split}_{i:05d}.txt").write_text("\n".join(labels))
    if i % 500 == 0:
        print(f"  {split} {i}/{n} {time.time() - t0:.0f}s", flush=True)
    return kind, [int(l.split()[0]) for l in labels]

def build_split(split, n):
    (OUT_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    # fork: workers inherit the leaf/background caches without pickling them
    with ProcessPoolExecutor(WORKERS, mp_context=mp.get_context("fork")) as ex:
        res = list(ex.map(scene_job, [(split, i, n, t0) for i in range(n)], chunksize=16))
    kinds = Counter(k for k, _ in res)
    cnt = Counter(c for _, cs in res for c in cs)
    inst = [len(cs) for k, cs in res if k == "dense"]
    print(f"SPLIT {split} scenes={n} kinds={dict(kinds)} dense_inst_median={int(np.median(inst)) if inst else 0}",
          {CLASSES[k]: cnt[k] for k in range(len(CLASSES))}, f"{time.time() - t0:.0f}s", flush=True)

assert OUT_DIR.resolve().is_relative_to(ROOT)
if OUT_DIR.exists():
    shutil.rmtree(OUT_DIR)
for s, n in N_SCENES.items():
    build_split(s, n)

yaml = OUT_DIR / "rose_seg_v2.yaml"
yaml.write_text(f"path: {OUT_DIR.resolve().as_posix()}\ntrain: images/train\nval: images/val\ntest: images/test\n"
                "names:\n" + "".join(f"  {i}: {c}\n" for i, c in enumerate(CLASSES)))
print("YAML", yaml)
print("DONE", flush=True)
