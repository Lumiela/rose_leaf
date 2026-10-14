# Render the bbox-variant classifier dataset (CLS_VARIANT=bbox): every Dataset leaf cutout is pasted at the centre of a
# square patch cut from a real cube photo (white-balanced), so the training input looks like export_bbox.py crops:
# a leaf filling ~1/BOX_SCALE of the square, surrounded by real leaves / tray / pots.
# Scale-matched (2026-10-14): the pasted leaf gets the size of the real leaves in that background photo
# (bg_leaf_size.json x U(0.7, 1.3)) and the patch is cut at the photo's native resolution, so leaf and surroundings
# share one scale and one sharpness. Clean = that patch; degraded = the whole patch further shrunk x U(0.4, 1.0),
# blurred and colour-jittered (real fixed-camera leaves are ~70-100 px long).
#   train: 1 clean + N degraded per image   val: clean + 1 degraded   test: clean   test_deg: 1 degraded (fixed seed)
# Background photos: common/real_images_cube/*.jpg + rose_real.webp, rose_real2.webp (camfix photos are kept out:
# they are the real-crop check).  Needs bg_leaf_size.py first.
# Usage: CLS_VARIANT=bbox python v7_leafid/cls/scripts/render_data_bbox.py
import os, sys  # noqa: E401
os.environ.setdefault("CLS_VARIANT", "bbox")
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from cls_common import DATA, CLS, NAMES, SHORT, Progress, load_rgba, gray_world, square_crop, to_input, paths  # noqa: E402

import json, shutil  # noqa: E401,E402
from multiprocessing import Pool  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image, ImageFilter, ImageEnhance  # noqa: E402

N_DEG = {"Black": 2, "Fresh": 1, "Hole": 2, "Yellow": 10}
BG_FILES = sorted(paths.REAL_IMAGES_CUBE.glob("*.jpg")) + [paths.REAL_IMAGES / "rose_real.webp", paths.REAL_IMAGES / "rose_real2.webp"]
LEAF_SCALE = (0.7, 1.3)      # pasted leaf long side / median real leaf long side of the background photo
DEG_SHRINK = (0.4, 1.0)      # degraded: extra downscale of the whole patch
BG = []


def init():
    size = json.loads((CLS / "bg_leaf_size.json").read_text())
    for p in BG_FILES:
        img = gray_world(np.asarray(Image.open(p).convert("RGB")))
        f = img.astype(np.float32) / 255
        exg = (2 * f[..., 1] - f[..., 0] - f[..., 2]) / np.maximum(f.sum(-1), 1e-6)
        ys, xs = np.nonzero(exg > 0.12)
        sel = np.random.default_rng(0).choice(len(ys), min(20000, len(ys)), replace=False) if len(ys) else []
        BG.append((img, ys[sel], xs[sel], size.get(p.name, size["_global"])["L"]))


def sample(rgba, rng, deg):
    img, ys, xs, Lbg = BG[rng.integers(len(BG))]
    H, W = img.shape[:2]
    L = max(24.0, Lbg * rng.uniform(*LEAF_SCALE))
    S = L * rng.uniform(1.15, 1.35)
    if len(ys) and rng.random() < 0.75:                     # mostly centred among leaves, like real stage-1 crops
        k = rng.integers(len(ys)); cx, cy = xs[k], ys[k]
    else:
        cx, cy = rng.uniform(0, W), rng.uniform(0, H)
    patch = square_crop(img, cx, cy, S)                       # native resolution of the background photo
    leaf = rgba.rotate(rng.uniform(0, 360), resample=Image.BICUBIC, expand=True)
    if rng.random() < 0.5:
        leaf = leaf.transpose(Image.FLIP_LEFT_RIGHT)
    bb = leaf.getchannel("A").point(lambda v: 255 if v > 16 else 0).getbbox()
    leaf = leaf.crop(bb) if bb else leaf
    s = L / max(leaf.size)
    leaf = leaf.resize((max(2, round(leaf.width * s)), max(2, round(leaf.height * s))), Image.LANCZOS)
    off = rng.uniform(-0.05, 0.05, 2) * patch.width
    patch.paste(leaf, (int((patch.width - leaf.width) / 2 + off[0]), int((patch.height - leaf.height) / 2 + off[1])), leaf)
    if deg:
        n = max(16, round(patch.width * rng.uniform(*DEG_SHRINK)))
        patch = patch.resize((n, n), Image.BILINEAR)
        if rng.random() < 0.7:
            patch = patch.filter(ImageFilter.GaussianBlur(rng.uniform(0.3, 1.2)))
        patch = ImageEnhance.Brightness(patch).enhance(rng.uniform(0.8, 1.2))
        patch = ImageEnhance.Contrast(patch).enhance(rng.uniform(0.85, 1.15))
        patch = ImageEnhance.Color(patch).enhance(rng.uniform(0.85, 1.15))
    return to_input(patch)


def job(args):
    split, rel, ci, idx = args
    rng = np.random.default_rng([7, {"train": 0, "val": 1, "test": 2}[split], idx])
    rgba = load_rgba(paths.DATASET / rel)
    stem, c = Path(rel).stem, NAMES[ci]
    outs = []
    if split == "train":
        outs.append((f"train/{c}/{stem}_c.jpg", sample(rgba, rng, False)))
        for k in range(N_DEG[SHORT[ci]]):
            outs.append((f"train/{c}/{stem}_d{k}.jpg", sample(rgba, rng, True)))
    elif split == "val":
        outs += [(f"val/{c}/{stem}_c.jpg", sample(rgba, rng, False)),
                 (f"val/{c}/{stem}_d.jpg", sample(rgba, rng, True))]
    else:
        outs += [(f"test/{c}/{stem}.jpg", sample(rgba, rng, False)),
                 (f"test_deg/{c}/{stem}.jpg", sample(rgba, rng, True))]
    for rp, im in outs:
        im.save(DATA / rp, quality=95)
    return len(outs)


def check_sheet(out, per=8):
    """Clean (top rows) and degraded (bottom rows) train samples per class, for an eye check of the scale match."""
    rng = np.random.default_rng(0)
    cells = []
    for kind in ("_c", "_d"):
        for c in NAMES:
            fs = sorted(p for p in (DATA / "train" / c).glob("*.jpg") if kind in p.stem)
            cells += [fs[i] for i in rng.choice(len(fs), min(per, len(fs)), replace=False)]
    t = 160
    sheet = Image.new("RGB", (per * t, (len(cells) + per - 1) // per * t), (40, 40, 40))
    for i, p in enumerate(cells):
        sheet.paste(Image.open(p).resize((t, t)), (i % per * t, i // per * t))
    sheet.save(out, quality=88)


if __name__ == "__main__":
    Progress.log("render", f"start (backgrounds: {len(BG_FILES)} photos, scale-matched)", "running")
    split = json.loads((CLS / "split.json").read_text())
    if DATA.exists():
        shutil.rmtree(DATA)
    for s in ("train", "val", "test", "test_deg"):
        for c in NAMES:
            (DATA / s / c).mkdir(parents=True, exist_ok=True)
    with Pool(4, initializer=init) as pool:
        for s in ("train", "val", "test"):
            jobs = [(s, rel, ci, i) for i, (rel, ci) in enumerate(split[s])]
            n = 0
            for k, r in enumerate(pool.imap_unordered(job, jobs, chunksize=16), 1):
                n += r
                if k % 500 == 0:
                    Progress.log("render", f"{s} {k}/{len(jobs)} originals")
            Progress.log("render", f"{s}: {n} images")
    check_sheet(CLS / "bbox_samples_check.jpg")
    Progress.log("render", "dataset ready -> " + str(DATA) + " (check: bbox_samples_check.jpg)", "done")
