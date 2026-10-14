"""Shared helpers for the stage-2 leaf classifier comparison (v7_leafid/cls).

Images: RGBA leaf cutouts (common/Dataset for training, v7 leaf_search crops for inference) are letterboxed on a
neutral gray square (SIZE px). Training variants imitate the cube crops: shrunk to 40-150 px then upscaled, blur,
brightness/contrast/saturation jitter. Progress of all stages goes to progress.json / progress.md (open the .md in
JupyterLab: right click -> Open With -> Markdown Preview).

Variants (env CLS_VARIANT): "" = leaf cutouts on gray (data/, runs/, progress.md);
"bbox" = square box crops WITH real background (data_bbox/, runs_bbox/, progress_bbox.md);
"bbox_real" = bbox + reviewed real cube leaves as Fresh (data_bbox_real/, runs_bbox_real/, see real_prep.py). In the bbox variant the
training leaf is pasted on a patch of a real cube photo (common/real_images_cube + rose_real 1,2; the camfix photos used
for the real-crop check are NOT used) and the whole patch is degraded, imitating a box crop around a stage-1 leaf.
"""
import os
import sys
import json
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "common"))
import paths  # noqa: E402  shared paths + opencv workaround (before cv2/ultralytics)

import numpy as np  # noqa: E402
from PIL import Image, ImageFilter, ImageEnhance  # noqa: E402

CLS = paths.V7 / "cls"
VARIANT = os.environ.get("CLS_VARIANT", "")
SUF = f"_{VARIANT}" if VARIANT else ""
DATA = CLS / f"data{SUF}"
RUNS = CLS / f"runs{SUF}"
BOX_SCALE = 1.25          # bbox variant: square side = BOX_SCALE x leaf long side
NAMES = paths.CLASS_NAMES
SHORT = [n.split()[0] for n in NAMES]
SIZE = 224
GRAY = (128, 128, 128)
V7_RUN = paths.V7 / "runs" / "real_images_camfix_20261014_005116"     # stage-1 output used for the real-crop check


def load_rgba(path):
    im = Image.open(path).convert("RGBA")
    bb = im.getchannel("A").point(lambda v: 255 if v > 16 else 0).getbbox()
    return im.crop(bb) if bb else im


def compose(rgba, size=SIZE):
    """Letterbox an RGBA cutout on a gray square (5% margin)."""
    w, h = rgba.size
    s = size * 0.9 / max(w, h)
    im = rgba.resize((max(1, round(w * s)), max(1, round(h * s))), Image.LANCZOS)
    bg = Image.new("RGB", (size, size), GRAY)
    bg.paste(im, ((size - im.width) // 2, (size - im.height) // 2), im)
    return bg


def degrade(rgba, rng):
    """Imitate a stage-1 cube crop: small (40-150 px long side), often blurred, colour/brightness shifted."""
    w, h = rgba.size
    s = rng.uniform(40, 150) / max(w, h)
    small = rgba.resize((max(4, round(w * s)), max(4, round(h * s))), Image.BILINEAR)
    rgb, a = small.convert("RGB"), small.getchannel("A")
    if rng.random() < 0.7:
        rgb = rgb.filter(ImageFilter.GaussianBlur(rng.uniform(0.3, 1.2)))
    rgb = ImageEnhance.Brightness(rgb).enhance(rng.uniform(0.8, 1.2))
    rgb = ImageEnhance.Contrast(rgb).enhance(rng.uniform(0.85, 1.15))
    rgb = ImageEnhance.Color(rgb).enhance(rng.uniform(0.85, 1.15))
    out = rgb.convert("RGBA")
    out.putalpha(a)
    return out


def metrics(y_true, y_pred):
    from sklearn.metrics import accuracy_score, f1_score, recall_score, confusion_matrix
    y_true, y_pred = list(map(int, y_true)), list(map(int, y_pred))
    return dict(acc=round(accuracy_score(y_true, y_pred), 4),
                macro_f1=round(f1_score(y_true, y_pred, average="macro", labels=list(range(4)), zero_division=0), 4),
                recall={SHORT[k]: round(float(r), 3) for k, r in enumerate(recall_score(y_true, y_pred, average=None, labels=list(range(4)), zero_division=0))},
                confusion=confusion_matrix(y_true, y_pred, labels=list(range(4))).tolist(), n=len(y_true))


def folder_items(split):
    return [(p, ci) for ci, c in enumerate(NAMES) for p in sorted((DATA / split / c).glob("*.jpg"))]


def v7_crops():
    return sorted((V7_RUN / "crops").glob("*/*.png"))


def v7_items():
    """(paths, loader) of the stage-1 real leaves for the current variant (bbox crops come from export_bbox.py)."""
    if VARIANT.startswith("bbox"):      # bbox, bbox_real
        return sorted((V7_RUN / "bbox").glob("*/*.jpg")), (lambda p: Image.open(p).convert("RGB"))
    return v7_crops(), (lambda p: compose(load_rgba(p)))


def v7_hold():
    """id -> True if the stage-1 leaf should not be judged (several leaves merged or cut by the frame)."""
    rows = json.loads((V7_RUN / "leaves.json").read_text())
    return {r["id"]: bool(r["solidity"] < 0.8 or r["touches_border"]) for r in rows}


def gray_world(img):
    hsv_s = (img.max(-1).astype(np.int16) - img.min(-1)) / np.maximum(img.max(-1), 1)
    v = img.max(-1)
    ref = (hsv_s < 60 / 255) & (v > 60) & (v < 240)
    if ref.sum() < 1000:
        ref = np.ones(ref.shape, bool)
    mean = img[ref].reshape(-1, 3).mean(0)
    return np.clip(img.astype(np.float32) * (mean.mean() / mean), 0, 255).astype(np.uint8)


def square_crop(img, cx, cy, side, fill=GRAY):
    """Square crop centred at (cx, cy); outside the image is filled with gray."""
    H, W = img.shape[:2]
    side = max(4, int(round(side)))
    x0, y0 = int(round(cx - side / 2)), int(round(cy - side / 2))
    out = np.empty((side, side, 3), np.uint8); out[:] = fill
    sx0, sy0, sx1, sy1 = max(0, x0), max(0, y0), min(W, x0 + side), min(H, y0 + side)
    if sx1 > sx0 and sy1 > sy0:
        out[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = img[sy0:sy1, sx0:sx1]
    return Image.fromarray(out)


def to_input(pil, size=SIZE):
    return pil.resize((size, size), Image.LANCZOS if pil.width > size else Image.BILINEAR)


class Progress:
    """Stage status + last lines -> progress.json / progress.md (re-read each update so stages can run separately)."""
    J, M = CLS / f"progress{SUF}.json", CLS / f"progress{SUF}.md"
    ORDER = ["download", "export_bbox", "bg_size", "render", "yolo11s_cls", "effnetv2s", "dinov2_vits14", "compare"]

    @classmethod
    def log(cls, stage, msg, status=None):
        st = json.loads(cls.J.read_text()) if cls.J.exists() else {}
        s = st.setdefault(stage, dict(status="running", start=time.strftime("%H:%M:%S"), lines=[]))
        if status:
            s["status"] = status
        s["updated"] = time.strftime("%H:%M:%S")
        if msg:
            s["lines"] = (s["lines"] + [f"{time.strftime('%H:%M:%S')} {msg}"])[-25:]
            print(f"[{stage}] {msg}", flush=True)
        cls.J.write_text(json.dumps(st, indent=1, ensure_ascii=False))
        icon = {"running": "진행 중", "done": "완료", "failed": "실패"}
        md = [f"# v7 2단계 분류 모델 비교 학습 진행 상황 ({VARIANT or 'cutout'})", f"마지막 갱신: {time.strftime('%Y-%m-%d %H:%M:%S')} (UTC)", "",
              "| 단계 | 상태 | 시작 | 갱신 | 최근 메시지 |", "|---|---|---|---|---|"]
        for k in cls.ORDER + [k for k in st if k not in cls.ORDER]:
            if k in st:
                v = st[k]
                md.append(f"| {k} | {icon.get(v['status'], v['status'])} | {v.get('start', '')} | {v.get('updated', '')} | {(v['lines'][-1][9:] if v['lines'] else '').replace('|', '/')} |")
        for k in cls.ORDER + [k for k in st if k not in cls.ORDER]:
            if k in st:
                md += ["", f"## {k}", "```"] + st[k]["lines"] + ["```"]
        cmp_md = CLS / f"compare{SUF}.md"
        if cmp_md.exists():
            md += ["", cmp_md.read_text()]
        cls.M.write_text("\n".join(md))


def save_v7_sheet(rows, out_path, max_per_class=24, loader=None):
    """rows: list of (png path, pred idx, conf). Sheet of non-Fresh predictions (highest confidence first)."""
    from PIL import ImageDraw, ImageFont
    try:
        F = ImageFont.load_default(size=16)
    except TypeError:
        F = ImageFont.load_default()
    sel = []
    for k in (0, 2, 3):
        sel += sorted([r for r in rows if r[1] == k], key=lambda r: -r[2])[:max_per_class]
    if not sel:
        return
    cell, cols = 130, 8
    sheet = Image.new("RGB", (cols * cell, -(-len(sel) // cols) * (cell + 20)), (30, 30, 30))
    d = ImageDraw.Draw(sheet)
    col = {0: (255, 80, 80), 2: (80, 150, 255), 3: (255, 220, 0)}
    for i, (p, k, c) in enumerate(sel):
        t = (loader(p) if loader else compose(load_rgba(p))).resize((cell - 10, cell - 10))
        x, y = (i % cols) * cell, (i // cols) * (cell + 20)
        sheet.paste(t, (x + 5, y + 20))
        d.text((x + 4, y + 2), f"{SHORT[k]} {c:.2f} {Path(p).stem[-9:]}", font=F, fill=col[k])
    sheet.save(out_path, quality=88)
