# v8 shared helpers: SAM2.1 runner, colour masks, instance-map storage, per-instance properties.
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
import paths  # noqa: F401,E402  (opencv workaround first)
import numpy as np  # noqa: E402
import cv2  # noqa: E402
from PIL import Image, ImageOps  # noqa: E402

V8 = paths.MODEL / "v8_reallabel"
WORK = V8 / "work"
REVIEW = V8 / "review"
_SAM = None


def sam():
    global _SAM
    if _SAM is None:
        from ultralytics import SAM
        _SAM = SAM(str(paths.WEIGHTS / "sam2.1_s.pt"))
    return _SAM


def load_rgb(p):
    return np.asarray(ImageOps.exif_transpose(Image.open(p)).convert("RGB"))


def gray_world(img):
    f = img.astype(np.float32)
    m = f.reshape(-1, 3).mean(0)
    f *= m.mean() / np.maximum(m, 1)
    return np.clip(f, 0, 255).astype(np.uint8)


def leaf_color(img_wb):
    """green/yellow leaf pixels (brown dry leaves are NOT included -> reviewer adds them)"""
    f = img_wb.astype(np.float32) + 1.0
    exg = (2 * f[..., 1] - f[..., 0] - f[..., 2]) / f.sum(2)
    hsv = cv2.cvtColor(img_wb, cv2.COLOR_RGB2HSV)
    return (exg > 0.06) & (hsv[..., 1] > 50) & (hsv[..., 2] > 35)


def window(cx, cy, T, W, H):
    T = int(min(T, W, H))
    x0 = int(np.clip(round(cx - T / 2), 0, W - T))
    y0 = int(np.clip(round(cy - T / 2), 0, H - T))
    return (x0, y0, x0 + T, y0 + T)


def run_sam(img, win, boxes=None, points=None, labels=None):
    """SAM on a window of the full RGB image. boxes [[x0,y0,x1,y1]...] or points [[[x,y],...] per object] + labels,
    all in full-image coords. Returns list of (bool mask in window coords, score), one per prompt."""
    x0, y0, x1, y1 = win
    crop = np.ascontiguousarray(img[y0:y1, x0:x1, ::-1])
    kw = {}
    if boxes is not None:
        kw["bboxes"] = [[b[0] - x0, b[1] - y0, b[2] - x0, b[3] - y0] for b in boxes]
    if points is not None:
        kw["points"] = [[[p[0] - x0, p[1] - y0] for p in grp] for grp in points]
        kw["labels"] = labels
    r = sam()(crop, verbose=False, **kw)[0]
    if r.masks is None:
        return []
    M = r.masks.data.cpu().numpy().astype(bool)
    h, w = crop.shape[:2]
    if M.shape[1:] != (h, w):
        M = np.stack([cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool) for m in M])
    sc = r.boxes.conf.cpu().numpy() if (r.boxes is not None and len(r.boxes) == len(M)) else np.ones(len(M))
    return list(zip(M, sc))


def paste(full_shape, win, m):
    out = np.zeros(full_shape, bool)
    x0, y0, x1, y1 = win
    out[y0:y1, x0:x1] = m
    return out


def save(stem, inst, meta, tag=None):
    d = WORK / stem
    d.mkdir(parents=True, exist_ok=True)
    if tag is not None and (d / "inst.npz").exists():
        (d / "history").mkdir(exist_ok=True)
        (d / "inst.npz").replace(d / "history" / f"inst_{tag}.npz")
        if (d / "meta.json").exists():
            (d / "meta.json").replace(d / "history" / f"meta_{tag}.json")
    np.savez_compressed(d / "inst.npz", inst=inst)
    (d / "meta.json").write_text(json.dumps(meta, indent=1))


def load(stem):
    d = WORK / stem
    return np.load(d / "inst.npz")["inst"], json.loads((d / "meta.json").read_text())


def props(inst, leaf, L):
    H, W = inst.shape
    out = {}
    ids = [int(i) for i in np.unique(inst) if i]
    for i in ids:
        ys, xs = np.nonzero(inst == i)
        x0, y0, x1, y1 = xs.min(), ys.min(), xs.max() + 1, ys.max() + 1
        m = (inst[y0:y1, x0:x1] == i).astype(np.uint8)
        n, lab, st, _ = cv2.connectedComponentsWithStats(m, 8)
        big = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA])) if n > 1 else 0
        cs, _ = cv2.findContours((lab == big).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        hull = cv2.contourArea(cv2.convexHull(cs[0])) if cs else 0
        area = int(m.sum())
        out[i] = dict(id=i, x0=int(x0), y0=int(y0), x1=int(x1), y1=int(y1), area=area,
                      cx=float(xs.mean()), cy=float(ys.mean()),
                      solidity=round(float(st[big, cv2.CC_STAT_AREA] / hull), 3) if hull > 0 else 0.0,
                      parts=int(n - 1), main_frac=round(float(st[big, cv2.CC_STAT_AREA] / area), 3),
                      leaf_frac=round(float(leaf[ys, xs].mean()), 3),
                      border=bool(x0 <= 1 or y0 <= 1 or x1 >= W - 1 or y1 >= H - 1))
    if out:
        med = float(np.median([p["area"] for p in out.values()]))
        for p in out.values():
            f = []
            if p["solidity"] < 0.85: f.append("solidity")
            if p["area"] > 3.0 * med: f.append("big")
            if p["area"] < 0.15 * med: f.append("small")
            if p["leaf_frac"] < 0.5: f.append("color")
            if p["main_frac"] < 0.9: f.append("parts")
            p["flags"] = f
    return out


def peaks(mask, L, min_sep=0.35):
    """leaflet centres = local maxima of the distance transform, at least min_sep*L apart"""
    dt = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    r = max(2, int(min_sep * L))
    mx = cv2.dilate(dt, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1)))
    ys, xs = np.nonzero((dt >= mx - 1e-6) & (dt > 0.12 * L))
    order = np.argsort(-dt[ys, xs])
    pts = []
    for k in order:
        p = (int(xs[k]), int(ys[k]))
        if all((p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2 >= r * r for q in pts):
            pts.append(p)
    return pts


def split_by_points(img, region, pts, T, L, clip=True):
    """one SAM object per point (other points as negatives). Returns list of full-size bool masks (clipped to region)."""
    H, W = region.shape
    ys, xs = np.nonzero(region) if region.any() else (np.array([p[1] for p in pts]), np.array([p[0] for p in pts]))
    cx, cy = (xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2
    span = max(xs.max() - xs.min(), ys.max() - ys.min())
    win = window(cx, cy, max(T, int(1.6 * span)), W, H)
    prom = [[p] + [q for q in pts if q != p] for p in pts]
    lab = [[1] + [0] * (len(pts) - 1) for _ in pts]
    out = []
    reg = cv2.dilate(region.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool) if clip else None
    for m, s in run_sam(img, win, points=prom, labels=lab):
        full = paste((H, W), win, m)
        if clip:
            full &= reg
        out.append(full)
    return out


def is_thin(mask, L):
    cs, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cs:
        return False
    c = max(cs, key=cv2.contourArea)
    (_, _), (w, h), _ = cv2.minAreaRect(c)
    return min(w, h) < 0.18 * L or max(w, h) > 5 * max(1.0, min(w, h))


def split_region(img, region, pts, T, L):
    """split region into len(pts) parts with no pixel lost: SAM masks (one per point, others negative) give
    markers, watershed on the image fills the rest of the region. Returns list of bool masks (full size)."""
    H, W = region.shape
    ms = split_by_points(img, region, pts, T, L)
    ys, xs = np.nonzero(region)
    x0, y0, x1, y1 = max(0, xs.min() - 3), max(0, ys.min() - 3), min(W, xs.max() + 4), min(H, ys.max() + 4)
    reg = region[y0:y1, x0:x1]
    mk = np.zeros(reg.shape, np.int32)
    mk[~cv2.dilate(reg.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)] = 1  # background marker
    er = np.ones((5, 5), np.uint8)
    for k, p in enumerate(pts):
        m = ms[k][y0:y1, x0:x1] & reg if k < len(ms) else np.zeros_like(reg)
        m = cv2.erode(m.astype(np.uint8), er).astype(bool)
        # keep only the component of the SAM mask that contains / is nearest to the click
        if m.any():
            n, lab = cv2.connectedComponents(m.astype(np.uint8))
            px, py = int(p[0] - x0), int(p[1] - y0)
            if 0 <= py < lab.shape[0] and 0 <= px < lab.shape[1] and lab[py, px] > 0:
                m = lab == lab[py, px]
            else:
                yy, xx = np.nonzero(m); j = int(np.argmin((xx - px) ** 2 + (yy - py) ** 2)); m = lab == lab[yy[j], xx[j]]
        mk[m & (mk == 0)] = k + 2
        cv2.circle(mk, (int(p[0] - x0), int(p[1] - y0)), max(2, int(0.08 * L)), k + 2, -1)
    mk[~reg & (mk > 1)] = 1
    ws = cv2.watershed(np.ascontiguousarray(img[y0:y1, x0:x1, ::-1]), mk.copy())
    out = []
    for k in range(len(pts)):
        full = np.zeros((H, W), bool)
        full[y0:y1, x0:x1] = (ws == k + 2) & reg
        out.append(full)
    # region pixels left unassigned (watershed boundaries, or flooded from the background marker) -> nearest part
    lab = np.zeros(reg.shape, np.int32)
    for k, m in enumerate(out):
        lab[m[y0:y1, x0:x1]] = k + 1
    for _ in range(200):
        left = reg & (lab == 0)
        if not left.any() or not lab.any():
            break
        dil = cv2.dilate(lab.astype(np.float32), np.ones((3, 3), np.uint8)).astype(np.int32)
        lab[left & (dil > 0)] = dil[left & (dil > 0)]
    for k in range(len(out)):
        out[k][y0:y1, x0:x1] = (lab == k + 1) & reg
    return out


def auto_masks(img, win):
    """SAM 'segment everything' on a window -> list of full-size bool masks"""
    x0, y0, x1, y1 = win
    crop = np.ascontiguousarray(img[y0:y1, x0:x1, ::-1])
    r = sam()(crop, verbose=False)[0]
    if r.masks is None:
        return []
    M = r.masks.data.cpu().numpy().astype(bool)
    h, w = crop.shape[:2]
    out = []
    for m in M:
        if m.shape != (h, w):
            m = cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
        out.append(paste(img.shape[:2], win, m))
    return out


def solidity_of(m):
    cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cs:
        return 0.0
    c = max(cs, key=cv2.contourArea)
    hull = cv2.contourArea(cv2.convexHull(c))
    return float(cv2.contourArea(c) / hull) if hull > 0 else 0.0


def add_auto(img, inst, leaf, L, T, box=None, min_new=0.6):
    """fill uncovered leaves with 'segment everything' proposals. Returns list of new masks (already filtered)."""
    H, W = inst.shape
    bx0, by0, bx1, by1 = box if box else (0, 0, W, H)
    st = max(1, int(0.7 * T))
    wins = sorted({window(cx, cy, T, W, H) for cy in range(by0 + T // 2, by1 + st, st) for cx in range(bx0 + T // 2, bx1 + st, st)})
    props_ = []
    for w in wins:
        for m in auto_masks(img, w):
            ar = int(m.sum())
            if ar < 0.1 * L * L or ar > 4 * L * L:
                continue
            props_.append(m)
    props_.sort(key=lambda m: m.sum())  # small first: single leaflets before merged clusters
    taken = inst > 0
    new = []
    for m in props_:
        free = m & ~taken
        if free.sum() < min_new * m.sum() or leaf[free].mean() < 0.6 or solidity_of(free) < 0.8:
            continue
        new.append(free); taken |= free
    return new


def add_grid(img, inst, leaf, L, T, box=None, spacing=0.5, min_new=0.5):
    """dense point prompts on uncovered leaf pixels (grid spacing*L); keep masks that are mostly new, leaf-coloured,
    compact and leaf-sized. Returns list of new bool masks."""
    H, W = inst.shape
    bx0, by0, bx1, by1 = box if box else (0, 0, W, H)
    unc = leaf & (inst == 0)
    unc = cv2.erode(unc.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    g = max(4, int(spacing * L))
    hb = 0.65 * L  # half box: a leaf-sized box around each grid point keeps SAM from returning the whole plant
    pts = [(x, y) for y in range(by0 + g // 2, by1, g) for x in range(bx0 + g // 2, bx1, g) if unc[y, x]]
    st = max(1, int(0.7 * T))
    wins = sorted({window(cx, cy, T, W, H) for cy in range(by0 + T // 2, by1 + st, st) for cx in range(bx0 + T // 2, bx1 + st, st)})
    groups = {}
    for p in pts:
        best = min(wins, key=lambda w: ((w[0] + w[2]) / 2 - p[0]) ** 2 + ((w[1] + w[3]) / 2 - p[1]) ** 2)
        groups.setdefault(best, []).append(p)
    cand = []
    for w, ps in groups.items():
        for k in range(0, len(ps), 64):
            chunk = ps[k:k + 64]
            bxs = [[max(w[0], q[0] - hb), max(w[1], q[1] - hb), min(w[2], q[0] + hb), min(w[3], q[1] + hb)] for q in chunk]
            for (m, s), p in zip(run_sam(img, w, boxes=bxs), chunk):
                full = paste((H, W), w, m)
                ar = int(full.sum())
                if 0.1 * L * L <= ar <= 3 * L * L:
                    cand.append((float(s), full))
    cand.sort(key=lambda t: (-t[0], t[1].sum()))
    taken = inst > 0
    new = []
    for s, m in cand:
        free = m & ~taken
        fa = free.sum()
        if fa < min_new * m.sum() or fa < 0.1 * L * L or leaf[free].mean() < 0.55 or solidity_of(free) < 0.75:
            continue
        new.append(free); taken |= free
    return new
