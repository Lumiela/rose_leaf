"""Stage-1 candidates for the two-stage pipeline: color-based leaf mask and plant ROIs.

hsv_leaf_mask : raw green/yellow-green pixel mask (same thresholds as v2/scripts/make_backgrounds.py)
refine_mask   : closing + hole filling + dilation, sizes given in pixels
plant_rois    : merge leaf pixels into plant-level boxes (with margin), overlapping boxes are unioned
"""
import numpy as np
import cv2

H_RANGE = (22, 95)   # OpenCV hue 0-180: yellow-green .. green
S_MIN, V_MIN = 45, 40


def _disk(px):
    px = max(1, int(round(px)))
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * px + 1, 2 * px + 1))


def hsv_leaf_mask(rgb):
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    m = ((h >= H_RANGE[0]) & (h <= H_RANGE[1]) & (s >= S_MIN) & (v >= V_MIN)).astype(np.uint8)
    return cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))


def fill_holes(m):
    """Fill background regions not connected to the image border."""
    pad = np.pad(m, 1).astype(np.uint8)
    flood = pad.copy()
    cv2.floodFill(flood, None, (0, 0), 1)
    return (pad | (1 - flood))[1:-1, 1:-1].astype(np.uint8)


def refine_mask(m, close_px=0, dilate_px=0, fill=True):
    if close_px:
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, _disk(close_px))
    if fill:
        m = fill_holes(m)
    if dilate_px:
        m = cv2.dilate(m, _disk(dilate_px))
    return m


def _union_overlapping(boxes):
    boxes = [list(b) for b in boxes]
    changed = True
    while changed:
        changed = False
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                a, b = boxes[i], boxes[j]
                if a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]:
                    boxes[i] = [min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])]
                    boxes.pop(j)
                    changed = True
                    break
            if changed:
                break
    return [tuple(b) for b in boxes]


def plant_rois(mask, merge_px, min_area_frac=0.01, margin=0.15):
    """Plant-level boxes (x0, y0, x1, y1) from a leaf mask."""
    H, W = mask.shape
    blobs = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, _disk(merge_px))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(blobs, 8)
    boxes = []
    for k in range(1, n):
        x, y, w, h, area = stats[k]
        if area < min_area_frac * H * W:
            continue
        mx, my = int(w * margin), int(h * margin)
        boxes.append((max(0, x - mx), max(0, y - my), min(W, x + w + mx), min(H, y + h + my)))
    return _union_overlapping(boxes)
