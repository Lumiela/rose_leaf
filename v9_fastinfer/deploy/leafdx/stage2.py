"""Stage 2: per-leaf state (v9: same output as v8; gray_world is faster, see its docstring).

Classifier input (same as training, v7 cls_common): gray-world white balance of the whole photo, square crop of
side box_scale x leaf long side centred on the leaf box (outside the photo filled with gray 128), resized to 224.
Model: torchvision EfficientNetV2-S, 4 outputs in the order CLASSES.
States (thresholds from states.json):
  HOLD_SHAPE  solidity < min_solidity or the leaf touches the photo border (several leaves merged / cut)
  HOLD        quality gate: sharpness < s_min or dark share > d_max or leaf long side < L_min
  ANOMALY     max(p_Black, p_Hole, p_Yellow) >= t_anom  (class = the largest of the three)
  FRESH       p_Fresh >= t_fresh
  UNKNOWN     otherwise (abnormal-looking but unsure -> watch, not an alarm)
"""
from . import _compat  # noqa: F401
import json
import numpy as np
import cv2
import torch
import torch.nn as nn
from torchvision import transforms as T
from torchvision.models import efficientnet_v2_s
from PIL import Image

CLASSES = ["Black Spot", "Fresh", "Hole", "Yellow"]
ANOM = {0: "Black Spot", 2: "Hole", 3: "Yellow"}
GRAY = (128, 128, 128)


def apply_gain(img, k):
    """np.clip(img.astype(np.float32) * k, 0, 255).astype(np.uint8) for per-channel gains k, via a 256-entry lookup
    table (same values; the product dtype follows k as in the direct expression)."""
    lut = np.clip(np.arange(256, dtype=np.float32)[:, None] * np.asarray(k)[None, :], 0, 255).astype(np.uint8)
    return cv2.LUT(np.ascontiguousarray(img), lut.reshape(1, 256, 3))


def gray_world(img, device=None):
    """White balance from low-saturation mid-brightness pixels (output identical to v7 cls_common.gray_world).

    v9: the reference-pixel test runs on device as the exact integer form of s < 60/255 (s = (max - min) / max),
    and the gain is applied with a lookup table. v8 did both in float over the whole photo (~2 s for 12MP).
    """
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    t = torch.from_numpy(img if img.flags.writeable else img.copy()).to(dev)
    mx, mn = t.amax(-1).int(), t.amin(-1).int()
    ref = (255 * (mx - mn) < 60 * mx) & (mx > 60) & (mx < 240)
    n = int(ref.sum())
    if n < 1000:
        sums, n = t.reshape(-1, 3).long().sum(0), mx.numel()
    else:
        sums = t[ref].long().sum(0)
    mean = sums.cpu().numpy().astype(np.float64) / n
    return apply_gain(img, mean.mean() / mean)


def square_crop(img, cx, cy, side, fill=GRAY):
    H, W = img.shape[:2]
    side = max(4, int(round(side)))
    x0, y0 = int(round(cx - side / 2)), int(round(cy - side / 2))
    out = np.empty((side, side, 3), np.uint8); out[:] = fill
    sx0, sy0, sx1, sy1 = max(0, x0), max(0, y0), min(W, x0 + side), min(H, y0 + side)
    if sx1 > sx0 and sy1 > sy0:
        out[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = img[sy0:sy1, sx0:sx1]
    return Image.fromarray(out)


def to_input(pil, size=224):
    return pil.resize((size, size), Image.LANCZOS if pil.width > size else Image.BILINEAR)


class LeafClassifier:
    def __init__(self, weights, states_json, box_scale=1.25, min_solidity=0.8, device=None, batch=128):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        m = efficientnet_v2_s(); m.classifier[1] = nn.Linear(m.classifier[1].in_features, 4)
        m.load_state_dict(torch.load(str(weights), map_location="cpu")); self.model = m.to(self.device).eval()
        self.th = json.loads(open(states_json).read())["thresholds"]
        self.box_scale, self.min_solidity, self.batch = box_scale, min_solidity, batch
        self.tf = T.Compose([T.ToTensor(), T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

    def probs(self, wb, leaves):
        crops = [to_input(square_crop(wb, (l["box"][0] + l["box"][2]) / 2, (l["box"][1] + l["box"][3]) / 2,
                                      self.box_scale * max(l["box"][2] - l["box"][0], l["box"][3] - l["box"][1]))) for l in leaves]
        out = []
        torch.backends.cudnn.benchmark = False  # ultralytics may enable it; avoid re-tuning on a new batch shape
        mean = torch.tensor((0.485, 0.456, 0.406), device=self.device).view(1, 3, 1, 1)
        std = torch.tensor((0.229, 0.224, 0.225), device=self.device).view(1, 3, 1, 1)
        with torch.no_grad():
            for i in range(0, len(crops), self.batch):
                # batched ToTensor + Normalize on the device (per-image torchvision transforms were ~18 ms/leaf on CPU)
                arr = np.stack([np.asarray(c, dtype=np.uint8) for c in crops[i:i + self.batch]])
                n = len(arr)
                if n < self.batch:  # fixed batch size -> one input shape for every call
                    arr = np.concatenate([arr, np.zeros((self.batch - n,) + arr.shape[1:], np.uint8)])
                x = torch.from_numpy(arr).to(self.device).permute(0, 3, 1, 2).float().div_(255).sub_(mean).div_(std)
                out.append(self.model(x).float().softmax(1)[:n].cpu())
        return (torch.cat(out).numpy() if out else np.zeros((0, 4), np.float32)), crops

    def state(self, leaf, p):
        th = self.th
        if leaf["solidity"] < self.min_solidity or leaf["touches_border"]:
            return "HOLD_SHAPE", None
        if leaf["sharpness"] < th["s_min"] or leaf["dark"] > th["d_max"] or leaf["L"] < th["L_min"]:
            return "HOLD", None
        k = max(ANOM, key=lambda j: p[j])
        if p[k] >= th["t_anom"]:
            return "ANOMALY", ANOM[k]
        if p[1] >= th["t_fresh"]:
            return "FRESH", None
        return "UNKNOWN", None
