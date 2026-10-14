# Model A: YOLO11s-cls (Ultralytics, ImageNet-pretrained) fine-tuned on the rendered dataset.
# Usage: python v7_leafid/cls/scripts/train_yolo.py [--epochs 30]
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from cls_common import DATA, RUNS, SHORT, Progress, metrics, folder_items, v7_items, v7_hold, save_v7_sheet, paths  # noqa: E402

import argparse, csv, json, os, time, traceback  # noqa: E401,E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402
import torchvision  # noqa: E402

# Ultralytics 8.4 passes ImageFolder(allow_empty=...), which torchvision 0.18 on this server does not accept.
_imagefolder_init = torchvision.datasets.ImageFolder.__init__
def _imagefolder_init_compat(self, *args, allow_empty=None, **kwargs):  # noqa: E302
    return _imagefolder_init(self, *args, **kwargs)
torchvision.datasets.ImageFolder.__init__ = _imagefolder_init_compat  # noqa: E305

from ultralytics import YOLO  # noqa: E402

NAME = "yolo11s_cls"
ap = argparse.ArgumentParser(); ap.add_argument("--epochs", type=int, default=30); a = ap.parse_args()
out = RUNS / NAME
try:
    Progress.log(NAME, "start", "running")
    os.chdir(paths.COMMON)                       # AMP check resolves weights/yolo26n.pt relative to cwd
    model = YOLO(str(paths.WEIGHTS / "yolo11s-cls.pt"))
    def _on_epoch(tr):
        try:
            loss = float(tr.tloss.mean()) if hasattr(tr.tloss, "mean") else float(tr.tloss)
        except Exception:
            loss = float("nan")
        Progress.log(NAME, f"epoch {tr.epoch + 1}/{tr.epochs} train_loss {loss:.4f} val_top1 {(tr.metrics or {}).get('metrics/accuracy_top1')}")
    model.add_callback("on_fit_epoch_end", _on_epoch)
    t0 = time.time()
    model.train(data=str(DATA), imgsz=224, epochs=a.epochs, patience=10, batch=64, workers=4, seed=42,
                project=str(RUNS), name=NAME, exist_ok=True, hsv_h=0.005, fliplr=0.5, flipud=0.5, plots=True, verbose=False)
    Progress.log(NAME, f"training done in {(time.time() - t0) / 60:.1f} min, evaluating")
    best = YOLO(str(out / "weights" / "best.pt"))
    names = [best.names[i] for i in range(len(best.names))]
    assert [n.split()[0] for n in names] == SHORT, names

    def predict(imgs):
        preds, confs = [], []
        for i in range(0, len(imgs), 64):
            for r in best.predict(imgs[i:i + 64], imgsz=224, device=0, verbose=False):
                p = r.probs.data.cpu().numpy(); preds.append(int(p.argmax())); confs.append(float(p.max()))
        return preds, confs
    res = {"model": NAME}
    for split in ("test", "test_deg"):
        items = folder_items(split)
        t1 = time.time(); pr, _ = predict([str(p) for p, _ in items]); dt = (time.time() - t1) / len(items) * 1000
        res[split] = metrics([c for _, c in items], pr); res[split]["ms_per_img"] = round(dt, 2)
        Progress.log(NAME, f"{split}: acc {res[split]['acc']} macroF1 {res[split]['macro_f1']} recall {res[split]['recall']}")
    crops, loader = v7_items()
    pr, cf = predict([loader(p) for p in crops])
    res["v7_crops"] = {SHORT[k]: pr.count(k) for k in range(4)}
    with open(out / "v7_crop_preds.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["id", "pred", "conf"]); w.writerows([[p.stem, SHORT[k], round(c, 3)] for p, k, c in zip(crops, pr, cf)])
    save_v7_sheet(list(zip(crops, pr, cf)), out / "v7_nonfresh_sheet.jpg", loader=loader)
    (out / "metrics.json").write_text(json.dumps(res, indent=1))
    Progress.log(NAME, f"v7 real crops ({len(crops)}): {res['v7_crops']}", "done")
except Exception as e:
    Progress.log(NAME, "ERROR " + repr(e)[:300], "failed"); traceback.print_exc()
