# Print checkpoint metadata (epoch, best_fitness, saved metrics) for weights files. Read-only.
# Usage: python common/tools/ckpt_info.py <weights.pt> [...]
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import paths  # noqa: E402  shared paths + opencv workaround (must come before cv2/ultralytics)

import os, time
import torch
import ultralytics  # noqa: F401  (needed to unpickle the model)

for p in sys.argv[1:]:
    ck = torch.load(p, map_location="cpu", weights_only=False)
    tm = ck.get("train_metrics") or {}
    print("CKPT", p, "| size %.1fMB" % (os.path.getsize(p) / 1e6),
          "| mtime", time.strftime("%m-%d %H:%M", time.gmtime(os.path.getmtime(p))),
          "| epoch", ck.get("epoch"), "| best_fitness", ck.get("best_fitness"),
          "| mask mAP50 %s mAP50-95 %s" % (tm.get("metrics/mAP50(M)"), tm.get("metrics/mAP50-95(M)")),
          "| fitness", tm.get("fitness"), "| has_optimizer", ck.get("optimizer") is not None, flush=True)
