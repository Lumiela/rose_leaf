# v8 stage 2: EfficientNetV2-S fine-tuned on v8_reallabel/cls/data (same recipe as v7 train_effnet.py; only data/output paths differ).
# Saves best.pt (val macro-F1) and per-split probabilities for calibration (probs_<split>.csv: path, label, p_Black, p_Fresh, p_Hole, p_Yellow).
# Usage: python v8_reallabel/cls/scripts/train_cls_v8.py [--epochs 20] [--name effnetv2s]
import sys, argparse, csv, json, time  # noqa: E401
from pathlib import Path
MODEL = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(MODEL / "v7_leafid" / "cls" / "scripts"))
from cls_common import NAMES, SHORT, metrics, paths  # noqa: E402
import torch, torch.nn as nn  # noqa: E401,E402
from torchvision import transforms as T  # noqa: E402
from torchvision.models import efficientnet_v2_s  # noqa: E402
from PIL import Image  # noqa: E402

ap = argparse.ArgumentParser(); ap.add_argument("--epochs", type=int, default=20); ap.add_argument("--name", default="effnetv2s")
a = ap.parse_args()
DATA = MODEL / "v8_reallabel" / "cls" / "data"; out = MODEL / "v8_reallabel" / "cls" / "runs" / a.name; out.mkdir(parents=True, exist_ok=True)
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
TF_TRAIN = T.Compose([T.RandomHorizontalFlip(), T.RandomVerticalFlip(), T.ColorJitter(0.1, 0.1, 0.1, 0.01), T.ToTensor(), T.Normalize(MEAN, STD)])
TF_EVAL = T.Compose([T.ToTensor(), T.Normalize(MEAN, STD)])


def items(split):
    return [(p, ci) for ci, c in enumerate(NAMES) for p in sorted((DATA / split / c).glob("*.jpg"))]


class DS(torch.utils.data.Dataset):
    def __init__(self, it, tf): self.it, self.tf = it, tf
    def __len__(self): return len(self.it)
    def __getitem__(self, i):
        p, c = self.it[i]; return self.tf(Image.open(p).convert("RGB")), c


def probs(model, it):
    dl = torch.utils.data.DataLoader(DS(it, TF_EVAL), batch_size=128, shuffle=False, num_workers=4)
    model.eval(); P = []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        for x, _ in dl:
            P.append(model(x.cuda()).float().softmax(1).cpu())
    return torch.cat(P).numpy() if P else None


torch.manual_seed(42)
model = efficientnet_v2_s()
model.load_state_dict(torch.load(paths.WEIGHTS / "torch_hub/checkpoints/efficientnet_v2_s-dd5fe13b.pth", map_location="cpu"))
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 4); model = model.cuda()
tr = torch.utils.data.DataLoader(DS(items("train"), TF_TRAIN), batch_size=64, shuffle=True, num_workers=4, pin_memory=True)
va_items = items("val")
opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.05)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-4, total_steps=a.epochs * len(tr), pct_start=0.1)
scaler, crit, best = torch.cuda.amp.GradScaler(), nn.CrossEntropyLoss(label_smoothing=0.1), -1
print("train", len(tr.dataset), "val", len(va_items), flush=True)
t0 = time.time()
for ep in range(a.epochs):
    model.train(); tl = n = 0
    for x, y in tr:
        x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
        with torch.autocast("cuda", dtype=torch.float16):
            loss = crit(model(x), y)
        opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
        tl += loss.item() * len(y); n += len(y)
    P = probs(model, va_items); m = metrics([c for _, c in va_items], P.argmax(1))
    if m["macro_f1"] > best:
        best = m["macro_f1"]; torch.save(model.state_dict(), out / "best.pt")
    print(f"epoch {ep + 1}/{a.epochs} loss {tl / n:.4f} val acc {m['acc']} macroF1 {m['macro_f1']} recall {m['recall']} (best {best}) {time.time() - t0:.0f}s", flush=True)
model.load_state_dict(torch.load(out / "best.pt")); res = {}
for split in ("val", "test", "test_deg", "test_real", "test_real_anom"):
    it = items(split)
    if not it:
        continue
    P = probs(model, it); res[split] = metrics([c for _, c in it], P.argmax(1))
    with open(out / f"probs_{split}.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["path", "label"] + [f"p_{s}" for s in SHORT])
        w.writerows([[str(p), SHORT[c]] + [round(float(v), 4) for v in pr] for (p, c), pr in zip(it, P)])
    print(split, res[split]["acc"], res[split]["recall"], flush=True)
(out / "metrics.json").write_text(json.dumps(res, indent=1)); print("TRAIN_DONE")
