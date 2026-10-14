# Model B: EfficientNetV2-S (torchvision, ImageNet-pretrained) fine-tuned on the rendered dataset.
# Usage: python v7_leafid/cls/scripts/train_effnet.py [--epochs 20]
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from cls_common import RUNS, SHORT, Progress, metrics, folder_items, v7_items, save_v7_sheet, paths  # noqa: E402

import argparse, csv, json, time, traceback  # noqa: E401,E402
import torch, torch.nn as nn  # noqa: E401,E402
from torchvision import transforms as T  # noqa: E402
from torchvision.models import efficientnet_v2_s  # noqa: E402
from PIL import Image  # noqa: E402

NAME = "effnetv2s"
ap = argparse.ArgumentParser(); ap.add_argument("--epochs", type=int, default=20); a = ap.parse_args()
out = RUNS / NAME; out.mkdir(parents=True, exist_ok=True)
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
TF_TRAIN = T.Compose([T.RandomHorizontalFlip(), T.RandomVerticalFlip(), T.ColorJitter(0.1, 0.1, 0.1, 0.01), T.ToTensor(), T.Normalize(MEAN, STD)])
TF_EVAL = T.Compose([T.ToTensor(), T.Normalize(MEAN, STD)])


class DS(torch.utils.data.Dataset):
    def __init__(self, items, tf, loader=None): self.items, self.tf, self.loader = items, tf, loader
    def __len__(self): return len(self.items)
    def __getitem__(self, i):
        p, c = self.items[i]
        im = self.loader(p) if self.loader else Image.open(p).convert("RGB")
        return self.tf(im), c


def run_eval(model, loader):
    model.eval(); P, C, Y = [], [], []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        for x, y in loader:
            pr = model(x.cuda(non_blocking=True)).float().softmax(1)
            P += pr.argmax(1).tolist(); C += pr.max(1).values.tolist(); Y += list(y)
    return P, C, Y


try:
    Progress.log(NAME, "start", "running")
    torch.manual_seed(42)
    model = efficientnet_v2_s()
    model.load_state_dict(torch.load(paths.WEIGHTS / "torch_hub/checkpoints/efficientnet_v2_s-dd5fe13b.pth", map_location="cpu"))
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 4)
    model = model.cuda()
    dl = lambda items, tf, sh, loader=None: torch.utils.data.DataLoader(DS(items, tf, loader), batch_size=64, shuffle=sh, num_workers=4, pin_memory=True)
    tr, va = dl(folder_items("train"), TF_TRAIN, True), dl(folder_items("val"), TF_EVAL, False)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-4, total_steps=a.epochs * len(tr), pct_start=0.1)
    scaler, crit, best_f1 = torch.cuda.amp.GradScaler(), nn.CrossEntropyLoss(label_smoothing=0.1), -1
    t0 = time.time()
    for ep in range(a.epochs):
        model.train(); tl, n = 0.0, 0
        for x, y in tr:
            x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
            with torch.autocast("cuda", dtype=torch.float16):
                loss = crit(model(x), y)
            opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
            tl += loss.item() * len(y); n += len(y)
        P, _, Y = run_eval(model, va); m = metrics(Y, P)
        if m["macro_f1"] > best_f1:
            best_f1 = m["macro_f1"]; torch.save(model.state_dict(), out / "best.pt")
        Progress.log(NAME, f"epoch {ep + 1}/{a.epochs} train_loss {tl / n:.4f} val_acc {m['acc']} val_macroF1 {m['macro_f1']} (best {best_f1})")
    Progress.log(NAME, f"training done in {(time.time() - t0) / 60:.1f} min, evaluating best")
    model.load_state_dict(torch.load(out / "best.pt")); res = {"model": NAME}
    for split in ("test", "test_deg"):
        items = folder_items(split); t1 = time.time()
        P, _, Y = run_eval(model, dl(items, TF_EVAL, False)); res[split] = metrics(Y, P)
        res[split]["ms_per_img"] = round((time.time() - t1) / len(items) * 1000, 2)
        Progress.log(NAME, f"{split}: acc {res[split]['acc']} macroF1 {res[split]['macro_f1']} recall {res[split]['recall']}")
    crops, loader = v7_items()
    P, Cf, _ = run_eval(model, dl([(p, 0) for p in crops], TF_EVAL, False, loader=loader))
    res["v7_crops"] = {SHORT[k]: P.count(k) for k in range(4)}
    with open(out / "v7_crop_preds.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["id", "pred", "conf"]); w.writerows([[p.stem, SHORT[k], round(c, 3)] for p, k, c in zip(crops, P, Cf)])
    save_v7_sheet(list(zip(crops, P, Cf)), out / "v7_nonfresh_sheet.jpg", loader=loader)
    (out / "metrics.json").write_text(json.dumps(res, indent=1))
    Progress.log(NAME, f"v7 real crops ({len(crops)}): {res['v7_crops']}", "done")
except Exception as e:
    Progress.log(NAME, "ERROR " + repr(e)[:300], "failed"); traceback.print_exc()
