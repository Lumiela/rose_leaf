# Model C: DINOv2 ViT-S/14 (frozen, self-supervised features) + logistic regression on [CLS, mean patch] features.
# Train features use the rendered train set plus one horizontally flipped pass. C is chosen on val macro-F1.
# Usage: python v7_leafid/cls/scripts/train_dino.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from cls_common import RUNS, SHORT, Progress, metrics, folder_items, v7_items, save_v7_sheet, paths  # noqa: E402

import csv, json, pickle, time, traceback  # noqa: E401,E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from torchvision import transforms as T  # noqa: E402
from PIL import Image  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.metrics import f1_score  # noqa: E402

NAME = "dinov2_vits14"
out = RUNS / NAME; out.mkdir(parents=True, exist_ok=True)
TF = T.Compose([T.ToTensor(), T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])


@torch.no_grad()
def feats(model, ims, flip=False):
    F = []
    for i in range(0, len(ims), 128):
        b = []
        for im in ims[i:i + 128]:
            im = Image.open(im).convert("RGB") if isinstance(im, (str, Path)) else im
            b.append(TF(im.transpose(Image.FLIP_LEFT_RIGHT) if flip else im))
        x = torch.stack(b).cuda()
        with torch.autocast("cuda", dtype=torch.float16):
            o = model.forward_features(x)
        F.append(torch.cat([o["x_norm_clstoken"], o["x_norm_patchtokens"].mean(1)], 1).float().cpu().numpy())
    return np.concatenate(F)


try:
    Progress.log(NAME, "start", "running")
    model = torch.hub.load(str(paths.WEIGHTS / "dinov2_repo"), "dinov2_vits14", source="local", pretrained=False)
    model.load_state_dict(torch.load(paths.WEIGHTS / "dinov2_vits14_pretrain.pth", map_location="cpu"))
    model = model.cuda().eval()
    t0 = time.time()
    tr, va = folder_items("train"), folder_items("val")
    Xtr = np.concatenate([feats(model, [p for p, _ in tr]), feats(model, [p for p, _ in tr], flip=True)])
    ytr = np.array([c for _, c in tr] * 2)
    Progress.log(NAME, f"train features {Xtr.shape} in {time.time() - t0:.0f}s")
    Xva, yva = feats(model, [p for p, _ in va]), np.array([c for _, c in va])
    sc = StandardScaler().fit(Xtr); Xtr_s, Xva_s = sc.transform(Xtr), sc.transform(Xva)
    best = None
    for C in (0.01, 0.03, 0.1, 0.3, 1.0):
        clf = LogisticRegression(C=C, max_iter=3000, class_weight="balanced").fit(Xtr_s, ytr)
        f1 = f1_score(yva, clf.predict(Xva_s), average="macro")
        Progress.log(NAME, f"C={C} val_macroF1 {f1:.4f}")
        if best is None or f1 > best[0]:
            best = (f1, C, clf)
    clf = best[2]; pickle.dump(dict(scaler=sc, clf=clf, C=best[1]), open(out / "head.pkl", "wb"))
    Progress.log(NAME, f"best C={best[1]} val_macroF1 {best[0]:.4f}; evaluating")
    res = {"model": NAME, "C": best[1]}
    for split in ("test", "test_deg"):
        items = folder_items(split); t1 = time.time()
        P = clf.predict(sc.transform(feats(model, [p for p, _ in items])))
        res[split] = metrics([c for _, c in items], P); res[split]["ms_per_img"] = round((time.time() - t1) / len(items) * 1000, 2)
        Progress.log(NAME, f"{split}: acc {res[split]['acc']} macroF1 {res[split]['macro_f1']} recall {res[split]['recall']}")
    crops, loader = v7_items()
    pr = clf.predict_proba(sc.transform(feats(model, [loader(p) for p in crops])))
    P, Cf = pr.argmax(1).tolist(), pr.max(1).tolist()
    res["v7_crops"] = {SHORT[k]: P.count(k) for k in range(4)}
    with open(out / "v7_crop_preds.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["id", "pred", "conf"]); w.writerows([[p.stem, SHORT[k], round(c, 3)] for p, k, c in zip(crops, P, Cf)])
    save_v7_sheet(list(zip(crops, P, Cf)), out / "v7_nonfresh_sheet.jpg", loader=loader)
    (out / "metrics.json").write_text(json.dumps(res, indent=1))
    Progress.log(NAME, f"v7 real crops ({len(crops)}): {res['v7_crops']}", "done")
except Exception as e:
    Progress.log(NAME, "ERROR " + repr(e)[:300], "failed"); traceback.print_exc()
