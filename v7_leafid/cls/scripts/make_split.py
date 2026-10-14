# Stage-2 classifier data: split common/Dataset cutouts by original photo (stem), same rule and seed as v1/v2
# (no augmented copy of a train leaf in val/test; val/test use originals only). Writes split lists only.
# Usage: python v7_leafid/cls/scripts/make_split.py  -> v7_leafid/cls/split.json
import sys, re, json, random
from pathlib import Path
from collections import defaultdict
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "common"))
import paths  # noqa: E402

SEED, SPLIT = 42, (0.70, 0.15, 0.15)
stem_of = lambda n: re.split(r"_aug_", Path(n).stem)[0]
random.seed(SEED)
out = {"classes": paths.CLASS_NAMES, "train": [], "val": [], "test": []}
for ci, c in enumerate(paths.CLASS_NAMES):
    groups = defaultdict(list)
    for f in sorted((paths.DATASET / c).glob("*.png")):
        groups[stem_of(f.name)].append(f)
    stems = sorted(groups); random.shuffle(stems)
    n = len(stems); n_tr, n_va = int(n * SPLIT[0]), int(n * SPLIT[1])
    for s_, ss in (("train", stems[:n_tr]), ("val", stems[n_tr:n_tr + n_va]), ("test", stems[n_tr + n_va:])):
        for st in ss:
            files = groups[st] if s_ == "train" else ([f for f in groups[st] if "_aug_" not in f.name] or groups[st][:1])
            out[s_] += [[str(f.relative_to(paths.DATASET)), ci] for f in files]
(paths.V7 / "cls" / "split.json").write_text(json.dumps(out, indent=0, ensure_ascii=False))
for s_ in ("train", "val", "test"):
    cnt = [sum(1 for _, c in out[s_] if c == k) for k in range(4)]
    print("SPLIT", s_, len(out[s_]), dict(zip([n.split()[0] for n in paths.CLASS_NAMES], cnt)))
