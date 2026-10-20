# v8 step 3: apply review edits to an instance map (each run is saved to work/<stem>/history for undo).
# edits JSON = list of ops (coords in full-image px):
#   {"op":"del","ids":[..]}                       remove instances
#   {"op":"split","id":k,"pts":[[x,y],..]}        one leaf per point inside instance k (SAM, other points negative)
#   {"op":"autosplit","ids":[..]}                 split by distance-transform peaks
#   {"op":"add","pts":[[x,y],..],"force":false}   new leaf per point (positive click); force=overwrite existing pixels
#   {"op":"addpts","pos":[[x,y]..],"neg":[[x,y]..]}  one new leaf from several clicks
#   {"op":"addbox","boxes":[[x0,y0,x1,y1],..]}
#   {"op":"fix","id":k,"pos":[..],"neg":[..]}     re-segment instance k from clicks (replaces it)
#   {"op":"merge","ids":[a,b,..]}                 merge into a
#   {"op":"trim","id":k,"box":[x0,y0,x1,y1]}      erase k's pixels inside box
#   {"op":"ignore","boxes":[[x0,y0,x1,y1],..]}    region excluded from training (grey at export)
#   {"op":"unignore","idx":[..]}
# Usage: python v8_reallabel/scripts/apply_edits.py <stem> <edits.json | '<json>'>
import sys, json, time  # noqa: E401
from pathlib import Path
from sam_common import load, save, load_rgb, run_sam, window, paste, peaks, split_region, props, gray_world, leaf_color, WORK, add_auto, add_grid
import numpy as np

stem, src = sys.argv[1], sys.argv[2]
if src == "undo":
    hd = WORK / stem / "history"
    last = sorted(hd.glob("inst_*.npz"))[-1]; tag = last.stem[5:]
    (WORK / stem / "inst.npz").unlink(); last.replace(WORK / stem / "inst.npz")
    if (hd / f"meta_{tag}.json").exists():
        (hd / f"meta_{tag}.json").replace(WORK / stem / "meta.json")
    print("UNDO ->", tag); sys.exit(0)
ops = json.loads(Path(src).read_text()) if src.endswith(".json") else json.loads(src)
inst, meta = load(stem)
img = load_rgb(meta["photo"]); H, W = inst.shape; L, T = meta["L"], meta["T"]
nid = int(inst.max())
rep = []


def new_id(src_tag):
    global nid
    nid += 1
    meta["ids"][str(nid)] = dict(src=src_tag)
    return nid


def put(mask, tag, force=False):
    m = mask if force else (mask & (inst == 0))
    if m.sum() < 0.04 * L * L:
        return None
    i = new_id(tag); inst[m] = i
    return i


for op in ops:
    o = op["op"]
    if o == "del":
        for i in op["ids"]:
            inst[inst == i] = 0; meta["ids"].pop(str(i), None)
        rep.append(f"del {op['ids']}")
    elif o in ("split", "autosplit"):
        for k in ([op["id"]] if o == "split" else op["ids"]):
            region = inst == k
            pts = [tuple(p) for p in op["pts"]] if o == "split" else peaks(region, L)
            if len(pts) < 2:
                rep.append(f"{o} {k}: <2 points, skipped"); continue
            ms = split_region(img, region, pts, T, L)
            inst[region] = 0; meta["ids"].pop(str(k), None)
            got = []
            for m in ms:
                if m.sum() == 0:
                    continue
                i = new_id(f"split{k}"); inst[m] = i; got.append(i)
            rep.append(f"{o} {k} -> {got}")
    elif o == "add":
        for p in op["pts"]:
            win = window(p[0], p[1], T, W, H)
            res = run_sam(img, win, points=[[p]], labels=[[1]])
            i = put(paste((H, W), win, res[0][0]), "add", op.get("force", False)) if res else None
            rep.append(f"add {p} -> {i}")
    elif o == "addpts":
        pos, neg = op["pos"], op.get("neg", [])
        cx, cy = np.mean([q[0] for q in pos]), np.mean([q[1] for q in pos])
        win = window(cx, cy, T, W, H)
        res = run_sam(img, win, points=[pos + neg], labels=[[1] * len(pos) + [0] * len(neg)])
        i = put(paste((H, W), win, res[0][0]), "addpts", op.get("force", False)) if res else None
        rep.append(f"addpts {pos} -> {i}")
    elif o == "addbox":
        for b in op["boxes"]:
            win = window((b[0] + b[2]) / 2, (b[1] + b[3]) / 2, max(T, int(1.6 * max(b[2] - b[0], b[3] - b[1]))), W, H)
            res = run_sam(img, win, boxes=[b])
            i = put(paste((H, W), win, res[0][0]), "addbox", op.get("force", False)) if res else None
            rep.append(f"addbox {b} -> {i}")
    elif o == "fix":
        k = op["id"]; pos, neg = op["pos"], op.get("neg", [])
        cx, cy = np.mean([q[0] for q in pos]), np.mean([q[1] for q in pos])
        win = window(cx, cy, T, W, H)
        res = run_sam(img, win, points=[pos + neg], labels=[[1] * len(pos) + [0] * len(neg)])
        if res:
            inst[inst == k] = 0
            m = paste((H, W), win, res[0][0])
            m = m if op.get("force", True) else (m & (inst == 0))
            inst[m] = k
        rep.append(f"fix {k}")
    elif o == "merge":
        a = op["ids"][0]
        for b in op["ids"][1:]:
            inst[inst == b] = a; meta["ids"].pop(str(b), None)
        rep.append(f"merge {op['ids']}")
    elif o == "trim":
        x0, y0, x1, y1 = op["box"]; sub = inst[y0:y1, x0:x1]; sub[sub == op["id"]] = 0
        rep.append(f"trim {op['id']}")
    elif o == "auto":
        leafm = leaf_color(gray_world(img))
        got = []
        for m in add_auto(img, inst, leafm, L, T, box=op.get("box"), min_new=op.get("min_new", 0.6)):
            i = put(m, "auto")
            if i: got.append(i)
        rep.append(f"auto {op.get('box')} -> {len(got)} new")
    elif o == "grid":
        leafm = leaf_color(gray_world(img))
        got = []
        for m in add_grid(img, inst, leafm, L, T, box=op.get("box"), spacing=op.get("spacing", 0.5)):
            i = put(m, "grid")
            if i: got.append(i)
        rep.append(f"grid {op.get('box')} -> {len(got)} new")
    elif o == "mark":
        meta["reviewed"] = bool(op.get("reviewed", True)); rep.append(f"reviewed={meta['reviewed']}")
    elif o == "ignore":
        meta["ignore"] += op["boxes"]; rep.append(f"ignore +{len(op['boxes'])}")
    elif o == "unignore":
        meta["ignore"] = [b for j, b in enumerate(meta["ignore"]) if j not in op["idx"]]; rep.append("unignore")
meta["log"].append(dict(t=time.strftime("%Y-%m-%d %H:%M:%S"), ops=ops))
save(stem, inst, meta, tag=time.strftime("%H%M%S"))
P = props(inst, leaf_color(gray_world(img)), L)
print("\n".join(rep)); print(f"SAVED {stem}: instances={len(P)} flagged={sum(1 for p in P.values() if p['flags'])}")
