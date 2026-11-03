"""같은 장면을 찍은 new_cam 사진끼리 정합한 뒤 잎을 박스 IoU로 짝지어, 검출 재현과 판정 일치를 잽니다.
python v10_farm_infer/esp_cam_20261103/stability.py   (model_git에서)"""
import json, glob, sys, itertools, collections
sys.path.insert(0, 'v9_fastinfer/deploy')
import leafdx
import cv2, numpy as np
B = 'v10_farm_infer/esp_cam_20261103'
fs = sorted(glob.glob(f'{B}/inputs/*'))
R = {f: json.load(open(f'{B}/runs/new_cam/' + f.split('/')[-1].rsplit('.', 1)[0] + '.json')) for f in fs}
G = {f: cv2.imread(f, 0) for f in fs}
orb = cv2.ORB_create(6000); KD = {f: orb.detectAndCompute(G[f], None) for f in fs}
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
def iou(a, b):
    x0, y0, x1, y1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    i = max(0, x1 - x0) * max(0, y1 - y0); u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i
    return i / u if u else 0
out = []; conf = collections.Counter(); good = collections.Counter(); MIN_INLIERS = 400
for a, b in itertools.combinations(fs, 2):
    ka, da = KD[a]; kb, db = KD[b]; m = bf.match(da, db)
    H, mask = cv2.findHomography(np.float32([ka[x.queryIdx].pt for x in m]), np.float32([kb[x.trainIdx].pt for x in m]), cv2.RANSAC, 6.0)
    la, lb = R[a]['leaves'], R[b]['leaves']; used = set(); n = same = 0
    for l in la:
        x0, y0, x1, y1 = l['box']; p = cv2.perspectiveTransform(np.float32([[[x0, y0], [x1, y1]]]), H)[0].ravel().tolist()
        best = max(((iou(p, o['box']), i) for i, o in enumerate(lb) if i not in used), default=(0, -1))
        if best[0] >= 0.5:
            used.add(best[1]); n += 1; o = lb[best[1]]; same += l['state'] == o['state']; k = tuple(sorted((l['state'], o['state']))); conf[k] += 1; good[k] += int(mask.sum()) >= MIN_INLIERS
    out.append(dict(a=a.split('/')[-1][:6], b=b.split('/')[-1][:6], inliers=int(mask.sum()), la=len(la), lb=len(lb), matched=n, repeat=round(2 * n / (len(la) + len(lb)), 3), same_state=round(same / max(n, 1), 3)))
res = dict(pairs=out, min_inliers=MIN_INLIERS, state_pairs={' / '.join(k): v for k, v in sorted(conf.items(), key=lambda t: -t[1])}, state_pairs_aligned={' / '.join(k): v for k, v in sorted(good.items(), key=lambda t: -t[1])})
json.dump(res, open(f'{B}/review/stability.json', 'w'), indent=1, ensure_ascii=False)
print(json.dumps(res, ensure_ascii=False))
