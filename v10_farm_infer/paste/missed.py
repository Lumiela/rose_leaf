import json,numpy as np
rows=json.load(open('v10_farm_infer/paste/results.json'))
for c in ('asis','transp'):
  for cls in ('Hole','Fresh'):
    R=[r for r in rows if r['cond']==c and r['cls']==cls and not r['found']]
    print(c,cls,'missed',len(R),'best IoU',sorted(round(r['iou'],2) for r in R))
# transp-only misses
a={r['id']:r for r in rows if r['cond']=='asis'}; t={r['id']:r for r in rows if r['cond']=='transp'}
print('asis Hole -> transp change:',{f"{a[i]['state']} -> {t[i]['state']}":0 for i in a if a[i]['cls']=='Hole' and a[i]['state']!=t[i]['state']}.keys())
import collections
print(collections.Counter(f"{a[i]['state']} -> {t[i]['state']}" for i in a if a[i]['cls']=='Hole' and a[i]['state']!=t[i]['state']))
