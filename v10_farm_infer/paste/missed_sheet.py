import json
from PIL import Image, ImageDraw
B='v10_farm_infer'
rows=json.load(open(f'{B}/paste/results.json'))
R=[r for r in rows if r['cond']=='asis' and not r['found']][:16]
C=300; sheet=Image.new('RGB',(4*C,4*C))
for i,r in enumerate(R):
  det=json.load(open(f'{B}/runs_paste/asis/{r["photo"][:-4]}.json'))['leaves']
  im=Image.open(f'{B}/inputs_paste/asis/{r["photo"]}').convert('RGB'); d=ImageDraw.Draw(im)
  x0,y0,x1,y1=r['box']; cx,cy=(x0+x1)/2,(y0+y1)/2; s=2.2*max(x1-x0,y1-y0)
  for q in det:
    b=q['box']
    if b[2]<cx-s or b[0]>cx+s or b[3]<cy-s or b[1]>cy+s: continue
    d.line([tuple(p) for p in q['polygon']]+[tuple(q['polygon'][0])],fill=(255,0,255) if q['state']=='HOLD_SHAPE' else (0,255,255),width=4)
  d.rectangle(r['box'],outline=(255,255,0),width=3)
  cr=im.crop((int(cx-s/2),int(cy-s/2),int(cx+s/2),int(cy+s/2))).resize((C,C))
  ImageDraw.Draw(cr).text((3,3),f"{r['id']} {r['cls']} iou{r['iou']}",fill=(255,255,0))
  sheet.paste(cr,((i%4)*C,(i//4)*C))
sheet.save(f'{B}/paste/missed_asis.jpg',quality=85)
