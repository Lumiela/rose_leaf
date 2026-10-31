import json,numpy as np
from PIL import Image
from scipy import ndimage as ndi
s=json.load(open('v7_leafid/cls/split.json'))
for cls in (2,1,0):
  fs=[f for f,c in s['test'] if c==cls]
  sz=[];hf=[]
  for f in fs:
    a=np.array(Image.open('common/Dataset/'+f).convert('RGBA'))
    m=a[...,3]>127
    fill=ndi.binary_fill_holes(m)
    hole=fill&~m
    sz.append(max(m.shape)); hf.append(hole.sum()/max(1,fill.sum()))
  hf=np.array(hf)
  print(cls,len(fs),'size med',int(np.median(sz)),'min',min(sz),'max',max(sz),'transp-hole>1%:',(hf>0.01).mean().round(2),'med',np.median(hf).round(3),'aug',sum('_aug' in f for f in fs))
