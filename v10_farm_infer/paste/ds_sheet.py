import json,numpy as np
from PIL import Image
from scipy import ndimage as ndi
s=json.load(open('v7_leafid/cls/split.json'))
fs=[f for f,c in s['test'] if c==2]
C=200; cols=8; rows=(len(fs)+cols-1)//cols
sheet=Image.new('RGB',(cols*C,rows*C),(255,0,255))
for i,f in enumerate(fs):
  a=Image.open('common/Dataset/'+f).convert('RGBA'); a.thumbnail((C-6,C-6))
  bg=Image.new('RGBA',(C,C),(255,0,255,255)); bg.alpha_composite(a,((C-a.width)//2,(C-a.height)//2))
  sheet.paste(bg.convert('RGB'),((i%cols)*C,(i//cols)*C))
sheet.save('v10_farm_infer/paste/ds_hole_test_sheet.jpg',quality=88); print(sheet.size)
