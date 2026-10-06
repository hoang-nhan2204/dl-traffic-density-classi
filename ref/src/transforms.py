import random
from PIL import Image,ImageOps,ImageEnhance,ImageFilter
from torchvision.transforms import functional as F
MEAN=(.485,.456,.406);STD=(.229,.224,.225)
def letterbox(im,size=224):
 im.thumbnail((size,size),Image.Resampling.LANCZOS); bg=Image.new('RGB',(size,size),(124,116,103));bg.paste(im,((size-im.width)//2,(size-im.height)//2));return bg
class Transform:
 def __init__(self,aug='A0'):self.aug=aug
 def __call__(self,im):
  im=letterbox(im)
  if self.aug!='A0':
   if random.random()<.5:im=ImageOps.mirror(im)
   s=.1 if self.aug=='A1' else .2
   for C in (ImageEnhance.Brightness,ImageEnhance.Contrast,ImageEnhance.Color):im=C(im).enhance(random.uniform(1-s,1+s))
   if self.aug=='A2' and random.random()<.25:im=im.filter(ImageFilter.GaussianBlur(random.uniform(.1,1.2)))
  return F.normalize(F.to_tensor(im),MEAN,STD)
