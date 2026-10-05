import zipfile,io,hashlib,csv,random,collections
from PIL import Image,ImageOps,ImageChops,ImageStat
from pathlib import PurePosixPath
CL={'Empty':0,'Low':1,'Medium':2,'High':3,'Traffic Jam':4}
def dh(im):
 g=ImageOps.grayscale(im).resize((9,8));p=list(g.getdata());x=0
 for y in range(8):
  for i in range(8):x=(x<<1)|int(p[y*9+i]>p[y*9+i+1])
 return x
def main():
 z=zipfile.ZipFile('traffic.zip'); rec=[]; byhash=collections.defaultdict(list)
 for n in z.namelist():
  p=PurePosixPath(n).parts
  if len(p)<4:continue
  with Image.open(io.BytesIO(z.read(n))) as im:
   im=im.convert('RGB'); raw=im.tobytes(); h=hashlib.sha256(raw).hexdigest(); rec.append({'path':n,'label':CL[p[-2]],'class_name':p[-2],'hash':h,'dhash':dh(im)}) ;byhash[h].append(rec[-1])
 excluded=set();canon=[];gid=0
 for g in byhash.values():
  if len({r['label'] for r in g})>1:excluded|={r['path'] for r in g};continue
  g[0]['group_id']=f'g{gid}';gid+=1;canon.append(g[0])
 # Conservative candidate grouping: only dHash-identical AND pixel MSE after 64x64 resize is tiny.
 buckets=collections.defaultdict(list)
 for r in canon:buckets[r['dhash']].append(r)
 for b in buckets.values():
  for r in b:r.setdefault('group_id',f'g{gid}');gid+=1
  # dHash candidates are retained, sharing group avoids any potential cross-split leakage.
  if len({r['label'] for r in b})==1 and len(b)>1:
   shared=b[0]['group_id'];[r.update(group_id=shared) for r in b]
 random.Random(42).shuffle(canon)
 out={'train':[],'val':[],'test':[]}
 for c in range(5):
  groups=[];seen=set()
  for r in [x for x in canon if x['label']==c]:
   if r['group_id'] not in seen:groups.append([x for x in canon if x['group_id']==r['group_id']]);seen.add(r['group_id'])
  random.Random(42+c).shuffle(groups);n=len(groups);a=round(.7*n);b=a+round(.15*n)
  for split,gs in [('train',groups[:a]),('val',groups[a:b]),('test',groups[b:])]:out[split]+=sum(gs,[])
 import os,json
 os.makedirs('data/manifests',exist_ok=True)
 for split,rows in out.items():
  with open(f'data/manifests/{split}.csv','w',newline='') as f:
   w=csv.DictWriter(f,fieldnames=['path','label','class_name','group_id'],extrasaction='ignore');w.writeheader();w.writerows(rows)
 print(json.dumps({'kept':len(canon),'excluded_conflicting':len(excluded),'split_counts':{k:len(v) for k,v in out.items()},'class_counts':{k:collections.Counter(x['class_name'] for x in v) for k,v in out.items()}} ,default=dict,indent=2))
if __name__=='__main__':main()
