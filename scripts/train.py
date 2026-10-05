import argparse,os,json,random,sys
sys.path.insert(0,'.')
import numpy as np,torch
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast,GradScaler
from sklearn.metrics import accuracy_score,precision_recall_fscore_support,balanced_accuracy_score,confusion_matrix
from src.dataset import ZipTrafficDataset
from src.transforms import Transform
from src.models import SimpleCNN,ComplexCNN,mobile
CL=['empty','low','medium','high','traffic_jam']
def seed(s):random.seed(s);np.random.seed(s);torch.manual_seed(s);torch.cuda.manual_seed_all(s)
def metrics(y,p):
 q=precision_recall_fscore_support(y,p,labels=range(5),zero_division=0);return {'accuracy':accuracy_score(y,p),'macro_precision':q[0].mean(),'macro_recall':q[1].mean(),'macro_f1':q[2].mean(),'weighted_f1':precision_recall_fscore_support(y,p,average='weighted',zero_division=0)[2],'balanced_accuracy':balanced_accuracy_score(y,p),'ordinal_mae':float(np.mean(np.abs(np.array(y)-np.array(p)))),'within_one':float(np.mean(np.abs(np.array(y)-np.array(p))<=1)),'per_class':{CL[i]:{'precision':float(q[0][i]),'recall':float(q[1][i]),'f1':float(q[2][i])} for i in range(5)}}
def main():
 a=argparse.ArgumentParser();a.add_argument('--model',choices=['simple','complex','mobile'],required=True);a.add_argument('--aug',default='A0');a.add_argument('--loss',default='weighted');a.add_argument('--seed',type=int,default=42);a.add_argument('--epochs',type=int,default=40);x=a.parse_args();seed(x.seed);dev='cuda' if torch.cuda.is_available() else 'cpu';out=f'outputs/{x.model}_{x.aug}_{x.loss}_{x.seed}';os.makedirs(out,exist_ok=True)
 tr=ZipTrafficDataset('data/manifests/train.csv','traffic.zip',Transform(x.aug));va=ZipTrafficDataset('data/manifests/val.csv','traffic.zip',Transform());te=ZipTrafficDataset('data/manifests/test.csv','traffic.zip',Transform());dl=lambda d,sh:DataLoader(d,batch_size=64,shuffle=sh,num_workers=4,pin_memory=True,persistent_workers=True)
 model={'simple':SimpleCNN(),'complex':ComplexCNN(),'mobile':mobile()}[x.model].to(dev)
 if x.model=='mobile':
  for p in model.features.parameters():p.requires_grad=False
 counts=np.bincount([int(r['label']) for r in tr.rows],minlength=5);w=torch.tensor(len(tr)/(5*counts),dtype=torch.float,device=dev) if x.loss=='weighted' else None
 crit=torch.nn.CrossEntropyLoss(weight=w);opt=torch.optim.AdamW(filter(lambda p:p.requires_grad,model.parameters()),lr=1e-3,weight_decay=1e-4);sch=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,mode='max',patience=3,factor=.3);sc=GradScaler();best=-1;bad=0;hist=[]
 for ep in range(x.epochs):
  # Transfer-learning stage 2: unlock the last MobileNet blocks after head warm-up.
  if x.model=='mobile' and ep==5:
   for p in list(model.features.children())[-4:]:
    for q in p.parameters(): q.requires_grad=True
   opt=torch.optim.AdamW(filter(lambda p:p.requires_grad,model.parameters()),lr=1e-4,weight_decay=1e-4)
  model.train()
  for im,y,_ in dl(tr,True):
   opt.zero_grad();im,y=im.to(dev),y.to(dev)
   with autocast():loss=crit(model(im),y)
   sc.scale(loss).backward();sc.step(opt);sc.update()
  model.eval();Y=[];P=[]
  with torch.no_grad():
   for im,y,_ in dl(va,False):Y+=y.tolist();P+=model(im.to(dev)).argmax(1).cpu().tolist()
  m=metrics(Y,P);hist.append(m);sch.step(m['macro_f1']);print(ep,m['macro_f1'],flush=True)
  if m['macro_f1']>best:best=m['macro_f1'];bad=0;torch.save({'model':model.state_dict(),'args':vars(x),'epoch':ep},out+'/best.pt')
  else:bad+=1
  if bad>=7:break
 ck=torch.load(out+'/best.pt',map_location=dev);model.load_state_dict(ck['model']);model.eval();Y=[];P=[];paths=[]
 with torch.no_grad():
  for im,y,pa in dl(te,False):Y+=y.tolist();P+=model(im.to(dev)).argmax(1).cpu().tolist();paths+=list(pa)
 m=metrics(Y,P);m.update({'best_epoch':ck['epoch'],'confusion_matrix':confusion_matrix(Y,P,labels=range(5)).tolist(),'args':vars(x)});json.dump(m,open(out+'/metrics.json','w'),indent=2);import csv
 with open(out+'/predictions.csv','w',newline='') as f:w=csv.writer(f);w.writerow(['path','true','pred']);w.writerows(zip(paths,Y,P))
 json.dump(hist,open(out+'/history.json','w'),indent=2);print(json.dumps(m,indent=2))
if __name__=='__main__':main()
