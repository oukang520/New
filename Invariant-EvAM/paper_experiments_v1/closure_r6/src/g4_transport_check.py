"""Paired held-patient decomposition, internal vs external; explanatory reanalysis."""
from pathlib import Path
import sys,json
import numpy as np,pandas as pd
P=Path(__file__).resolve().parents[2];G=P/'g4/revision_r6';sys.path.insert(0,str(P/'g4/revision_r4/src'));import bridge as b
TR=json.loads((P/'g4/data_private/selection_truth/operators.json').read_text());DES=list(b.gp.DESIGNS);rows=[];boots=[];raw=[]
for ci,cancer in enumerate(['LUAD','COAD']):
 x=b.gp.roledata(b.gp.records(cancer),'evaluation');n=len(x);mat={};weights={}
 for des in DES:
  sp=b.gp.DESIGNS[des];preds={k:[] for k in ['MORE','ALIGNED','DEASSOCIATED','HALF','TWO_STEP','POOLED']}
  for rep in range(50):
   z=json.loads((G/f'runs/{cancer}_{des}_{rep:03d}.json').read_text());old=json.loads((P/f'g4/revision_r4/runs/evaluation_{cancer}_{des}_{rep:03d}.json').read_text())
   if z['status']!='COMPLETE':continue
   for k,m in [('MORE','MORE_CTMC'),('ALIGNED','BRIDGE_CTMC'),('TWO_STEP','BRIDGE_TWO_STEP'),('POOLED','BRIDGE_POOLED')]:preds[k].append(old['methods'][m]['probabilities'])
   for k,dose in [('DEASSOCIATED','0.0'),('HALF','0.5')]:preds[k].append(z['results'][dose]['probabilities'])
  w=np.array([.5*np.exp(TR[des]['alpha'][e]+np.array(sp['features'][e])[x]@TR[des]['phi'][e]) for e in range(2)]);weights[des]=w.sum(axis=0)
  losses={}
  for name,ps in preds.items():
   ps=np.array(ps);loss=np.zeros((len(ps),n))
   for e in range(2):
    A=np.array(sp['measurement'][e]);yi=np.argmax(A,axis=0)[x];hidden=np.flatnonzero(1-np.array(sp['masks'][e]))[0];y=b.X[x,hidden];prob=np.clip(ps[:,e,yi],1e-8,1-1e-8);loss+=w[e]*( -y*np.log(prob)-(1-y)*np.log1p(-prob))
   losses[name]=loss
  mat[des]=losses;np.savez_compressed(G/f'results/internal_losses_{cancer}_{des}.npz',weights=weights[des],**losses)
 contrasts=[('total','MORE','ALIGNED'),('deassociated_gain','MORE','DEASSOCIATED'),('pairing','DEASSOCIATED','ALIGNED'),('optimizer','TWO_STEP','ALIGNED'),('model','POOLED','ALIGNED')]
 mean=np.mean([[((mat[des][a]-mat[des][c]).sum(axis=1)/weights[des].sum()).mean() for _,a,c in contrasts] for des in DES],axis=0)
 rng=np.random.default_rng([2026092764,77,ci]);boot=[]
 for rep in range(2000):
  ix=rng.integers(0,n,n);vals=[]
  for des in DES:
   m=mat[des];si=rng.integers(0,len(m['MORE']),len(m['MORE']));vals.append([((m[a]-m[c])[si][:,ix].sum(axis=1)/weights[des][ix].sum()).mean() for _,a,c in contrasts])
  boot.append(np.mean(vals,axis=0))
 boot=np.array(boot)
 for j,(name,_,_) in enumerate(contrasts):
  lo,hi=np.quantile(boot[:,j],[.025,.975]);rows.append(dict(cancer=cancer,cohort='GENIE held evaluation patients',contrast=name,gain=mean[j],lo=lo,hi=hi,n_patients=n,scope='Scorer integrates registered perturbation over held patient records; fits and patients previously explored'))
  boots.extend(dict(cancer=cancer,contrast=name,bootstrap=i,gain=v) for i,v in enumerate(boot[:,j]))
 print(cancer,flush=True)
pd.DataFrame(rows).to_csv(G/'results/internal_patient_summary.tsv',sep='\t',index=False);pd.DataFrame(boots).to_csv(G/'results/internal_patient_bootstrap.tsv',sep='\t',index=False);print(pd.DataFrame(rows).to_string(index=False))
