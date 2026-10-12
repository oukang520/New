from pathlib import Path
import sys,json,copy,hashlib,time
from concurrent.futures import ProcessPoolExecutor
import numpy as np,pandas as pd
P=Path(__file__).resolve().parents[2];G=P/'g4/revision_r6';sys.path.insert(0,str(P/'g4/revision_r4/src'));import bridge as b
C=json.loads((P/'closure_r6/configs/G4_PROTOCOL.json').read_text());OLD=P/'g4/revision_r4';R5=P/'g4/revision_r5';X=b.X
sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
def one(t):
 cancer,design,rep=t;out=G/f'runs/{cancer}_{design}_{rep:03d}.json'
 if out.exists():return
 z=json.loads((OLD/f'runs/evaluation_{cancer}_{design}_{rep:03d}.json').read_text())
 if z['status']!='COMPLETE':out.write_text(json.dumps(dict(cancer=cancer,design=design,replicate=rep,status=z['status'])));return
 spec=b.gp.DESIGNS[design];ds,ev,stats=b.generate(cancer,design,rep,'evaluation');assert ds==z['public_inputs']
 results={};check=[]
 for dose in [0.,.5]:
  data=copy.deepcopy(ds['BRIDGE'])
  for e in range(2):
   c=np.array(data['bridge'][e],float);A=np.array(spec['measurement'][e]);h=np.flatnonzero(1-np.array(spec['masks'][e]))[0];yi=np.argmax(A,axis=0);margin=A@c;ph=(c@X[:,h])/c.sum();ind=margin[yi]*np.where(X[:,h],ph,1-ph);new=dose*c+(1-dose)*ind
   assert np.allclose(A@new,A@c) and np.isclose(new@X[:,h],c@X[:,h]) and np.isclose(new.sum(),c.sum())
   data['bridge'][e]=new.tolist();check.append(dict(dose=dose,env=e,mass=float(new.sum()),max_margin_error=float(abs(A@new-A@c).max())))
  fit=b.estimate(data,spec,'CTMC');pred=b.pred(fit,spec);sc,pat=b.score(pred,fit['source'],ev,spec,json.loads((P/'g4/data_private/selection_truth/operators.json').read_text())[design]);results[str(dose)]=dict(fit=fit,probabilities=pred.tolist(),scores=sc,patient=pat)
 out.write_text(json.dumps(dict(cancer=cancer,design=design,replicate=rep,status='COMPLETE',results=results,checks=check,old_fit_sha256=sha(OLD/f'runs/evaluation_{cancer}_{design}_{rep:03d}.json'))))

def finish():
 raw=[];matrices={};checks=[];internal=[]
 truth=json.loads((P/'g4/data_private/selection_truth/operators.json').read_text())
 for cancer in C['cancers']:
  records=pd.read_csv(R5/f'data_external/{cancer}_records.tsv',sep='\t');states=records.state.to_numpy();n=len(states)
  for design in C['designs']:
   spec=b.gp.DESIGNS[design];arr={'DEASSOCIATED':[],'HALF_ASSOCIATION':[]};oldmat=np.load(R5/f'results/external_patient_loss_{cancer}_{design}.npz');ids=oldmat['replicate'];
   for rep in ids:
    z=json.loads((G/f'runs/{cancer}_{design}_{rep:03d}.json').read_text());checks+=z['checks'];old=json.loads((OLD/f'runs/evaluation_{cancer}_{design}_{rep:03d}.json').read_text())
    rng=np.random.default_rng([202609265,7,False,C['cancers'].index(cancer),int(rep)]);env=rng.integers(0,2,n);sel=rng.random(n)<np.array([np.exp(truth[design]['alpha'][e]+np.array(spec['features'][e])[s]@truth[design]['phi'][e]) for e,s in zip(env,states)]);y=np.array([X[s,np.flatnonzero(1-np.array(spec['masks'][e]))[0]] for e,s in zip(env,states)]);visible=np.array([np.argmax(np.array(spec['measurement'][e]),axis=0)[s] for e,s in zip(env,states)])
    for dose,name in [('0.0','DEASSOCIATED'),('0.5','HALF_ASSOCIATION')]:
     r=z['results'][dose];pred=np.array(r['probabilities']);pr=np.clip(pred[env,visible],1e-8,1-1e-8);loss=-(y*np.log(pr)+(1-y)*np.log1p(-pr));loss[~sel]=np.nan;arr[name].append(loss);raw.append(dict(cancer=cancer,design=design,replicate=rep,method=name,loss=float(np.nanmean(loss)),fit_success=r['fit']['success']))
     internal.append(dict(cancer=cancer,design=design,replicate=rep,method=name,loss=r['scores']['integrated_log_score'],fit_success=r['fit']['success']))
    for meth,name in [('MORE_CTMC','MORE'),('BRIDGE_CTMC','ALIGNED'),('BRIDGE_POOLED','POOLED'),('BRIDGE_TWO_STEP','TWO_STEP')]:internal.append(dict(cancer=cancer,design=design,replicate=rep,method=name,loss=old['methods'][meth]['scores']['integrated_log_score'],fit_success=old['methods'][meth]['fit']['success']))
   for name,om in [('MORE','MORE_CTMC'),('ALIGNED','BRIDGE_CTMC'),('POOLED','BRIDGE_POOLED'),('TWO_STEP','BRIDGE_TWO_STEP')]:arr[name]=oldmat[om]
   arr={k:np.asarray(v) for k,v in arr.items()};matrices[cancer,design]=arr;np.savez_compressed(G/f'results/patient_losses_{cancer}_{design}.npz',patient_id=records.PATIENT_ID.to_numpy(dtype=str),replicate=ids,**arr)
 pd.DataFrame(raw).to_csv(G/'results/new_fit_raw.tsv',sep='\t',index=False);pd.DataFrame(internal).to_csv(G/'results/internal_raw.tsv',sep='\t',index=False)
 contrasts=[('total','MORE','ALIGNED'),('margin','MORE','DEASSOCIATED'),('pairing','DEASSOCIATED','ALIGNED'),('dose_first','DEASSOCIATED','HALF_ASSOCIATION'),('dose_second','HALF_ASSOCIATION','ALIGNED'),('optimizer','TWO_STEP','ALIGNED'),('model','POOLED','ALIGNED')];summ=[];boots=[];cells=[]
 for cancer in C['cancers']:
  n=len(pd.read_csv(R5/f'data_external/{cancer}_records.tsv',sep='\t'));rng=np.random.default_rng([C['seed'],C['cancers'].index(cancer)]);est=[];boot=[]
  for des in C['designs']:
   m=matrices[cancer,des];vals=np.stack([m[a]-m[b] for _,a,b in contrasts]);est.append(np.nanmean(vals,axis=2).mean(axis=1))
   for (name,_,_),v in zip(contrasts,est[-1]):cells.append(dict(cancer=cancer,design=des,contrast=name,gain=v))
  for rep in range(2000):
   ix=rng.integers(0,n,n);means=[]
   for des in C['designs']:
    m=matrices[cancer,des];si=rng.integers(0,len(m['MORE']),len(m['MORE']));means.append([float(np.nanmean((m[a]-m[b])[si][:,ix],axis=1).mean()) for _,a,b in contrasts])
   boot.append(np.mean(means,axis=0))
  boot=np.array(boot);assert np.allclose(boot[:,0],boot[:,1]+boot[:,2]);mean=np.mean(est,axis=0)
  for j,(name,_,_) in enumerate(contrasts):
   lo,hi=np.quantile(boot[:,j],[.025,.975]);summ.append(dict(cancer=cancer,contrast=name,gain=mean[j],lo=lo,hi=hi,n_patients=n))
   boots.extend(dict(cancer=cancer,contrast=name,bootstrap=i,gain=v) for i,v in enumerate(boot[:,j]))
 pd.DataFrame(summ).to_csv(G/'results/attribution_summary.tsv',sep='\t',index=False);pd.DataFrame(boots).to_csv(G/'results/attribution_bootstrap.tsv',sep='\t',index=False);pd.DataFrame(cells).to_csv(G/'results/attribution_by_design.tsv',sep='\t',index=False)
 (G/'results/VALIDATION.json').write_text(json.dumps(dict(margin_checks=len(checks),max_margin_error=max(c['max_margin_error'] for c in checks),decomposition_exact=True,new_fit_failures=sum(not x['fit_success'] for x in raw),planned=300,completed=len(list((G/'runs').glob('*.json')))),indent=2));print(pd.DataFrame(summ).to_string(index=False),flush=True)
if __name__=='__main__':
 tasks=list(__import__('itertools').product(C['cancers'],C['designs'],range(50)))
 with ProcessPoolExecutor(2) as pool:
  for i,_ in enumerate(pool.map(one,tasks),1):
   if i%25==0:print(i,len(tasks),flush=True)
 finish()
