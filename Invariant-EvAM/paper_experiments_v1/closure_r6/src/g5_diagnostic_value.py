from pathlib import Path
import sys,json,hashlib,time,argparse,copy
from concurrent.futures import ProcessPoolExecutor
import numpy as np,pandas as pd
from scipy.stats import chi2
from g5_explore import Obs,spec,generate,public_policy,MASKS,SEED,P,G
from g5_model import fit,X,snapshot
from g5_interval import constrained,endpoints
HERE=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def dump(path,obj):path.write_text(json.dumps(obj,default=lambda x:x.item() if isinstance(x,np.generic) else x.tolist()))
def generate_case(stage,case,rep):
 theta,p,q,base,records,selected,acts=generate(stage,min(case,5),rep)
 if case>=6:
  # Dedicated out-of-family controls: unmodelled selection interaction.
  rng=np.random.default_rng([SEED,stage,case,rep]);theta=np.r_[rng.uniform(-1.5,-.3,4),rng.uniform(-.7,.7,6)];theta[4]=rng.choice([-1.,-.5,0.,.5,1.]);p,_=snapshot(theta);z=.7*X[:,0]+.6*X[:,0]*X[:,1]+2*X[:,1]*X[:,2];q=p*np.exp(z);q/=q.sum();ns=[240,40] if case==6 else [40,240];records=[rng.multinomial(n,p) for n in ns];selected=rng.multinomial(120,q);base=[dict(env=0,kind='source',A=MASKS[j].tolist(),counts=(MASKS[j]@c).tolist()) for j,c in enumerate(records)]+[dict(env=0,kind='selected',A=np.eye(16).tolist(),counts=selected.tolist())];acts=[dict(name='COMPLETE_A',complete=0,new_source=(600-ns[0])//4,new_selected=0,cost=600),dict(name='COMPLETE_B',complete=1,new_source=(600-ns[1])//4,new_selected=0,cost=600),dict(name='NEW_SOURCE',complete=-1,new_source=150,new_selected=0,cost=600),dict(name='MORE_SELECTED',complete=-1,new_source=0,new_selected=150,cost=600)]
 return theta,p,q,base,records,selected,acts

def one(task):
 stage,case,rep=task;path=G/f'runs/{stage}_{case}_{rep:04d}.json'
 if path.exists():return
 start=time.time();st=1 if stage=='calibration' else 2;theta,p,q,base,records,selected,acts=generate_case(st,case,rep);choice,dopt,diag,br=public_policy(base,acts);out=[];cut=json.loads((G/'configs/FREEZE.json').read_text())['cutoff'] if stage=='evaluation' else None
 for j,a in enumerate(acts+([dict(name='NO_ACTION',cost=0)] if stage=='evaluation' else [])):
  if a['name']=='NO_ACTION':data=base;r=br;lik=Obs(base,spec())
  else:
   rng=np.random.default_rng([SEED,st,case,rep,j,91]);data=[]
   for k,c in enumerate(records):data.append(dict(env=0,kind='source',A=(np.eye(16) if k==a['complete'] else MASKS[k]).tolist(),counts=(c if k==a['complete'] else MASKS[k]@c).tolist()))
   data+=[dict(env=0,kind='selected',A=np.eye(16).tolist(),counts=(selected+rng.multinomial(a['new_selected'],q)).tolist()),dict(env=0,kind='source',A=np.eye(16).tolist(),counts=rng.multinomial(a['new_source'],p).tolist())]
   actual=sum(records[a['complete']]) if a['complete']>=0 else 0;actual+=4*(a['new_source']+a['new_selected']);assert actual==600
   lik=Obs(data,spec());r=fit(lik)
  t=float(theta[4]);rec=dict(action=a['name'],cost=a['cost'],data=data,fit=r,estimate=r['x'][4],squared_error=(r['x'][4]-t)**2 if r['success'] else 36.)
  if stage=='calibration':
   ct=constrained(lik,r,t);lr=2*(ct['nll']-r['nll']);rec.update(lr=lr,valid=bool(ct['success'] and r['success'] and lr>=-1e-5))
  else:
   pr=endpoints(lik,r,cut);valid=pr['status'] in ['COMPLETE','UNRESOLVED'];called=valid and (pr['lo']>0 or pr['hi']<0);wrong=called and ((pr['lo']>0 and t<=0) or (pr['hi']<0 and t>=0));pr.update(valid=valid,covered=bool(valid and pr['lo']<=t<=pr['hi']),called=called,wrong=wrong,correct=bool(called and not wrong),finite=bool(valid and np.isfinite([pr['lo'],pr['hi']]).all()));rec['interval']=pr
  out.append(rec)
 dump(path,dict(stage=stage,case=case,replicate=rep,truth=t,theta=theta,baseline=base,choices={'DIAGNOSTIC':choice,'C_OPT':choice,'D_OPT':dopt,'LARGEST_PANEL':'COMPLETE_A' if sum(base[0]['counts'])>=sum(base[1]['counts']) else 'COMPLETE_B'},diagnostics=diag,base_fit=br,outcomes=out,runtime=time.time()-start))

def freeze():
 docs=[json.loads(f.read_text()) for f in sorted((G/'runs').glob('calibration_*.json'))];assert len(docs)==240
 rows=[dict(case=z['case'],replicate=z['replicate'],action=o['action'],lr=o['lr'],valid=o['valid'],squared_error=o['squared_error']) for z in docs for o in z['outcomes']];d=pd.DataFrame(rows);d.to_csv(G/'results/calibration_raw.tsv',sep='\t',index=False)
 assert d.valid.mean()>=.99
 # Pool six equally weighted development conditions, take the maximum across actions.
 cut=max(chi2.ppf(.95,1),d[d.valid].groupby('action').lr.quantile(.95,interpolation='higher').max());fixed=d.groupby('action').squared_error.mean().idxmin()
 output=dict(cutoff=float(cut),fixed_action=fixed,calibration_datasets=240,files={str(f.relative_to(P)):sha(f) for f in sorted((G/'runs').glob('calibration_*.json'))},code_sha256={f.name:sha(f) for f in [Path(__file__),HERE/'g5_explore.py']},evaluation_per_case=150,cases=list(range(8)),core_cases=list(range(6)),family_outside=[6,7],seed=SEED,cutoff_rule='maximum action-wise pooled equal-case calibration q95, lower bounded chi-square95',fixed_rule='minimum mean development squared error; no evaluation selection',target='original sparse D4 CTMC theta[4], one source reference population',scope='Two partial source panels, unknown selected-source bias; 600 extra assays, known public measurement operators. C-opt is an identical framework module, not a competing end-to-end solution.',score8_requirements=['at least two actions each used>=20% core','pooled core coverage>=.93 and each case>=.90; wrong<=.05 pooled; failures<=.02','paired core MSE gain over frozen best-fixed, random, shuffled diagnosis all lower95>0','correct-answer gain over best-fixed and random positive'],score8plus_requirements=['correct-answer paired lower95>0 vs best-fixed and shuffled diagnosis','paired MSE gain vs every fixed action positive after Holm adjustment; include source-only and no-action','out-of-family controls and largest-panel/c-opt/D-opt outcomes shown without omission'])
 dump(G/'configs/FREEZE.json',output);print('FREEZE cutoff',cut,'fixed',fixed,flush=True)

def finish():
 docs=[json.loads(f.read_text()) for f in sorted((G/'runs').glob('evaluation_*.json'))];assert len(docs)==1200;freeze=json.loads((G/'configs/FREEZE.json').read_text());rows=[];raw=[]
 # Permutation of diagnostic actions is independent of action outcomes, preserving action proportions within core/stress.
 permutations={}
 for core in [True,False]:
  ids=[i for i,z in enumerate(docs) if (z['case']<6)==core];rr=np.random.default_rng([SEED,701,int(core)]);shuffled=rr.permutation(ids)
  permutations.update({i:docs[k]['choices']['DIAGNOSTIC'] for i,k in zip(ids,shuffled)})
 for i,z in enumerate(docs):
  choices=dict(z['choices']);choices.update(BEST_FIXED=freeze['fixed_action'],RANDOM=['COMPLETE_A','COMPLETE_B','NEW_SOURCE','MORE_SELECTED'][np.random.default_rng([SEED,702,z['case'],z['replicate']]).integers(4)],SHUFFLED_DIAGNOSIS=permutations[i],COUNT_ONLY='MORE_SELECTED',NO_ACTION='NO_ACTION')
  choices.update({a:a for a in ['COMPLETE_A','COMPLETE_B','NEW_SOURCE','MORE_SELECTED']})
  for o in z['outcomes']:raw.append(dict(case=z['case'],replicate=z['replicate'],truth=z['truth'],action=o['action'],cost=o['cost'],squared_error=o['squared_error'],**o['interval']))
  for policy,act in choices.items():
   o=next(o for o in z['outcomes'] if o['action']==act);rows.append(dict(case=z['case'],replicate=z['replicate'],truth=z['truth'],policy=policy,action=act,cost=o['cost'],squared_error=o['squared_error'],**o['interval']))
 pd.DataFrame(raw).to_csv(G/'results/action_raw.tsv',sep='\t',index=False);pd.DataFrame(rows).to_csv(G/'results/policy_raw.tsv',sep='\t',index=False)
 print('Evaluation complete:',len(docs),'datasets',flush=True)
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('stage',choices=['calibration','evaluation']);a.add_argument('--workers',type=int,default=2);args=a.parse_args()
 if args.stage=='evaluation':
  f=json.loads((G/'configs/FREEZE.json').read_text());assert all(sha(HERE/k)==v for k,v in f['code_sha256'].items());n=f['evaluation_per_case'];cases=f['cases']
 else:n=40;cases=range(6)
 tasks=[(args.stage,c,r) for c in cases for r in range(n)]
 with ProcessPoolExecutor(args.workers) as pool:
  for i,_ in enumerate(pool.map(one,tasks),1):
   if i%25==0:print(i,len(tasks),flush=True)
 freeze() if args.stage=='calibration' else finish()
