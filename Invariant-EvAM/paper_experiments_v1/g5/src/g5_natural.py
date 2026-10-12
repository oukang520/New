from pathlib import Path
import json,copy,time,argparse,sys
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np,pandas as pd
from g5_model import Model,fit,profile,features,geometry,deviance,snapshot
R=Path(__file__).resolve().parents[3];G=R/'paper_experiments_v1/g5';P=json.loads((G/'configs/PROTOCOL.json').read_text());PANELS=P['panels'];T=pd.read_csv(G/'configs/target_registry.tsv',sep='\t')
def put(p,o):p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix('.tmp');q.write_text(json.dumps(o,ensure_ascii=False));q.replace(p)
def rng(purpose,rep):return np.random.default_rng(np.random.SeedSequence([P['master_seed'],5,purpose,rep]))
def rows():return json.loads((G/'data_private/natural_records.json').read_text())
def counts_for(rr,panels):return [np.bincount([z['reference_state'] for z in rr if z['environment']==e and z['natural_training_eligible']],minlength=16) for e in panels]
def make_spec(reg,E,topology='SPARSE',eta=0):
 f=features(reg['family']);known=[None]*E;known[0]=np.zeros(f.shape[1]).tolist() if eta==0 else None
 return dict(topology=topology,features=[f.tolist()]*E,known_phi=known,phi_bound=P['phi_bound'],reference_phi_bound=eta if eta else P['phi_bound'],bound=P['theta_bound'],epsilon=reg.get('epsilon',0),penalty=reg.get('penalty',0),mode=reg.get('mode','ctmc'))
def dataset(counts):return [dict(env=e,A=np.eye(16).tolist(),counts=np.asarray(c).tolist()) for e,c in enumerate(counts)]
def do_fit(reg,counts,topology='SPARSE',profiles=True,eta=0):
 spec=make_spec(reg,len(counts),topology,eta)
 if reg.get('independent',False):
  parts=[do_fit(dict(id='SINGLE',family='identity'),[c],topology,False) for c in counts];r=copy.deepcopy(parts[0]['fit']);return dict(fit=r,parts=parts,spec=spec,probabilities=[q['probabilities'][0] for q in parts],nll=sum(q['fit']['nll'] for q in parts),deviance=sum(q['deviance'] for q in parts),targets={},success=all(q['success'] for q in parts),reference_panel=PANELS[0])
 lik=Model(dataset(counts),spec);r=fit(lik);x=np.array(r['x']);prob=[lik.selected(x,e)[0].tolist() for e in range(len(counts))];tg={};diagnostic=None
 if reg.get('mode')=='fusion':
  source=lik.source(x)[0];pr=do_fit(dict(id='PROJECTION',family='identity'),[source],topology,False);r['projection']=pr['fit'];p=np.array(pr['probabilities'][0]);prob=[]
  for e in range(len(counts)):
   f=np.array(spec['features'][e]);ph=x[lik.phi[e]] if lik.phi[e] is not None else np.zeros(f.shape[1]);v=p*np.exp(f@ph);prob.append((v/v.sum()).tolist())
 else:
  tt=T[T.model_id.eq('D4_SPARSE10' if topology=='SPARSE' else 'D4_FULL16')]
  for z in tt.itertuples():
   a=json.loads(z.vector);tg[z.target_id]=profile(lik,r,a,len(tt)) if profiles else {'estimate':float(np.array(a)@x[:lik.p])}
  diagnostic=geometry(lik,x,json.loads(tt[tt.primary.eq(True)].iloc[0].vector))
 return dict(fit=r,spec=spec,probabilities=prob,nll=sum(float(-c@np.log(np.maximum(p,1e-280))) for c,p,j in lik.outputs(x)),deviance=deviance(lik,r),targets=tg,geometry=diagnostic,success=r['success'],reference_panel=PANELS[0])
def mainfit():
 rr=rows();counts=counts_for(rr,PANELS)
 for reg in P['natural_specs']:
  for top in ['SPARSE']+(['FULL'] if reg['id'] in P['competition_specs'] else []):
   path=G/f"runs/natural_main/{reg['id']}_{top}.json"
   if path.exists():continue
   r=do_fit(reg,counts,top,profiles=reg['id'] in P['natural_bootstrap_specs']);put(path,dict(spec_id=reg['id'],topology=top,counts=[c.tolist() for c in counts],result=r));print(reg['id'],top,'success',r['success'],'deviance',round(r['deviance'],3),flush=True)
 for eps in P['sensitivity_epsilon']:
  for eta in P['sensitivity_eta_reference_gene_selection']:
   path=G/f'runs/sensitivity/eps{eps}_eta{eta}.json'
   if path.exists():continue
   r=do_fit(dict(id='SENSITIVITY',family='genes',epsilon=eps),counts,profiles=True,eta=eta);put(path,dict(epsilon=eps,eta=eta,result=r));print('sensitivity',eps,eta,flush=True)

def heldout():
 rr=rows()
 for held in PANELS[1:]:
  trainpan=[p for p in PANELS if p!=held];counts=counts_for(rr,trainpan);cal=np.bincount([z['reference_state'] for z in rr if z['environment']==held and z['heldout_role']=='calibration'],minlength=16);ev=np.bincount([z['reference_state'] for z in rr if z['environment']==held and z['heldout_role']=='evaluation'],minlength=16)
  for reg in P['natural_specs']:
   path=G/f"runs/heldout/{held}_{reg['id']}.json"
   if path.exists():continue
   train=do_fit(reg,counts,profiles=False);f=features(reg['family']);sourcefit=train['fit'].get('projection',train['fit']);theta=np.array(sourcefit['x'][:10]);cs=make_spec({k:v for k,v in reg.items() if k not in ['mode','independent']},2);cd=dataset([np.zeros(16),cal]);lik=Model(cd,cs)
   if not reg.get('independent'):lik.bounds[:10]=[(v,v) for v in theta]
   # Independent baseline uses only the same allowed heldout calibration, fitting its own generator.
   cr=fit(lik);prob=lik.selected(np.array(cr['x']),1)[0];zero=snapshot(theta)[0];freq=ev/ev.sum();score=float(-freq@np.log(np.maximum(prob,1e-280)));tv=float(.5*np.abs(prob-freq).sum());stat=float(2*np.sum(ev[ev>0]*np.log(freq[ev>0]/np.maximum(prob[ev>0],1e-280))));random=rng(510+PANELS.index(held),0);sim=random.multinomial(int(ev.sum()),prob,size=1000);dev=[]
   for y in sim:
    nz=y>0;dev.append(float(2*np.sum(y[nz]*np.log((y[nz]/y.sum())/np.maximum(prob[nz],1e-280)))))
   pv=(1+sum(v>=stat for v in dev))/1001
   # Independent evaluation-patient resampling of scores, conditional on fixed training/calibration fit.
   random=rng(520+PANELS.index(held),0);boot=random.multinomial(int(ev.sum()),freq,size=200);scores=-(boot/ev.sum())@np.log(np.maximum(prob,1e-280))
   put(path,dict(spec_id=reg['id'],heldout=held,reference_panel=PANELS[0],training_panels=trainpan,n_training=[int(c.sum()) for c in counts],n_calibration=int(cal.sum()),n_evaluation=int(ev.sum()),train=train,calibration_fit=cr,prediction=prob.tolist(),evaluation_counts=ev.tolist(),log_score=score,tv=tv,zero_calibration_log_score=float(-freq@np.log(np.maximum(zero,1e-280))),zero_rule='Frozen reference-record distribution; assumes no new selection/biology',predictive_deviance=stat,conditional_multinomial_p=pv,compatibility_label='Conditional fixed-prediction multinomial diagnostic; calibration-estimation uncertainty not integrated; not natural-biology test',evaluation_bootstrap_scores=scores.tolist(),success=train['success'] and cr['success']));print('heldout',held,reg['id'],round(score,4),flush=True)

def batch_one(task):
 purpose,regid,rep=task;path=G/f'runs/{purpose}/{regid}_{rep:03d}.json'
 if path.exists():return purpose,regid,rep,'CACHED'
 reg=next(z for z in P['natural_specs'] if z['id']==regid);rr=rows();original=counts_for(rr,PANELS);random=rng({'bootstrap':530,'compatibility':540,'partitions':550}[purpose],rep)
 if purpose=='bootstrap':counts=[random.multinomial(int(c.sum()),c/c.sum()) for c in original];result=do_fit(reg,counts)
 elif purpose=='compatibility':
  main=json.loads((G/f'runs/natural_main/{regid}_SPARSE.json').read_text())['result'];counts=[random.multinomial(int(c.sum()),p) for c,p in zip(original,main['probabilities'])];result=do_fit(reg,counts,profiles=False)
 else:
  x=np.repeat(np.arange(16),original[0]);random.shuffle(x);counts=[np.bincount(a,minlength=16) for a in np.array_split(x,3)];result=do_fit(reg,counts,profiles=False);single=do_fit(dict(id='SINGLE',family='identity'),[original[0]],profiles=False);one=Model(dataset([original[0]]),make_spec(dict(family='identity'),1));many=Model(dataset(counts),make_spec(dict(family='identity'),3));a=np.eye(10)[4];theta=np.array(single['fit']['x']);result['same_operator_rank_one']=geometry(one,theta,a)['theta_rank'];result['same_operator_rank_split']=geometry(many,theta,a)['theta_rank'];result['independent']=do_fit(dict(id='INDEPENDENT',family='identity',independent=True),counts,profiles=False)
 put(path,dict(purpose=purpose,spec_id=regid,replicate=rep,result=result));return purpose,regid,rep,result['success']

def main():
 a=argparse.ArgumentParser();a.add_argument('--task',choices=['main','heldout','bootstrap','compatibility','partitions'],required=True);a.add_argument('--workers',type=int,default=4);args=a.parse_args();assert (G/'configs/EVALUATION_FREEZE.json').exists()
 if args.task=='main':return mainfit()
 if args.task=='heldout':return heldout()
 ids=P['natural_bootstrap_specs'] if args.task in ['bootstrap','compatibility'] else ['IDENTITY_SHARED'];n=P['random_partitions'] if args.task=='partitions' else 200;tasks=[(args.task,id,i) for id in ids for i in range(n)]
 with ProcessPoolExecutor(max_workers=args.workers) as ex:
  for i,f in enumerate(as_completed([ex.submit(batch_one,t) for t in tasks]),1):print(i,'/',len(tasks),*f.result(),flush=True)
if __name__=='__main__':main()
