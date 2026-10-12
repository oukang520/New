from pathlib import Path
import sys,json,time,argparse,copy,hashlib
import numpy as np,pandas as pd
from concurrent.futures import ProcessPoolExecutor,as_completed
R=Path(__file__).resolve().parents[3];P=R/'paper_experiments_v1';D=Path(__file__).resolve().parent
sys.path[:0]=[str(P/'g4/src'),str(P/'g5/src')]
CFG=json.loads((D/'PROTOCOL.json').read_text());MASTER=CFG['master_seed']
def rng(g,*keys):return np.random.default_rng(np.random.SeedSequence([MASTER,g,*keys]))
def serial(x):
 if isinstance(x,np.ndarray):return x.tolist()
 if isinstance(x,np.generic):return x.item()
 raise TypeError(str(type(x)))
def put(p,x):p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix('.tmp');q.write_text(json.dumps(x,default=serial));q.replace(p)
def readrows(cancer):return json.loads((P/f'g4/data_private/reference_records/{cancer}.json').read_text())
def fusion(data,spec):
 from g4_estimator import Likelihood,fit
 lik=Likelihood(data,spec,'fusion');r=fit(lik);x=np.array(r['x']);src=lik.source(x)[0];selected=[]
 for e in range(lik.E):
  phi=x[lik.phi[e]];v=src*np.exp(np.asarray(spec['features'][e])@phi);selected.append((v/v.sum()).tolist())
 r.update(source=src.tolist(),selected=selected,method='UNPROJECTED_FUSION',estimate=None);return r,lik

def g4_one(task):
 from g4_pipeline import perturb,score,DESIGNS,P as OLD
 from g4_estimator import train,Likelihood
 import g4_pipeline
 purpose,cancer,design,rep=task;path=P/f'g4/revision_r2/runs/{purpose}_{cancer}_{design}_{rep:03d}.json'
 if path.exists():return str(path.name),'CACHED'
 start=time.time();g4_pipeline.P=copy.deepcopy(OLD);g4_pipeline.P['master_seed']=MASTER;g4_pipeline.P['purpose']['perturbation']=804;g4_pipeline.P['purpose']['bootstrap']=805
 data,private,stats=perturb(cancer,design,rep,purpose,fixed=purpose!='bootstrap')
 if purpose=='bootstrap' and data is not None:
  # Fixed number of bootstrap slots, preserving patient identities/multiplicities.
  tr=private['train'];xx=np.array(tr['states']);env=np.array(tr['environment']);chosen=np.array(tr['selected']);w=np.array(tr['multiplicity']);neww=w.copy()
  for e in range(2):
   ids=np.flatnonzero((env==e)&chosen);slots=np.repeat(ids,w[ids])
   if len(slots)<40:
    put(path,dict(status='CAPACITY_FAILURE',cancer=cancer,design=design,replicate=rep,purpose=purpose,reason='FEWER_THAN_40_WEIGHTED_ENROLLED_SLOTS'));return path.name,'CAPACITY_FAILURE'
   take=rng(4,810,['LUAD','COAD'].index(cancer),rep,e).choice(slots,40,replace=False);neww[ids]=0;neww+=np.bincount(take,minlength=len(w))
   a=np.array(DESIGNS[design]['measurement'][e]);yy=np.argmax(a[:,xx],axis=0);use=(env==e)&chosen;data['counts'][e]=np.bincount(yy[use],weights=neww[use],minlength=8).tolist();assert sum(data['counts'][e])==40
   stats['train']['by_environment'][e]['selected']=40
  tr['multiplicity']=neww.tolist();stats['train']['n_selected']=80;stats['train']['selected_unique']=int(sum(chosen&(neww>0)));stats['actual_gene_assays']=3*80+4*80;stats['fixedN_unit']='80 weighted bootstrap slots, not80 independent new patients'
 if data is None:put(path,dict(status='CAPACITY_FAILURE',cancer=cancer,design=design,replicate=rep,purpose=purpose));return path.name,'CAPACITY_FAILURE'
 # Validation split uses observed count categories, not hidden outcomes or evaluation records.
 train_data=copy.deepcopy(data);valid_data=copy.deepcopy(data);random=rng(4,806,['LUAD','COAD'].index(cancer),rep,0 if purpose=='perturbation' else 1)
 for field in ['counts','calibration']:
  for e,cts in enumerate(data[field]):
   c=np.array(cts,int);v=random.binomial(c,.2);train_data[field][e]=(c-v).tolist();valid_data[field][e]=v.tolist()
 spec=DESIGNS[design];fm=train(train_data,spec,'M',profile=False);ff,lf=fusion(train_data,spec);lm=Likelihood(train_data,spec)
 xm=np.array(fm['x']);xf=np.array(ff['x']);outs_m=lm.outputs(xm);outs_f=lf.outputs(xf);validlik=Likelihood(valid_data,spec);validouts=validlik.outputs(xm);val=[]
 for w in CFG['g4']['mixing_weights']:
  loss=sum(float(-c@np.log(np.maximum(w*om[1]+(1-w)*of[1],1e-280))) for (c,_,_),om,of in zip(validouts,outs_m,outs_f));val.append(loss)
 chosen=CFG['g4']['mixing_weights'][int(np.argmin(val))];development_ok=fm['success'] and ff['success']
 m=train(data,spec,'M',profile=False);f,_=fusion(data,spec);pooled=train(data,spec,'B1',profile=False);mix=dict(success=bool(m['success'] and f['success'] and development_ok),source=(chosen*np.array(m['source'])+(1-chosen)*np.array(f['source'])).tolist(),selected=(chosen*np.array(m['selected'])+(1-chosen)*np.array(f['selected'])).tolist(),method='TRAIN_ONLY_MIXTURE',estimate=None)
 results={}
 for method,result in [('CTMC_M',m),('UNPROJECTED_FUSION',f),('POOLED',pooled),('TRAIN_ONLY_MIXTURE',mix)]:results[method]=dict(success=result['success'],scores=score(result,private,spec),source=result['source'])
 put(path,dict(status='COMPLETE',cancer=cancer,design=design,replicate=rep,purpose=purpose,stats=stats,chosen_ctmc_weight=chosen,validation_losses=val,validation_fits_success=development_ok,methods=results,public_data=data,validation_split=valid_data,fit_audit={'ctmc':m,'fusion':f},runtime_seconds=time.time()-start,evidence_type='NEW_PERTURBATION_PREVIOUSLY_EXPLORED_PATIENTS',natural_theta_truth=False));return path.name,round(time.time()-start,2)

def generate_direct(cancer,rep):
 from g5_model import X
 rows=readrows(cancer);random=rng(5,807,['LUAD','COAD'].index(cancer),rep);role={key:np.array([z['reference_state'] for z in rows if z['role']==key]) for key in ['train','calibration','evaluation']};envs={};sel={};stats={}
 for key,x in role.items():
  env=random.integers(0,2,len(x));prob=np.exp(-2.2+np.where(env==0,.5*X[x,0],X[x,1]));chosen=random.random(len(x))<prob;envs[key]=env;sel[key]=chosen;stats[key]=dict(n=len(x),selected=int(chosen.sum()))
 types=np.zeros(len(role['calibration']),int)
 for e in range(2):
  ids=np.flatnonzero(envs['calibration']==e);random.shuffle(ids);types[ids[len(ids)//2:]]=1
 out=dict(base=[],add=[],calx=[],cals=[],evaluation=[],source_evaluation=role['evaluation'],baseids=[],futureids=[],calids=[],true_target=None,base_unique=100)
 for e in range(2):
  b=np.flatnonzero((envs['train']==e)&sel['train']);random.shuffle(b)
  if len(b)<50:return dict(capacity_failure=True,reason='BASE_CAPACITY',stats=stats)
  a=np.flatnonzero((envs['calibration']==e)&(types==0)&sel['calibration']);c=np.flatnonzero((envs['calibration']==e)&(types==1));random.shuffle(a);random.shuffle(c)
  out['base'].append(role['train'][b[:50]]);out['add'].append(role['calibration'][a]);out['calx'].append(role['calibration'][c]);out['cals'].append(sel['calibration'][c]);out['evaluation'].append(role['evaluation'][(envs['evaluation']==e)&sel['evaluation']]);out['baseids'].append(b[:50]);out['futureids'].append(a);out['calids'].append(c)
 out['capacities']=dict(add_pairs=min(map(len,out['add'])),calibration_pairs=min(map(len,out['calx'])));out['stats']=stats;return out

def g5_one(task):
 from g5_design_evaluator import baseline_data,reveal,scores
 from g5_design_strategy import recommend,design_spec
 from g5_model import Model,fit,geometry
 cancer,rep=task;path=P/f'g5/revision_r2/runs/{cancer}_{rep:03d}.json'
 if path.exists():return path.name,'CACHED'
 start=time.time();private=generate_direct(cancer,rep)
 if private.get('capacity_failure'):put(path,dict(status='CAPACITY_FAILURE',cancer=cancer,replicate=rep,details=private));return path.name,'CAPACITY_FAILURE'
 base=baseline_data(private);spec=design_spec();bm=Model(base,spec);br=fit(bm);before=scores(bm,br,private);out=[];decisions=[];cache={}
 for cap in CFG['g5']['budget_caps']:
  # Public candidate-pool capacity decides a common feasible cost BEFORE outcomes are revealed.
  budget=min(cap,6*private['capacities']['add_pairs']);budget=6*(budget//6)
  if budget==0:out.append(dict(cap=cap,status='NO_COMMON_POSITIVE_BUDGET'));continue
  decision=recommend(base,budget,private['capacities'],MASTER+rep);decisions.append(dict(cap=cap,budget=budget,decision=decision))
  for strategy,action in decision['choices'].items():
   key=action['candidate_id']
   if key not in cache:
    data,counts=reveal(private,action);lik=Model(data,spec);fitresult=fit(lik);geom=geometry(lik,np.array(fitresult['x']),np.eye(10)[4]);cache[key]=dict(fit=fitresult,geometry=geom,scores=scores(lik,fitresult,private),counts=counts,public_data=data)
   result=cache[key];out.append(dict(cap=cap,budget=budget,strategy=strategy,action=action,result=result))
 put(path,dict(status='COMPLETE',cancer=cancer,replicate=rep,baseline_scores=before,baseline_fit=br,decisions=decisions,results=out,runtime_seconds=time.time()-start,evidence_type='NEW_PERTURBATION_PREVIOUSLY_EXPLORED_PATIENTS',natural_theta_truth=False,capacities=private['capacities'],private_reveal_audit=private));return path.name,round(time.time()-start,2)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('group',choices=['g4','g5']);ap.add_argument('--workers',type=int,default=4);ap.add_argument('--probe',action='store_true');a=ap.parse_args()
 if a.group=='g4':tasks=[('perturbation',c,d,i) for c in CFG['g4']['cancers'] for d in CFG['g4']['designs'] for i in range(CFG['g4']['replicates_per_cell'])]+[('bootstrap',c,'HETEROGENEOUS',i) for c in CFG['g4']['cancers'] for i in range(CFG['g4']['bootstrap_replicates'])];fn=g4_one
 else:tasks=[(c,i) for c in CFG['g5']['cancers'] for i in range(CFG['g5']['replicates'])];fn=g5_one
 if a.probe:tasks=tasks[:1]
 with ProcessPoolExecutor(max_workers=a.workers) as pool:
  for i,f in enumerate(as_completed([pool.submit(fn,t) for t in tasks]),1):print(i,'/',len(tasks),*f.result(),flush=True)
if __name__=='__main__':main()
