"""Private generator + isolated estimator invocation + private scorer; never passes truth to K1."""
from pathlib import Path
import sys,json,copy,hashlib,time,argparse,os
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
R=Path(__file__).resolve().parents[3];G=R/'paper_experiments_v1/g4'
from g4_estimator import train,full_reference,states,snapshot
P=json.loads((G/'configs/PROTOCOL.json').read_text());DESIGNS=json.loads((G/'data_public/contracts/designs.json').read_text());X=states(4)
def put(p,o):p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(o,ensure_ascii=False));tmp.replace(p)
def rng(purpose,cancer,rep,stream=0):return np.random.default_rng(np.random.SeedSequence([P['master_seed'],4,P['purpose'][purpose],['LUAD','COAD'].index(cancer),rep,stream]))
def records(cancer):return json.loads((G/f'data_private/reference_records/{cancer}.json').read_text())
def roledata(rows,role):return np.array([z['reference_state'] for z in rows if z['role']==role],dtype=int)
def perturb(cancer,design,rep,purpose='perturbation',fixed=False):
 rows=records(cancer);truth=json.loads((G/'data_private/selection_truth/operators.json').read_text())[design];spec=DESIGNS[design];data=dict(counts=[],calibration=[],calibration_kind='XS');private={};stats={};weights={}
 for ri,role in enumerate(['train','calibration','evaluation']):
  if purpose=='development' and role=='evaluation':continue
  x=roledata(rows,role);w=np.ones(len(x),int)
  if purpose=='bootstrap':w=rng(purpose,cancer,rep,10+ri).multinomial(len(x),np.ones(len(x))/len(x))
  weights[role]=w;random=rng(purpose,cancer,rep,100+ri);env=random.integers(0,2,len(x));u=random.random(len(x));ss=np.array([np.exp(truth['alpha'][e]+np.array(spec['features'][e])[xx]@truth['phi'][e]) for xx,e in zip(x,env)]);chosen=u<ss
  if role=='calibration':
   # Draw 40 patient slots per environment from the resampled role, without replacement of slots.
   cw=np.zeros(len(x),int)
   for e in range(2):
    pool=np.repeat(np.flatnonzero(env==e),w[env==e]);idx=random.choice(pool,40,replace=False);cw+=np.bincount(idx,minlength=len(x))
   w=cw
  if fixed and role=='train':
   for e in range(2):
    idx=np.flatnonzero((env==e)&chosen&(w>0))
    if len(idx)<P['fixed_enrolled_n_per_environment']:return None,None,dict(capacity_failure=True,n_selected=[int(sum((env==j)&chosen)) for j in range(2)])
    keep=random.choice(idx,P['fixed_enrolled_n_per_environment'],replace=False);chosen[idx]=False;chosen[keep]=True
  private[role]=dict(states=x.tolist(),environment=env.tolist(),selected=chosen.tolist(),multiplicity=w.tolist())
  stats[role]=dict(n_screened=int(w.sum()),n_unique=int(sum(w>0)),n_selected=int(w[chosen].sum()),selected_unique=int(sum(chosen&(w>0))),by_environment=[])
  for e in range(2):
   use=(env==e)&chosen;stats[role]['by_environment'].append(dict(screened=int(w[env==e].sum()),selected=int(w[use].sum())))
   if role=='train':
    y=np.argmax(np.array(spec['measurement'][e])[:,x],axis=0);data['counts'].append(np.bincount(y[use],weights=w[use],minlength=8).tolist())
   if role=='calibration':data['calibration'].append(np.bincount(x[env==e]+16*chosen[env==e],weights=w[env==e],minlength=32).tolist())
 full=np.zeros(16)
 for role in ['train','calibration']:
  xx=roledata(rows,role);full+=np.bincount(xx,weights=weights[role],minlength=16)
 private['full_reference_counts']=full.tolist();stats['budget_cap']=3*len(roledata(rows,'train'))+4*80;stats['actual_gene_assays']=3*stats['train']['n_selected']+4*80;stats['selection_metadata']=80;stats['capacity_failure']=False
 return data,private,stats

def score(result,private,spec):
 if not result['success']:return dict(score_status='FIT_FAILED',hidden_brier=None,hidden_log_score=None,hidden_joint_log_score=None,source_log_score=None,source_tv=None)
 ev=private['evaluation'];x=np.array(ev['states']);w=np.array(ev['multiplicity']);env=np.array(ev['environment']);sel=np.array(ev['selected']);p=np.maximum(np.array(result['source']),1e-280);p/=p.sum();freq=np.bincount(x,weights=w,minlength=16)/w.sum();out=dict(score_status='COMPLETE',source_log_score=float(-freq@np.log(p)),source_tv=float(.5*np.abs(freq-p).sum()),n_source_evaluation=int(w.sum()),n_unique_source_evaluation=int(sum(w>0)));den=0;bs=0.;logs=0.;bins=[];eventrows=[]
 for e in range(2):
  a=np.asarray(spec['measurement'][e]);s=np.array(result['selected'][e]);qy=a@s;post=a*s[None,:]/np.maximum(qy[:,None],1e-280);hidden=np.flatnonzero(1-np.array(spec['masks'][e]));use=(env==e)&sel&(w>0);xx=x[use];ww=w[use];yy=np.argmax(a[:,xx],axis=0);prob=post[yy]@X[:,hidden];target=X[xx][:,hidden];nn=ww.sum()
  b=float(np.sum(ww[:,None]*(prob-target)**2));lp=np.clip(prob,1e-12,1-1e-12);l=float(-np.sum(ww[:,None]*(target*np.log(lp)+(1-target)*np.log1p(-lp))));bs+=b;logs+=l;den+=nn*len(hidden)
  joint=float(-np.sum(ww*np.log(np.maximum(post[yy,xx],1e-280))))
  eventrows.append(dict(environment=e,hidden_indices=hidden.tolist(),n=int(nn),brier=b/nn if nn else None,log_score=l/nn if nn else None,joint_log_score=joint/nn if nn else None))
  for k in range(5):
   mask=(prob[:,0]>=k/5)&(prob[:,0]<(k+1)/5 if k<4 else prob[:,0]<=1);nw=ww[mask].sum();bins.append(dict(environment=e,bin=k,n=int(nw),prediction_sum=float(ww[mask]@prob[mask,0]),event_sum=float(ww[mask]@target[mask,0])))
 out.update(hidden_brier=bs/den if den else None,hidden_log_score=logs/den if den else None,hidden_joint_log_score=sum(z['n']*(z['joint_log_score'] or 0) for z in eventrows)/den if den else None,n_hidden_evaluation=int(den),events=eventrows,calibration_bins=bins)
 return out

def reference_prediction(ref,spec,phi):
 r=copy.deepcopy(ref);r['selected']=[]
 for e in range(2):
  s=np.array(ref['source'])*np.exp(np.array(spec['features'][e])@phi[e]);r['selected'].append((s/s.sum()).tolist())
 return r

def run_one(task):
 cancer,design,rep,purpose,fixed=task;name=f'{purpose}_{cancer}_{design}_{rep:03d}'+('_fixedN' if fixed else '');path=G/f'runs/{purpose}/{name}.json'
 if path.exists():return name,'CACHED'
 start=time.perf_counter();data,private,stats=perturb(cancer,design,rep,purpose,fixed)
 if data is None:put(path,dict(name=name,cancer=cancer,design=design,replicate=rep,purpose=purpose,fixed_n=fixed,stats=stats,status='DESIGN_CAPACITY_FAILURE'));return name,'CAPACITY_FAILURE'
 spec=DESIGNS[design];truth=json.loads((G/'data_private/selection_truth/operators.json').read_text())[design];methods=['M','B1','B2','B3','B4','K0','M_NO_CAL'] if purpose!='bootstrap' else ['M','B1','B3','B4'];out={}
 public=dict(data=data,spec=spec,stats=stats);put(G/f'data_public/estimator_inputs/{name}.json',public);put(G/f'data_private/selection_truth/{name}.json',private)
 for method in methods:
  ms=copy.deepcopy(spec)
  if method=='K0':ms['known_phi']=truth['phi']
  r=train(copy.deepcopy(data),ms,method,profile=purpose=='perturbation' and method in ['M','K0','M_NO_CAL'] and not fixed);out[method]=r
  if purpose!='development':r['scores']=score(r,private,spec)
 out['B5']=dict(alias_of='M',method='B5',success=out['M']['success'],estimate=out['M']['estimate'],scores=out['M'].get('scores'),interval_type='WALD_EXPLORATORY',same_likelihood_same_estimate=True)
 if purpose!='bootstrap':
  refpath=G/f'runs/reference/{cancer}.json'
  if purpose=='development':ref=full_reference(private['full_reference_counts'],profile=False)
  else:ref=json.loads(refpath.read_text())
  ref=reference_prediction(ref,spec,truth['phi']);ref['extra_information']='All train+cal full records AND known selection for selected posterior; not equal-information';out['FULL_RECORD_REFERENCE']=ref
  if purpose!='development':ref['scores']=score(ref,private,spec)
 result=dict(name=name,protocol_version=P['version'],group_id='G4',evidence_type='DIRECT_RECORD_PROXY_REFERENCE',natural_theta_truth_available=False,cancer=cancer,design=design,replicate=rep,purpose=purpose,fixed_n=fixed,stats=stats,methods=out,runtime_seconds=time.perf_counter()-start,status='COMPLETE',cache_key=hashlib.sha256((json.dumps(public,sort_keys=True)+P['data_freeze_sha256']+json.dumps(P,sort_keys=True)).encode()).hexdigest());put(path,result);return name,round(result['runtime_seconds'],2)

def prepare_reference():
 for cancer in ['LUAD','COAD']:
  p=G/f'runs/reference/{cancer}.json'
  if p.exists():continue
  rows=records(cancer);x=[z['reference_state'] for z in rows if z['role']!='evaluation'];r=full_reference(np.bincount(x,minlength=16));r['n_train_calibration']=len(x);r['evaluation_used']=False;put(p,r)

def main():
 a=argparse.ArgumentParser();a.add_argument('--purpose',choices=['development','perturbation','bootstrap'],required=True);a.add_argument('--workers',type=int,default=4);a.add_argument('--limit',type=int);a.add_argument('--fixed-n',action='store_true');args=a.parse_args()
 if args.purpose!='development':
  assert (G/'configs/EVALUATION_FREEZE.json').exists(),'Freeze before evaluation';prepare_reference()
 n=2 if args.purpose=='development' else P['bootstrap_replicates'] if args.purpose=='bootstrap' else P['perturbation_replicates'];n=min(n,args.limit) if args.limit else n;designs=['HETEROGENEOUS','REDUNDANT'] if args.purpose=='bootstrap' else P['designs'];tasks=[(c,d,i,args.purpose,args.fixed_n) for c in ['LUAD','COAD'] for d in designs for i in range(n)]
 with ProcessPoolExecutor(max_workers=args.workers) as ex:
  fs=[ex.submit(run_one,t) for t in tasks]
  for j,f in enumerate(as_completed(fs),1):print(j,'/',len(tasks),*f.result(),flush=True)
if __name__=='__main__':main()
