"""Information-isolated local alignment: truth determines DGM direction only, never estimator feature values."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
from pathlib import Path
import json,time,hashlib,sys
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
from g3_model import Joint,inference,states,snapshot
from g3_design import G,R,theta,cells

def public_arms():
 x=states(4);f=np.column_stack([x[:,i] for i in range(4)]+[x[:,i]*x[:,j] for i in range(4) for j in range(i+1,4)]);base=dict(d=4,features=[x[:,[0]].tolist(),f.tolist()],measurement=[np.eye(16).tolist()]*2,epsilon=0)
 flex={**base,'features':[np.column_stack([x[:,0],x[:,0]*x[:,1]]).tolist(),f.tolist()]}
 return [dict(name='SHARED_K1',spec=base),dict(name='BOUNDED_BIOLOGY',spec={**base,'epsilon':.5}),dict(name='FLEXIBLE_SELECTION',spec=flex),dict(name='ORACLE_STRATIFIED',spec={**base,'epsilon':3.,'known_phi':[[.5],[.5]+[0.]*9]})]

def generator_direction():
 t=theta(4);p,j=snapshot(t,4);f=np.array(public_arms()[0]['spec']['features'][1]);w=np.sqrt(p);a=j/w[:,None];b=w[:,None]*(f-p@f);u,sv,_=np.linalg.svd(b,full_matrices=False);nr=sum(sv>sv[0]*1e-8);projected=a-u[:,:nr]@(u[:,:nr].T@a);_,s,vh=np.linalg.svd(projected,full_matrices=True);rank=sum(s>s[0]*1e-8);null=vh[rank:].T;v=null@(null.T@np.eye(10)[4]);v/=max(abs(v));return v

def run(rep,purpose=305):
 folder='alignment_revision' if purpose==305 else 'alignment_development';(G/'runs'/folder).mkdir(exist_ok=True);path=G/'runs'/folder/f'cell04_rep{rep:04d}.json'
 if path.exists():return json.loads(path.read_text())['runtime_seconds']
 start=time.time();t=theta(4);v=generator_direction();pop=[snapshot(t,4)[0],snapshot(t+.5*v,4)[0]];sp=public_arms()[0]['spec'];phis=[np.array([.5]),np.r_[.5,np.zeros(9)]];prob=[]
 for p,f,phi in zip(pop,sp['features'],phis):w=np.exp(np.asarray(f)@phi);prob.append(p*w/(p@w))
 rng=np.random.default_rng(np.random.SeedSequence([20260923,3,purpose,4,0,rep]));data=dict(counts=[rng.multinomial(500,p).tolist() for p in prob],calibration=None,calibration_kind=None);out=[]
 for arm in public_arms():
  begin=time.time();lik=Joint(data,arm['spec']);r=inference(lik);r['prediction_KL']=float(np.mean([sum(p*(np.log(p)-np.log(np.maximum(z[1],1e-280)))) for p,z in zip(prob,lik.outputs(np.array(r['x'])))]));defined=r['status']=='COMPLETE';called=r['sign'] in ['POSITIVE','NEGATIVE'];correct=called and r['sign']=='NEGATIVE';r.update(coverage_defined=defined,covers_truth=bool(r['lo']<=t[4]<=r['hi']) if defined else None,called=called,correct_call=correct,wrong_call=called and not correct,nuisance_rmse=None)
  meta=dict(target=float(t[4]),n_observed=1000,n_calibration=0,n_budget_used=1000,n_screened_calibration=None,population_selected=[p.tolist() for p in prob],phi_true=[p.tolist() for p in phis],raw_subgroup_counts=[],seed_sequence=[20260923,3,purpose,4,0,rep])
  out.append(dict(method=arm['name'],spec=arm['spec'],data=data,evaluator=meta,result=r,runtime_seconds=time.time()-begin))
 obj=dict(protocol='G3_1.1_ALIGNMENT_INFORMATION_ISOLATED',evidence_type='NEW_RUN' if purpose==305 else 'DEVELOPMENT_ONLY',cell=cells()[4],replicate=rep,purpose=purpose,arms=out,runtime_seconds=time.time()-start);tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(obj));tmp.replace(path);return obj['runtime_seconds']

def main():
 config=G/'configs/LOCAL_ALIGNMENT_REVISION.json'
 if not config.exists():
  runtime=run(0,306);obj=dict(status='FROZEN',reason='Remove truth-derived score values from estimator information. Fixed single-event and pairwise monomial basis is public; only DGM direction uses truth geometry.',evaluation_purpose=305,development_purpose=306,n_replicates=200,supersedes='core/cell04 only for primary inferential evaluation; previous score-feature records preserved as truth-informed geometry reference',public_arms=public_arms(),development_runtime_seconds=runtime,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest());config.write_text(json.dumps(obj,indent=2));print('New alignment specification frozen after independent development',runtime,flush=True)
 start=time.time()
 with ProcessPoolExecutor(max_workers=2) as ex:
  fs=[ex.submit(run,r) for r in range(200)]
  for i,f in enumerate(as_completed(fs),1):
   f.result()
   if i%25==0:
    status=dict(run_status='COMPLETE' if i==200 else 'RUNNING',completed=i,total=200,elapsed_seconds=time.time()-start);(G/'runs/alignment_STATUS.json').write_text(json.dumps(status,indent=2));print(status,flush=True)
if __name__=='__main__':main()
