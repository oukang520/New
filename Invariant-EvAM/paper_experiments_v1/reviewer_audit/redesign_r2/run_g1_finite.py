from pathlib import Path
import sys,json,time
import numpy as np,pandas as pd
from concurrent.futures import ProcessPoolExecutor,as_completed
R=Path(__file__).resolve().parents[3];P=R/'paper_experiments_v1';D=Path(__file__).resolve().parent;sys.path[:0]=[str(P/'g2/src')]
from g2_engine import snapshot,operators,fit,Likelihood
MASTER=2026092402

def one(task):
 draw,N,design=task;out=P/f'g1/revision_r2/runs/finite_{draw}_{N}_{design}.json'
 if out.exists():return json.loads(out.read_text())
 random=np.random.default_rng(np.random.SeedSequence([MASTER,1,808,draw]));region=draw%3;br=[[-4,-3],[-1.8,-.2],[.2,.8]][region];theta=np.r_[random.uniform(*br,3),random.choice([-1.,-.5,-.15,0,.15,.5,1.],2)];pub=json.loads((P/'g2/configs/public_contract.json').read_text());specs=pub['operators'][{'heterogeneous':'strong','duplicated':'weak','missing_joint':'nonidentifying'}[design]];ops=operators(specs);p=snapshot(theta)[0];phis=[[.5],[1.]] if design=='heterogeneous' else [[.5],[.5]] if design=='duplicated' else [[],[]];probs=[]
 for (a,f),phi in zip(ops,phis):q=p*np.exp(f@phi);q/=q.sum();probs.append(a@q)
 nper=N//2 if design!='missing_joint' else 3*N//2;rows=[]
 for rep in range(50):
  gen=np.random.default_rng(np.random.SeedSequence([MASTER,1,809,draw,N,rep]));cts=[gen.multinomial(nper,q) for q in probs];lik=Likelihood(cts,ops);z=np.zeros(lik.dim);z[:3]=-1.;z2=z.copy();z2[:3]=-2.5;z2[3:5]=.15;r=fit(lik,bound=6,starts=[z,z2]);err=float(r['x'][3]-theta[3]);rows.append(dict(draw=draw,regime=['rare','ordinary','frequent'][region],n_standard=N,n_actual=2*nper,measurement_cost=3*N,design=design,replicate=rep,truth=float(theta[3]),estimate=float(r['x'][3]),error=err,abs_error=abs(err),squared_error=err*err,success=r['success'],boundary=r['boundary'],counts=[c.tolist() for c in cts]))
 result={'theta':theta.tolist(),'rows':rows};out.write_text(json.dumps(result));return result
if __name__=='__main__':
 tasks=[(d,n,s) for d in range(12) for n in [400,4000] for s in ['heterogeneous','duplicated','missing_joint']];rows=[]
 with ProcessPoolExecutor(max_workers=2) as pool:
  for i,f in enumerate(as_completed([pool.submit(one,t) for t in tasks]),1):rows.extend(f.result()['rows']);print(i,'/',len(tasks),flush=True)
 df=pd.DataFrame(rows);df.to_csv(P/'g1/revision_r2/results/G1_R2_finite_raw.tsv',sep='\t',index=False);df.groupby(['regime','design','n_standard']).agg(n=('success','size'),success_rate=('success','mean'),median_absolute_error=('abs_error','median'),mse=('squared_error','mean'),boundary_fraction=('boundary','mean')).reset_index().to_csv(P/'g1/revision_r2/results/G1_R2_finite_summary.tsv',sep='\t',index=False)
