"""Independent confirmation after the classical solver's numerical repair."""
from pathlib import Path
import json,hashlib
from concurrent.futures import ProcessPoolExecutor
import numpy as np,pandas as pd
import run_g2 as m
import refine_profile_v2 as ref
from exact_target import confidence
OUT=m.G/'fresh_confirmation';(OUT/'runs').mkdir(parents=True,exist_ok=True);ref.D=OUT/'refined';ref.D.mkdir(exist_ok=True)
def one(t):
 des,effect,rep=t;path=OUT/'runs'/f'{des}_16000_{effect}_{rep:03d}.json'
 if not path.exists():
  rng=np.random.default_rng([m.SEED,2,3,m.DES.index(des),m.EFF.index(effect),rep]);theta=np.r_[m.EV['baseline'],effect,m.EV['other_interaction']];p,j=m.snapshot(theta);ops=m.operators(m.PUB['operators'][des]);cts=[];groups=[]
  for k,((A,F),phi) in enumerate(zip(ops,m.EV['true_phis'][des])):
   q=p*np.exp(F@np.array(phi));q/=q.sum();c=rng.multinomial(8000,A@q);cts.append(c.tolist());groups.append(dict(kind='SELECT_A' if k==0 or des=='weak' else 'SELECT_B',counts=c.reshape(4,2).sum(axis=1).tolist()))
  z=confidence(groups);z['covered']=bool(z['status']=='COMPLETE' and z['lo']<=effect<=z['hi']);z['correct']=bool((z['lo']>0 and effect>0)or(z['hi']<0 and effect<0));z['wrong']=bool((z['lo']>0 and effect<=0)or(z['hi']<0 and effect>=0));path.write_text(json.dumps(dict(design=des,n=16000,truth=effect,replicate=rep,full_counts=cts,groups=groups,exact=z)))
 ref.one(path)
if __name__=='__main__':
 f=m.G/'configs/FRESH_CONFIRMATION_FREEZE.json';h=hashlib.sha256(Path(__file__).read_bytes()).hexdigest();rh=hashlib.sha256(Path(ref.__file__).read_bytes()).hexdigest()
 if f.exists():assert json.loads(f.read_text())['code']==h and json.loads(f.read_text())['solver_code']==rh
 else:f.write_text(json.dumps(dict(code=h,solver_code=rh,designs=['strong','weak'],n=16000,effects=m.EFF,replicates_per_cell=25,seed_namespace=3,reason='fresh confirmation after fixed numerical repair; preserve original 1800 evaluation records'),indent=2))
 with ProcessPoolExecutor(2) as pool:
  for i,_ in enumerate(pool.map(one,[(d,e,r) for d in ['strong','weak'] for e in m.EFF for r in range(25)]),1):
   if i%20==0:print(i,200,flush=True)
 rows=[]
 for p in (OUT/'runs').glob('*.json'):
  z=json.loads(p.read_text());rows.append(dict(design=z['design'],truth=z['truth'],replicate=z['replicate'],method='EXACT_CONDITIONAL_OUTER',**z['exact']));rows.append(json.loads((ref.D/p.name).read_text()))
 d=pd.DataFrame(rows);d.to_csv(OUT/'results.tsv',sep='\t',index=False);print(d.groupby(['design','method']).agg(n=('replicate','size'),coverage=('covered','mean'),finite=('finite','mean'),correct=('correct','mean'),wrong=('wrong','mean')).to_string())
