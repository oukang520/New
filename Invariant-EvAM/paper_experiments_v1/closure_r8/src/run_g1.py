from pathlib import Path
import sys,json,hashlib,time,argparse
from concurrent.futures import ProcessPoolExecutor
import numpy as np,pandas as pd
from complex_geometry import *
P=Path(__file__).resolve().parents[2];G=P/'g1/revision_r8';SEED=202609288;DS=[3,4,6,8];KINDS=['CHAIN','DENSE_DAG','FULL'];REGIMES=['ordinary','rare','strong']
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def one(t):
 stage,d,kind,regime,rep=t;path=G/('development' if stage=='development' else 'runs')/f'{d}_{kind}_{regime}_{rep:02d}.json'
 if path.exists():return
 start=time.time();rng=np.random.default_rng([SEED,1,0 if stage=='development' else 1,d,KINDS.index(kind),REGIMES.index(regime),rep]);edges=topology(d,kind);base=rng.uniform(-3.8,-2.8,d) if regime=='rare' else rng.uniform(-1.5,-.3,d);mag=.7 if regime=='strong' else .35;theta=np.r_[base,rng.uniform(-mag,mag,len(edges))];p,j,X=snapshot(theta,d,edges);target=d+edges.index((1,0));full=np.ones(d,bool);drop0=full.copy();drop0[0]=False;drop1=full.copy();drop1[1]=False;only0=np.arange(d)==0;only1=np.arange(d)==1
 source=operator(p,j,X,full);a=operator(p,j,X,full,0,.7);b=operator(p,j,X,full,1,-.7);p2,j2,_=snapshot(theta,d,edges,.5)
 designs={'SOURCE_FULL':([source],[10000/d]),'DUPLICATE':([a,a],[5000/d]*2),'COMPLEMENTARY':([a,b],[5000/d]*2),'MASK_COMPLEMENTARY':([operator(p,j,X,drop0),operator(p,j,X,drop1)],[5000/(d-1)]*2),'NO_JOINT_VIEW':([operator(p,j,X,only0),operator(p,j,X,only1)],[5000]*2),'NEW_TIME':([source,operator(p2,j2,X,full)],[5000/d]*2)}
 rows=[]
 for design,(out,n) in designs.items():rows.append(dict(d=d,topology=kind,regime=regime,replicate=rep,design=design,measurement_budget=10000,**geometry(out,n,target)))
 # Identical copies at equal total cost exactly match one operator's information matrix.
 single=geometry([a],[10000/d],target);dup=next(z for z in rows if z['design']=='DUPLICATE');assert single['rank']==dup['rank'];assert np.isclose(single['target_information'],dup['target_information'],rtol=1e-6,atol=1e-8)
 path.write_text(json.dumps(dict(stage=stage,theta=theta.tolist(),edges=edges,results=rows,runtime=time.time()-start)))
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['development','evaluation']);ap.add_argument('--workers',type=int,default=3);args=ap.parse_args();code={p.name:sha(p) for p in [Path(__file__),Path(__file__).with_name('complex_geometry.py')]}
 if args.stage=='evaluation':
  f=json.loads((G/'configs/FREEZE.json').read_text());assert code==f['code'];n=10
 else:n=1
 tasks=[(args.stage,d,k,r,i) for d in DS for k in KINDS for r in REGIMES for i in range(n)]
 with ProcessPoolExecutor(args.workers) as ex:
  for i,_ in enumerate(ex.map(one,tasks),1):
   if i%12==0:print(args.stage,i,len(tasks),flush=True)
 files=sorted((G/('development' if args.stage=='development' else 'runs')).glob('*.json'));rows=[r for p in files for r in json.loads(p.read_text())['results']];df=pd.DataFrame(rows);df.to_csv(G/f'results/{args.stage}_geometry.tsv',sep='\t',index=False)
 if args.stage=='development':(G/'configs/FREEZE.json').write_text(json.dumps(dict(seed=SEED,code=code,dimensions=DS,topologies=KINDS,regimes=REGIMES,evaluation_draws=10,scope='regular-local population geometry, not global uniqueness or finite-sample coverage',all_designs_retained=True),indent=2))
 else:df.groupby(['d','topology','regime','design']).agg(n=('replicate','size'),identified=('target_identified','mean'),rank=('rank','mean'),dimension=('dimension','first'),information=('target_information','median'),stable=('rank_tolerance_stable','mean')).reset_index().to_csv(G/'results/summary.tsv',sep='\t',index=False)
