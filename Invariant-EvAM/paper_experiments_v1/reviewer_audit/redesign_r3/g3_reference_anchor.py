"""Reference-target calibration in the ORIGINAL generator-scale sensitivity model."""
from pathlib import Path
import sys,json,copy,time,argparse,hashlib
from concurrent.futures import ProcessPoolExecutor
import numpy as np,pandas as pd
P=Path(__file__).resolve().parents[2];D=Path(__file__).resolve().parent;G=P/'g3/revision_r3'
sys.path.insert(0,str(P/'g3/src'))
from g3_design import cells,specifications
from g3_generator import distributions
from g3_model import Joint,inference
C=json.loads((D/'PROTOCOL.json').read_text())
BUDGET=4000
def one(task):
    stage,cellid,rep=task;path=G/f'runs/anchor_{stage}_{cellid}_{rep:03d}.json'
    if path.exists():return path.name,'CACHED'
    start=time.time();cell=cells()[cellid];truth=distributions(cell);_,spec=specifications(cell);spec={**spec,'epsilon':.5};n=8
    rng=np.random.default_rng(np.random.SeedSequence([C['seed'],3,960 if stage=='development' else 961,cellid,rep]));obs=[];allobs=[];cal=[]
    for e in range(2):
        p=truth['pops'][e];w=np.exp(truth['features'][e]@truth['phis'][e]);q=p*w/(p@w)
        # Prefixes provide paired data while every arm has exactly cost 4000.
        xx=rng.choice(n,2000,p=q);obs.append(np.bincount(xx[:1000],minlength=n).tolist());allobs.append(np.bincount(xx,minlength=n).tolist());cal.append(rng.choice(n,1000,p=p))
    reference=[np.bincount(cal[0],minlength=n).tolist(),[0]*n]
    balanced=[np.bincount(cal[e][:500],minlength=n).tolist() for e in range(2)]
    datasets={'NO_ANCHOR':dict(counts=allobs,calibration=None,calibration_kind=None),'BALANCED_X':dict(counts=obs,calibration=balanced,calibration_kind='X'),'REFERENCE_X':dict(counts=obs,calibration=reference,calibration_kind='X')}
    out={}
    for method,data in datasets.items():
        r=inference(Joint(data,spec));valid=r['status']=='COMPLETE';lo=r.get('lo');hi=r.get('hi');target=truth['target']
        out[method]=dict(result=r,data=data,covers=bool(lo<=target<=hi) if valid else None,called=bool(valid and (lo>0 or hi<0)),wrong=bool(valid and ((lo>0 and target<=0) or (hi<0 and target>=0))),finite=bool(valid and np.isfinite([lo,hi]).all()),width=hi-lo if valid else None)
    path.write_text(json.dumps(dict(cell=cellid,replicate=rep,stage=stage,budget=BUDGET,target=truth['target'],epsilon_log_rate=.5,eta_selection_coefficient=1,methods=out,runtime=time.time()-start)))
    return path.name,round(time.time()-start,2)

def summarize(stage):
    rows=[]
    for f in sorted((G/'runs').glob('anchor_'+stage+'_*.json')):
        z=json.loads(f.read_text())
        for method,r in z['methods'].items():rows.append(dict(cell=z['cell'],replicate=z['replicate'],method=method,budget=z['budget'],truth=z['target'],status=r['result']['status'],**{k:v for k,v in r.items() if k not in ['data','result']}))
    d=pd.DataFrame(rows);d.to_csv(G/f'results/anchor_{stage}_raw.tsv',sep='\t',index=False)
    d.groupby(['cell','method']).agg(n=('replicate','size'),valid=('covers','count'),coverage=('covers','mean'),called=('called','mean'),wrong=('wrong','mean'),finite=('finite','mean'),median_width=('width','median')).reset_index().to_csv(G/f'results/anchor_{stage}_summary.tsv',sep='\t',index=False)

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('stage',choices=['development','evaluation']);a.add_argument('--workers',type=int,default=2);a.add_argument('--probe',action='store_true');args=a.parse_args()
    freeze=G/'configs/REFERENCE_ANCHOR_FREEZE.json'
    if not freeze.exists():freeze.write_text(json.dumps(dict(budget=BUDGET,cells=[3,10],methods=['NO_ANCHOR','BALANCED_X','REFERENCE_X'],evaluation_replicates=500,development_replicates=20,scope='Original reference interaction log-rate, epsilon .5, declared pair-selection coefficient bound 1; fixed extra-budget experiment, not same budget as old1000',code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),indent=2))
    n=20 if args.stage=='development' else 500;tasks=[(args.stage,c,r) for c in [3,10] for r in range(n)]
    if args.probe:tasks=tasks[:1]
    with ProcessPoolExecutor(args.workers) as ex:
        for i,r in enumerate(ex.map(one,tasks),1):print(i,len(tasks),r,flush=True)
    summarize(args.stage)
