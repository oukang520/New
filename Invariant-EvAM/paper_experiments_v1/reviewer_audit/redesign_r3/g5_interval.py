"""Independent calibration of post-acquisition LR cutoff under fixed simulator family.

Not a distribution-free guarantee; all misspecification cells stay in validation.
"""
from pathlib import Path
import sys,json,copy,time,argparse,hashlib
from concurrent.futures import ProcessPoolExecutor
import numpy as np,pandas as pd
from scipy.optimize import minimize
from scipy.stats import chi2
P=Path(__file__).resolve().parents[2];D=Path(__file__).resolve().parent;G=P/'g5/revision_r3'
sys.path.insert(0,str(P/'g5/src'))
import g5_design_evaluator as evaluator
from g5_model import Model,fit,geometry
from g5_design_strategy import design_spec,information_for,target_metrics
from lbfgsb_direct import minimize_direct
C=json.loads((D/'PROTOCOL.json').read_text());BUDGET=74

def choose(base,br):
    lik=Model(base,design_spec());units=information_for(lik,np.array(br['x']))
    actions=[]
    for a,b,c in [(0,37,0),(12,1,0),(0,2,7)]:
        info=sum(50*units[e] for e in range(2))+a*(units[0]+units[1])+b*(units[2]+units[3]-units[0]-units[1])+c*(units[4]+units[5])
        score=target_metrics(info)
        actions.append(dict(candidate_id=f'A{a}_R{b}_C{c}',add_pairs=a,reveal_pairs=b,calibration_pairs=c,cost=6*a+2*b+10*c,**score))
    return min(actions,key=lambda z:(z['null_leakage']>=1e-7,z['variance'] if np.isfinite(z['variance']) else z['null_leakage'],z['candidate_id'])),actions

def constrained(lik,r,t):
    bounds=list(lik.bounds);bounds[4]=(t,t);start=np.array(r['x']);start[4]=t
    o=minimize_direct(lik.objective,start,bounds,maxiter=650,ftol=1e-10,gtol=1e-6,maxls=40)
    return dict(nll=float(o.fun),success=bool(o.success),x=o.x.tolist())

def endpoints(lik,r,cut):
    x=np.array(r['x']);geom=geometry(lik,x,np.eye(10)[4]);base=dict(estimate=x[4],identified=geom['null_leakage']<1e-7)
    if not r['success']:return dict(**base,lo=-np.inf,hi=np.inf,status='FIT_FAILED')
    if not base['identified']:return dict(**base,lo=-np.inf,hi=np.inf,status='UNRESOLVED')
    ends=[];ok=[];cap=r['nll']+cut/2
    for s in [1.,-1.]:
        grad=np.zeros(lik.dim);grad[4]=s
        o=minimize(lambda z:s*z[4],x,jac=lambda z:grad,bounds=lik.bounds,constraints=[dict(type='ineq',fun=lambda z:cap-lik.objective(z)[0],jac=lambda z:-lik.objective(z)[1])],method='SLSQP',options=dict(maxiter=300,ftol=1e-8))
        feasible=o.success and lik.objective(o.x)[0]<=cap+1e-5;ok.append(bool(feasible));v=float(o.x[4])
        if (s==1 and v<=lik.bounds[4][0]+.001):v=-np.inf
        if (s==-1 and v>=lik.bounds[4][1]-.001):v=np.inf
        ends.append(v)
    if not all(ok):return dict(**base,lo=-np.inf,hi=np.inf,status='PROFILE_FAILED')
    return dict(**base,lo=ends[0],hi=ends[1],status='COMPLETE')

def one(task):
    stage,scenario,rep=task;path=G/f'runs/interval_{stage}_{scenario}_{rep:03d}.json'
    if path.exists():return path.name,'CACHED'
    start=time.time();evaluator.MASTER=C['seed'];private=evaluator.generate_sim(scenario,rep,950 if stage=='development' else 951)
    base=evaluator.baseline_data(private);lik=Model(base,design_spec());br=fit(lik);action,actions=choose(base,br)
    data,cost=evaluator.reveal(private,action);lik=Model(data,design_spec());r=fit(lik);truth=private['true_target'];ct=constrained(lik,r,truth)
    # A constrained solution better than the original fit reveals a numerical
    # issue. Do not clip such LR values into successful outcomes.
    lr=2*(ct['nll']-r['nll']);valid=r['success'] and ct['success'] and lr>=-1e-5
    result=dict(stage=stage,scenario=scenario,replicate=rep,action=action,candidates=actions,cost=cost,truth=truth,truth_parameters=private['truth'],fit=r,constrained_true=ct,lr_true=lr,valid=bool(valid),public_data=data)
    if stage=='evaluation':
        freeze=json.loads((G/'configs/INTERVAL_METHOD_FREEZE.json').read_text())
        result['intervals']={}
        for name,cut in [('CLASSIC',float(chi2.ppf(.95,1))),('CALIBRATED',freeze['cutoff'])]:
            pr=endpoints(lik,r,cut);pr.update(cutoff=cut,covered=bool(pr['lo']<=truth<=pr['hi']),called=bool(pr['lo']>0 or pr['hi']<0),wrong=bool((pr['lo']>0 and truth<=0) or (pr['hi']<0 and truth>=0)),width=pr['hi']-pr['lo'])
            result['intervals'][name]=pr
    result['runtime']=time.time()-start
    path.write_text(json.dumps(result,default=lambda x:x.item() if isinstance(x,np.generic) else x.tolist()))
    return path.name,round(result['runtime'],2)

def finish(stage):
    files=sorted((G/'runs').glob('interval_'+stage+'_*.json'));raw=[json.loads(f.read_text()) for f in files]
    if stage=='development':
        audits=[]
        for s in range(3):
            rr=[z for z in raw if z['scenario']==s];v=[z['lr_true'] for z in rr if z['valid']]
            audits.append(dict(scenario=s,total=len(rr),valid=len(v),q975=float(np.quantile(v,.975,method='higher'))))
        cutoff=max(float(chi2.ppf(.95,1)),max(z['q975'] for z in audits))
        out=dict(cutoff=cutoff,development=audits,budget=BUDGET,selection='Common maximum 97.5% development LR quantile across three prespecified scenarios; not tuned on final evaluation',code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        (G/'configs/INTERVAL_METHOD_FREEZE.json').write_text(json.dumps(out,indent=2));print('INTERVAL FREEZE',out,flush=True)
    else:
        out=[]
        for z in raw:
            for method,pr in z['intervals'].items():out.append(dict(scenario=z['scenario'],replicate=z['replicate'],method=method,truth=z['truth'],cost=z['cost']['cost'],action=z['action']['candidate_id'],fit_valid=z['valid'],lr_true=z['lr_true'],**pr))
        d=pd.DataFrame(out);d['finite']=np.isfinite(d.width);d.to_csv(G/'results/interval_evaluation_raw.tsv',sep='\t',index=False)
        d.groupby(['scenario','method']).agg(n=('replicate','size'),coverage=('covered','mean'),called=('called','mean'),wrong=('wrong','mean'),finite=('finite','mean'),fit_valid=('fit_valid','mean'),median_width=('width','median')).reset_index().to_csv(G/'results/interval_evaluation_summary.tsv',sep='\t',index=False)

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('stage',choices=['development','evaluation']);a.add_argument('--workers',type=int,default=4);a.add_argument('--probe',action='store_true');args=a.parse_args()
    n=C['development_replicates'] if args.stage=='development' else C['evaluation_replicates'];tasks=[(args.stage,s,r) for s in range(3) for r in range(n)]
    if args.probe:tasks=tasks[:1]
    with ProcessPoolExecutor(args.workers) as ex:
        for i,r in enumerate(ex.map(one,tasks),1):print(i,len(tasks),r,flush=True)
    if not args.probe:finish(args.stage)
