"""Paid random bridge measurements on real records; estimator accepts counts only."""
from pathlib import Path
import sys,json,copy,hashlib,argparse,time
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd
from scipy.special import expit,logit
from scipy.optimize import minimize
G=Path(__file__).resolve().parents[1];P=G.parents[1]
sys.path.insert(0,str(P/'g4/src'))
from g4_estimator import Likelihood,fit,selected,states
import g4_pipeline as gp
X=states(4);CANCERS=['LUAD','COAD'];DESIGNS=list(gp.DESIGNS)
SEED=2026092604
def put(path,x):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(x,ensure_ascii=False,indent=2))
class BridgeLikelihood(Likelihood):
    def outputs(self,x):
        out=super().outputs(x).copy()
        for e,c in enumerate(self.data['bridge']):
            p,J=self.source(x,e);f=np.asarray(self.spec['features'][e]);phi=np.asarray(self.spec['known_phi'][e]) if self.phi[e] is None else x[self.phi[e]]
            q,j,jp=selected(p,J,f,phi,np.eye(16))
            if self.phi[e] is not None:j[:,self.phi[e]]+=jp
            out.append((np.asarray(c),q,j))
        return out
def estimate(data,spec,method):
    data=copy.deepcopy(data);spec=copy.deepcopy(spec)
    if method=='POOLED':
        spec['known_phi']=[[0.],[0.]];spec['features']=[np.zeros((16,1)).tolist()]*2
        data['calibration']=[(np.array(c[:16])+c[16:]).tolist() for c in data['calibration']];data['calibration_kind']='X'
    mode={'CTMC':'joint','POOLED':'joint','HIERARCHICAL':'hierarchical','FUSION':'fusion'}[method]
    m=BridgeLikelihood(data,spec,mode);r=fit(m);z=np.array(r['x']);src=[];sel=[]
    for e in range(2):
        p,_=m.source(z,e);src.append(p)
        phi=np.array(spec['known_phi'][e]) if m.phi[e] is None else z[m.phi[e]]
        q=p*np.exp(np.array(spec['features'][e])@phi);sel.append((q/q.sum()).tolist())
    return dict(success=bool(r['success']),source=src[0].tolist(),selected=sel,x=r['x'],nll=r['nll'])
def pred(r,spec):
    out=[]
    for e in range(2):
        A=np.array(spec['measurement'][e]);q=np.array(r['selected'][e]);h=np.flatnonzero(1-np.array(spec['masks'][e]))[0]
        out.append(((A@(q*X[:,h]))/np.maximum(A@q,1e-12)).tolist())
    return np.array(out)
def adjust(base,bridge,spec,method,lam):
    """Audited labels only. No evaluation labels, truth selection or private records."""
    out=[]
    for e in range(2):
        A=np.array(spec['measurement'][e]);c=np.array(bridge[e]);h=np.flatnonzero(1-np.array(spec['masks'][e]))[0];n=A@c;y=A@(c*X[:,h]);b=np.clip(base[e],1e-5,1-1e-5)
        if method=='CELL':p=(y+lam*b)/(n+lam)
        elif method=='DIRECT':p=(y+lam*.5)/(n+lam)
        elif method=='MARGINAL':p=np.full(8,(y.sum()+.5)/(n.sum()+1))
        else:
            z=logit(b)
            def obj(v):
                eta=z+v[0];p=expit(eta)
                return float(np.sum(n*np.logaddexp(0,eta)-y*eta)+lam*v[0]**2/2),np.array([np.sum(n*p-y)+lam*v[0]])
            rr=minimize(obj,[0.],jac=True,method='L-BFGS-B');p=expit(z+rr.x[0])
        out.append(p.tolist())
    return np.array(out)
def generate(cancer,design,rep,stage):
    spec=gp.DESIGNS[design];truth=json.loads((P/'g4/data_private/selection_truth/operators.json').read_text())[design]
    rng=np.random.default_rng(np.random.SeedSequence([SEED,{'development':1,'evaluation':2,'bootstrap':3}[stage],CANCERS.index(cancer),rep]))
    rows=gp.records(cancer);roles={k:gp.roledata(rows,k) for k in ['train','calibration','evaluation']}
    # Development scoring uses held train patients, never the final evaluation role.
    dev=np.array([int(hashlib.sha256(r['patient_key'].encode()).hexdigest()[:8],16)%2==0 for r in rows if r['role']=='calibration'])
    if stage=='development':roles['evaluation']=roles['calibration'][dev];roles['calibration']=roles['calibration'][~dev]
    if stage=='bootstrap':roles={k:rng.choice(v,len(v),replace=True) for k,v in roles.items()}
    data=dict(counts=[],calibration=[],calibration_kind='XS',bridge=[]);more=copy.deepcopy(data);bridge=copy.deepcopy(data);ev={};stats={}
    for role,x in roles.items():
        env=rng.integers(0,2,len(x));s=np.array([np.exp(truth['alpha'][e]+np.array(spec['features'][e])[xx]@truth['phi'][e]) for xx,e in zip(x,env)]);sel=rng.random(len(x))<s
        if role=='evaluation':ev=dict(states=x.tolist(),environment=env.tolist(),selected=sel.tolist(),multiplicity=np.ones(len(x),int).tolist());continue
        for e in range(2):
            A=np.array(spec['measurement'][e]);yi=np.argmax(A,axis=0)
            if role=='calibration':
                ids=np.flatnonzero(env==e)
                if len(ids)<40:return None,None,dict(failure='CAL_CAPACITY')
                ids=rng.choice(ids,40,replace=False);cts=np.bincount(x[ids]+16*sel[ids],minlength=32).tolist()
                for d in [data,more,bridge]:d['calibration'].append(cts)
            else:
                ids=rng.permutation(np.flatnonzero((env==e)&sel));stats[str(e)]=len(ids)
                if len(ids)<53:return None,None,dict(failure='TRAIN_CAPACITY',available=stats)
                initial=ids[:40];extra=ids[40:53];audit=initial[:39];rest=initial[39:]
                data['counts'].append(np.bincount(yi[x[initial]],minlength=8).tolist());data['bridge'].append([0]*16)
                more['counts'].append(np.bincount(yi[x[np.r_[initial,extra]]],minlength=8).tolist());more['bridge'].append([0]*16)
                bridge['counts'].append(np.bincount(yi[x[rest]],minlength=8).tolist());bridge['bridge'].append(np.bincount(x[audit],minlength=16).tolist())
    return {'BASE':data,'MORE':more,'BRIDGE':bridge},ev,dict(train_selected_available=stats,base_assays=560,extra_assays=78,total_assays=638,n_base=80,n_more=106,n_bridge=80,audited=78)
def score(p,source,ev,spec):
    x=np.array(ev['states']);env=np.array(ev['environment']);sel=np.array(ev['selected']);pr=[];ys=[];idx=[]
    for e in range(2):
        ids=np.flatnonzero((env==e)&sel);yi=np.argmax(np.array(spec['measurement'][e]),axis=0);h=np.flatnonzero(1-np.array(spec['masks'][e]))[0]
        pr.extend(p[e,yi[x[ids]]]);ys.extend(X[x[ids],h]);idx.extend(ids)
    pr=np.clip(np.array(pr),1e-8,1-1e-8);ys=np.array(ys);ece=0.
    for j in range(5):
        use=(pr>=j/5)&(pr<(j+1)/5)
        if use.any():ece+=abs((pr[use]-ys[use]).sum())/len(pr)
    return dict(brier=float(np.mean((pr-ys)**2)),log_score=float(-np.mean(ys*np.log(pr)+(1-ys)*np.log1p(-pr))),ece=float(ece),bias=float(np.mean(pr-ys)),source_log_score=float(-np.mean(np.log(np.maximum(np.array(source)[x],1e-12)))),n=len(pr)),dict(index=list(map(int,idx)),p=pr.tolist(),y=ys.tolist())
def one(task):
    stage,cancer,design,rep=task;path=G/f'runs/{stage}_{cancer}_{design}_{rep:03d}.json'
    if path.exists():return path.name,'CACHED'
    start=time.time();ds,ev,stats=generate(cancer,design,rep,stage)
    if ds is None:put(path,dict(status='CAPACITY_FAILURE',stage=stage,cancer=cancer,design=design,replicate=rep,stats=stats));return path.name,'CAPACITY_FAILURE'
    spec=gp.DESIGNS[design];out={};base=None
    for arm in ds:
        for m in ['CTMC','POOLED','HIERARCHICAL','FUSION']:
            r=estimate(ds[arm],spec,m);p=pred(r,spec);s,patient=score(p,r['source'],ev,spec)
            out[arm+'_'+m]=dict(fit=r,probabilities=p.tolist(),scores=s,patient=patient)
            if arm=='BASE' and m=='CTMC':base=r;basep=p
    candidates=[('CELL',x) for x in [1.,4.,16.,64.]]+[('OFFSET',x) for x in [1.,4.,16.]]+[('DIRECT',x) for x in [1.,4.,16.]]+[('MARGINAL',1.)]
    if stage!='development':
        f=json.loads((G/'configs/EVALUATION_FREEZE.json').read_text());candidates=[tuple(x) for x in f['calibration_candidates']]
    for m,lam in candidates:
        p=adjust(basep,ds['BRIDGE']['bridge'],spec,m,lam);s,patient=score(p,base['source'],ev,spec)
        out[f'AUDIT_{m}_{lam:g}']=dict(probabilities=p.tolist(),scores=s,patient=patient)
    put(path,dict(status='COMPLETE',stage=stage,cancer=cancer,design=design,replicate=rep,stats=stats,public_inputs=ds,methods=out,runtime=time.time()-start))
    return path.name,round(time.time()-start,1)
def summarize(stage):
    rows=[]
    for f in sorted((G/'runs').glob(stage+'_*.json')):
        z=json.loads(f.read_text())
        for m,r in z.get('methods',{}).items():rows.append(dict(stage=stage,cancer=z['cancer'],design=z['design'],replicate=z['replicate'],method=m,success=r.get('fit',{}).get('success',True),**r['scores']))
    d=pd.DataFrame(rows);d.to_csv(G/f'results/{stage}_raw.tsv',sep='\t',index=False);d.groupby(['cancer','design','method'])[['brier','log_score','ece','bias','source_log_score','success']].mean().to_csv(G/f'results/{stage}_summary.tsv',sep='\t')
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('stage',choices=['development','evaluation','bootstrap']);a.add_argument('--workers',type=int,default=4);a.add_argument('--n',type=int);args=a.parse_args()
    if args.stage!='development':assert (G/'configs/EVALUATION_FREEZE.json').exists()
    n=args.n or {'development':10,'evaluation':50,'bootstrap':200}[args.stage]
    tasks=[(args.stage,c,d,r) for c in CANCERS for d in DESIGNS for r in range(n)]
    with ProcessPoolExecutor(args.workers) as ex:
        for i,v in enumerate(ex.map(one,tasks),1):print(i,len(tasks),v,flush=True)
    summarize(args.stage)
