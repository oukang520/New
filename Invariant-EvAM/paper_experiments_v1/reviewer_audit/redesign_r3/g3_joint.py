"""Two-phase source estimation; generator truths are supplied only to the scorer."""
from pathlib import Path
import sys,json,argparse,hashlib
import numpy as np
import pandas as pd
P=Path(__file__).resolve().parents[2];D=Path(__file__).resolve().parent
sys.path.insert(0,str(D.parent/'redesign_r2'))
from run_simulations import snapshot,exact_sign,wilson
C=json.loads((D/'PROTOCOL.json').read_text());G=P/'g3/revision_r3'
SIGN=np.array([1.,-1.,-1.,1.])

def lor(p):return float(SIGN@np.log(p))
def estimate(obs,xs):
    # p = Pr(S=1) p(X|S=1) + Pr(S=0) p(X|S=0).
    # Fixed Jeffreys smoothing is part of the estimator, not a data filter.
    xs=np.asarray(xs,float);n=xs.sum();k=xs[1].sum()
    pi=(k+.5)/(n+1);q=(obs+xs[1]+.5)/(sum(obs)+k+2)
    r=(xs[0]+.5)/(n-k+2);p=pi*q+(1-pi)*r
    cov=pi*pi*(np.diag(q)-np.outer(q,q))/(sum(obs)+k+2)
    cov+=(1-pi)**2*(np.diag(r)-np.outer(r,r))/(n-k+2)
    cov+=pi*(1-pi)/(n+1)*np.outer(q-r,q-r)
    grad=SIGN/p
    return lor(p),float(np.sqrt(max(grad@cov@grad,1e-16))),p

def run(stage):
    nrep=C['development_replicates'] if stage=='development' else C['evaluation_replicates']
    frozen=json.loads((G/'configs/METHOD_FREEZE.json').read_text()) if stage=='evaluation' else None
    rows=[];counts=[]
    for budget in C['g3']['budgets']:
      for ei,effect in enumerate(C['g3']['effects']):
       p=snapshot(np.array([-1.,-1.2,-.8,effect,.5]))[0].reshape(4,2).sum(1)
       for bi,bias in enumerate(C['g3']['biases']):
        states=np.array([[0,0],[0,1],[1,0],[1,1]])
        ss=np.exp(-2.2+.5*states[:,0]+bias*states[:,0]*states[:,1]);q=p*ss;q/=q.sum()
        for fi,frac in enumerate(C['g3']['fractions']):
         calbudget=6*int(budget*frac/6);nxs=calbudget//3;nx=calbudget//2;nobs=budget-calbudget
         for rep in range(nrep):
          rng=np.random.default_rng(np.random.SeedSequence([C['seed'],3,0 if stage=='development' else 1,budget,ei,bi,fi,rep]))
          obs=rng.multinomial(nobs,q);xs=rng.multinomial(nxs,np.r_[p*(1-ss),p*ss]).reshape(2,4);cx=rng.multinomial(nx,p)
          counts.append(dict(budget=budget,effect=effect,bias=bias,fraction=frac,replicate=rep,observed=json.dumps(obs.tolist()),xs=json.dumps(xs.tolist()),random_x=json.dumps(cx.tolist()),nobs=nobs,nxs=nxs,nx=nx,cost_xs=nobs+3*nxs,cost_x=nobs+2*nx))
          for method in ['XS_JOINT','XS_SOURCE']:
           if method=='XS_JOINT':est,se,_=estimate(obs,xs)
           else:
            cc=xs.sum(0)+.5;est=lor(cc/cc.sum());se=float(np.sqrt(np.sum(1/cc)))
           mult=frozen['multipliers'][method] if frozen else 1.96
           lo=est-mult*se;hi=est+mult*se;sg=1 if lo>0 else -1 if hi<0 else 0
           rows.append(dict(budget=budget,effect=effect,bias=bias,fraction=frac,replicate=rep,method=method,estimate=est,se=se,z_truth=abs(est-lor(p))/se,lo=lo,hi=hi,width=hi-lo,coverage=lo<=lor(p)<=hi,sign=sg,correct=bool(sg and sg==np.sign(effect)),wrong=bool(sg and sg!=np.sign(effect)),truth_logOR=lor(p),multiplier=mult))
          for method,sg in [('RANDOM_X',exact_sign([cx.reshape(2,2)])),('NO_CAL_NAIVE',exact_sign([obs.reshape(2,2)])),('NO_CAL_BOUND1',exact_sign([obs.reshape(2,2)],eta=1))]:
           rows.append(dict(budget=budget,effect=effect,bias=bias,fraction=frac,replicate=rep,method=method,sign=sg,correct=bool(sg and sg==np.sign(effect)),wrong=bool(sg and sg!=np.sign(effect)),truth_logOR=lor(p)))
         print(stage,budget,effect,bias,frac,flush=True)
    d=pd.DataFrame(rows);d.to_csv(G/f'results/{stage}_raw.tsv',sep='\t',index=False)
    pd.DataFrame(counts).to_csv(G/f'results/{stage}_counts.tsv',sep='\t',index=False)
    if stage=='development':
      multi={};audit=[]
      for method in ['XS_JOINT','XS_SOURCE']:
       sub=d[d.method.eq(method)]
       for keys,z in sub.groupby(['budget','effect','bias','fraction']):
        audit.append(dict(method=method,cell=list(keys),q95=float(np.quantile(z.z_truth,.95,method='higher'))))
       multi[method]=max(1.96,max(v['q95'] for v in audit if v['method']==method))
      freeze=dict(multipliers=multi,development_quantiles=audit,rule='One global multiplier per method, maximum development-cell 95th percentile; empirical calibration, no finite-sample theorem',code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
      (G/'configs/METHOD_FREEZE.json').write_text(json.dumps(freeze,indent=2,default=lambda x:x.item()));print('FREEZE',multi,flush=True)
    out=[]
    for key,z in d.groupby(['budget','effect','bias','fraction','method']):
     cl,cu=wilson(int(z.correct.sum()),len(z));wl,wu=wilson(int(z.wrong.sum()),len(z))
     out.append(dict(zip(['budget','effect','bias','fraction','method'],key),n=len(z),correct=z.correct.mean(),correct_lo=cl,correct_hi=cu,wrong=z.wrong.mean(),wrong_lo=wl,wrong_hi=wu,coverage=z.coverage.mean(),mean_width=z.width.mean()))
    pd.DataFrame(out).to_csv(G/f'results/{stage}_summary.tsv',sep='\t',index=False)
    if stage=='evaluation':
     # A bounded biological/reference shift on this explicit log-OR scale.
     # This is not the old generator-scale epsilon contract.
     sens=[]
     for delta in [-.25,0,.25]:
      for bound in [0,.25,.5]:
       z=d[d.method.eq('XS_JOINT')].copy();target=z.truth_logOR-delta
       z['covered_reference']=(z.lo-bound<=target)&(target<=z.hi+bound)
       z['correct_reference']=((z.lo-bound>0)&(target>0))|((z.hi+bound<0)&(target<0))
       for key,w in z.groupby(['budget','effect','bias','fraction']):sens.append(dict(zip(['budget','effect','bias','fraction'],key),delta=delta,bound=bound,inside=abs(delta)<=bound,coverage=w.covered_reference.mean(),correct=w.correct_reference.mean(),n=len(w)))
     pd.DataFrame(sens).to_csv(G/'results/sensitivity_summary.tsv',sep='\t',index=False)

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('stage',choices=['development','evaluation']);run(a.parse_args().stage)
