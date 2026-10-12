from pathlib import Path
import sys,json,hashlib,time,argparse
from concurrent.futures import ProcessPoolExecutor
import numpy as np,pandas as pd
from scipy.stats import chi2
from exact_target import confidence,probs,invert
P=Path(__file__).resolve().parents[2];G=P/'g2/revision_r8';sys.path.insert(0,str(P/'g2/src'))
from g2_engine import snapshot,operators,Likelihood,fit,profile,interval
SEED=202609288;PUB=json.loads((P/'g2/configs/public_contract.json').read_text());EV=json.loads((P/'g2/configs/evaluator_contract.json').read_text());DES=['strong','weak','nonidentifying'];N=[1000,4000,16000];EFF=[-.8,0,.15,.8]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def one(t):
 stage,des,n,effect,rep=t;name=f'{des}_{n}_{effect}_{rep:03d}.json';path=G/('development' if stage=='development' else 'runs')/name
 if path.exists():return
 start=time.time();rng=np.random.default_rng([SEED,2,0 if stage=='development' else 1,DES.index(des),N.index(n),EFF.index(effect),rep]);theta=np.r_[EV['baseline'],effect,EV['other_interaction']];p,j=snapshot(theta);assert np.max(abs(p.reshape(4,2).sum(axis=1)-probs(np.exp([theta[0],theta[1],theta[1]+effect]))))<1e-10;ops=operators(PUB['operators'][des]);cts=[];groups=[]
 for k,((A,F),phi) in enumerate(zip(ops,EV['true_phis'][des])):
  q=p*np.exp(F@np.array(phi));q/=q.sum();c=rng.multinomial(n//2,A@q);cts.append(c);kind=('SELECT_A' if k==0 or des=='weak' else 'SELECT_B') if des!='nonidentifying' else ('MARGINAL_A' if k==0 else 'MARGINAL_B');groups.append(dict(kind=kind,counts=(c.reshape(4,2).sum(axis=1) if len(c)==8 else c).tolist()))
 exact=confidence(groups);m=Likelihood(cts,ops);z0=np.r_[[-1.,-1.,-1.,0.,0.],np.zeros(m.dim-5)];z1=z0.copy();z1[3:5]=[.5,-.5];r=fit(m,starts=[z0,z1]);pts=profile(m,r,index=3,bound=3.,extra_restart=True);best=min(pts,key=lambda z:z['nll'])
 if best['nll']<r['nll']-1e-5:r=fit(m,starts=[r['x'],np.array(best['x'])]);pts=profile(m,r,index=3,bound=3.,extra_restart=True)
 classic=interval(pts,r['nll'],chi2.ppf(.95,1),3.);classic['valid']=bool(r['success'] and all(z['success'] for z in pts));exact['valid']=exact['status']=='COMPLETE';out={}
 for method,z in [('EXACT_CONDITIONAL_OUTER',exact),('CLASSIC_PROFILE',classic)]:
  lo=z['lo'];hi=z['hi'];valid=z['valid'];called=bool(valid and (lo>0 or hi<0));wrong=bool(called and ((lo>0 and effect<=0)or(hi<0 and effect>=0)));out[method]={**z,'covered':bool(valid and lo<=effect<=hi),'called':called,'correct':called and not wrong,'wrong':wrong,'width':hi-lo,'finite':bool(valid and np.isfinite([lo,hi]).all())}
 path.write_text(json.dumps(dict(stage=stage,design=des,n=n,truth=effect,replicate=rep,groups=groups,full_counts=[c.tolist() for c in cts],methods=out,runtime=time.time()-start),default=lambda v:v.item() if isinstance(v,np.generic) else v.tolist()))
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['development','evaluation']);ap.add_argument('--workers',type=int,default=3);a=ap.parse_args();code={p.name:sha(p) for p in [Path(__file__),Path(__file__).with_name('exact_target.py')]}
 if a.stage=='evaluation':assert json.loads((G/'configs/FREEZE.json').read_text())['code']==code
 nrep=2 if a.stage=='development' else 50;tasks=[(a.stage,des,n,e,r) for des in DES for n in N for e in EFF for r in range(nrep)]
 with ProcessPoolExecutor(a.workers) as pool:
  for i,_ in enumerate(pool.map(one,tasks),1):
   if i%24==0:print(a.stage,i,len(tasks),flush=True)
 docs=[json.loads(p.read_text()) for p in (G/('development' if a.stage=='development' else 'runs')).glob('*.json') if p.name!='probe.json'];rows=[dict(design=z['design'],n=z['n'],truth=z['truth'],replicate=z['replicate'],method=k,**v) for z in docs for k,v in z['methods'].items()];d=pd.DataFrame(rows);d.to_csv(G/f'results/{a.stage}_raw.tsv',sep='\t',index=False);s=d.groupby(['design','n','truth','method']).agg(replicates=('replicate','size'),coverage=('covered','mean'),finite=('finite','mean'),correct=('correct','mean'),wrong=('wrong','mean'),valid=('valid','mean'),median_width=('width','median')).reset_index();s.to_csv(G/f'results/{a.stage}_summary.tsv',sep='\t',index=False)
 if a.stage=='development':(G/'configs/FREEZE.json').write_text(json.dumps(dict(seed=SEED,code=code,designs=DES,n=N,effects=EFF,evaluation_per_cell=50,confidence=.95,subdivisions=96,scope='original sparse D3 topology and public selection families; closed A-B subsystem, global probability-set projection with floating-point geometry checks',primary='coverage + finite fraction + correct/wrong answers; separate structural refusal and numerical failures; no requirement to beat classical profile in interval length'),indent=2))
 print(s.to_string(index=False),flush=True)
