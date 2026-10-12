from pathlib import Path
import sys,json,time,argparse
import numpy as np,pandas as pd
from scipy.stats import hypergeom,nchypergeom_fisher,norm,beta
from scipy.special import gammaln,softmax
R=Path(__file__).resolve().parents[3];P=R/'paper_experiments_v1';D=Path(__file__).resolve().parent
sys.path[:0]=[str(R/'src'),str(P/'g1/src'),str(P/'g2/src')]
from g2_engine import snapshot,operators,STATES
from g2_certified_sign import certify
CFG=json.loads((D/'PROTOCOL.json').read_text());MASTER=CFG['master_seed']
def rng(g,*keys):return np.random.default_rng(np.random.SeedSequence([MASTER,g,*keys]))
def write(df,g,name):df.to_csv(P/f'g{g}/revision_r2/results/{name}.tsv',sep='\t',index=False)
def wilson(k,n):
 if not n:return np.nan,np.nan
 z=1.96;q=k/n;den=1+z*z/n;m=(q+z*z/(2*n))/den;h=z*np.sqrt(q*(1-q)/n+z*z/(4*n*n))/den;return m-h,m+h

def conditional_tail(tables,logodds=0.):
 # Condition each stratum on its row/column totals. Evaluate T=sum(n00).
 mass=np.array([1.]);offset=0;obs=0
 for tab in tables:
  a,b,c,d=map(int,np.asarray(tab).ravel());M=a+b+c+d;n=a+b;N=a+c;lo=max(0,N-(M-n));hi=min(n,N);xx=np.arange(lo,hi+1)
  logpm=gammaln(n+1)-gammaln(xx+1)-gammaln(n-xx+1)+gammaln(M-n+1)-gammaln(N-xx+1)-gammaln(M-n-N+xx+1)+logodds*xx
  pm=softmax(logpm);mass=np.convolve(mass,pm);offset+=lo;obs+=a
 idx=obs-offset;return float(mass[:idx+1].sum()),float(mass[idx:].sum())
def exact_sign(tables,eta=0.,alpha=.05):
 lo=conditional_tail(tables,-eta)[0];hi=conditional_tail(tables,eta)[1]
 return -1 if lo<=alpha/2 else 1 if hi<=alpha/2 else 0

def metrics(rows,groups,g,name):
 d=pd.DataFrame(rows);write(d,g,name+'_raw');out=[]
 for key,q in d.groupby(groups,dropna=False):
  if not isinstance(key,tuple):key=(key,)
  n=len(q);k=int(q.correct.sum());wrong=int(q.wrong.sum());called=int(q.called.sum());wl,wu=wilson(wrong,n);cl,cu=wilson(k,n)
  out.append(dict(zip(groups,key),n=n,correct=k,correct_rate=k/n,correct_lo=cl,correct_hi=cu,called=called,call_rate=called/n,wrong=wrong,wrong_rate=wrong/n,wrong_lo=wl,wrong_hi=wu,conditional_wrong=wrong/called if called else np.nan,sign_set_coverage=1-wrong/n))
 write(pd.DataFrame(out),g,name+'_summary')

def g1():
 from g1_v1_1 import load_v11,sentinels
 from g1_core import latent_geometry,primary_target_index,expand_design,target_metrics
 from g1_pipeline import _geometry_for
 cfg,models,ops,designs,_=load_v11();rows=[];params=[]
 for mi,(mid,m) in enumerate(models.items()):
  for i in range(CFG['g1']['draws_per_model']):
   region=i%3;gen=rng(1,801,mi,i);br=CFG['g1']['rate_regimes'][region];t=np.r_[gen.uniform(*br,m.d),gen.choice([-1.,-.5,-.15,0.,.15,.5,1.],len(m.edges))];params.append(dict(model=mid,draw=i,region=region,theta=json.dumps(t.tolist())));src=latent_geometry(m,t);a=np.eye(m.parameter_count)[primary_target_index(m)]
   for name,(des,_) in sentinels(mid,designs).items():
    E=1 if name=='single' else 2;ids,groups=expand_design(des,E);cost=sum(ops[k]['panel_width'] for k in ids)/E;N=1000*m.d/cost
    for knowledge in CFG['g1']['knowledge']:
     geom,_,_=_geometry_for(m,t,ids,groups,ops,knowledge,1e-8,[1e-6,1e-7,1e-8,1e-9,1e-10]);z=target_metrics(a,geom);rows.append(dict(model=mid,draw=i,regime=['rare','ordinary','frequent'][region],design=name,knowledge=knowledge,n_total=N,cost=1000*m.d,rank=geom.rank,source_rank=src.rank,identified=z['identified'],information=N*z['target_information_per_sample'],leak=z['null_leakage'],tolerance_stable=len(set(geom.rank_by_tolerance.values()))==1))
  print('G1',mid,'done',flush=True)
 write(pd.DataFrame(params),1,'G1_R2_parameters');d=pd.DataFrame(rows);write(d,1,'G1_R2_geometry');summary=d.groupby(['model','regime','design','knowledge']).agg(n=('draw','size'),identified_fraction=('identified','mean'),information_median=('information','median'),rank_median=('rank','median')).reset_index();write(summary,1,'G1_R2_summary')

def g2():
 pub=json.loads((P/'g2/configs/public_contract.json').read_text());ev=json.loads((P/'g2/configs/evaluator_contract.json').read_text());rows=[];raw=[]
 for di,design in enumerate(CFG['g2']['designs']):
  specs=pub['operators'][design];ops=operators(specs)
  for ei,effect in enumerate(CFG['g2']['effects']):
   theta=np.r_[ev['baseline'],effect,ev['other_interaction']];p=snapshot(theta)[0];probs=[]
   for (a,f),phi in zip(ops,ev['true_phis'][design]):q=p*np.exp(f@phi);q/=q.sum();probs.append(a@q)
   allcounts=[[],[]]
   for rep in range(CFG['g2']['replicates_per_cell']):
    random=rng(2,802,di,ei,rep);cts=[random.multinomial(500,q) for q in probs]
    for j,c in enumerate(cts):allcounts[j].append(c)
    raw.append(dict(design=design,effect=effect,replicate=rep,counts=json.dumps([c.tolist() for c in cts])))
   bands,_,_=certify([np.array(c) for c in allcounts],specs)
   for rep in range(CFG['g2']['replicates_per_cell']):
    band=bands[rep];preds={'OLD_BONFERRONI_BANDS':1 if band=='POSITIVE_OPEN_HALF_LINE' else -1 if band=='NEGATIVE_OPEN_HALF_LINE' else 0}
    if design=='nonidentifying':preds.update(EXACT_STRATIFIED=0,EXACT_PER_ENV_BONFERRONI=0,LOG_OR_WALD=0)
    else:
     tabs=[np.asarray(allcounts[e][rep]).reshape(4,2).sum(1).reshape(2,2) for e in range(2)];preds['EXACT_STRATIFIED']=exact_sign(tabs);sg=[exact_sign([t],alpha=.025) for t in tabs];preds['EXACT_PER_ENV_BONFERRONI']=1 if 1 in sg and -1 not in sg else -1 if -1 in sg and 1 not in sg else 0
     vals=[];var=[]
     for t in tabs:
      a,b,c,d=t.ravel()+.5;vals.append(np.log(a*d/(b*c)));var.append(1/a+1/b+1/c+1/d)
     weights=1/np.array(var);est=np.dot(weights,vals)/weights.sum();se=np.sqrt(1/weights.sum());preds['LOG_OR_WALD']=1 if est-1.96*se>0 else -1 if est+1.96*se<0 else 0
    for method,sg in preds.items():rows.append(dict(design=design,effect=effect,replicate=rep,method=method,sign=sg,called=sg!=0,correct=sg!=0 and sg==np.sign(effect),wrong=sg!=0 and sg!=np.sign(effect)))
   print('G2',design,effect,'done',flush=True)
 write(pd.DataFrame(raw),2,'G2_R2_observed_counts');metrics(rows,['design','effect','method'],2,'G2_R2')

def sel_bounds(cal_counts,alpha=.025):
 # Bounds on selection probabilities in each 2x2 source-state cell; union over 4 cells.
 selected=cal_counts[1];total=cal_counts.sum(0);a=alpha/4
 lo=np.where(selected==0,0,beta.ppf(a/2,selected,total-selected+1));hi=np.where(selected==total,1,beta.ppf(1-a/2,selected+1,total-selected));lo=np.nan_to_num(lo,nan=0);hi=np.nan_to_num(hi,nan=1)
 low=lo[0]*lo[3]/max(hi[1]*hi[2],1e-300);high=hi[0]*hi[3]/max(lo[1]*lo[2],1e-300)
 return np.log(max(low,1e-300)),np.log(max(high,1e-300))

def g3():
 rows=[];raw=[];B=CFG['g3']['budget'];effects=CFG['g3']['effects'];biases=CFG['g3']['pair_biases'];fracs=CFG['g3']['calibration_fractions']
 for ei,effect in enumerate(effects):
  p=snapshot(np.array([-1.,-1.2,-.8,effect,.5]))[0].reshape(4,2).sum(1);states=np.array([[0,0],[0,1],[1,0],[1,1]])
  for bi,bias in enumerate(biases):
   # Pair selection log-OR shift exactly bias. Same source estimand in every arm.
   ss=np.exp(-2.2+.5*states[:,0]+bias*states[:,0]*states[:,1]);q=p*ss;q/=q.sum()
   for fi,frac in enumerate(fracs):
    nx=int(B*frac/2);nxs=int(B*frac/3);nobs=B-int(B*frac)
    for rep in range(CFG['g3']['replicates_per_cell']):
     random=rng(3,803,ei,bi,fi,rep);obs=random.multinomial(nobs,q);cx=random.multinomial(nx,p);xs=random.multinomial(nxs,np.r_[p*(1-ss),p*ss]).reshape(2,4);preds={}
     for eta in CFG['g3']['eta_grid']:preds[f'BOUND_{eta}']=exact_sign([obs.reshape(2,2)],eta=eta)
     preds['RANDOM_X_ONLY']=exact_sign([cx.reshape(2,2)]) if nx else 0
     if nxs:
      lo,hi=sel_bounds(xs);lower=conditional_tail([obs.reshape(2,2)],float(np.clip(lo,-40,40)))[0];upper=conditional_tail([obs.reshape(2,2)],float(np.clip(hi,-40,40)))[1];preds['XS_BIAS_CALIBRATION']=-1 if lo>-40 and lower<=.0125 else 1 if hi<40 and upper<=.0125 else 0
      preds['XS_SOURCE_ONLY']=exact_sign([xs.sum(0).reshape(2,2)])
     else:preds.update(XS_BIAS_CALIBRATION=0,XS_SOURCE_ONLY=0)
     raw.append(dict(effect=effect,bias=bias,fraction=frac,replicate=rep,n_observed=nobs,n_x=nx,n_xs=nxs,observed=json.dumps(obs.tolist()),cal_x=json.dumps(cx.tolist()),cal_xs=json.dumps(xs.tolist())))
     for method,sg in preds.items():rows.append(dict(effect=effect,bias=bias,fraction=frac,replicate=rep,method=method,inside_bias_class=(bias<=float(method.split('_')[1])) if method.startswith('BOUND_') else True,sign=sg,called=sg!=0,correct=sg!=0 and sg==np.sign(effect),wrong=sg!=0 and sg!=np.sign(effect)))
    print('G3',effect,bias,frac,'done',flush=True)
 write(pd.DataFrame(raw),3,'G3_R2_observed_counts');metrics(rows,['effect','bias','fraction','method','inside_bias_class'],3,'G3_R2')
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('group',choices=['g1','g2','g3']);args=a.parse_args();globals()[args.group]()
