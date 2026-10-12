"""Population diagnostics; no finite-sample calibration claims."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
from pathlib import Path
import json,sys,hashlib
import numpy as np,pandas as pd
from scipy.linalg import expm
from g3_model import *
from g3_design import theta,cells,arms,specifications,G
from g3_generator import distributions

def normalized(p,w):q=p*np.exp(w-np.max(w));return q/q.sum()
def kl(p,q):return float(sum(p*(np.log(np.maximum(p,1e-280))-np.log(np.maximum(q,1e-280)))))
def population_fit(probs,spec):
 l=Joint(dict(counts=[(1000/len(probs)*p).tolist() for p in probs],calibration=None,calibration_kind=None),spec);o=fit(l);pr=[z[1] for z in l.outputs(np.array(o['x']))];geo=l.geometry(np.array(o['x']));return dict(estimate=o['x'][l.index],population_KL=float(np.mean([kl(p,q) for p,q in zip(probs,pr)])),max_probability_error=float(max(np.max(abs(p-q)) for p,q in zip(probs,pr))),success=o['success'],source_attribution_allowed=False,**geo)

def run():
 rows=[]
 for d in [3,4]:
  t=theta(d);p,j=snapshot(t,d);x=states(d)
  v=np.zeros(len(t));v[d]=1.;score=j@v/p;score/=max(abs(score));phi=.5
  rng=np.random.default_rng(np.random.SeedSequence([20260923,3,304,99,d]));vr=rng.normal(size=len(t));vr/=max(abs(vr))
  for mechanism,direction,f in [('ALIGNED_LOCAL',v,score),('OUTSIDE_NUISANCE',np.eye(len(t))[-1],x[:,0]),('UNSELECTED_RANDOM',vr,x[:,0])]:
   base=normalized(p,f*phi);tangent=j@direction;jphi=base*(f-base@f)
   for amp in [0,.01,.05,.2,.5]:
    pshift,jshift=snapshot(t+amp*direction,d)
    # Optimize only selection coefficient, with base target fixed: test if biology can be absorbed.
    from scipy.optimize import minimize_scalar
    target=normalized(pshift,f*phi)
    o=minimize_scalar(lambda z:kl(target,normalized(p,f*z)),bounds=(-3,3),method='bounded',options={'xatol':1e-13});q=normalized(p,f*o.x)
    rows.append(dict(d=d,mechanism=mechanism,amplitude=amp,selection_adjustment=o.x-phi,population_KL=kl(target,q),max_probability_error=max(abs(target-q)),exact_claim=False,evidence_type='NEW_RUN_POPULATION'))
  b=np.zeros(len(t));b[d]=1
  for amp in [0,.2,.5]:
   a=snapshot(t+b*.5+b*amp,d)[0];z=snapshot(t+b*(.5+amp),d)[0]
   rows.append(dict(d=d,mechanism='ALIGNED_EXACT_SHIFT_ANALYTIC_REFERENCE',amplitude=amp,selection_adjustment=amp,population_KL=kl(a,z),max_probability_error=max(abs(a-z)),exact_claim=True,evidence_type='NEW_RUN_POPULATION'))
 pd.DataFrame(rows).to_csv(G/'results/G3_population_nonseparability.tsv',sep='\t',index=False)
 # All four quadrants plus D4 controlled directions under every prespecified specification.
 rows=[]
 for c in cells()[:6]:
  tr=distributions(c);probs=[a@normalized(p,f@phi) for p,f,phi,a in zip(tr['pops'],tr['features'],tr['phis'],tr['measurement'])]
  for arm in arms(c):rows.append(dict(cell=c['cell_id'],kind=c['kind'],d=c['d'],method=arm['name'],truth=tr['target'],**population_fit(probs,arm['spec'])))
 pd.DataFrame(rows).to_csv(G/'results/G3_population_crossed.tsv',sep='\t',index=False)
 # Common omitted state function and exact wrong-theta explanation.
 rows=[]
 for c in cells()[8:12]:
  tr=distributions(c);probs=[a@normalized(p,f@phi) for p,f,phi,a in zip(tr['pops'],tr['features'],tr['phis'],tr['measurement'])]
  for arm in arms(c):
   if arm['envs']!=2 or arm['calibration_kind']:continue
   rows.append(dict(cell=c['cell_id'],kind=c['kind'],method=arm['name'],truth=tr['target'],**population_fit(probs,arm['spec'])))
  if c['kind']=='common_exact':
   alt=theta(3);alt[3]=.4;pa=snapshot(alt,3)[0];base,_=specifications(c);other=[normalized(pa,np.array(f)[:,0]*.5) for f in base['features']]
   rows.append(dict(cell=c['cell_id'],kind=c['kind'],method='EXPLICIT_WRONG_THETA_ALIAS',truth=tr['target'],estimate=.4,population_KL=max(kl(p,q) for p,q in zip(probs,other)),max_probability_error=max(np.max(abs(p-q)) for p,q in zip(probs,other)),success=True,source_attribution_allowed=False))
 pd.DataFrame(rows).to_csv(G/'results/G3_population_common_bias.tsv',sep='\t',index=False)
 # Mechanism grid: distinct units, no pooled coverage of a nonexistent CTMC mixture target.
 rows=[]
 for d in [3,4]:
  c={'d':d,'kind':'same_different'};base,_=specifications(c);t=theta(d);p,j=snapshot(t,d);f=[np.asarray(z) for z in base['features']];select=lambda ps:[normalized(q,ff[:,0]*.5) for q,ff in zip(ps,f)]
  for cv in [0,.5,1.]:
   tp=timed_snapshot(t,d,cv)[0];probs=select([tp,tp]);
   for label,sp in [('WRONG_FIXED',base),('KNOWN_TIME',{**base,'cv':cv}),('ESTIMATED_TIME',{**base,'estimate_cv':True})]:
    rows.append(dict(d=d,mechanism='TIME_GAMMA',level=cv,fit=label,target_defined=True,**population_fit(probs,sp)))
  x=states(d)
  for amp in [0,.3,.6]:
   q=np.zeros((2**d,2**d))
   for s in range(2**d):
    for event in range(d):
     if x[s,event]:continue
     log=t[event]+sum(t[d+k]*x[s,src] for k,(dest,src) in enumerate(EDGES[d]) if dest==event)
     if event==d-1:log+=amp*x[s,0]*x[s,1]
     rate=np.exp(log);q[s,s|(1<<(d-1-event))]+=rate;q[s,s]-=rate
   pp=expm(q.T)[:,0];rows.append(dict(d=d,mechanism='OMITTED_HIGHER_ORDER',level=amp,fit='PAIRWISE',target_defined=False,**population_fit(select([pp,pp]),base)))
  t2=t.copy();t2[d]+=.8;p2=snapshot(t2,d)[0]
  for mix in [.3,.5,.7]:
   probs=select([mix*p+(1-mix)*p2,(1-mix)*p+mix*p2]);rows.append(dict(d=d,mechanism='LATENT_SUBGROUP_MIXTURE',level=mix,fit='SHARED_CTMC',target_defined=False,**population_fit(probs,base)))
  ordinary=select([p,p])
  cases=[('FALSE_NEGATIVE',z,measure(d,z,0)) for z in [1,.95,.85]]+[('FALSE_POSITIVE',z,measure(d,1,z)) for z in [0,.01,.05]]+[('PANEL_ANNOTATION_SWAP',1,measure(d,1,0,True))]
  for name,level,a in cases:
   probs=[a@q for q in ordinary]
   for label,sp in [('WRONG_IDENTITY',base),('KNOWN_MEASUREMENT',{**base,'measurement':[a.tolist()]*2})]:rows.append(dict(d=d,mechanism=name,level=level,fit=label,target_defined=True,**population_fit(probs,sp)))
  for mix in [0,.1,.25]:
   probs=[(1-mix)*ordinary[0]+mix*ordinary[1],mix*ordinary[0]+(1-mix)*ordinary[1]];rows.append(dict(d=d,mechanism='ENVIRONMENT_LABEL_MIX',level=mix,fit='WRONG_UNMIXED',target_defined=True,**population_fit(probs,base)))
 pd.DataFrame(rows).to_csv(G/'results/G3_population_misspecification.tsv',sep='\t',index=False)
 rows=[]
 for d in [3,4]:
  t=theta(d);a=np.eye(len(t))[d]
  for eps in [0,.2,.5]:
   for eta in [0,.2,.5]:
    radius=sum(abs(a))*(eps+eta);lower=a@t-radius;upper=a@t+radius
    rows.append(dict(d=d,epsilon_log_rate=eps,eta_effective_log_rate=eta,center=a@t,lower=lower,upper=upper,max_abs_linear_error=radius,proof='Triangle inequality; attained by aligned box vertices',scope='EFFECTIVE_PARAMETER_BOX_ONLY_NOT_FORWARD_OPERATOR_ERROR',evidence_type='DERIVED_FOR_THIS_PROTOCOL'))
 pd.DataFrame(rows).to_csv(G/'results/G3_parameter_scale_envelope.tsv',sep='\t',index=False)
 print('Population diagnostics complete',flush=True)
if __name__=='__main__':run()
