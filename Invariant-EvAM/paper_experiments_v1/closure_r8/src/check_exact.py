from pathlib import Path
import json
import numpy as np
from exact_target import probs,invert,selected,confidence,polytope
from acquisition import derivative,distribution
r=np.random.default_rng(992188);errors=[];bad=[];derr=[];contained=0;tested=0;outer_violations=[]
for i in range(1000):
 rates=np.exp(r.uniform(-2.5,1,3));p=probs(rates);x=np.array([p[0],sum(p[:2]),p[2]]);truth=np.log(rates[2]/rates[1]);errors.append(abs(invert(*x)-truth));g=[]
 for k in range(3):
  h=1e-7*min(x[k],1-x[k]);e=np.eye(3)[k]*h;g.append((invert(*(x+e))-invert(*(x-e)))/(2*h))
 if not(g[0]>0 and g[1]<0 and g[2]<0):bad.append([rates.tolist(),g])
 if i<100:
  z=np.r_[np.log(rates),r.uniform(-3,3,2)]
  for kind in ['SOURCE','SELECT_A','SELECT_B','MARGINAL_A','MARGINAL_B']:
   _,j=derivative(z,kind);fd=np.column_stack([(distribution(z+np.eye(5)[k]*1e-5,kind)-distribution(z-np.eye(5)[k]*1e-5,kind))/2e-5 for k in range(5)]);derr.append(float(np.max(abs(j-fd))))
for i in range(30):
 rates=np.exp(r.uniform(-1.5,-.3,3));p=probs(rates);groups=[dict(kind=k,counts=r.multinomial(1000,selected(p,k,phi)).tolist()) for k,phi in [('SELECT_A',.7),('SELECT_B',-.7)]];ci=confidence(groups);A,B=polytope(groups)
 for k in range(1500):
  rr=rates*np.exp(r.normal(0,.15,3));pp=probs(rr);tested+=1
  if np.max(A@pp-B)<=0:
   contained+=1;v=np.log(rr[2]/rr[1])
   if not ci['lo']-1e-9<=v<=ci['hi']+1e-9:outer_violations.append([i,v,ci])
res=dict(inverse_cases=1000,max_inverse_error=float(max(errors)),monotonicity_failures=bad,max_derivative_error=max(derr),outer_candidate_points=tested,feasible_physical_points=contained,outer_violations=outer_violations,scope='numerical stress check, not machine-certified interval arithmetic')
Path(__file__).resolve().parents[1].joinpath('reports/EXACT_NUMERICAL_CHECK.json').write_text(json.dumps(res,indent=2));print(res)
assert not bad and not outer_violations and max(errors)<1e-8 and max(derr)<1e-7
