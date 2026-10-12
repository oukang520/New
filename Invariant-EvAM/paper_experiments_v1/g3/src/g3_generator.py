"""Generator and scorer-side truth only. Estimator receives observed counts/public specification."""
import numpy as np
from scipy.linalg import expm
from g3_model import snapshot,timed_snapshot,states,measure,EDGES
from g3_design import theta,specifications

def distributions(cell,E=2):
 d=cell['d'];kind=cell['kind'];t=theta(d);base,flex=specifications(cell);pops=[];fs=[];phis=[];measurement=[];targets=[];pis=[]
 for e in range(E):
  z=t.copy();f=np.asarray(base['features'][e%2]);phi=np.array([.5]);a=np.eye(2**d);target=float(t[d]);pi=None
  if kind in ['different_same','different_different','local_aligned'] and e%2==1:z[d]+=.5
  if kind=='outside_nuisance' and e%2==1:z[-1]+=.5
  p=snapshot(z,d)[0]
  if kind=='time':p=timed_snapshot(z,d,.5)[0]
  if kind=='higher_order':
   x=states(d);n=2**d;q=np.zeros((n,n))
   for s in range(n):
    for event in range(d):
     if x[s,event]:continue
     rate=z[event]+sum(z[d+k]*x[s,source] for k,(target_i,source) in enumerate(EDGES[d]) if target_i==event)
     if event==d-1:rate+=.6*x[s,0]*x[s,1]
     rate=np.exp(rate);dest=s|(1<<(d-1-event));q[s,dest]+=rate;q[s,s]-=rate
   p=expm(q.T)[:,0];target=None
  if kind=='mixture':
   z2=t.copy();z2[d]+=.8;pi=.3 if e%2==0 else .7;p=pi*snapshot(t,d)[0]+(1-pi)*snapshot(z2,d)[0];target=None
  if kind=='measurement':a=measure(d,.85,.02)
  if kind.startswith('common_'):
   strength={'common_zero':0.,'common_mild':.5,'common_moderate':1.,'common_exact':1.}[kind];f=np.asarray(flex['features'][e%2]);phi=np.array([.5,strength])
  pops.append(p);fs.append(f);phis.append(phi);measurement.append(a);targets.append(target);pis.append(pi)
 return dict(pops=pops,features=fs,phis=phis,measurement=measurement,target=targets[0],reference_theta=t,mixture=pis)

def generate(cell,arm,rep,purpose=302):
 E=arm['envs'];truth=distributions(cell,E);ss=np.random.SeedSequence([20260923,3,purpose,cell['cell_id'],rep]);children=ss.spawn(20)
 share=arm['share'];kind=arm['calibration_kind'];cost=2 if kind=='X' else 3 if kind=='XS' else 0;ncal=int(1000*share/cost) if cost else 0;nobs=1000-int(1000*share);counts=[];cal=[];raw=[];population_selected=[]
 for e in range(E):
  rng=np.random.default_rng(children[e]);p=truth['pops'][e];f=truth['features'][e];phi=truth['phis'][e];a=truth['measurement'][e];n=nobs//E+(e<nobs%E)
  if cell['kind']=='mixture' and arm['name'] in ['LABEL_KNOWN_SUBGROUP','COARSE_SUBGROUP']:
   # Full candidate sample with latent subgroup; retain observed subgroup labels (or noisy surrogate).
   d=cell['d'];t=truth['reference_theta'];t2=t.copy();t2[d]+=.8;pi=truth['mixture'][e];p0=snapshot(t,d)[0];p1=snapshot(t2,d)[0];w=np.exp(f@phi);joint=np.r_[pi*p0*w,(1-pi)*p1*w];joint/=joint.sum();record=rng.multinomial(n,joint).reshape(2,-1)
   if arm['name']=='LABEL_KNOWN_SUBGROUP':c=record[0];truth['target']=float(t[d])
   else:
    # 80% sensitivity/specificity surrogate; target distribution mixture, not a CTMC truth.
    c=rng.binomial(record[0],.8)+rng.binomial(record[1],.2);truth['target']=None
   counts.append(c.tolist());population_selected.append((p0*w/(p0@w)).tolist());raw.append(record.tolist())
  else:
   w=np.exp(f@phi);obs=a@(p*w/(p@w));c=rng.multinomial(n,obs);counts.append(c.tolist());population_selected.append(obs.tolist())
  if kind:
   crng=np.random.default_rng(children[8+e]);m=ncal//E+(e<ncal%E)
   if kind=='X':cal.append(crng.multinomial(m,p).tolist())
   else:
    alpha=-float(abs(f).max(axis=0).sum())-.8;s=np.exp(alpha+f@phi);assert max(s)<1
    cal.append(crng.multinomial(m,np.r_[p*(1-s),p*s]).tolist())
 return dict(counts=counts,calibration=cal if kind else None,calibration_kind=kind),dict(target=truth['target'],n_observed=sum(map(sum,counts)),n_calibration=ncal,n_budget_used=nobs+ncal*cost,n_screened_calibration=ncal if kind=='XS' else None,population_selected=population_selected,phi_true=[p.tolist() for p in truth['phis']],raw_subgroup_counts=raw,seed_sequence=[20260923,3,purpose,cell['cell_id'],rep])
