"""Estimator only. Public forward specification and observed counts; no truth registry access."""
from pathlib import Path
import ctypes,sys
import numpy as np
from scipy.special import roots_genlaguerre,gamma
from scipy.optimize import minimize
from scipy.stats import chi2
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'paper_experiments_v1/g2/src'))
from lbfgsb_direct import minimize_direct
LIB=ctypes.CDLL(str(Path(__file__).with_name('ctmc_generic.dylib')))
A=np.ctypeslib.ndpointer(dtype=np.float64,flags='C_CONTIGUOUS');IA=np.ctypeslib.ndpointer(dtype=np.int32,flags='C_CONTIGUOUS')
LIB.snapshot_generic.argtypes=[ctypes.c_int,ctypes.c_int,IA,A,ctypes.c_double,A,A]
EDGES={3:[(1,0),(2,1)],4:[(1,0),(2,1),(3,2),(3,0),(2,0),(3,1)]}
def snapshot(theta,d=3,time=1):
 edges=np.ascontiguousarray(EDGES[d],dtype=np.int32);t=np.ascontiguousarray(theta,dtype=float);p=np.empty(2**d);j=np.empty((2**d,len(t)));LIB.snapshot_generic(d,len(edges),edges,t,float(time),p,j);return p,j

def timed_snapshot(theta,d,cv):
 if cv<1e-5:return snapshot(theta,d)
 shape=1/cv**2;nodes,weights=roots_genlaguerre(16,shape-1);weights=weights/gamma(shape);out=[snapshot(theta,d,x/shape) for x in nodes]
 return sum(w*z[0] for w,z in zip(weights,out)),sum(w*z[1] for w,z in zip(weights,out))

def states(d):return np.array([[s>>(d-1-i)&1 for i in range(d)] for s in range(2**d)])
def measure(d,sensitivity=1.,false_positive=0.,swap=False):
 x=states(d);a=np.ones((2**d,2**d))
 for j in range(d):
  truth=x[:,1-j] if swap and j<2 else x[:,j];prob=np.where(truth,sensitivity,false_positive);a*=np.where(x[:,j,None],prob[None,:],1-prob[None,:])
 return a

def selected(p,j,f,phi,a):
 z=f@phi;w=np.exp(z-z.max());u=p*w;s=u/u.sum();jt=w[:,None]*j/u.sum()-s[:,None]*(w@j/u.sum());jp=s[:,None]*(f-s@f)
 return a@s,a@jt,a@jp

class Joint:
 def __init__(self,data,spec):
  self.data=data;self.spec=spec;self.d=spec['d'];self.p=self.d+len(EDGES[self.d]);self.E=len(data['counts']);self.index=self.d
  self.bounds=[(-3.,3.)]*self.p;self.delta=[]
  eps=spec.get('epsilon',0)
  for e in range(self.E):
   self.delta.append(None if e==0 or eps==0 else slice(len(self.bounds),len(self.bounds)+self.p))
   if self.delta[-1] is not None:self.bounds += [(-eps,eps)]*self.p
  self.phi=[];self.alpha=[]
  for e,f in enumerate(spec['features']):
   if spec.get('known_phi') is not None:self.phi.append(None)
   else:self.phi.append(slice(len(self.bounds),len(self.bounds)+len(f[0])));self.bounds += [(-1.,1.)]*len(f[0])
   if data.get('calibration_kind')=='XS':self.alpha.append(len(self.bounds));self.bounds.append((-12.,-float(np.abs(np.asarray(f)).max(axis=0).sum())-.1))
   else:self.alpha.append(None)
  self.time_index=None
  if spec.get('estimate_cv',False):self.time_index=len(self.bounds);self.bounds.append((.1,1.2))
  self.dim=len(self.bounds);self.cache=None
 def initial(self):
  x=np.zeros(self.dim);x[:self.d]=-1
  for e,a in enumerate(self.alpha):
   if a is not None:x[a]=-float(np.abs(np.asarray(self.spec['features'][e])).max(axis=0).sum())-1.
  if self.time_index is not None:x[self.time_index]=.5
  return x
 def outputs(self,x):
  if self.cache is not None and np.array_equal(x,self.cache[0]):return self.cache[1]
  out=[]
  for e,c in enumerate(self.data['counts']):
   theta=x[:self.p].copy()
   if self.delta[e] is not None:theta+=x[self.delta[e]]
   cv=x[self.time_index] if self.time_index is not None else self.spec.get('cv',0)
   p,j=timed_snapshot(theta,self.d,cv);J=np.zeros((len(p),self.dim));J[:,:self.p]=j
   if self.delta[e] is not None:J[:,self.delta[e]]=j
   if self.time_index is not None:J[:,self.time_index]=(timed_snapshot(theta,self.d,cv+1e-5)[0]-timed_snapshot(theta,self.d,cv-1e-5)[0])/2e-5
   f=np.asarray(self.spec['features'][e]);phi=np.asarray(self.spec['known_phi'][e]) if self.phi[e] is None else x[self.phi[e]]
   prob,jt,jp=selected(p,J,f,phi,np.asarray(self.spec['measurement'][e]))
   if self.phi[e] is not None:jt[:,self.phi[e]]+=jp
   out.append((np.asarray(c),prob,jt))
   if self.data.get('calibration') is not None:
    counts=np.asarray(self.data['calibration'][e]);kind=self.data['calibration_kind']
    if kind=='X':out.append((counts,p,J))
    elif kind=='XS':
     s=np.exp(x[self.alpha[e]]+f@phi);ds=np.zeros((len(p),self.dim));ds[:,self.alpha[e]]=s
     if self.phi[e] is not None:ds[:,self.phi[e]]=s[:,None]*f
     assert np.max(s)<1
     out.append((counts,np.r_[p*(1-s),p*s],np.vstack(((1-s[:,None])*J-p[:,None]*ds,s[:,None]*J+p[:,None]*ds))))
  self.cache=(x.copy(),out);return out
 def objective(self,x):
  v=0.;g=np.zeros(self.dim)
  for c,p,j in self.outputs(x):
   p=np.maximum(p,1e-280);v-=c@np.log(p);g-=j.T@(c/p)
  return float(v),g
 def geometry(self,x):
  h=np.vstack([np.sqrt(c.sum())*j/np.sqrt(np.maximum(p[:,None],1e-15)) for c,p,j in self.outputs(x) if c.sum()>0]);a=np.zeros(self.dim);a[self.index]=1
  _,s,vh=np.linalg.svd(h,full_matrices=True);rank=sum(s>s[0]*1e-8);leak=np.linalg.norm(vh[rank:]@a);cov=(vh[:rank].T/(s[:rank]**2))@vh[:rank]
  nuisance=np.delete(h,self.index,axis=1);u,sv,_=np.linalg.svd(nuisance,full_matrices=False);r=sum(sv>sv[0]*1e-8) if len(sv) else 0;proj=h[:,self.index]-u[:,:r]@(u[:,:r].T@h[:,self.index]);info=proj@proj
  return dict(rank=int(rank),dim=self.dim,null_leakage=float(leak),target_effective_information=float(info),se=float(np.sqrt(max(0,cov[self.index,self.index]))),rank_tolerance={str(t):int(sum(s>s[0]*t)) for t in [1e-6,1e-7,1e-8,1e-9,1e-10]})

def fit(lik):
 z=lik.initial();starts=[z,z+np.r_[np.zeros(lik.d),np.full(lik.dim-lik.d,.15)]];attempts=[]
 for x in starts:
  x=np.clip(x,np.array(lik.bounds)[:,0],np.array(lik.bounds)[:,1]);o=minimize_direct(lik.objective,x,lik.bounds,maxiter=600,ftol=1e-10,gtol=1e-6,maxls=40);attempts.append(dict(x=o.x.tolist(),nll=float(o.fun),success=bool(o.success),message=str(o.message),nit=int(o.nit)))
 best=min(attempts,key=lambda a:a['nll']);return {**best,'attempts':attempts}

def inference(lik,profile=True):
 r=fit(lik);x=np.array(r['x']);geo=lik.geometry(x);r.update(geo);r['estimate']=float(x[lik.index]);r['source_attribution_allowed']=False
 sat=0
 for c,p,j in lik.outputs(x):
  ok=c>0
  if c.sum()>0:sat-=float(c[ok]@np.log(c[ok]/c.sum()))
 r['deviance']=max(0,2*(r['nll']-sat));r['compatibility_status']='DESCRIPTIVE_DEVIANCE_NO_UNVALIDATED_CHI2_THRESHOLD';r['interval_type']='LOCAL_PROFILE_EXPLORATORY';r['interval_attempts']=[]
 if not r['success']:r.update(lo=None,hi=None,sign='NUMERICAL_FAILURE',status='FIT_FAILED');return r
 if geo['null_leakage']>1e-7:
  r.update(lo=-np.inf,hi=np.inf,sign='UNKNOWN',status='COMPLETE',interval_type='LOCAL_TANGENT_UNRESOLVED_OUTER_R');return r
 if not profile:
  r.update(lo=float(x[lik.index]-1.96*geo['se']),hi=float(x[lik.index]+1.96*geo['se']),status='COMPLETE',interval_type='WALD_EXPLORATORY');r['sign']='POSITIVE' if r['lo']>0 else ('NEGATIVE' if r['hi']<0 else 'UNKNOWN');return r
 # Likelihood-set extrema; local constrained optimizer, not a global projection certificate.
 limit=r['nll']+chi2.ppf(.95,1)/2;ends=[]
 for direction in [1,-1]:
  grad=np.zeros(lik.dim);grad[lik.index]=direction
  o=minimize(lambda z:direction*z[lik.index],x,jac=lambda z:grad,bounds=lik.bounds,constraints=[{'type':'ineq','fun':lambda z:limit-lik.objective(z)[0],'jac':lambda z:-lik.objective(z)[1]}],method='SLSQP',options={'maxiter':250,'ftol':1e-8})
  feasible=lik.objective(o.x)[0]<=limit+1e-5
  r['interval_attempts'].append(dict(success=bool(o.success),feasible=bool(feasible),x=o.x.tolist(),nll=lik.objective(o.x)[0],message=str(o.message)))
  ends.append(float(o.x[lik.index]))
 if not all(z['success'] and z['feasible'] for z in r['interval_attempts']):r.update(lo=None,hi=None,sign='NUMERICAL_FAILURE',status='INTERVAL_FAILED');return r
 r['lo']=-np.inf if ends[0]<-2.999 else ends[0];r['hi']=np.inf if ends[1]>2.999 else ends[1];r['boundary_hit']=not np.isfinite([r['lo'],r['hi']]).all();r['status']='COMPLETE';r['sign']='POSITIVE' if r['lo']>0 else ('NEGATIVE' if r['hi']<0 else 'UNKNOWN')
 if r['boundary_hit']:r['interval_type']='BOUND_TRUNCATED_EXTENDED_TO_INFINITY'
 if lik.spec.get('epsilon',0)>0:r['interval_type']='SENSITIVITY_ENVELOPE_LOCAL_NUMERICAL'
 return r
