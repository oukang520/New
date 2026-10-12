"""Public likelihood/geometry only; no private record or truth reads."""
from pathlib import Path
import sys,ctypes,time
import numpy as np
from scipy.special import softmax
from scipy.optimize import minimize
from scipy.stats import chi2
R=Path(__file__).resolve().parents[3];sys.path.insert(0,str(R/'paper_experiments_v1/g2/src'))
from lbfgsb_direct import minimize_direct
LIB=ctypes.CDLL(str(Path(__file__).with_name('ctmc_g5.dylib')));A=np.ctypeslib.ndpointer(dtype=np.float64,flags='C_CONTIGUOUS');IA=np.ctypeslib.ndpointer(dtype=np.int32,flags='C_CONTIGUOUS');LIB.snapshot_generic.argtypes=[ctypes.c_int,ctypes.c_int,IA,A,ctypes.c_double,A,A]
EDGES={'SPARSE':[(1,0),(2,1),(3,2),(3,0),(2,0),(3,1)],'FULL':[(i,j) for i in range(4) for j in range(4) if i!=j]};X=np.array([[s>>(3-i)&1 for i in range(4)] for s in range(16)])
def snapshot(theta,topology='SPARSE'):
 edges=np.ascontiguousarray(EDGES[topology],dtype=np.int32);p=np.empty(16);j=np.empty((16,len(theta)));LIB.snapshot_generic(4,len(edges),edges,np.ascontiguousarray(theta,dtype=float),1.,p,j);return p,j

def mask_matrix(mask):
 v=X[:,np.array(mask,bool)];yi=v@2**np.arange(v.shape[1]-1,-1,-1);return (np.arange(2**v.shape[1])[:,None]==yi).astype(float)

def features(kind):return np.zeros((16,0)) if kind=='identity' else X.mean(1,keepdims=True) if kind=='burden' else X.astype(float)

class Model:
 def __init__(self,data,spec):
  self.data=data;self.spec=spec;self.topology=spec.get('topology','SPARSE');self.p=4+len(EDGES[self.topology]);self.mode=spec.get('mode','ctmc');self.nbase=15 if self.mode=='fusion' else self.p;bound=spec.get('bound',3);self.bounds=[(-12.,12.)]*15 if self.mode=='fusion' else [(-bound,bound)]*self.p;self.E=len(spec['features']);self.delta=[];self.phi=[];self.alpha=[];eps=spec.get('epsilon',0)
  for e in range(self.E):
   if e and (eps>0 or spec.get('penalty',0)>0):self.delta.append(slice(len(self.bounds),len(self.bounds)+self.p));self.bounds += [(-eps,eps)]*self.p if eps else [(-2.,2.)]*self.p
   else:self.delta.append(None)
  for e,f in enumerate(spec['features']):
   known=spec.get('known_phi',[None]*self.E)[e];pb=spec.get('reference_phi_bound',spec.get('phi_bound',2)) if e==0 else spec.get('phi_bound',2)
   if known is not None or len(f[0])==0:self.phi.append(None)
   else:self.phi.append(slice(len(self.bounds),len(self.bounds)+len(f[0])));self.bounds += [(-pb,pb)]*len(f[0])
   if spec.get('xs',False):self.alpha.append(len(self.bounds));self.bounds.append((-12.,-float(np.abs(f).max(axis=0).sum())*pb-.1))
   else:self.alpha.append(None)
  self.dim=len(self.bounds);self.cache=None
 def initial(self):
  x=np.zeros(self.dim)
  if self.mode!='fusion':x[:4]=-1.
  for e,k in enumerate(self.alpha):
   if k is not None:x[k]=self.bounds[k][1]-.9
  return x
 def source(self,x,e=0):
  J=np.zeros((16,self.dim))
  if self.mode=='fusion':p=softmax(np.r_[0,x[:15]]);J[:,:15]=p[:,None]*(np.eye(16)[:,1:]-p[None,1:])
  else:
   t=x[:self.p].copy()
   if self.delta[e] is not None:t+=x[self.delta[e]]
   p,j=snapshot(t,self.topology);J[:,:self.p]=j
   if self.delta[e] is not None:J[:,self.delta[e]]=j
  return p,J
 def selected(self,x,e):
  p,J=self.source(x,e);f=np.asarray(self.spec['features'][e]);sl=self.phi[e];known=self.spec.get('known_phi',[None]*self.E)[e];phi=np.asarray(known) if known is not None else x[sl] if sl is not None else np.zeros(f.shape[1]);z=f@phi;w=np.exp(z-z.max());u=p*w;den=u.sum();s=u/den;js=w[:,None]*J/den-s[:,None]*(w@J/den)
  if sl is not None:js[:,sl]+=s[:,None]*(f-s@f)
  return s,js
 def outputs(self,x):
  if self.cache is not None and np.array_equal(x,self.cache[0]):return self.cache[1]
  out=[];selected={};source={}
  for group in self.data:
   e=group['env'];kind=group.get('kind','selected');c=np.asarray(group['counts'])
   if kind=='selected':
    if e not in selected:selected[e]=self.selected(x,e)
    p,j=selected[e];a=np.asarray(group['A']);out.append((c,a@p,a@j))
   elif kind=='XS':
    p,J=self.source(x,e);f=np.asarray(self.spec['features'][e]);sl=self.phi[e];phi=x[sl] if sl is not None else np.asarray(self.spec['known_phi'][e]);s=np.exp(x[self.alpha[e]]+f@phi);ds=np.zeros((16,self.dim));ds[:,self.alpha[e]]=s
    if sl is not None:ds[:,sl]=s[:,None]*f
    assert s.max()<1
    out.append((c,np.r_[p*(1-s),p*s],np.vstack(((1-s[:,None])*J-p[:,None]*ds,s[:,None]*J+p[:,None]*ds))))
  self.cache=(x.copy(),out);return out
 def objective(self,x):
  v=0.;g=np.zeros(self.dim)
  for c,p,j in self.outputs(x):p=np.maximum(p,1e-280);v-=c@np.log(p);g-=j.T@(c/p)
  for sl in self.delta:
   if sl is not None and self.spec.get('penalty',0):v+=self.spec['penalty']*.5*(x[sl]@x[sl]);g[sl]+=self.spec['penalty']*x[sl]
  return float(v),g
 def whitened(self,x):return np.vstack([np.sqrt(c.sum())*j/np.sqrt(np.maximum(p[:,None],1e-15)) for c,p,j in self.outputs(x) if c.sum()>0])

def fit(lik):
 t=time.perf_counter();z=lik.initial();attempts=[]
 for step in [0.,.15]:
  start=np.clip(z+np.r_[np.zeros(4),np.full(lik.dim-4,step)],np.array(lik.bounds)[:,0],np.array(lik.bounds)[:,1]);o=minimize_direct(lik.objective,start,lik.bounds,maxiter=650,ftol=1e-10,gtol=1e-6,maxls=40);attempts.append(dict(x=o.x.tolist(),nll=float(o.fun),success=bool(o.success),nit=int(o.nit),message=str(o.message)))
 best=min(attempts,key=lambda z:z['nll']);return {**best,'attempts':attempts,'runtime_seconds':time.perf_counter()-t}

def geometry(lik,x,a):
 a=np.pad(np.array(a),(0,lik.dim-len(a)));h=lik.whitened(x);u,s,vh=np.linalg.svd(h,full_matrices=True);rank=int(sum(s>s[0]*1e-8));leak=float(np.linalg.norm(vh[rank:]@a)/max(np.linalg.norm(a),1e-15));var=float(np.sum((vh[:rank]@a/s[:rank])**2));n=sum(np.sum(z['counts']) for z in lik.data);tol={str(t):dict(rank=int(sum(s>s[0]*t)),leak=float(np.linalg.norm(vh[int(sum(s>s[0]*t)):]@a)/max(np.linalg.norm(a),1e-15))) for t in [1e-6,1e-7,1e-8,1e-9,1e-10]};ht=h[:,:lik.p];hn=h[:,lik.p:]
 if hn.shape[1]:un,sn,_=np.linalg.svd(hn,full_matrices=False);rn=sum(sn>max(sn[0],1e-100)*1e-8);ht=ht-un[:,:rn]@(un[:,:rn].T@ht)
 sv=np.linalg.svd(ht/np.sqrt(max(n,1)),compute_uv=False)
 return dict(rank=rank,dim=lik.dim,null_leakage=leak,target_variance=var if leak<1e-7 else np.inf,target_information=1/var if leak<1e-7 and var>0 else 0.,sigma_min_per_sample=float(sv[-1]),theta_rank=int(sum(sv>max(sv[0],1e-100)*1e-8)),tolerance=tol,tolerance_stable=all((v['leak']<1e-7)==(leak<1e-7) for v in tol.values()))

def profile(lik,r,a,family_size=1):
 a=np.pad(np.asarray(a),(0,lik.dim-len(a)));x=np.array(r['x']);g=geometry(lik,x,a);out={**g,'estimate':float(a@x),'lo':None,'hi':None,'sign':'NUMERICAL_FAILURE','status':'FIT_FAILED','interval_type':'LOCAL_PROFILE_EXPLORATORY_BONFERRONI' if family_size>1 else 'LOCAL_PROFILE_EXPLORATORY','family_size':family_size,'nominal_alpha':.05/family_size}
 if not r['success']:return out
 if g['null_leakage']>1e-7:out.update(lo=-np.inf,hi=np.inf,sign='UNKNOWN',status='COMPLETE',interval_type='LOCAL_TANGENT_UNRESOLVED_OUTER_R');return out
 cutoff=r['nll']+chi2.ppf(1-.05/family_size,1)/2;ends=[];attempts=[]
 for direction in [1.,-1.]:
  o=minimize(lambda z:direction*(a@z),x,jac=lambda z:direction*a,bounds=lik.bounds,constraints=[dict(type='ineq',fun=lambda z:cutoff-lik.objective(z)[0],jac=lambda z:-lik.objective(z)[1])],method='SLSQP',options=dict(maxiter=300,ftol=1e-8));v=lik.objective(o.x)[0];ok=bool(o.success and v<=cutoff+1e-5);ends.append(float(a@o.x));active=np.flatnonzero(a);touch=any(abs(o.x[k]-lik.bounds[k][0])<.001 or abs(o.x[k]-lik.bounds[k][1])<.001 for k in active);attempts.append(dict(success=ok,message=str(o.message),nll=v,active_target_boundary=touch))
 out['attempts']=attempts
 if not all(z['success'] for z in attempts):out['status']='PROFILE_FAILED';return out
 out.update(lo=-np.inf if attempts[0]['active_target_boundary'] else ends[0],hi=np.inf if attempts[1]['active_target_boundary'] else ends[1],status='COMPLETE');out['boundary_hit']=not np.isfinite([out['lo'],out['hi']]).all();out['sign']='POSITIVE' if out['lo']>0 else 'NEGATIVE' if out['hi']<0 else 'UNKNOWN'
 if out['boundary_hit']:out['interval_type']='BOUND_TRUNCATED_EXTENDED_TO_INFINITY'
 return out

def deviance(lik,r):
 sat=0
 for c,p,j in lik.outputs(np.array(r['x'])):
  ok=c>0
  if c.sum()>0:sat-=float(c[ok]@np.log(c[ok]/c.sum()))
 nll=sum(float(-c@np.log(np.maximum(p,1e-280))) for c,p,j in lik.outputs(np.array(r['x'])))
 return max(0,2*(nll-sat))
