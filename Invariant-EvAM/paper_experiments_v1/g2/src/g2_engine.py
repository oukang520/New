"""Estimator-only code: observed counts + public operators; no truth files."""
from pathlib import Path
import ctypes
import numpy as np
from scipy.optimize import minimize
from lbfgsb_direct import minimize_direct
from scipy.special import softmax
from scipy.stats import norm,chi2
LIB=ctypes.CDLL(str(Path(__file__).with_name('ctmc_fast_v3.dylib')))
ARR=np.ctypeslib.ndpointer(dtype=np.float64,flags='C_CONTIGUOUS')
LIB.snapshot.argtypes=[ARR,ARR,ARR]
IARR=np.ctypeslib.ndpointer(dtype=np.int32,flags="C_CONTIGUOUS")
LIB.joint_fg.argtypes=[ctypes.c_void_p]*6+[ctypes.c_int]+[ctypes.c_void_p]*2
STATES=np.array([[s>>2&1,s>>1&1,s&1] for s in range(8)])
def snapshot(theta):
 t=np.ascontiguousarray(theta,dtype=float);p=np.empty(8);j=np.empty((8,5));LIB.snapshot(t,p,j);return p,j

def operators(specs):
 result=[]
 for s in specs:
  mask=np.asarray(s['panel_mask'],bool);obs=STATES[:,mask];ids=obs@(2**np.arange(obs.shape[1]-1,-1,-1));a=np.zeros((2**mask.sum(),8));a[ids,np.arange(8)]=1
  f=s['feature']
  if f['kind']=='none':features=np.empty((8,0))
  elif f['kind']=='burden':features=STATES.mean(axis=1,keepdims=True)
  else:features=np.column_stack([STATES[:,t].prod(axis=1) for t in f['terms']])
  result.append((a,features))
 return result

class Likelihood:
 def __init__(self,counts,ops,mode='joint',penalty=1.):
  self.counts=counts;self.ops=ops;self.mode=mode;self.penalty=penalty;self.e=len(ops)
  self.p=7 if mode=='fusion' else (5*self.e if mode=='hierarchical' else 5)
  self.q=[f.shape[1] for a,f in ops];self.dim=self.p+sum(self.q)
  self.fast_ready=mode=='joint' and max(self.q,default=0)<=2
  if self.fast_ready:
   self.fc=np.zeros((self.e,8));self.fa=np.zeros((self.e,8,8));self.ff=np.zeros((self.e,8,2));self.fw=np.array([len(a) for a,f in ops],dtype=np.int32);self.fq=np.array(self.q,dtype=np.int32)
   for e,(a,f) in enumerate(ops):self.fc[e,:len(a)]=counts[e];self.fa[e,:len(a)]=a;self.ff[e,:,:f.shape[1]]=f
   self.fptr=[a.ctypes.data for a in [self.fc,self.fa,self.ff,self.fw,self.fq]]
 def outputs(self,x):
  if self.mode=='fusion':
   p=softmax(np.r_[x[:7],0]);j=(np.diag(p)-np.outer(p,p))[:,:7]
  elif self.mode!='hierarchical':p,j=snapshot(x[:5])
  result=[];off=self.p
  for e,(a,f) in enumerate(self.ops):
   if self.mode=='hierarchical':p,je=snapshot(x[e*5:e*5+5]);j=np.zeros((8,self.p));j[:,e*5:e*5+5]=je
   phi=x[off:off+f.shape[1]];logs=f@phi;w=np.exp(logs-logs.max());u=p*w;z=u.sum();ps=u/z
   jt=w[:,None]*j/z-ps[:,None]*(w@j/z)[None,:]
   jp=ps[:,None]*(f-ps@f)
   dy=np.zeros((a.shape[0],self.dim));dy[:,:self.p]=a@jt;dy[:,off:off+f.shape[1]]=a@jp
   result.append((a@ps,dy));off+=f.shape[1]
  return result
 def objective(self,x):
  if self.fast_ready:
   x=np.ascontiguousarray(x,dtype=np.float64);value=np.empty(1);grad=np.empty(self.dim);LIB.joint_fg(x.ctypes.data,*self.fptr,self.e,value.ctypes.data,grad.ctypes.data);return float(value[0]),grad
  value=0.;grad=np.zeros(self.dim)
  for c,(p,j) in zip(self.counts,self.outputs(x)):
   p=np.maximum(p,1e-300);value-=c@np.log(p);grad-=j.T@(c/p)
  if self.mode=='hierarchical':
   t=x[:self.p].reshape(self.e,5);dev=t-t.mean(axis=0);value+=.5*self.penalty*np.sum(dev**2);grad[:self.p]+=self.penalty*dev.ravel()
  return value,grad
 def fisher(self,x):
  h=np.vstack([np.sqrt(c.sum())*j/np.sqrt(np.maximum(p[:,None],1e-12)) for c,(p,j) in zip(self.counts,self.outputs(x))]);info=h.T@h
  if self.mode=='hierarchical':info[:self.p,:self.p]+=self.penalty*np.kron(np.eye(self.e)-np.ones((self.e,self.e))/self.e,np.eye(5))
  return info

def fit(lik,bound=3.,starts=None,fixed=None,maxiter=400):
 bounds=[(-bound,bound)]*lik.p+[(-2.,2.) if lik.mode=='fusion' else (-2*bound/3,2*bound/3)]*(lik.dim-lik.p)
 if starts is None:
  x=np.zeros(lik.dim)
  if lik.mode not in ['fusion','hierarchical']:x[:3]=-1
  if lik.mode=='hierarchical':x[:lik.p]=np.tile([-1,-1,-1,0,0],lik.e)
  starts=[x]
 free=np.ones(lik.dim,bool)
 if fixed is not None:free[fixed[0]]=False
 best=None;attempts=[]
 for seed in starts:
  x=np.clip(np.asarray(seed,float),np.array(bounds)[:,0],np.array(bounds)[:,1])
  if fixed is not None:x[fixed[0]]=fixed[1]
  def fun(v):
   z=x.copy();z[free]=v;f,g=lik.objective(z);return f,g[free]
  opt=minimize_direct(fun,x[free],[b for i,b in enumerate(bounds) if free[i]],maxiter=maxiter,ftol=1e-10,gtol=1e-6,maxls=30)
  x[free]=opt.x;attempts.append({'success':bool(opt.success),'nll':float(opt.fun),'nit':int(opt.nit),'x':x.tolist(),'message':str(opt.message)})
  if best is None or opt.fun<best['nll']:best={'x':x,'nll':float(opt.fun),'success':bool(opt.success),'nit':int(opt.nit)}
 best['attempts']=attempts;best['boundary']=bool(np.any(np.abs(best['x'])>=np.array(bounds)[:,1]-1e-4));return best

def covariance(lik,x):
 h=np.vstack([np.sqrt(c.sum())*j/np.sqrt(np.maximum(p[:,None],1e-12)) for c,(p,j) in zip(lik.counts,lik.outputs(x))])
 if lik.mode=='hierarchical':
  penalty=np.zeros((lik.p,lik.dim));penalty[:,:lik.p]=np.sqrt(lik.penalty)*np.kron(np.eye(lik.e)-np.ones((lik.e,lik.e))/lik.e,np.eye(5));h=np.vstack([h,penalty])
 _,s,vh=np.linalg.svd(h,full_matrices=True);rank=int(np.sum(s>s[0]*1e-8));v=vh[:rank].T;inv=(v/(s[:rank]**2))@v.T;null=vh[rank:].T;return inv,null

def wald(lik,fitresult,a,alpha=.05):
 a=np.asarray(a);cov,null=covariance(lik,fitresult['x']);est=float(a@fitresult['x']);leak=float(np.linalg.norm(null.T@a))
 variance=float(a@cov@a);se=np.sqrt(max(0,variance));z=norm.ppf(1-alpha/2)
 return {'estimate':est,'lo':est-z*se if leak<=1e-7 else -np.inf,'hi':est+z*se if leak<=1e-7 else np.inf,'se':se,'null_leakage':leak}

def profile(lik,fitresult,index=3,bound=3.,grid=None,extra_restart=True):
 if grid is None:grid=np.unique(np.r_[np.linspace(-bound,bound,25),0.,fitresult['x'][index]])
 points=[];warm=fitresult['x'].copy();alternative=np.zeros(lik.dim);alternative[:min(3,lik.p)]=-1.5
 for v in sorted(grid):
  starts=[warm,fitresult['x']]
  if extra_restart:starts.append(alternative)
  out=fit(lik,bound,starts=starts,fixed=(index,float(v)),maxiter=250);warm=out['x'];points.append({'c':float(v),'nll':out['nll'],'success':out['success'],'x':out['x'].tolist(),'attempts':out['attempts']})
 return points

def interval(points,reference_nll,cutoff,bound):
 # Confidence components from a computed grid; crossings are interpolated and explicitly numerical.
 c=np.array([p['c'] for p in points]);lr=2*(np.array([p['nll'] for p in points])-reference_nll);ok=lr<=cutoff
 if not ok.any():return {'lo':np.nan,'hi':np.nan,'components':[],'boundary':False,'status':'EMPTY_OR_UNRESOLVED'}
 components=[]
 starts=np.flatnonzero(ok&~np.r_[False,ok[:-1]]);ends=np.flatnonzero(ok&~np.r_[ok[1:],False])
 for i,j in zip(starts,ends):
  lo=-np.inf if i==0 else float(np.interp(cutoff,[lr[i],lr[i-1]],[c[i],c[i-1]]))
  hi=np.inf if j==len(c)-1 else float(np.interp(cutoff,[lr[j],lr[j+1]],[c[j],c[j+1]]))
  components.append([lo,hi])
 return {'lo':components[0][0],'hi':components[-1][1],'components':components,'boundary':bool(ok[0] or ok[-1]),'status':'UNBOUNDED_OR_UNRESOLVED' if ok[0] or ok[-1] else 'LOCAL_PROFILE_EXPLORATORY'}

def sign(lo,hi,status='COMPLETE'):
 if status!='COMPLETE' or np.isnan(lo) or np.isnan(hi):return 'NUMERICAL_FAILURE'
 return 'POSITIVE' if lo>0 else ('NEGATIVE' if hi<0 else 'UNKNOWN')
