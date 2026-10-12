"""Observed-input estimator. Never reads G4 private records or generator/scorer modules."""
from pathlib import Path
import sys,copy,time
import numpy as np
from scipy.special import softmax
from scipy.optimize import minimize
from scipy.stats import chi2
R=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(R/'paper_experiments_v1/g3/src'))
from g3_model import snapshot,states,selected,Joint,fit

class Likelihood(Joint):
 def __init__(self,data,spec,mode='joint',bound=3):
  self.mode=mode;self.data=data;self.spec=spec;self.d=4;self.p=15 if mode=='fusion' else 10;self.E=len(data['counts']);self.index=4
  self.bounds=[(-12.,12.)]*15 if mode=='fusion' else [(-float(bound),float(bound))]*10
  self.delta=[None]*self.E
  if mode=='hierarchical':
   for e in range(1,self.E):self.delta[e]=slice(len(self.bounds),len(self.bounds)+10);self.bounds += [(-2.,2.)]*10
  self.phi=[];self.alpha=[]
  for f in spec['features']:
   if spec.get('known_phi') is not None:self.phi.append(None)
   else:self.phi.append(slice(len(self.bounds),len(self.bounds)+len(f[0])));self.bounds += [(-1.,1.)]*len(f[0])
   if data.get('calibration_kind')=='XS':self.alpha.append(len(self.bounds));self.bounds.append((-12.,-float(np.abs(f).max(axis=0).sum())-.1))
   else:self.alpha.append(None)
  self.dim=len(self.bounds);self.cache=None;self.time_index=None
 def initial(self):
  z=np.zeros(self.dim)
  if self.mode!='fusion':z[:4]=-1
  for e,a in enumerate(self.alpha):
   if a is not None:z[a]=-float(np.abs(self.spec['features'][e]).max(axis=0).sum())-1.
  return z
 def source(self,x,e=0):
  if self.mode=='fusion':
   p=softmax(np.r_[0,x[:15]]);j=p[:,None]*(np.eye(16)[:,1:]-p[None,1:]);J=np.zeros((16,self.dim));J[:,:15]=j
  else:
   theta=x[:10].copy()
   if self.delta[e] is not None:theta+=x[self.delta[e]]
   p,j=snapshot(theta,4);J=np.zeros((16,self.dim));J[:,:10]=j
   if self.delta[e] is not None:J[:,self.delta[e]]=j
  return p,J
 def outputs(self,x):
  if self.cache is not None and np.array_equal(x,self.cache[0]):return self.cache[1]
  out=[]
  for e,c in enumerate(self.data['counts']):
   p,J=self.source(x,e);f=np.asarray(self.spec['features'][e]);phi=np.asarray(self.spec['known_phi'][e]) if self.phi[e] is None else x[self.phi[e]]
   prob,jt,jp=selected(p,J,f,phi,np.asarray(self.spec['measurement'][e]))
   if self.phi[e] is not None:jt[:,self.phi[e]]+=jp
   out.append((np.asarray(c),prob,jt))
   if self.data.get('calibration') is not None:
    counts=np.asarray(self.data['calibration'][e]);kind=self.data['calibration_kind']
    if kind=='X':out.append((counts,p,J))
    else:
     s=np.exp(x[self.alpha[e]]+f@phi);ds=np.zeros((16,self.dim));ds[:,self.alpha[e]]=s
     if self.phi[e] is not None:ds[:,self.phi[e]]=s[:,None]*f
     if np.max(s)>=1:raise ValueError('Selection probability exceeds one')
     out.append((counts,np.r_[p*(1-s),p*s],np.vstack(((1-s[:,None])*J-p[:,None]*ds,s[:,None]*J+p[:,None]*ds))))
  self.cache=(x.copy(),out);return out
 def objective(self,x):
  v,g=super().objective(x)
  if self.mode=='hierarchical':
   for sl in self.delta:
    if sl is not None:v+=.5*float(x[sl]@x[sl]);g[sl]+=x[sl]
  return v,g

def projection(p,bound=3):
 spec=dict(d=4,features=[np.zeros((16,1)).tolist()],measurement=[np.eye(16).tolist()],known_phi=[[0.]])
 lik=Likelihood(dict(counts=[np.asarray(p).tolist()]),spec,bound=bound);r=fit(lik);return r,lik

def interval(lik,r):
 x=np.array(r['x']);g=lik.geometry(x);out={**g,'estimate':float(x[4]),'interval_type':'LOCAL_PROFILE_EXPLORATORY','sign':'UNKNOWN','lo':None,'hi':None,'interval_status':'NOT_RUN'}
 if not r['success']:out['interval_status']='FIT_FAILED';return out
 if g['null_leakage']>1e-7:out.update(lo=-np.inf,hi=np.inf,interval_status='COMPLETE',interval_type='LOCAL_TANGENT_UNRESOLVED_OUTER_R');return out
 limit=r['nll']+chi2.ppf(.95,1)/2;ends=[];attempts=[]
 for direction in [1,-1]:
  grad=np.zeros(lik.dim);grad[4]=direction
  o=minimize(lambda z:direction*z[4],x,jac=lambda z:grad,bounds=lik.bounds,constraints=[dict(type='ineq',fun=lambda z:limit-lik.objective(z)[0],jac=lambda z:-lik.objective(z)[1])],method='SLSQP',options=dict(maxiter=250,ftol=1e-8));feasible=lik.objective(o.x)[0]<=limit+1e-5;attempts.append(dict(success=bool(o.success),feasible=bool(feasible),nll=float(lik.objective(o.x)[0]),message=str(o.message)));ends.append(float(o.x[4]))
 out['interval_attempts']=attempts
 if not all(z['success'] and z['feasible'] for z in attempts):out['interval_status']='INTERVAL_FAILED';return out
 b=lik.bounds[4];out.update(lo=-np.inf if ends[0]<=b[0]+.001 else ends[0],hi=np.inf if ends[1]>=b[1]-.001 else ends[1],interval_status='COMPLETE');out['boundary_hit']=not np.isfinite([out['lo'],out['hi']]).all();out['sign']='POSITIVE' if out['lo']>0 else 'NEGATIVE' if out['hi']<0 else 'UNKNOWN'
 if out['boundary_hit']:out['interval_type']='BOUND_TRUNCATED_EXTENDED_TO_INFINITY'
 return out

def train(data,spec,method,profile=False):
 t=time.perf_counter();data=copy.deepcopy(data);spec=copy.deepcopy(spec)
 if method=='M_NO_CAL':data.pop('calibration',None);data.pop('calibration_kind',None)
 if method=='B1':
  spec['known_phi']=[[0.] for _ in spec['features']];spec['features']=[np.zeros((16,1)).tolist() for _ in spec['features']]
  if data.get('calibration_kind')=='XS':data['calibration']=[(np.array(c[:16])+np.array(c[16:])).tolist() for c in data['calibration']];data['calibration_kind']='X'
 if method=='B2':
  parts=[]
  for e in range(len(data['counts'])):
   de={k:([v[e]] if k in ['counts','calibration'] else v) for k,v in data.items()};sp={k:([v[e]] if k in ['features','measurement','known_phi','masks','operator_ids'] else v) for k,v in spec.items()};parts.append(train(de,sp,'M',profile=False))
  p=np.mean([z['source'] for z in parts],axis=0);proj,_=projection(p);ok=all(z['success'] for z in parts) and proj['success'];return dict(method=method,success=ok,status='COMPLETE' if ok else 'FIT_FAILED',parts=parts,source=p.tolist(),selected=[z['selected'][0] for z in parts],estimate=float(proj['x'][4]),interval_type='META_TARGET_REFUSED_UNLESS_ENV_IDENTIFIED',meta_target_allowed=all(z.get('null_leakage',1)<1e-7 for z in parts),runtime_seconds=time.perf_counter()-t)
 mode='fusion' if method=='B4' else 'hierarchical' if method=='B3' else 'joint';lik=Likelihood(data,spec,mode);r=fit(lik);x=np.array(r['x']);ps=[];ss=[]
 for e in range(lik.E):
  p,_=lik.source(x,e);ps.append(p);phi=np.asarray(spec['known_phi'][e]) if lik.phi[e] is None else x[lik.phi[e]];w=np.exp(np.asarray(spec['features'][e])@phi);s=p*w;s/=s.sum();ss.append(s)
 r.update(method=method,status='COMPLETE' if r['success'] else 'FIT_FAILED',source=ps[0].tolist(),selected=[z.tolist() for z in ss],runtime_seconds=time.perf_counter()-t,estimate=float(x[4]),interval_type='NOT_REQUESTED')
 if method=='B4':
  pr,_=projection(ps[0]);r['fusion_unprojected_source']=r['source'];r['projection']=pr;r['estimate']=float(pr['x'][4]);r['source']=snapshot(pr['x'][:10],4)[0].tolist();r['selected']=[]
  for e in range(lik.E):
   phi=x[lik.phi[e]];w=np.exp(np.asarray(spec['features'][e])@phi);s=np.array(r['source'])*w;r['selected'].append((s/s.sum()).tolist())
  r['success']=r['success'] and pr['success'];r['status']='COMPLETE' if r['success'] else 'FIT_FAILED'
 if method in ['M','M_NO_CAL','K0']:
  r.update(interval(lik,r) if profile else lik.geometry(x))
 r['runtime_seconds']=time.perf_counter()-t;return r

def full_reference(counts,profile=True,bound=3):
 spec=dict(d=4,features=[np.zeros((16,1)).tolist()],measurement=[np.eye(16).tolist()],known_phi=[[0.]])
 lik=Likelihood(dict(counts=[np.asarray(counts).tolist()]),spec,bound=bound);r=fit(lik);r.update(method='FULL_RECORD_REFERENCE',source=snapshot(r['x'][:10],4)[0].tolist(),estimate=float(r['x'][4]),status='COMPLETE' if r['success'] else 'FIT_FAILED')
 if profile:r.update(interval(lik,r))
 return r
