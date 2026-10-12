from pathlib import Path
import numpy as np
from scipy.sparse import coo_matrix

def topology(d,kind):
 if kind=='CHAIN':return [(i,i-1) for i in range(1,d)]
 if kind=='DENSE_DAG':return [(i,j) for i in range(d) for j in range(i)]
 if kind=='FULL':return [(i,j) for i in range(d) for j in range(d) if i!=j]
 raise ValueError(kind)

def snapshot(theta,d,edges,time=1.):
 n=1<<d;p=len(theta);X=(np.arange(n)[:,None]>>(d-1-np.arange(d)))&1;src=[];dst=[];F=[]
 for s in range(n):
  for e in range(d):
   if X[s,e]:continue
   f=np.zeros(p);f[e]=1
   for k,(i,j) in enumerate(edges):
    if i==e:f[d+k]=X[s,j]
   src.append(s);dst.append(s|(1<<(d-1-e)));F.append(f)
 src=np.array(src);dst=np.array(dst);F=np.array(F);rates=np.exp(F@theta)*time;exit=np.bincount(src,weights=rates,minlength=n);lam=max(exit.max(),1e-12);ids=np.arange(len(src));B=coo_matrix((np.r_[np.ones(len(src)),-np.ones(len(src))],(np.r_[dst,src],np.r_[ids,ids])),shape=(n,len(src))).tocsr();Q=coo_matrix((np.r_[rates,-exit],(np.r_[dst,np.arange(n)],np.r_[src,np.arange(n)])),shape=(n,n)).tocsr();v=np.zeros(n);v[0]=1;dv=np.zeros((n,p));w=np.exp(-lam);prob=w*v;jac=w*dv
 if lam>600:raise ValueError('uniformization rate too large; do not silently underflow')
 for k in range(1,2000):
  nd=dv+Q@dv/lam+B@((rates*v[src])[:,None]*F)/lam;nv=v+Q@v/lam;v=nv;dv=nd;w*=lam/k;prob+=w*v;jac+=w*dv
  if k>lam+15 and w<1e-17:break
 assert abs(prob.sum()-1)<1e-10 and np.max(abs(jac.sum(axis=0)))<1e-9
 return prob,jac,X

def operator(prob,jac,X,mask,feature=None,phi=0.):
 if feature is None:q=prob;j=jac;jn=np.empty((len(q),0))
 else:
  f=X[:,feature];w=np.exp(phi*f);q=prob*w;q/=q.sum();j=w[:,None]*jac/(prob@w)-q[:,None]*(w@jac/(prob@w));jn=q[:,None]*(f-q@f)[:,None]
 inds=np.flatnonzero(mask);code=X[:,inds]@2**np.arange(len(inds)-1,-1,-1);A=(np.arange(2**len(inds))[:,None]==code).astype(float);return A@q,A@j,A@jn

def geometry(outputs,ns,target):
 blocks=[]
 for (q,j,jn),n in zip(outputs,ns):
  s=np.sqrt(np.maximum(q,1e-300));H=j/s[:,None];N=jn/s[:,None]
  if N.shape[1]:
   u,sv,_=np.linalg.svd(N,full_matrices=False);r=sum(sv>sv[0]*1e-8);H-=u[:,:r]@(u[:,:r].T@H)
  blocks.append(np.sqrt(n)*H)
 H=np.vstack(blocks);u,s,v=np.linalg.svd(H,full_matrices=False);r=sum(s>s[0]*1e-8);a=np.eye(H.shape[1])[target];leak=np.linalg.norm(a-v[:r].T@(v[:r]@a));var=np.sum(((v[:r]@a)/s[:r])**2) if leak<1e-7 else np.inf;leaks=np.sqrt(np.maximum(0,1-np.sum(v[:r]**2,axis=0)));ranks={str(t):int(sum(s>s[0]*t)) for t in [1e-6,1e-7,1e-8,1e-9,1e-10]}
 return dict(rank=int(r),dimension=H.shape[1],target_identified=bool(leak<1e-7),target_leak=float(leak),target_information=float(1/var) if np.isfinite(var) and var>0 else 0.,identified_coordinates=int(sum(leaks<1e-6)),sigma_min=float(s[-1]) if r==H.shape[1] else 0.,rank_tolerance_stable=len(set(ranks.values()))==1,rank_by_tolerance=ranks,singular_values=s.tolist())
