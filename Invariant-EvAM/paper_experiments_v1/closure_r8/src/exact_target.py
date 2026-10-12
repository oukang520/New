"""Classical conditional binomial confidence bounds projected through linear constraints.
Target: theta_(B<-A) of a CLOSED two-event irreversible subsystem with A autonomous.
Model: rates a (A), b (B before A), c (B after A); theta=log(c/b), horizon=1.
Unknown selection on A or B preserves opposite conditional distributions.
"""
import numpy as np
from scipy.stats import beta
from scipy.optimize import linprog,brentq

def probs(rates):
 a,b,c=np.asarray(rates);L=a+b;p00=np.exp(-L)
 p10=a*np.exp(-L)*(-np.expm1(-(c-L)))/(c-L) if abs(c-L)>1e-8 else a*np.exp(-L)
 u=np.exp(-a);return np.array([p00,u-p00,p10,1-u-p10])
def invert(p00,u,v):
 if not (0<p00<u<1) or v<=0:return np.nan
 a=-np.log(u);L=-np.log(p00);b=L-a
 vmax=a*(-np.expm1(-L))/L
 if v>=vmax:return -np.inf
 def fun(c):return probs([a,b,c])[2]-v
 hi=max(1.,L)
 while fun(hi)>0 and hi<1e6:hi*=2
 if fun(hi)>0:return np.inf
 c=brentq(fun,0.,hi,xtol=1e-11);return np.log(c/b) if c>0 else -np.inf

def cp(k,n,alpha):
 if n==0:return 0.,1.
 return (0. if k==0 else beta.ppf(alpha/2,k,n-k+1),1. if k==n else beta.ppf(1-alpha/2,k+1,n-k))

def polytope(groups,alpha=.05):
 # Each source group supplies three event probabilities. Selected-A/B supplies two conditionals.
 # Groups must be independent confirmation samples, not pilot-selected reuse.
 constraints=sum(3 if g['kind']=='SOURCE' else 1 if g['kind'].startswith('MARGINAL') else 2 for g in groups)
 A=[];B=[]
 def bound(num,den,k,n):
  lo,hi=cp(k,n,alpha/max(constraints,1));num=np.array(num,float);den=np.array(den,float);A.extend([num-hi*den,lo*den-num]);B.extend([0.,0.])
 for g in groups:
  c=np.array(g['counts'],int)
  if g['kind']=='SOURCE':
   for vec in [[1,0,0,0],[1,1,0,0],[0,0,1,0]]:bound(vec,[1,1,1,1],int(np.array(vec)@c),int(c.sum()))
  elif g['kind']=='MARGINAL_A':bound([0,0,1,1],[1,1,1,1],c[1],c.sum())
  elif g['kind']=='MARGINAL_B':bound([0,1,0,1],[1,1,1,1],c[1],c.sum())
  elif g['kind']=='SELECT_A':
   bound([0,1,0,0],[1,1,0,0],c[1],c[0]+c[1]);bound([0,0,0,1],[0,0,1,1],c[3],c[2]+c[3])
  elif g['kind']=='SELECT_B':
   bound([0,0,1,0],[1,0,1,0],c[2],c[0]+c[2]);bound([0,0,0,1],[0,1,0,1],c[3],c[1]+c[3])
  else:raise ValueError(g['kind'])
 return np.asarray(A),np.asarray(B)

def box_bounds(vertices):
 (plo,ulo,vlo)=vertices.min(axis=0);(phi,uhi,vhi)=vertices.max(axis=0)
 pad=1e-10;plo-=pad;ulo-=pad;vlo-=pad;phi+=pad;uhi+=pad;vhi+=pad
 lo=invert(plo,uhi,vhi) if 0<plo<uhi<1 and vhi>0 else -np.inf
 hi=invert(phi,ulo,vlo) if 0<phi<ulo<1 and vlo>0 else np.inf
 if np.isnan(lo):lo=-np.inf
 if np.isnan(hi) or hi==-np.inf:hi=np.inf
 return lo,hi

def confidence(groups,alpha=.05,max_splits=96):
 from scipy.spatial import HalfspaceIntersection,ConvexHull,QhullError
 A,B=polytope(groups,alpha);A=np.vstack([A,-np.eye(4)]);B=np.r_[B,np.zeros(4)];T=np.array([[1,0,0],[-1,1,0],[0,0,1],[0,-1,-1]]);offset=np.array([0,0,0,1]);H=A@T;b=B-A@offset;norm=np.linalg.norm(H,axis=1);use=norm>1e-14
 if np.any(b[~use]<-1e-10):return dict(lo=-np.inf,hi=np.inf,status='EMPTY_OUTER_R',finite=False)
 H=H[use];b=b[use];norm=norm[use]
 center=linprog([0,0,0,-1],A_ub=np.column_stack([H,norm]),b_ub=b,bounds=[(None,None)]*3+[(0,None)],method='highs')
 if not center.success or center.x[3]<1e-11:return dict(lo=-np.inf,hi=np.inf,status='EMPTY_OR_NUMERICAL_OUTER_R',finite=False)
 try:vertices=HalfspaceIntersection(np.column_stack([H,-b]),center.x[:3]).intersections
 except QhullError:return dict(lo=-np.inf,hi=np.inf,status='NUMERICAL_OUTER_R',finite=False)
 scale=np.maximum(np.ptp(vertices,axis=0),1e-10);nodes=[(vertices,box_bounds(vertices))]
 for step in range(max_splits):
  los=[x[1][0] for x in nodes];his=[x[1][1] for x in nodes];idx=int(np.argmin(los)) if step%2==0 else int(np.argmax(his));v,bounds=nodes.pop(idx);axis=int(np.argmax(np.ptp(v,axis=0)/scale));mid=(v[:,axis].min()+v[:,axis].max())/2
  try:
   hull=ConvexHull(v);edges=set()
   for tri in hull.simplices:
    for i in range(3):edges.add(tuple(sorted((tri[i],tri[(i+1)%3]))))
   cross=[]
   for i,j in edges:
    if (v[i,axis]-mid)*(v[j,axis]-mid)<0:cross.append(v[i]+(mid-v[i,axis])/(v[j,axis]-v[i,axis])*(v[j]-v[i]))
   if not cross:nodes.append((v,bounds));break
   cross=np.asarray(cross);children=[]
   for keep in [v[:,axis]<=mid,v[:,axis]>=mid]:
    child=np.vstack([v[keep],cross]);child=child[ConvexHull(child).vertices];children.append((child,box_bounds(child)))
   nodes.extend(children)
  except QhullError:nodes.append((v,bounds));break
 lo=min(x[1][0] for x in nodes);hi=max(x[1][1] for x in nodes)
 return dict(lo=float(lo),hi=float(hi),status='COMPLETE',finite=bool(np.isfinite([lo,hi]).all()),subdivisions=len(nodes)-1)

def selected(p,kind,phi):
 f=np.array([0,0,1,1]) if kind=='SELECT_A' else np.array([0,1,0,1]);q=p*np.exp(phi*f);return q/q.sum()
