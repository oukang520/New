import numpy as np
from scipy.optimize import minimize
from exact_target import probs,selected

def distribution(x,kind):
 p=probs(np.exp(x[:3]))
 if kind=='SOURCE':return p
 if kind=='MARGINAL_A':return np.array([p[0]+p[1],p[2]+p[3]])
 if kind=='MARGINAL_B':return np.array([p[0]+p[2],p[1]+p[3]])
 return selected(p,kind,x[3] if kind=='SELECT_A' else x[4])
def derivative(x,kind):
 p=distribution(x,kind);j=np.zeros((len(p),5))
 for k in range(5):
  z=np.array(x,dtype=complex);z[k]+=1e-20j;j[:,k]=distribution(z,kind).imag/1e-20
 return p,j

def fit(pilot):
 def fg(x):
  v=0.;g=np.zeros(5)
  for row in pilot:
   q,j=derivative(x,row['kind']);c=np.array(row['counts']);q=np.maximum(q,1e-100);v-=c@np.log(q);g-=j.T@(c/q)
  return v,g
 rs=[minimize(fg,x,jac=True,bounds=[(-4,1.5)]*3+[(-5,5)]*2,method='L-BFGS-B',options=dict(maxiter=500,ftol=1e-12)) for x in [[-1,-1,-1,0,0],[-1,-1,-.5,.5,-.5]]];r=min(rs,key=lambda z:z.fun);return dict(x=r.x.tolist(),success=bool(r.success),nll=float(r.fun))

def choose(pilot,designs):
 f=fit(pilot);x=np.array(f['x']);a=np.array([0,-1,1,0,0]);rows=[]
 for name,groups in designs.items():
  H=[]
  for kind,n in groups.items():
   if n==0:continue
   p,j=derivative(x,kind);H.append(np.sqrt(n)*j/np.sqrt(np.maximum(p[:,None],1e-15)))
  H=np.vstack(H);u,s,v=np.linalg.svd(H,full_matrices=False);rank=sum(s>s[0]*1e-8);leak=np.linalg.norm(a-v[:rank].T@(v[:rank]@a));var=np.sum((v[:rank]@a/s[:rank])**2) if leak<1e-7 else np.inf
  rows.append(dict(action=name,rank=int(rank),leak=leak,variance=var))
 choice=min(rows,key=lambda z:(z['leak']>=1e-7,z['variance'],z['action']))['action'];return choice,rows,f
