"""Thin analytic-gradient adapter to the SAME installed SciPy L-BFGS-B kernel.
No change to stopping rules, line-search budget, correction history or bounds.
Derived from installed scipy.optimize._lbfgsb_py._minimize_lbfgsb (BSD-3-Clause).
Generic finite differences/callbacks/Hessian wrappers are unnecessary here.
"""
import numpy as np
from scipy.optimize import _lbfgsb,OptimizeResult

def minimize_direct(fun,x0,bounds,maxiter=400,ftol=1e-10,gtol=1e-6,maxls=30):
 n=len(x0);m=10;dt=_lbfgsb.types.intvar.dtype;b=np.asarray(bounds,float);x=np.clip(np.asarray(x0,float),b[:,0],b[:,1]);lower=np.ascontiguousarray(b[:,0]);upper=np.ascontiguousarray(b[:,1]);nbd=np.full(n,2,dtype=dt)
 f=np.array(0.,float);g=np.zeros(n);wa=np.zeros(2*m*n+5*n+11*m*m+8*m);iwa=np.zeros(3*n,dt);task=np.zeros(1,'S60');csave=np.zeros(1,'S60');lsave=np.zeros(4,dt);isave=np.zeros(44,dt);dsave=np.zeros(29);task[:]='START';nit=nfev=0
 while True:
  _lbfgsb.setulb(m,x,lower,upper,nbd,f,g,ftol/np.finfo(float).eps,gtol,wa,iwa,task,-1,csave,lsave,isave,dsave,maxls)
  t=task.tobytes()
  if t.startswith(b'FG'):
   f,g=fun(x);g=np.asarray(g,dtype=np.float64);nfev+=1
  elif t.startswith(b'NEW_X'):
   nit+=1
   if nit>=maxiter:task[:]='STOP: TOTAL NO. of ITERATIONS REACHED LIMIT'
   elif nfev>15000:task[:]='STOP: TOTAL NO. of f AND g EVALUATIONS EXCEEDS LIMIT'
  else:break
 msg=task.tobytes().strip(b'\x00').strip().decode();return OptimizeResult(x=x,fun=float(f),jac=g,nit=nit,nfev=nfev,success=msg.startswith('CONV'),message=msg)
