import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import numpy as np,pandas as pd
from scipy.special import roots_genlaguerre
from g3_model import *
from g3_design import theta,G,specifications
from g3_population import population_fit,normalized

def main():
 rows=[]
 for d in [3,4]:
  t=theta(d);x=states(d)
  for strength in [0,.5,1.]:
   for biology in [0,.2,.5]:
    for different in [False,True]:
     fs=[x[:,[0]],x[:,[1 if different else 0]]];tt=t.copy();tt[d]+=biology;probs=[normalized(snapshot(z,d)[0],f[:,0]*strength) for z,f in zip([t,tt],fs)];sp=dict(d=d,features=[f.tolist() for f in fs],measurement=[np.eye(2**d).tolist()]*2,epsilon=0)
     rows.append(dict(d=d,selection_strength=strength,biology_shift=biology,different_observation=different,**population_fit(probs,sp)))
 pd.DataFrame(rows).to_csv(G/'results/G3_population_strength_grid.tsv',sep='\t',index=False)
 rows=[]
 for d in [3,4]:
  t=theta(d)
  for cv in [.1,.5,1.]:
   shape=1/cv**2;nodes,w=roots_genlaguerre(64,shape-1);w/=w.sum();p64=sum(ww*snapshot(t,d,z/shape)[0] for z,ww in zip(nodes,w));p16=timed_snapshot(t,d,cv)[0]
   rows.append(dict(d=d,cv=cv,scope='FROZEN_TRUE_PARAMETER_ONLY',max_abs_probability_difference=float(max(abs(p64-p16))),sum_probability_16=float(sum(p16))))
 pd.DataFrame(rows).to_csv(G/'results/G3_time_quadrature_audit.tsv',sep='\t',index=False)
 print('Additional population grids complete')
if __name__=='__main__':main()
