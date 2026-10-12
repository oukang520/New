"""Independent fixed-target reoptimization of predeclared endpoint witnesses, not a global proof."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import json,numpy as np,pandas as pd
from scipy.optimize import minimize
from scipy.stats import chi2
from g3_design import G
from g3_model import Joint

def main():
 rows=[]
 for cell in [0,3,6,10,12]:
  for rep in range(3):
   p=G/'runs/core'/f'cell{cell:02d}_rep{rep:04d}.json'
   if not p.exists():continue
   o=json.loads(p.read_text())
   for a in o['arms']:
    if a['method'] in ['SENSITIVITY_U','FLEXIBLE_U','FLEXIBLE_SELECTION']:continue
    l=Joint(a['data'],a['spec']);r=a['result']
    if r['status']!='COMPLETE':continue
    for endpoint in ['lo','hi']:
     value=r[endpoint]
     if not np.isfinite(value):continue
     free=np.arange(l.dim)!=l.index;base=np.array(r['x']);base[l.index]=value;limit=r['nll']+chi2.ppf(.95,1)/2;best=np.inf
     for start in [base,l.initial()]:
      start=start.copy();start[l.index]=value
      def fun(v):
       z=start.copy();z[free]=v;f,g=l.objective(z);return f,g[free]
      opt=minimize(fun,start[free],jac=True,bounds=[b for k,b in enumerate(l.bounds) if free[k]],method='L-BFGS-B',options={'maxiter':500,'ftol':1e-12,'gtol':1e-7});best=min(best,float(opt.fun))
     witness=r['interval_attempts'][0 if endpoint=='lo' else 1]['x'];witness_lr=2*(l.objective(np.array(witness))[0]-r['nll'])
     rows.append(dict(original_witness_lr=witness_lr,original_witness_feasible=bool(witness_lr<=chi2.ppf(.95,1)+2e-5),cell=cell,replicate=rep,method=a['method'],endpoint=endpoint,target=value,profile_lr_reoptimized=2*(best-r['nll']),nominal_lr_cutoff=chi2.ppf(.95,1),nll_below_reported_boundary=limit-best,feasible_witness=bool(best<=limit+1e-5),scope='LOCAL_ENDPOINT_REOPTIMIZATION_NOT_GLOBAL_SEARCH'))
 pd.DataFrame(rows).to_csv(G/'results/G3_interval_endpoint_audit.tsv',sep='\t',index=False);print('audited',len(rows),'finite endpoints')
if __name__=='__main__':main()
