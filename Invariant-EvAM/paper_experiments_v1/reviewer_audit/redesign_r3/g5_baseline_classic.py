"""Add the standard LR cutoff to the matched count-expansion control.
This prevents comparison only against the more conservative calibrated cutoff.
"""
import json
from concurrent.futures import ProcessPoolExecutor
from scipy.stats import chi2
import numpy as np,pandas as pd
from g5_anchor_interval import G,WithX,spec
from g5_interval import endpoints
def one(path):
 z=json.loads(path.read_text())
 if 'classic_interval' in z:return
 lik=WithX(z['public_data'],spec());pr=endpoints(lik,z['fit'],float(chi2.ppf(.95,1)));truth=z['truth'];pr.update(covered=bool(pr['lo']<=truth<=pr['hi']),called=bool(pr['lo']>0 or pr['hi']<0),wrong=bool((pr['lo']>0 and truth<=0) or (pr['hi']<0 and truth>=0)),width=pr['hi']-pr['lo']);z['classic_interval']=pr;path.write_text(json.dumps(z,default=lambda x:x.item() if isinstance(x,np.generic) else x.tolist()))
if __name__=='__main__':
 files=sorted((G/'runs').glob('anchor_baseline_*.json'))
 with ProcessPoolExecutor(4) as ex:
  for i,_ in enumerate(ex.map(one,files),1):
   if i%100==0:print(i,len(files),flush=True)
 rows=[]
 for f in files:
  z=json.loads(f.read_text())
  for m,k in [('COUNT_EXPANSION_CALIBRATED','interval'),('COUNT_EXPANSION_CLASSIC','classic_interval')]:rows.append(dict(scenario=z['scenario'],replicate=z['replicate'],method=m,cost=z['added_cost'],truth=z['truth'],**z[k]))
 d=pd.DataFrame(rows);d['finite']=np.isfinite(d.width);d.to_csv(G/'results/anchor_baseline_raw.tsv',sep='\t',index=False);d.groupby(['scenario','method']).agg(n=('replicate','size'),coverage=('covered','mean'),called=('called','mean'),wrong=('wrong','mean'),finite=('finite','mean')).reset_index().to_csv(G/'results/anchor_baseline_summary.tsv',sep='\t',index=False)
