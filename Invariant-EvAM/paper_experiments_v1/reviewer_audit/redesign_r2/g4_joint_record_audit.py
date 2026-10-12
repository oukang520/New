from pathlib import Path
import json,hashlib,datetime
import numpy as np,pandas as pd
from scipy.stats import hypergeom
P=Path(__file__).resolve().parents[2];D=Path(__file__).resolve().parent;G=P/'g4/revision_r2'

def finite_interval(k,n,N,alpha=.05):
 k=np.asarray(k,dtype=int);l=k.copy();h=N-n+k
 while np.any(l<h):
  mid=(l+h)//2;ok=hypergeom.sf(k-1,N,mid,n)>=alpha/2;h=np.where(ok,mid,h);l=np.where(ok,l,mid+1)
 low=l.copy();l=k.copy();h=N-n+k
 while np.any(l<h):
  mid=(l+h+1)//2;ok=hypergeom.cdf(k,N,mid,n)>=alpha/2;l=np.where(ok,mid,l);h=np.where(ok,h,mid-1)
 return low/N,l/N

def run():
 cfg=json.loads((D/'G4_JOINT_RECORD_AUDIT_FREEZE.json').read_text());rows=[];summaries=[]
 for ci,cancer in enumerate(['LUAD','COAD']):
  records=json.loads((P/f'g4/data_private/reference_records/{cancer}.json').read_text());states=np.array([z['reference_state'] for z in records]);x0=(states>>3)&1;x1=(states>>2)&1;joint=x0*x1;N=len(states);truth=float(joint.mean())
  for budget in cfg['budgets']:
   n=budget//2;cts=[]
   for rep in range(cfg['replicates']):
    rng=np.random.default_rng(np.random.SeedSequence([2026092402,4,811,ci,budget,rep]));ids=rng.permutation(N);i0=ids[:n];i1=ids[n:2*n];cts.append((int(x0[i0].sum()),int(x1[i1].sum()),int(joint[i0].sum())))
   c=np.array(cts);l0,u0=finite_interval(c[:,0],n,N,.025);l1,u1=finite_interval(c[:,1],n,N,.025);lj,uj=finite_interval(c[:,2],n,N,.05);intervals={'SEPARATE_SHARP':(np.maximum(0,l0+l1-1),np.minimum(u0,u1)),'JOINT_EXACT':(lj,uj),'INDEPENDENCE_IMPUTATION':(l0*l1,u0*u1)}
   for method,(lo,hi) in intervals.items():
    for rep in range(cfg['replicates']):rows.append(dict(cancer=cancer,budget=budget,replicate=rep,method=method,N_population=N,n_unique_observed=n if method=='JOINT_EXACT' else 2*n,cost=budget,target='reported_primary_pair_joint_frequency',truth=truth,count0=c[rep,0],count1=c[rep,1],count_joint=c[rep,2],lo=lo[rep],hi=hi[rep],width=hi[rep]-lo[rep],covered=bool(lo[rep]-1e-12<=truth<=hi[rep]+1e-12),natural_theta_truth=False,evidence_type='RANDOMIZED_FINITE_COHORT_AUDIT_PREVIOUSLY_EXPLORED'))
   print(cancer,budget,'done',flush=True)
 d=pd.DataFrame(rows);d.to_csv(G/'results/G4_R2_joint_audit_raw.tsv',sep='\t',index=False)
 for key,q in d.groupby(['cancer','budget','method']):
  summaries.append(dict(zip(['cancer','budget','method'],key),n=len(q),covered=int(q.covered.sum()),coverage=q.covered.mean(),mean_width=q.width.mean(),truth=q.truth.iloc[0],n_unique=q.n_unique_observed.iloc[0],cost=q.cost.iloc[0]))
 pd.DataFrame(summaries).to_csv(G/'results/G4_R2_joint_audit_summary.tsv',sep='\t',index=False)
 pairs=[]
 for key,q in d.groupby(['cancer','budget']):
  w=q.pivot(index='replicate',columns='method',values='width');v=w.SEPARATE_SHARP-w.JOINT_EXACT
  for rep,z in v.items():pairs.append(dict(cancer=key[0],budget=key[1],replicate=rep,width_reduction=z))
 pd.DataFrame(pairs).to_csv(G/'results/G4_R2_joint_audit_paired.tsv',sep='\t',index=False)
if __name__=='__main__':run()
