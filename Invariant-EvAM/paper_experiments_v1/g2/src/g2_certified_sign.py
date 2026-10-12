"""Analytic global SIGN outer projection, independently evaluated after freeze.
No nuisance truth or generator parameters are visible to certify().
"""
from pathlib import Path
import sys,json,hashlib
import numpy as np,pandas as pd
from scipy.stats import beta
G=Path(__file__).resolve().parents[1]
from g2_engine import snapshot,operators,STATES
from g2_summarize import wilson

def certify(batch_counts,specs):
 nrep=len(batch_counts[0]);n_env=len(specs)
 # Reject unsupported operator/topology adaptations rather than extrapolate the proof.
 supported=all(s['panel_mask']==[1,1,1] and s['feature']['kind']=='terms' and s['feature']['terms'] in [[[0]],[[1]]] for s in specs)
 if not supported:return np.array(['REAL_LINE']*nrep),np.full((nrep,n_env),np.nan),np.full((nrep,n_env),np.nan)
 ors=[];ors_hi=[];alpha=.05/(4*n_env)
 for c in batch_counts:
  k=c.reshape(nrep,4,2).sum(axis=2);n=k.sum(axis=1)[:,None]
  lo=np.where(k==0,0.,beta.ppf(alpha/2,k,n-k+1));hi=np.where(k==n,1.,beta.ppf(1-alpha/2,k+1,n-k))
  ol=lo[:,0]*lo[:,3]/(hi[:,1]*hi[:,2]);den=lo[:,1]*lo[:,2];oh=np.divide(hi[:,0]*hi[:,3],den,out=np.full(nrep,np.inf),where=den>0);ors.append(ol);ors_hi.append(oh)
 lower=np.column_stack(ors);upper=np.column_stack(ors_hi);positive=(lower>1).any(axis=1);negative=(upper<1).any(axis=1)
 out=np.full(nrep,'REAL_LINE',dtype=object);out[positive]='POSITIVE_OPEN_HALF_LINE';out[negative]='NEGATIVE_OPEN_HALF_LINE';out[positive&negative]='EMPTY_CONTRADICTORY_BANDS';return out,lower,upper

def main():
 cfg=G/'configs/CERTIFIED_SIGN_ADDENDUM.json';freeze=G/'configs/CERTIFIED_SIGN_ADDENDUM.sha256';h=hashlib.sha256(cfg.read_bytes()).hexdigest()
 if freeze.exists():assert freeze.read_text().strip()==h
 else:freeze.write_text(h+'\n')
 pub=json.loads((G/'configs/public_contract.json').read_text());ev=json.loads((G/'configs/evaluator_contract.json').read_text());rng=np.random.default_rng(2309208);checks=[]
 for i in range(500):
  theta=rng.uniform(-2,2,5);p=snapshot(theta)[0];phi=rng.uniform(-2,2)
  for event in [0,1]:
   q=p*np.exp(phi*STATES[:,event]);q/=q.sum();v=q.reshape(4,2).sum(axis=1);odds=v[0]*v[3]/(v[1]*v[2]);checks.append({'draw':i,'selection_event':event,'target':theta[3],'odds_ratio':odds,'sign_equal':bool(np.sign(np.log(odds))==np.sign(theta[3]))})
 check=pd.DataFrame(checks);assert check.sign_equal.all();check.to_csv(G/'results/G2_sign_identity_check.tsv',sep='\t',index=False)
 rows=[];raw=[];summary=[]
 for cell in range(12):
  label=['strong','weak','nonidentifying'][cell//4];specs=pub['operators'][label];ops=operators(specs);truth=ev['effects'][cell%4];theta=np.r_[ev['baseline'],truth,ev['other_interaction']];p=snapshot(theta)[0];probs=[]
  for (a,f),phi in zip(ops,ev['true_phis'][label]):q=p*np.exp(f@np.array(phi));q/=q.sum();probs.append(a@q)
  counts=[[],[]]
  for rep in range(500):
   seed=[20260923,2,208,cell,0,rep];rng=np.random.default_rng(np.random.SeedSequence(seed));c=[rng.multinomial(500,q) for q in probs]
   for e in range(2):counts[e].append(c[e])
   raw.append({'cell_id':cell,'replicate_id':rep,'seed':json.dumps(seed),'observed_counts':json.dumps([x.tolist() for x in c])})
  sets,lo,hi=certify([np.array(c) for c in counts],specs)
  for rep,ss in enumerate(sets):
   called=ss in ['POSITIVE_OPEN_HALF_LINE','NEGATIVE_OPEN_HALF_LINE'];covered=ss=='REAL_LINE' or (ss=='POSITIVE_OPEN_HALF_LINE' and truth>0) or (ss=='NEGATIVE_OPEN_HALF_LINE' and truth<0);correct=called and covered
   rows.append({'cell_id':cell,'design':label,'replicate_id':rep,'truth':truth,'confidence_set':ss,'called':called,'covered':covered,'correct_call':correct,'wrong_call':called and not covered,'OR_lower_env0':lo[rep,0],'OR_upper_env0':hi[rep,0],'OR_lower_env1':lo[rep,1],'OR_upper_env1':hi[rep,1],'evidence_type':'INDEPENDENT_ADDENDUM_PURPOSE208','global_sign_outer_certificate':True,'finite_magnitude_bounds_certified':False})
  f=pd.DataFrame(rows[-500:]);k=int(f.covered.sum());called=int(f.called.sum());wrong=int(f.wrong_call.sum());cl,cu=wilson(k,500);wl,wu=wilson(wrong,called)
  summary.append({'cell_id':cell,'design':label,'truth':truth,'n':500,'covered':k,'coverage':k/500,'coverage_wilson_low':cl,'coverage_wilson_high':cu,'called':called,'call_rate':called/500,'wrong':wrong,'conditional_wrong_risk':wrong/called if called else np.nan,'wrong_risk_wilson_low':wl,'wrong_risk_wilson_high':wu,'empty_sets':int(f.confidence_set.eq('EMPTY_CONTRADICTORY_BANDS').sum())});print('Certified sign addendum cell',cell,flush=True)
 pd.DataFrame(rows).to_csv(G/'results/G2_certified_sign_independent.tsv',sep='\t',index=False);pd.DataFrame(raw).to_csv(G/'results/G2_certified_sign_observed_counts.tsv.gz',sep='\t',index=False,compression='gzip');pd.DataFrame(summary).to_csv(G/'results/G2_certified_sign_summary.tsv',sep='\t',index=False)
if __name__=='__main__':main()
