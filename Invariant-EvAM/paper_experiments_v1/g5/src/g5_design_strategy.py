"""Strategy API: observed baseline, public model, public capacities/budget only. No generator import or private files."""
import numpy as np
from g5_model import Model,fit,mask_matrix
A0=mask_matrix([1,0,1,1]);A1=mask_matrix([0,1,1,1]);AS=[A0,A1]
def design_spec():
 from g5_model import X
 return dict(topology='SPARSE',features=[X[:,[0]].astype(float).tolist(),X[:,[1]].astype(float).tolist()],known_phi=[None,None],phi_bound=1.,bound=3,xs=True)
def make_data(partial,full=None,cal=None):
 if full is None:full=[np.zeros(16)]*2
 if cal is None:cal=[np.zeros(32)]*2
 return [dict(env=e,A=AS[e].tolist(),counts=np.asarray(partial[e]).tolist()) for e in range(2)]+[dict(env=e,A=np.eye(16).tolist(),counts=np.asarray(full[e]).tolist()) for e in range(2)]+[dict(env=e,kind='XS',counts=np.asarray(cal[e]).tolist()) for e in range(2)]
def information_for(lik,x):
 mats=[]
 for c,p,j in lik.outputs(x):mats.append(j.T@(j/np.maximum(p[:,None],1e-15)))
 return mats

def target_metrics(I,p=10,ridge=1e-8):
 h=I[:p,:p];n=I[p:,p:]
 if len(n):v,u=np.linalg.eigh(n);keep=v>max(v.max(),1e-100)*1e-10;ni=(u[:,keep]/v[keep])@u[:,keep].T if keep.any() else np.zeros_like(n);h=h-I[:p,p:]@ni@I[p:,:p]
 h=(h+h.T)/2;v,u=np.linalg.eigh(h);cut=max(v.max(),1e-100)*1e-8;keep=v>cut;a=np.eye(p)[4];leak=float(np.linalg.norm(u[:,~keep].T@a));var=float(np.sum((u[:,keep].T@a)**2/v[keep])) if leak<1e-7 else np.inf
 return dict(rank=int(sum(keep)),null_leakage=leak,variance=var,logdet_ridge=float(np.log(np.maximum(v,0)+ridge).sum()),sigma_min=float(max(0,v.min())))

def candidates(budget,capacities):
 out=[]
 for add in range(min(capacities['add_pairs'],budget//6)+1):
  for cal in range(min(capacities['calibration_pairs'],(budget-6*add)//10)+1):
   reveal=min(50,(budget-6*add-10*cal)//2);cost=6*add+10*cal+2*reveal
   out.append(dict(candidate_id=f'A{add}_R{reveal}_C{cal}',add_pairs=add,reveal_pairs=reveal,calibration_pairs=cal,cost=cost,unused=budget-cost))
 # Count-only explicitly allowed to leave integer remainder rather than reveal.
 add=min(capacities['add_pairs'],budget//6);z=dict(candidate_id=f'A{add}_R0_C0',add_pairs=add,reveal_pairs=0,calibration_pairs=0,cost=6*add,unused=budget-6*add)
 if z['candidate_id'] not in [x['candidate_id'] for x in out]:out.append(z)
 return out

def recommend(observed_baseline,budget,capacities,random_seed):
 spec=design_spec();lik=Model(observed_baseline,spec);r=fit(lik);representatives=[np.array(z['x']) for z in r['attempts'] if z['success'] and z['nll']<=r['nll']+1.92073];x=np.array(r['x'])
 if not representatives:representatives=[x]
 units=[information_for(lik,z) for z in representatives];point=information_for(lik,x);cand=candidates(budget,capacities);table=[]
 def info(units,c):
  return sum(50*units[e] for e in range(2))+c['add_pairs']*(units[0]+units[1])+c['reveal_pairs']*(units[2]+units[3]-units[0]-units[1])+c['calibration_pairs']*(units[4]+units[5])
 for c in cand:
  I=info(point,c);m=target_metrics(I);rs=[target_metrics(info(u,c)) for u in units];table.append({**c,**m,'robust_unresolved':any(z['null_leakage']>=1e-7 for z in rs),'robust_variance':max(z['variance'] for z in rs),'robust_max_leak':max(z['null_leakage'] for z in rs),'ridge_sensitivity':{str(r):target_metrics(I,ridge=r)['logdet_ridge'] for r in [1e-10,1e-8,1e-6]}})
 idmap={z['candidate_id']:z for z in table};random=np.random.default_rng(random_seed);chosen={};chosen['RANDOM_FEASIBLE']=table[int(random.integers(len(table)))];chosen['COUNT_ONLY']=max([z for z in table if z['reveal_pairs']==0 and z['calibration_pairs']==0],key=lambda z:z['add_pairs']);chosen['MAX_PANEL']=max(table,key=lambda z:(z['reveal_pairs']+z['calibration_pairs'],-z['add_pairs'],-z['unused']));chosen['D_OPT']=max(table,key=lambda z:z['logdet_ridge']);chosen['C_OPT']=min(table,key=lambda z:(z['null_leakage']>=1e-7,z['variance'] if np.isfinite(z['variance']) else z['null_leakage'],-z['rank'],z['candidate_id']));chosen['INVARIANT_ROBUST']=min(table,key=lambda z:(z['robust_unresolved'],z['robust_variance'] if np.isfinite(z['robust_variance']) else z['robust_max_leak'],-z['rank'],z['candidate_id']));ridgechoices={str(r):max(table,key=lambda z:z['ridge_sensitivity'][str(r)])['candidate_id'] for r in [1e-10,1e-8,1e-6]}
 return dict(baseline_fit=r,choices=chosen,candidates=table,representative_count=len(representatives),ridge_choices=ridgechoices,robust_equals_c=chosen['C_OPT']['candidate_id']==chosen['INVARIANT_ROBUST']['candidate_id'])
