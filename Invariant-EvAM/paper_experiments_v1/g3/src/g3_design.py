"""Public frozen sentinel specifications. Constants are artificial stress scales."""
from pathlib import Path
import json,hashlib
import numpy as np
from g3_model import EDGES,states,snapshot
R=Path(__file__).resolve().parents[3];G=R/'paper_experiments_v1/g3'
def theta(d):return np.array([-1.,-.8,-1.2,-.4,.4] if d==3 else [-1.,-.8,-1.2,-.9,-.4,.4,-.3,.2,-.2,.3])
def cells():
 rows=[]
 for i,(kind,d,n) in enumerate([('same_same',3,200),('same_different',3,200),('different_same',3,200),('different_different',3,200),('local_aligned',4,200),('outside_nuisance',4,200),('calibration',3,500),('calibration',4,500),('common_zero',3,500),('common_mild',3,500),('common_moderate',3,500),('common_exact',3,500),('time',3,200),('higher_order',4,200),('mixture',4,200),('measurement',3,200)]):
  rows.append(dict(cell_id=i,group='G3A' if i<6 else 'G3B' if i<8 else 'G3C' if i<12 else 'G3D',kind=kind,d=d,replicates=n,n_budget=1000,target_definition='REFERENCE_TARGET',reference_environment=0,primary_index=d,artificial_stress=True))
 return rows

def specifications(cell):
 d=cell['d'];kind=cell['kind'];x=states(d);t=theta(d);f=[x[:,[0]].astype(float),x[:,[1]].astype(float)]
 if kind in ['same_same','different_same','calibration']:f[1]=f[0].copy()
 if kind=='local_aligned':
  p,j=snapshot(t,d);score=j[:,d]/p;score/=abs(score).max();f[1]=score[:,None]
 if kind=='outside_nuisance':f[1]=x[:,[0]].astype(float)
 base=dict(d=d,features=[z.tolist() for z in f],measurement=[np.eye(2**d).tolist()]*2,epsilon=0)
 pair=(x[:,0]*x[:,1])[:,None]
 if kind=='common_exact':
  alt=t.copy();alt[d]=.4;pair=np.log(snapshot(alt,d)[0]/snapshot(t,d)[0])[:,None]
 flex={**base,'features':[np.column_stack((z,pair)).tolist() for z in f]}
 return base,flex

def arms(cell):
 base,flex=specifications(cell);kind=cell['kind'];d=cell['d'];out=[]
 def add(name,sp,share=0,cal=None,envs=2):out.append(dict(name=name,spec=sp,share=share,calibration_kind=cal,envs=envs))
 if cell['group']=='G3A':
  add('SHARED_K1',base);add('BOUNDED_BIOLOGY',{**base,'epsilon':.5});add('FLEXIBLE_SELECTION',flex);add('ORACLE_STRATIFIED',{**base,'epsilon':3.,'known_phi':[[.5],[.5]]})
 elif kind=='calibration':
  add('ORDINARY_ONLY',base)
  for share in [.10,.25]:
   add('RANDOM_X_'+str(share),base,share,'X');add('CALIBRATION_XS_'+str(share),base,share,'XS')
  add('KNOWN_OPERATOR_ORACLE',{**base,'known_phi':[[.5],[.5]]})
 elif cell['group']=='G3C':
  add('OMITTED_U',base);add('MORE_ENVS_SAME_BUDGET',{**base,'features':base['features']*2,'measurement':base['measurement']*2},envs=4)
  add('FLEXIBLE_U',flex);add('CALIBRATED_FLEXIBLE_U',flex,.25,'XS')
  # Zero-centered artificial U envelope includes 0/0.5/1 and the exact-ratio coefficient 1.
  add('SENSITIVITY_U',flex)
 else:
  add('MISSPECIFIED_SHARED',base)
  if kind=='time':
   add('KNOWN_TIME_FAMILY',{**base,'cv':.5});add('ESTIMATED_TIME_FAMILY',{**base,'estimate_cv':True})
  elif kind=='measurement':
   from g3_model import measure
   add('KNOWN_MEASUREMENT',{**base,'measurement':[measure(d,.85,.02).tolist()]*2});add('BIOLOGY_ENVELOPE',{**base,'epsilon':.5})
  elif kind=='higher_order':add('FLEXIBLE_SELECTION',flex);add('BIOLOGY_ENVELOPE',{**base,'epsilon':.5})
  elif kind=='mixture':add('LABEL_KNOWN_SUBGROUP',base);add('COARSE_SUBGROUP',base);add('BIOLOGY_ENVELOPE',{**base,'epsilon':.5})
 return out

if __name__=='__main__':
 import pandas as pd
 pd.DataFrame(cells()).to_csv(G/'configs/G3_SENTINEL_MATRIX.tsv',sep='\t',index=False)
 config=dict(status='DRAFT_IMPLEMENTATION_VALIDATION',master_seed=20260923,group=3,development_purpose=301,evaluation_purpose=302,bootstrap_purpose=303,population_purpose=304,cells=[dict(**c,arms=arms(c)) for c in cells()],costs=dict(selected_full_record=1,random_full_X=2,random_full_XS=3),interval_policy='Local constrained likelihood projection chi2_1 candidate; no inherited calibration guarantee; local null leakage -> R',shared_target='Reference environment or prespecified subgroup; no mean-generator interpretation')
 (G/'configs/G3_PROTOCOL.json').write_text(json.dumps(config,indent=2));print('Draft cells',len(config['cells']),'planned independent replicates',sum(c['replicates'] for c in cells()))
