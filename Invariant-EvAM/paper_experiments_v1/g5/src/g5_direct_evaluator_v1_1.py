"""Pre-direct-evaluation amendment: independent source-calibration and add-patient pools.
Original frozen evaluator is preserved. No natural outcome evaluation or direct strategy
scores were inspected to choose this correction. Source strata split before selection.
"""
import json
import numpy as np
import g5_design_evaluator as ev

def generate_direct(rep):
 rows=json.loads((ev.R/'paper_experiments_v1/g4/data_private/reference_records/LUAD.json').read_text());random=ev.rng(622,0,rep,0);role={role:np.array([z['reference_state'] for z in rows if z['role']==role]) for role in ['train','calibration','evaluation']};out={};selected={};environments={};stats={};poolrole={}
 for key,x in role.items():
  env=random.integers(0,2,len(x));u=random.random(len(x));prob=np.exp(-2.2+np.where(env==0,.5*ev.X[x,0],ev.X[x,1]));s=u<prob;selected[key]=s;environments[key]=env;stats[key]=dict(screened=len(x),selected=int(s.sum()));out[key]=x
 # This partition uses only random allocation and environment, never X or S.
 futuretype=np.zeros(len(out['calibration']),int)
 for e in range(2):
  ids=np.flatnonzero(environments['calibration']==e);random.shuffle(ids);futuretype[ids[len(ids)//2:]]=1
 base=[];add=[];calx=[];cals=[];evrows=[];baseids=[];futureids=[];calids=[]
 for e in range(2):
  ids=np.flatnonzero((environments['train']==e)&selected['train']);random.shuffle(ids)
  if len(ids)<50:return dict(capacity_failure=True,reason='BASE_ENROLLED_CAPACITY',stats=stats)
  base.append(out['train'][ids[:50]]);baseids.append(ids[:50].tolist())
  fi=np.flatnonzero((environments['calibration']==e)&(futuretype==0)&selected['calibration']);random.shuffle(fi)
  ci=np.flatnonzero((environments['calibration']==e)&(futuretype==1));random.shuffle(ci)
  assert not set(fi)&set(ci)
  add.append(out['calibration'][fi]);futureids.append(fi.tolist());calx.append(out['calibration'][ci]);cals.append(selected['calibration'][ci]);calids.append(ci.tolist());evrows.append(out['evaluation'][(environments['evaluation']==e)&selected['evaluation']])
 return dict(base=base,add=add,calx=calx,cals=cals,evaluation=evrows,source_evaluation=out['evaluation'],true_target=None,capacities=dict(add_pairs=min(map(len,add)),calibration_pairs=min(map(len,calx))),base_unique=100,evidence_type='DIRECT_RECORD_CONTROLLED_REVEAL_PREVIOUSLY_EXPLORED',n_screened=stats,baseids=baseids,futureids=futureids,calids=calids,pool_partition='Disjoint random 50/50 source strata before selection; no post-selection exclusion changes calibration distribution',direct_protocol='1.1',direct_purpose=622)

ev.generate_direct=generate_direct

if __name__=='__main__':
 assert (ev.G/'configs/DIRECT_EVALUATION_FREEZE_v1_1.json').exists()
 ev.main()
