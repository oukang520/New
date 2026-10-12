from pathlib import Path
import json,sys
import numpy as np,pandas as pd
import g5_closed_loop as g
from g5_model import geometry
P=Path(__file__).resolve().parents[2];rows=[]
for f in sorted((P/'g5/revision_r5/runs').glob('evaluation_*.json')):
 z=json.loads(f.read_text())
 for o in z['outcomes']:
  model=g.WithX(o['data'],g.spec());a=np.eye(model.dim)[4];q=geometry(model,np.array(o['fit']['x']),a)
  rows.append(dict(dataset=f.stem,action=o['action'],fit_success=o['fit']['success'],tolerance_stable=q['tolerance_stable'],rank=q['rank'],sigma_min_per_sample=q['sigma_min_per_sample'],null_leakage=q['null_leakage'],tolerance=json.dumps(q['tolerance'])))
d=pd.DataFrame(rows);d.to_csv(P/'g5/revision_r5/results/geometry_audit.tsv',sep='\t',index=False)
out=dict(fits=len(d),fit_failures=int((~d.fit_success).sum()),target_identification_tolerance_unstable=int((~d.tolerance_stable).sum()),tolerances=[1e-6,1e-7,1e-8,1e-9,1e-10],scope='SVD target identification stability at saved fitted points; not global identification or policy-optimization proof')
(P/'closure_r5/reports/GEOMETRY_AUDIT.json').write_text(json.dumps(out,indent=2));print(out)
