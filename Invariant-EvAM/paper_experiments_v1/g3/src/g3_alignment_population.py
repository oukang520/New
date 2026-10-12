import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import numpy as np,pandas as pd,json
from scipy.optimize import minimize
from g3_model import snapshot
from g3_design import G,theta
from g3_local_alignment_revision import public_arms,generator_direction
from g3_population import normalized,kl,population_fit

def main():
 t=theta(4);v=generator_direction();arms=public_arms();f=np.asarray(arms[0]['spec']['features'][1]);p,j=snapshot(t,4);coef=np.linalg.lstsq(f-p@f,(j@v)/p,rcond=1e-10)[0];phi=np.r_[.5,np.zeros(9)];base=normalized(p,f@phi);rows=[]
 for h in [0,.01,.05,.2,.5]:
  shifted=snapshot(t+h*v,4)[0];cancelled=normalized(shifted,f@(phi-h*coef));rows.append(dict(amplitude=h,probability_residual_after_first_order_cancel=float(max(abs(cancelled-base))),target_deviation=float(h*v[4]),public_feature_basis='fixed monomials, no truth-derived values',evidence_type='NEW_RUN_PUBLIC_FAMILY_POPULATION'))
 pd.DataFrame(rows).to_csv(G/'results/G3_local_alignment_v1_1.tsv',sep='\t',index=False)
 prob0=normalized(p,np.array(arms[0]['spec']['features'][0])[:,0]*.5);prob1=normalized(snapshot(t+.5*v,4)[0],f@phi);rows=[]
 for a in arms:rows.append(dict(method=a['name'],protocol='G3_1.1',**population_fit([prob0,prob1],a['spec'])))
 pd.DataFrame(rows).to_csv(G/'results/G3_population_alignment_v1_1.tsv',sep='\t',index=False)
 (G/'configs/GENERATOR_ONLY_ALIGNMENT_DIRECTION.json').write_text(json.dumps(dict(direction=v.tolist(),nuisance_cancel_derivative=coef.tolist(),tangent_residual=float(max(abs((f-p@f)@coef-j@v/p))),access='GENERATOR_EVALUATOR_ONLY_NOT_ESTIMATOR_INPUT'),indent=2))
 print('Public-family alignment population check complete')
if __name__=='__main__':main()
