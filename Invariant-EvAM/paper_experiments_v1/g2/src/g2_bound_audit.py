from pathlib import Path
import sys,json,time
import numpy as np,pandas as pd
from scipy.optimize import minimize,brentq
from scipy.stats import chi2,beta
R=Path(__file__).resolve().parents[3];G=R/'paper_experiments_v1/g2';sys.path.insert(0,str(G/'src'))
from g2_engine import Likelihood,operators,snapshot,fit,profile,interval

def main():
 pub=json.loads((G/'configs/public_contract.json').read_text());ev=json.loads((G/'configs/evaluator_contract.json').read_text())
 config={'rng_purpose':207,'cells':list(range(12)),'replicates_per_cell':3,'bounds':[3,4.5,6],'profiles_per_bound':49,'independent_restarts':True,'reference_projection':'SLSQP witnesses plus certified whole-real-line outer bound; numerical witnesses not certified outer set'}
 cp=G/'configs/BOUND_AUDIT_CONTRACT.json'
 if not cp.exists():cp.write_text(json.dumps(config,indent=2)+'\n')
 out=[];aliases=[];references=[];paths=[]
 for cell in range(12):
  label=['strong','weak','nonidentifying'][cell//4];ops=operators(pub['operators'][label]);theta=np.r_[ev['baseline'],ev['effects'][cell%4],ev['other_interaction']];p=snapshot(theta)[0]
  for rep in range(3):
   rng=np.random.default_rng(np.random.SeedSequence([20260923,2,207,cell,0,rep]));counts=[]
   for (a,f),phi in zip(ops,ev['true_phis'][label]):
    pp=p*np.exp(f@np.array(phi));pp/=pp.sum();counts.append(rng.multinomial(500,a@pp))
   lik=Likelihood(counts,ops);previous=None
   for bound in config['bounds']:
    x=np.r_[[-1,-1,-1,0,0],np.zeros(lik.dim-5)];alt=x.copy();alt[3:5]=[-1,1];starts=[x,alt]
    if previous is not None:starts.append(previous)
    f=fit(lik,bound,starts);previous=f['x'];ps=profile(lik,f,3,bound,np.unique(np.r_[np.linspace(-bound,bound,49),0,f['x'][3]]),True);ci=interval(ps,f['nll'],chi2.ppf(.95,1),bound)
    paths.append({'cell':cell,'rep':rep,'bound':bound,'fit_nll':f['nll'],'fit':f['x'].tolist(),'points':ps})
    out.append({'cell_id':cell,'replicate_id':rep,'design':label,'bound':bound,'estimate':float(f['x'][3]),'ci_lower':ci['lo'],'ci_upper':ci['hi'],'boundary':ci['boundary'],'components':json.dumps(ci['components']),'success':f['success'] and all(z['success'] for z in ps),'nll':f['nll']})
   if label=='nonidentifying':
    base=previous[:5].copy();pbase=snapshot(base)[0];target_prob=pbase[[2,3,6,7]].sum()
    for val in [-1.,1.]:
     candidate=base.copy();candidate[3]=val
     def objective(b):
      candidate[1]=b;return snapshot(candidate)[0][[2,3,6,7]].sum()-target_prob
     bb=brentq(objective,-15,6,xtol=1e-12);candidate[1]=bb;q=snapshot(candidate)[0];err=max(np.max(abs(a@q-a@pbase)) for a,feat in ops)
     aliases.append({'cell_id':cell,'replicate_id':rep,'target':val,'theta':json.dumps(candidate.tolist()),'max_observable_error':err,'reference_theta':json.dumps(base.tolist()),'opposite_sign_witness':True})
   # Bonferroni CP constraints at an independent replicate; extrema are feasible witnesses only.
   alpha=.05/sum(len(c) for c in counts);lo=[];hi=[]
   for c in counts:
    for k in c:lo.append(beta.ppf(alpha/2,k,501-k) if k else 0);hi.append(beta.ppf(1-alpha/2,k+1,500-k) if k<500 else 1)
   lo=np.array(lo);hi=np.array(hi)
   def constraints(z):
    outputs=lik.outputs(z);prob=np.concatenate([p for p,j in outputs]);return np.r_[prob-lo,hi-prob]
   def jac(z):
    j=np.vstack([j for p,j in lik.outputs(z)]);return np.vstack([j,-j])
   for direction in [-1.,1.]:
    obj=lambda z:direction*z[3]
    grad=np.eye(lik.dim)[3]*direction
    res=minimize(obj,previous,jac=lambda z:grad,method='SLSQP',bounds=[(-6,6)]*5+[(-4,4)]*(lik.dim-5),constraints={'type':'ineq','fun':constraints,'jac':jac},options={'maxiter':300,'ftol':1e-9})
    references.append({'cell_id':cell,'replicate_id':rep,'direction':direction,'target_witness':float(res.x[3]),'min_constraint':float(constraints(res.x).min()),'optimizer_success':bool(res.success),'certified_outer_lower':-np.inf,'certified_outer_upper':np.inf,'projection_status':'NUMERICAL_INNER_WITNESS_ONLY','global_coverage_claim':False})
  print('bound audit cell',cell,flush=True)
  pd.DataFrame(out).to_csv(G/'results/G2_bound_audit.tsv',sep='\t',index=False);pd.DataFrame(aliases).to_csv(G/'results/G2_opposite_sign_aliases.tsv',sep='\t',index=False);pd.DataFrame(references).to_csv(G/'results/G2_exact_reference_projection.tsv',sep='\t',index=False)
  (G/'results/G2_bound_profiles.json').write_text(json.dumps(paths))
if __name__=='__main__':main()
