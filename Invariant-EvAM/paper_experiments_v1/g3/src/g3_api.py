"""G3 interface for downstream consumers; explicit units and no residual attribution."""
import numpy as np
from g3_model import Joint,inference

def make_biological_deviation(target_definition,reference,offsets):
 if target_definition!='REFERENCE_TARGET':raise ValueError('This implementation publishes only the reference-target contract')
 out=np.asarray(offsets,float).copy()
 if np.any(out[reference]!=0):raise ValueError('Reference deviation must be exactly zero')
 return out

def make_selection_calibration_sample(source,selection,budget,seed):
 p=np.asarray(source,float);s=np.asarray(selection,float);cost=int(budget['cost_per_record']);n=int(budget['proxy_units'])//cost
 if not (np.all(p>=0) and np.isclose(p.sum(),1) and np.all((s>=0)&(s<=1))):raise ValueError('Invalid population/selection probabilities')
 rng=np.random.default_rng(seed);counts=rng.multinomial(n,np.r_[p*(1-s),p*s]);return dict(counts=counts.tolist(),n_calibration=n,spent=n*cost,kind='XS')

def fit_with_biological_and_observation_nuisance(data,constraints):return inference(Joint(data,constraints))

def profile_under_sensitivity(data,target,epsilon,operator_family,eta_selection_coefficient=None):
 spec={**operator_family,'epsilon':float(epsilon)};lik=Joint(data,spec)
 if target!=lik.index:raise ValueError('Published target is interaction 0->1 only')
 if eta_selection_coefficient is not None:
  eta=float(eta_selection_coefficient)
  if eta<=0:raise ValueError('At eta=0 remove the last uncertainty feature from the specification; do not count a fixed parameter as free nuisance')
  for sl in lik.phi:
   if sl is not None:lik.bounds[sl.stop-1]=(-eta,eta)
 result=inference(lik);result['epsilon_units']='log acquisition rate';result['eta_units']='coefficient of the declared selection feature';result['guarantee']='LOCAL_NUMERICAL_NO_GLOBAL_OR_FINITE_SAMPLE_CERTIFICATE';return result

def score_compatibility_without_attribution(fit,observed_data=None):
 return dict(deviance=fit.get('deviance'),compatibility_status=fit.get('compatibility_status','NOT_EVALUATED'),source_attribution_allowed=False,rejection_threshold=None,reason='No independently calibrated universal threshold; residuals cannot distinguish biology from observation')
