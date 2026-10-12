"""Development only: common CTMC target, competing observation channels."""
from pathlib import Path
import sys,json,itertools,time
import numpy as np,pandas as pd
P=Path(__file__).resolve().parents[2];sys.path[:0]=[str(P/'g5/src'),str(P/'reviewer_audit/redesign_r3')]
from g5_model import Model,fit,snapshot,X,mask_matrix,geometry
from g5_interval import endpoints,constrained
G=P/'g5/revision_r6'
SEED=202609276
class Obs(Model):
 def outputs(self,x):
  if self.cache is not None and np.array_equal(self.cache[0],x):return self.cache[1]
  out=[]
  for g in self.data:
   p,j=self.source(x,g['env']) if g['kind']=='source' else self.selected(x,g['env'])
   A=np.asarray(g['A']);out.append((np.array(g['counts']),A@p,A@j))
  self.cache=(x.copy(),out);return out

def spec():
 return dict(topology='SPARSE',features=[np.column_stack([X[:,0],X[:,0]*X[:,1]]).tolist()],known_phi=[None],phi_bound=1.5,bound=3.,epsilon=0)
MASKS=[mask_matrix([1,0,1,1]),mask_matrix([0,1,1,1]),np.eye(16)]
def group(n,p,A,kind):return dict(env=0,kind=kind,A=A.tolist(),counts=(n*(A@p)).tolist())
def generate(stage,case,rep,budget=600):
 rng=np.random.default_rng([SEED,stage,case,rep]);theta=np.r_[rng.uniform(-1.5,-.3,4),rng.uniform(-.7,.7,6)];theta[4]=rng.choice([-1.,-.5,0.,.5,1.])
 p,_=snapshot(theta);phi=[(.0,.0),(.7,.6),(-.7,-.6)][case//2];z=np.asarray(spec()['features'][0])@phi;q=p*np.exp(z);q/=q.sum()
 # Two different pre-existing source panels; no hidden truth enters the strategy.
 ns=[240,40] if case%2==0 else [40,240]
 records=[rng.multinomial(n,p) for n in ns];selected=rng.multinomial(120,q)
 base=[dict(env=0,kind='source',A=MASKS[j].tolist(),counts=(MASKS[j]@c).tolist()) for j,c in enumerate(records)]
 base.append(dict(env=0,kind='selected',A=np.eye(16).tolist(),counts=selected.tolist()))
 # Completion spends one assay per old record, then full new source records cost4.
 acts=[dict(name='COMPLETE_A',complete=0,new_source=(budget-ns[0])//4,new_selected=0,cost=budget),dict(name='COMPLETE_B',complete=1,new_source=(budget-ns[1])//4,new_selected=0,cost=budget),dict(name='NEW_SOURCE',complete=-1,new_source=budget//4,new_selected=0,cost=budget),dict(name='MORE_SELECTED',complete=-1,new_source=0,new_selected=budget//4,cost=budget)]
 return theta,p,q,base,records,selected,acts

def public_policy(base,acts):
 m=Obs(base,spec());r=fit(m);x=np.array(r['x']);p,_=m.source(x);q,_=m.selected(x,0);diag=[]
 for a in acts:
  d=[]
  for j,b in enumerate(base):
   A=np.eye(16) if j==a['complete'] else np.array(b['A']);d.append(group(sum(b['counts']),p if b['kind']=='source' else q,A,b['kind']))
  d += [group(a['new_source'],p,np.eye(16),'source'),group(a['new_selected'],q,np.eye(16),'selected')]
  z=Obs(d,spec());h=z.whitened(x);u,sv,vh=np.linalg.svd(h,full_matrices=True);rank=sum(sv>sv[0]*1e-8);leak=np.linalg.norm(vh[rank:]@np.eye(z.dim)[4]);var=np.sum((vh[:rank,4]/sv[:rank])**2) if leak<1e-7 else np.inf
  diag.append(dict(action=a['name'],variance=float(var),leak=float(leak),logdet=float(np.log(np.maximum(sv**2,1e-12)).sum())))
 choose=min(diag,key=lambda z:(z['leak']>=1e-7,z['variance'],z['action']))['action'];dopt=max(diag,key=lambda z:z['logdet'])['action']
 return choose,dopt,diag,r

def run(n=12):
 rows=[]
 for case in range(6):
  for rep in range(n):
   theta,p,q,base,records,selected,acts=generate(0,case,rep);choose,dopt,diag,r=public_policy(base,acts)
   row=dict(case=case,rep=rep,choice=choose,dopt=dopt,fit=r['success']);rows.append(row)
   for a in acts:
    rng=np.random.default_rng([SEED,99,case,rep,acts.index(a)]);data=[]
    for j,c in enumerate(records):data.append(dict(env=0,kind='source',A=(np.eye(16) if j==a['complete'] else MASKS[j]).tolist(),counts=(c if j==a['complete'] else MASKS[j]@c).tolist()))
    data += [dict(env=0,kind='selected',A=np.eye(16).tolist(),counts=(selected+rng.multinomial(a['new_selected'],q)).tolist()),dict(env=0,kind='source',A=np.eye(16).tolist(),counts=rng.multinomial(a['new_source'],p).tolist())]
    model=Obs(data,spec());f=fit(model);pr=endpoints(model,f,3.841459);row[a['name']+'_err']=(f['x'][4]-theta[4])**2;row[a['name']+'_correct']=bool((pr['lo']>0 and theta[4]>0)or(pr['hi']<0 and theta[4]<0));row[a['name']+'_covered']=pr['lo']<=theta[4]<=pr['hi']
   if rep==n-1:print(case,pd.DataFrame(rows)[pd.DataFrame(rows).case==case].choice.value_counts().to_dict(),flush=True)
 pd.DataFrame(rows).to_csv(G/'results/development_probe.tsv',sep='\t',index=False)
 print(pd.DataFrame(rows).groupby('case').mean(numeric_only=True).to_string())
if __name__=='__main__':run()
