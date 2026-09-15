"""Small paired B1 mechanism experiment. No fitting and no production edits."""
from pathlib import Path
import argparse, ast, hashlib, importlib.metadata, json, os, shutil, subprocess, sys, time
from datetime import datetime,timezone
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
REL=ROOT if (ROOT/'src/relobstq_mhn').is_dir() else ROOT/'public_release/RelObsTQ_MHN_reproducible_code'
BASE=REL/'reference_results/final_manuscript_evidence/simulation_dwell_gradient'
sys.path.insert(0,str(REL/'src'))
import numpy as np
import pandas as pd
from relobstq_mhn.simulation import generator as gen
from relobstq_mhn.workflows.simulation import _occupancy
from relobstq_mhn.workflows.controls import denominator_ablation
from relobstq_mhn.core.scoring import ScoreThresholds
CONDITIONS={'baseline':(False,1.),'inflow_only':(True,1.),'dwell_only':(False,2.),'joint':(True,2.)}
METHODS=['full_mhn','occupancy_only','uniform_inflow','frequency_inflow']
ORIGINAL_RATES=gen.event_rates_from_mask
def now():return datetime.now(timezone.utc).isoformat()
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def digest(data):return hashlib.sha256(json.dumps(data,sort_keys=True,allow_nan=False).encode()).hexdigest()
def clean(obj):
    if isinstance(obj,dict):return {str(k):clean(v) for k,v in obj.items()}
    if isinstance(obj,(list,tuple)):return [clean(v) for v in obj]
    if isinstance(obj,np.generic):return clean(obj.item())
    if isinstance(obj,float) and not np.isfinite(obj):return None
    return obj
def write(path,data):
    path=Path(path).resolve();assert path.is_relative_to(HERE)
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.partial')
    tmp.write_text(json.dumps(clean(data),indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8');os.replace(tmp,path)
def table(path,frame):
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.partial')
    frame.to_csv(tmp,index=False,compression='gzip' if path.name.endswith('.gz') else None);os.replace(tmp,path)
def plan():return json.loads((HERE/'plan.json').read_text(encoding='utf-8'))
def make_theta(cfg):
    tree=ast.parse((REL/'src/relobstq_mhn/workflows/simulation.py').read_text())
    fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='run_dwell_gradient')
    scaffold=ast.literal_eval(next(n.value for n in fn.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='scaffold_values' for t in n.targets)))
    return gen.create_sparse_theta([f'E{i+1}' for i in range(cfg['event_count'])],sparsity=cfg['theta_sparsity'],seed=cfg['random_seed'],forced_edges=scaffold)
def rates(mask,theta,targets,up):
    absent,values=ORIGINAL_RATES(mask,theta)
    if up and mask==0:
        total=values.sum();values=values.copy()
        for j,event in enumerate(absent):
            if (1<<int(event)) in targets:values[j]*=2.
        values*=total/values.sum()
    return absent,values
def initialize():
    assert not (HERE/'plan.json').exists(),'Existing B1 plan is immutable'
    cfg=json.loads((BASE/'resolved_config.json').read_text());truth=pd.read_csv(BASE/'tables/truth_states.tsv',sep='\t')
    chosen=truth[truth.event_count==1].sort_values(['pilot_count','state'],ascending=[False,True]).head(2)
    assert len(chosen)==2 and chosen.pilot_count.min()>=80
    events=[f'E{i+1}' for i in range(cfg['event_count'])];theta=make_theta(cfg);targets=chosen['mask'].astype(int).tolist()
    assert targets[0]&targets[1]==0
    paths=[Path(__file__),BASE/'resolved_config.json',BASE/'tables/truth_states.tsv']+[REL/'src/relobstq_mhn'/n for n in ['simulation/generator.py','workflows/simulation.py','workflows/controls.py','core/transitions.py','core/scoring.py','core/states.py']]
    environment=dict(python=sys.version,executable=sys.executable,packages={n:importlib.metadata.version(n) for n in ['numpy','pandas','scipy']},threads=1)
    p=dict(created=now(),stage_id='B1',base_generation=cfg,events=events,targets=chosen[['mask','state','genotype','pilot_count']].to_dict('records'),conditions={k:dict(upstream_reweight=v[0],target_D=v[1]) for k,v in CONDITIONS.items()},upstream_mask=0,upstream_target_weight_multiplier=2.,target_dwell_multiplier=2.,repeats=6,repeat_seeds=[20261001+i for i in range(6)],samples=cfg['samples_per_repeat'],seed_rule='Each patient uses default_rng(SeedSequence([repeat_seed, sample_id])); identical streams across all four conditions',budget_seconds=900,per_repeat_timeout_seconds=200,workers=1,max_retries=2,automatic_retries=0,pilot_repeat=1,pilot_included_in_plan=True,comparison_domain='all four methods eligible and finite on same target, condition and paired reference; primary contrasts require common across all four conditions',evaluation='within-state paired log2 response: inflow expected 0, dwell expected 1, joint expected 1; D is state-relative, not absolute holding time; descriptive 6-repeat summaries without bootstrap',occupancy_definition='Existing denominator_ablation occupancy_only: L divided by eligible L median; raw L also reported',frequency_definition='Existing denominator_ablation: current-condition empirical event frequencies, renormalized over absent events',reuse_search={'checked':['main simulation workflow/config/frozen output manifest','existing denominator_ablation implementation and E14 records','nc_revision completed task directories'],'finding':'No equivalent trajectory-level 2x2 intervention found; E6 changes D, E14 replaces denominators and compares rankings. Neither isolates arrival and dwell jointly.'},new_scope='Reuse original theta/scaffold,p=15,N=5000,max_time=10,max_events=8,thresholds and patient trajectory function. New neutral D=1 baseline and two-target 2x2 interventions; replace original 25-state gradient and use per-patient paired streams, so original observations cannot be reused. No pilot truth reselection or refit.',input_code_hashes={str(f.relative_to(ROOT)):sha(f) for f in paths},environment=environment,environment_hash=digest(environment),theta_sha256=digest(theta.tolist()),tolerance=dict(atol=1e-12,rtol=1e-9))
    write(HERE/'plan.json',p)
    for f in paths:
        dest=HERE/'before'/f.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(f,dest)
    audit=[]
    for condition,(up,D) in CONDITIONS.items():
        ar,vr=rates(0,theta,targets,up);_,rootbase=rates(0,theta,targets,False)
        assert np.isclose(vr.sum(),rootbase.sum(),rtol=0,atol=1e-12)
        for mask in targets:
            absent,v=rates(mask,theta,targets,up);_,v0=rates(mask,theta,targets,False)
            np.testing.assert_array_equal(v,v0)
            np.testing.assert_allclose((v/D)/(v/D).sum(),v0/v0.sum(),rtol=1e-12)
            audit.append(dict(condition=condition,target=gen.mask_to_state(mask,events,stage='s1'),mask=mask,D=D,baseline_exit_rate=v0.sum(),intervened_exit_rate=v.sum()/D,expected_untruncated_holding_time=D/v0.sum(),root_exit_rate=vr.sum(),root_to_target_probability=vr[list(ar).index(int(mask).bit_length()-1)]/vr.sum()))
    table(HERE/'intervention_contract.csv',pd.DataFrame(audit))
    write(HERE/'run_manifest.json',dict(stage_id='B1',start=now(),planned=24,attempted=0,completed=0,failed=0,skipped=24,plan_hash=sha(HERE/'plan.json'),commands=[sys.executable+' -B nc_revision/B1_inflow_dwell/run_b1.py init'],sessions=[],retry_log=[]))
    print('LOCKED:',chosen[['state','pilot_count']].to_dict('records'),'6 repeats x 4 conditions, no outcomes examined')

class TracedRNG:
    """Log the actual exponential draw without changing the simulator's draw order."""
    def __init__(self,seed):self.rng=np.random.default_rng(seed);self.waits=[]
    def exponential(self,scale):
        value=self.rng.exponential(scale);self.waits.append(float(value));return value
    def choice(self,*args,**kw):return self.rng.choice(*args,**kw)
    def uniform(self,*args,**kw):return self.rng.uniform(*args,**kw)

def cache_key(p,r,condition):
    return digest(dict(data=p['input_code_hashes'],event_order=p['events'],config=sha(HERE/'plan.json'),code=p['input_code_hashes'],environment=p['environment_hash'],fold='paired_mechanism',seed=p['repeat_seeds'][r-1],repeat=r,condition=condition))
def complete(p,r,condition):
    out=HERE/'runs'/f'{r:02d}'/condition;path=out/'complete.json'
    if not path.exists():return False
    ck=json.loads(path.read_text());assert ck['cache_key']==cache_key(p,r,condition)
    assert all(sha(out/name)==h for name,h in ck['artifacts'].items());return True

def repeat(r):
    p=plan();cfg=p['base_generation'];theta=make_theta(cfg);targets=[int(t['mask']) for t in p['targets']];events=p['events']
    for condition,(up,D) in CONDITIONS.items():
        if complete(p,r,condition):continue
        out=HERE/'runs'/f'{r:02d}'/condition;out.mkdir(parents=True,exist_ok=True)
        write(out/'attempt.json',dict(start=now(),repeat=r,condition=condition,seed=p['repeat_seeds'][r-1],cache_key=cache_key(p,r,condition)))
        trajectories=[];snapshots=[]
        gen.event_rates_from_mask=lambda mask,t:rates(mask,t,targets,up)
        try:
            simcfg=gen.SimulationConfig(samples=p['samples'],maximum_time=cfg['maximum_time'],maximum_events=cfg['maximum_events'],random_seed=p['repeat_seeds'][r-1])
            for sample in range(1,p['samples']+1):
                rng=TracedRNG(np.random.SeedSequence([p['repeat_seeds'][r-1],sample]))
                tr,snap=gen.simulate_patient_trajectory(theta,events,{mask:D for mask in targets},config=simcfg,rng=rng,sample_id=sample,repeat=r)
                assert len(rng.waits)==len(tr)
                tr['untruncated_wait_draw']=rng.waits
                trajectories.append(tr);snapshots.append(snap)
        finally:gen.event_rates_from_mask=ORIGINAL_RATES
        trajectory=pd.concat(trajectories,ignore_index=True);snapshots=pd.DataFrame(snapshots);occupancy=_occupancy(snapshots)
        def provider(g):
            present=set() if g=='WT' else set(g.split('+'));mask=sum(1<<i for i,e in enumerate(events) if e in present)
            absent,v=rates(mask,theta,targets,up)
            return {events[int(i)]:float(w/v.sum()) for i,w in zip(absent,v)} if len(absent) else {}
        details,_unused_similarity=denominator_ablation(occupancy,events,provider,snapshots[events].mean().to_dict(),thresholds=ScoreThresholds(**cfg['thresholds']))
        # Ignore the legacy ranking-similarity summaries; evaluate intervention truth instead.
        mechanism=[];target_rows=[]
        for target in p['targets']:
            mask=int(target['mask']);visits=trajectory[trajectory['mask']==mask]
            assert not visits.sample_id.duplicated().any()
            base_exit=ORIGINAL_RATES(mask,theta)[1].sum()
            mechanism.append(dict(repeat=r,condition=condition,state=target['state'],mask=mask,D=D,baseline_exit_rate=base_exit,expected_holding_time=D/base_exit,arrivals=len(visits),arrival_frequency=len(visits)/p['samples'],mean_simulated_untruncated_holding_time=visits.untruncated_wait_draw.mean(),mean_recorded_duration=visits.duration.mean(),right_censored_fraction=visits.event_added.eq('CENSORED_AT_HORIZON').mean(),snapshot_count=int((snapshots['mask']==mask).sum()),samples=p['samples'],mean_observation_time=snapshots.observation_time.mean()))
        for method in METHODS:
            values=details[details.variant==method].copy();eligible=values.eligible_relobstq.astype(bool)
            raw=values.L_v if method=='occupancy_only' else values.L_v/(values.F_hat+cfg['thresholds']['epsilon'])
            normalizer=float(raw[eligible].median());np.testing.assert_allclose(values.R_star,raw/normalizer,atol=1e-12,rtol=1e-9)
            for target in p['targets']:
                row=values[values.state==target['state']]
                if row.empty:
                    target_rows.append(dict(repeat=r,condition=condition,state=target['state'],method=method,eligible=False,reason='unobserved_target',N=0,L=None,F=None,raw_ratio=None,normalizer=normalizer,score=None))
                    continue
                row=row.iloc[0];ok=bool(row.eligible_relobstq)
                reason='eligible' if ok else ('count_below_5' if row.N_v<5 else 'inflow_below_1e-8')
                target_rows.append(dict(repeat=r,condition=condition,state=target['state'],method=method,eligible=ok,reason=reason,N=int(row.N_v),L=float(row.L_v),F=None if method=='occupancy_only' else float(row.F_hat),raw_ratio=float(row.L_v if method=='occupancy_only' else row.L_v/(row.F_hat+cfg['thresholds']['epsilon'])),normalizer=normalizer,score=float(row.R_star),normalization_domain_states=int(eligible.sum())))
        table(out/'mechanism.csv',pd.DataFrame(mechanism));table(out/'target_scores.csv',pd.DataFrame(target_rows));table(out/'occupancy.csv',occupancy);table(out/'all_method_scores.csv',details)
        table(out/'trajectories.csv.gz',trajectory);table(out/'snapshots.csv.gz',snapshots)
        files=[f for f in out.iterdir() if f.is_file() and f.name!='complete.json']
        write(out/'complete.json',dict(cache_key=cache_key(p,r,condition),end=now(),artifacts={f.name:sha(f) for f in files}))
        print(json.dumps(dict(repeat=r,condition=condition,complete=True)),flush=True)
    # D acts downstream of the root branch: identical patient streams imply identical arrivals.
    for left,right in [('baseline','dwell_only'),('inflow_only','joint')]:
        l=pd.read_csv(HERE/'runs'/f'{r:02d}'/left/'trajectories.csv.gz',usecols=['sample_id','mask'])
        rr=pd.read_csv(HERE/'runs'/f'{r:02d}'/right/'trajectories.csv.gz',usecols=['sample_id','mask'])
        for mask in targets:assert set(l.loc[l['mask']==mask,'sample_id'])==set(rr.loc[rr['mask']==mask,'sample_id'])
    write(HERE/'runs'/f'{r:02d}'/'pair_checks.json',dict(status='PASS',same_patient_streams=True,target_arrivals_unchanged_by_dwell=True))

def summarize():
    p=plan();sc=[];mech=[]
    for r in range(1,p['repeats']+1):
        for c in CONDITIONS:
            if complete(p,r,c):
                out=HERE/'runs'/f'{r:02d}'/c;sc.append(pd.read_csv(out/'target_scores.csv'));mech.append(pd.read_csv(out/'mechanism.csv'))
    if not sc:return
    scores=pd.concat(sc,ignore_index=True);mechanism=pd.concat(mech,ignore_index=True)
    table(HERE/'mechanism_figure_data.csv',mechanism);table(HERE/'all_target_scores.csv',scores)
    coverage=[];paired=[]
    for (r,state),block in scores.groupby(['repeat','state']):
        common=len(block)==16 and block.eligible.all() and np.isfinite(block.score).all()
        for c,cb in block.groupby('condition'):
            for row in cb.itertuples():coverage.append(dict(repeat=r,state=state,condition=c,method=row.method,eligible=row.eligible,reason=row.reason,common_four_conditions_all_methods=common))
        if not common:continue
        for method in METHODS:
            b=block[block.method==method].set_index('condition');base=b.loc['baseline']
            for condition,expected in [('inflow_only',0.),('dwell_only',1.),('joint',1.)]:
                z=b.loc[condition];change=np.log2(z.score/base.score)
                paired.append(dict(repeat=r,state=state,method=method,condition=condition,log2_score_change=change,expected_log2_D_change=expected,absolute_log2_error=abs(change-expected),log2_raw_change=np.log2(z.raw_ratio/base.raw_ratio),log2_normalizer_change=np.log2(z.normalizer/base.normalizer),log2_occupancy_change=np.log2(z.L/base.L),log2_F_change=None if method=='occupancy_only' else np.log2(z.F/base.F),joint_vs_dwell_log2_score=np.log2(b.loc['joint','score']/b.loc['dwell_only','score'])))
    table(HERE/'coverage.csv',pd.DataFrame(coverage));table(HERE/'unscored_targets.csv',pd.DataFrame(coverage).loc[lambda d:~d.eligible])
    if not paired:return
    pairs=pd.DataFrame(paired);table(HERE/'paired_responses.csv',pairs)
    reps=pairs.groupby(['repeat','method','condition'],as_index=False).agg(log2_score_change=('log2_score_change','mean'),log2_raw_change=('log2_raw_change','mean'),log2_normalizer_change=('log2_normalizer_change','mean'),absolute_log2_error=('absolute_log2_error','mean'),targets=('state','size'),joint_vs_dwell_log2_score=('joint_vs_dwell_log2_score','mean'))
    table(HERE/'repeat_method_metrics.csv',reps)
    key=[]
    for method in METHODS:
        row=dict(method=method,paired_state_repeats=int(pairs[(pairs.method==method)&(pairs.condition=='joint')].shape[0]),possible_state_repeats=12)
        for c in ['inflow_only','dwell_only','joint']:
            f=reps[(reps.method==method)&(reps.condition==c)]
            row[c+'_score_fold']=float(2**f.log2_score_change.median());row[c+'_raw_fold']=float(2**f.log2_raw_change.median());row[c+'_normalizer_fold']=float(2**f.log2_normalizer_change.median());row[c+'_mean_target_abs_log2_error_median']=float(f.absolute_log2_error.median())
            reference=reps[(reps.method=='full_mhn')&(reps.condition==c)].set_index('repeat').absolute_log2_error
            row[c+'_paired_error_difference_vs_SIRDwell']=float((f.set_index('repeat').absolute_log2_error-reference).median())
        row['joint_vs_dwell_score_fold']=float(2**f.joint_vs_dwell_log2_score.median());key.append(row)
    table(HERE/'KEY_RESULTS.csv',pd.DataFrame(key));print(pd.DataFrame(key).to_string(index=False))

def run(mode):
    p=plan()
    for f,h in p['input_code_hashes'].items():assert sha(ROOT/f)==h,'Locked input/code changed'
    manifest=json.loads((HERE/'run_manifest.json').read_text());started=time.monotonic()
    remaining=p['budget_seconds']-sum(s.get('seconds',0) for s in manifest['sessions']);deadline=started+remaining
    session=dict(mode=mode,start=now(),commands=[]);manifest['sessions'].append(session)
    if mode=='batch':assert (HERE/'runs/01/pair_checks.json').exists(),'Pilot must pass first'
    repeats=[1] if mode=='pilot' else list(range(1,7))
    for r in repeats:
        if all(complete(p,r,c) for c in CONDITIONS) and (HERE/'runs'/f'{r:02d}'/'pair_checks.json').exists():continue
        if time.monotonic()>=deadline:break
        cmd=[sys.executable,'-B',str(Path(__file__)),'repeat','--repeat',str(r)];session['commands'].append(dict(start=now(),command=subprocess.list2cmdline(cmd),repeat=r));write(HERE/'run_manifest.json',manifest)
        env=os.environ.copy();env.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
        with (HERE/f'repeat_{r:02d}.log').open('w') as log:
            proc=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
            try:proc.wait(timeout=min(p['per_repeat_timeout_seconds'],deadline-time.monotonic()))
            except subprocess.TimeoutExpired:
                subprocess.run(['taskkill.exe','/PID',str(proc.pid),'/T','/F'],capture_output=True);proc.wait()
                manifest.setdefault('failures',[]).append(dict(repeat=r,reason='budget_or_repeat_timeout',time=now()));break
        if proc.returncode:
            manifest.setdefault('failures',[]).append(dict(repeat=r,reason='worker_error',log=f'repeat_{r:02d}.log'));break
        print('Completed paired repeat',r,flush=True)
    session['end']=now();session['seconds']=time.monotonic()-started
    manifest['attempted']=len(list((HERE/'runs').glob('*/*/attempt.json')));manifest['completed']=sum(complete(p,r,c) for r in range(1,7) for c in CONDITIONS);manifest['failed']=len(manifest.get('failures',[]));manifest['skipped']=24-manifest['attempted'];manifest['end']=now();manifest['partial']=manifest['completed']<24
    write(HERE/'run_manifest.json',manifest)
    print(json.dumps({k:manifest[k] for k in ['planned','attempted','completed','failed','skipped','partial']}))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['init','pilot','batch','repeat','summary']);parser.add_argument('--repeat',type=int);args=parser.parse_args()
    if args.mode=='init':initialize()
    elif args.mode in ['pilot','batch']:run(args.mode)
    elif args.mode=='summary':summarize()
    else:
        assert 1<=args.repeat<=6;repeat(args.repeat)
