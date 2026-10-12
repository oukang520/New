from pathlib import Path
import sys,json,hashlib,argparse
import numpy as np,pandas as pd
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy.stats import gaussian_kde
R=Path(__file__).resolve().parents[3];P=R/'paper_experiments_v1';sys.path.insert(0,str(R/'src'))
from invariant_evam.figure_style import apply_publication_style
apply_publication_style({'svg.fonttype':'none','pdf.fonttype':42,'font.size':8,'axes.labelsize':8,'xtick.labelsize':7,'ytick.labelsize':7,'axes.linewidth':.75})
C=['#0072B2','#56B4E9','#009E73','#E69F00','#D55E00','#CC79A7'];CAT=json.loads((R/'configs/figure_reference_catalog.json').read_text())['records'];MAN={g:[] for g in range(1,6)}
def read(g,n):return pd.read_csv(P/f'g{g}/revision_r2/results/{n}.tsv',sep='\t')
def canvas(w=150,h=105):
 f,a=plt.subplots(figsize=(w/25.4,h/25.4));f.subplots_adjust(left=.14,right=.96,bottom=.24,top=.76);a.spines[['top','right']].set_visible(False);return f,a

def save(f,g,n,d,ref,adapt):
 G=P/f'g{g}/revision_r2';d.to_csv(G/f'figure_data/{n}.tsv',sep='\t',index=False)
 for ext in ['pdf','svg']:f.savefig(G/f'figures/{n}.{ext}',dpi=600)
 temp=Path('/tmp/invariant_r2_previews');temp.mkdir(exist_ok=True);f.savefig(temp/f'{n}.png',dpi=160)
 rr=next(r for r in CAT if r['id']==ref);fp=Path(rr['source_file_absolute']);MAN[g].append(dict(panel_id=n,reference=ref,reference_file=str(fp),actual_reference_sha256=hashlib.sha256(fp.read_bytes()).hexdigest(),physical_size_mm=(f.get_size_inches()*25.4).tolist(),dpi=600,palette=C,adaptations=adapt,data_sha256=hashlib.sha256((G/f'figure_data/{n}.tsv').read_bytes()).hexdigest(),review_status='AWAITING_ACTUAL_REVIEW',user_approved=False));plt.close(f)

def legend(f,labels,colors,y=.88):f.legend(handles=[Patch(fc=c,ec='#222222',lw=.7,label=l) for c,l in zip(colors,labels)],loc='upper left',bbox_to_anchor=(.13,y),ncol=len(labels),frameon=True,fancybox=False,edgecolor='#bbbbbb',fontsize=6.5,handlelength=1,handletextpad=.4,columnspacing=.9)
def foot(f,text):f.text(.14,.055,text,fontsize=6.1,linespacing=1.45)

def g1():
 d=read(1,'G1_R2_finite_raw');order=['duplicated','heterogeneous','missing_joint'];labels=['Repeated','Complementary','No joint path']
 for regime in ['rare','ordinary','frequent']:
  q=d[d.regime.eq(regime)].copy();v=q.groupby(['draw','design','n_standard']).abs_error.mean().reset_index();f,a=canvas();top=0
  for k,n in enumerate([400,4000]):
   for j,s in enumerate(order):
    vals=v[v.design.eq(s)&v.n_standard.eq(n)].abs_error.to_numpy();x=k+(j-1)*.24;m=vals.mean();a.bar(x,m,.205,color=C[j],ec='#222222',lw=.7);a.scatter(x+np.linspace(-.05,.05,len(vals)),vals,s=12,facecolor='white',edgecolor='#333333',lw=.6,zorder=4);a.errorbar(x,m,yerr=[[m-vals.min()],[vals.max()-m]],fmt='none',ecolor='#222222',capsize=2,lw=.7,zorder=3);top=max(top,vals.max());a.annotate(f'{m:.2f}',(x,vals.max()),xytext=(0,5),textcoords='offset points',ha='center',fontsize=6.2)
  a.set_ylim(0,top*1.22+.03);a.set_xticks([0,1],['1,200 measurements','12,000 measurements']);a.set_ylabel('Mean absolute target error ↓');legend(f,labels,C[:3]);f.text(.14,.95,f'G1 | Fixed-cost recovery · {regime} events',fontsize=9,weight='bold');foot(f,'Four independent parameter draws; 50 datasets per draw and design.\nOpen points: draw means. Bars: mean; whiskers: minimum–maximum across draws.\nAll outcomes retained, including numerical failures and boundary fits.');save(f,1,f'G1_R2_recovery_{regime}',q,'R21','Grouped outlined bars, raw parameter means, range whiskers, endpoint labels, boxed top legend. Range is heterogeneity, not confidence interval; no invented significance.')

def g2():
 d=read(2,'G2_R2_summary');methods=['OLD_BONFERRONI_BANDS','EXACT_PER_ENV_BONFERRONI','EXACT_STRATIFIED','LOG_OR_WALD'];labels=['Old bands','Separate exact','Stratified exact','OR-Wald'];cols=[C[1],C[3],C[0],C[5]]
 for design in ['strong','weak']:
  q=d[d.design.eq(design)];f,a=canvas(155,112)
  for k,effect in enumerate([-.8,0,.15,.8]):
   for j,s in enumerate(methods):
    z=q[q.effect.eq(effect)&q.method.eq(s)].iloc[0];x=k+(j-1.5)*.19;a.bar(x,z.correct_rate,width=.165,fc=cols[j],ec='#222222',lw=.6);a.errorbar(x,z.correct_rate,yerr=[[z.correct_rate-z.correct_lo],[z.correct_hi-z.correct_rate]],fmt='none',ecolor='#222222',capsize=2,lw=.65)
  a.set_ylim(0,1.10);a.set_yticks([0,.25,.5,.75,1],['0','25','50','75','100']);a.set_ylabel('Correct sign answers (%) ↑');a.set_xticks(range(4),['−0.8','0','+0.15','+0.8']);a.set_xlabel('True interaction (log-rate scale)');legend(f,labels,cols);f.text(.14,.96,f'G2 | Exact sign inference · {design} observation',fontsize=9,weight='bold');null=q[q.effect.eq(0)&q.method.eq('EXACT_STRATIFIED')].iloc[0];foot(f,f'500 paired datasets per effect; Wilson intervals. No raw 0/1 dot cloud.\nStratified exact: null wrong calls {int(null.wrong)}/500 ({null.wrong_rate:.1%}).\nClassical exact testing is a component; no new-test or finite-magnitude claim.');save(f,2,f'G2_R2_sign_{design}',q,'R21','Four effect groups and four matched methods; outlined narrow bars, black capped Wilson errors, boxed top legend. Binary replicate marks omitted; null risk explicit.')

def line_template(title,ylabel):
 f=plt.figure(figsize=(150/25.4,125/25.4));a=f.add_axes([.14,.23,.81,.59]);a.spines[['top','right']].set_visible(False);a.set_ylabel(ylabel);f.text(.14,.96,title,fontsize=8.6,weight='bold');ins=[a.inset_axes([x,.73,.30,.23]) for x in [.075,.625]]
 for b in ins:b.spines[['top','right']].set_visible(False);b.tick_params(labelsize=5.8,length=2,pad=2)
 return f,a,ins

def g3():
 d=read(3,'G3_R2_summary');cols=[C[0],C[3],C[4]]
 for effect in [-.8,0,.8]:
  q=d[d.effect.eq(effect)&d.fraction.eq(0)&d.method.str.startswith('BOUND_')].copy();q['eta']=q.method.str.replace('BOUND_','').astype(float);f,a,ins=line_template(f'G3 | Declared selection-bias bound · effect {effect:+g}','Wrong sign answers (%) ↓');a.set_ylim(-.05,1.70);a.spines['left'].set_bounds(-.05,1.04);a.set_yticks([0,.25,.5,.75,1],['0','25','50','75','100']);a.set_xlim(-.05,1.08);a.set_xticks([0,.5,1]);a.set_xlabel('Allowed |selection log-OR bias|');a.axhline(.05,color='#777777',ls=(0,(3,3)),lw=.7)
  for j,(bias,c) in enumerate(zip([0,.5,1],cols)):
   z=q[q.bias.eq(bias)].sort_values('eta');a.errorbar(z.eta,z.wrong_rate,yerr=[z.wrong_rate-z.wrong_lo,z.wrong_hi-z.wrong_rate],color=c,lw=1.8,marker='o',ms=5,mec='#222222',mew=.8,ecolor='#222222',capsize=2.5,elinewidth=.7)
   for ax,eta in zip(ins,[0,1]):
    t=z[z.eta.eq(eta)].iloc[0];ax.barh(j,t.correct_rate,.5,fc=c,ec='#222222',lw=.6);ax.errorbar(t.correct_rate,j,xerr=[[t.correct_rate-t.correct_lo],[t.correct_hi-t.correct_rate]],fmt='none',ecolor='#222222',capsize=1.5,lw=.6)
  for ax,eta in zip(ins,[0,1]):ax.set_xlim(0,1.03);ax.set_ylim(2.6,-.6);ax.set_yticks(range(3),['Bias 0','Bias 0.5','Bias 1']);ax.set_xticks([0,.5,1],['0','50','100%']);ax.set_title(f'Correct calls | bound {eta}',fontsize=6.5,pad=5)
  legend(f,['True bias 0','True bias 0.5','True bias 1'],cols,y=.91);foot(f,'500 datasets per point; Wilson intervals. Insets show the cost in correct answers.\nGuarantee applies only when the allowed bound contains the true bias.\nNo calibration in this panel; log-OR bias is not a generator-coefficient bound.');save(f,3,f'G3_R2_bias_{str(effect).replace("-","neg").replace(".","p")}',q,'R04','Colored measured-grid lines, black-edged circles, capped errors, top legend and two integrated horizontal-bar insets; safety line .05. No invented phase or smoothing.')
 q=d[d.effect.ne(0)&d.method.isin(['RANDOM_X_ONLY','XS_SOURCE_ONLY','XS_BIAS_CALIBRATION'])].copy();s=q.groupby(['fraction','method'])[['correct','n']].sum().reset_index();s['rate']=s.correct/s.n;f,a=canvas();methods=['RANDOM_X_ONLY','XS_SOURCE_ONLY','XS_BIAS_CALIBRATION'];labels=['Random X','XS: use X only','XS: bias bounds'];from run_simulations import wilson
 for k,frac in enumerate([0,.1,.25]):
  for j,m in enumerate(methods):
   z=s[s.fraction.eq(frac)&s.method.eq(m)].iloc[0];lo,hi=wilson(z.correct,z.n);x=k+(j-1)*.23;a.bar(x,z.rate,.195,fc=C[j],ec='#222222',lw=.6);a.errorbar(x,z.rate,yerr=[[max(0,z.rate-lo)],[max(0,hi-z.rate)]],fmt='none',ecolor='#222222',capsize=2,lw=.7)
 a.set_xticks(range(3),['0%','10%','25%']);a.set_xlabel('Budget assigned to calibration');a.set_ylabel('Correct sign answers (fraction) ↑');legend(f,labels,C[:3]);f.text(.14,.96,'G3 | Fixed-cost calibration: information versus conservatism',fontsize=8.5,weight='bold');foot(f,'Total proxy cost 1,000; X costs 2, XS costs 3, ordinary record costs 1.\nEqual-weight average across two nonzero effects × three bias settings.\nAll cell-specific outcomes and zero-effect risks are retained in source tables.');save(f,3,'G3_R2_calibration',q,'R21','Grouped narrow outlined bars, Wilson errors, top legend; declared equal-cell summary, not a new independent replicate. Zero XS gain remains visible.')

def g4():
 d=read(4,'G4_R2_paired');refs=['CTMC_M','UNPROJECTED_FUSION','POOLED'];labels=['CTMC','Unprojected fusion','Pooled'];ids=set(np.random.default_rng(2026092411).choice(200,40,replace=False))
 for cancer in ['LUAD','COAD']:
  q=d[d.purpose.eq('bootstrap')&d.cancer.eq(cancer)&d.metric.eq('source_log_score')].copy();q['displayed_point']=q.replicate.isin(ids);f,a=canvas(150,110);f.subplots_adjust(right=.78,top=.82);random=np.random.default_rng(911)
  for j,ref in enumerate(refs):
   z=q[q.reference.eq(ref)];v=z.gain.to_numpy();ys=np.linspace(v.min(),v.max(),240)
   if len(v)>5 and v.std()>1e-12:den=gaussian_kde(v)(ys);den*=.32/den.max();a.fill_betweenx(ys,j+.04,j+.04+den,color=C[j],ec='#333333',lw=.7,alpha=.85)
   a.scatter(j-random.uniform(.07,.25,len(z[z.displayed_point])),z[z.displayed_point].gain,s=4,c=C[j],alpha=.7,lw=0);a.boxplot(v,positions=[j],widths=.12,whis=(2.5,97.5),showfliers=False,patch_artist=True,boxprops=dict(fc='white',ec='#333333',lw=.8),medianprops=dict(c='#222222',lw=.9),whiskerprops=dict(c='#333333',lw=.8),capprops=dict(c='#333333',lw=.7),manage_ticks=False)
  a.spines[['top','right']].set_visible(True);a.spines[['top','right']].set_color('#aaaaaa');a.axhline(0,c='#555555',ls=(0,(4,3)),lw=.75);a.set_xlim(-.5,2.55);a.set_xticks(range(3),['CTMC','Fusion','Pooled']);a.set_ylabel('Mixture log-score gain over comparator ↑');a.legend(handles=[Patch(fc=c,ec='#333333',label=l) for c,l in zip(C[:3],labels)],loc='center left',bbox_to_anchor=(1.01,.5),frameon=False,fontsize=6.5);f.text(.14,.95,f'G4 | Patient-level predictive comparison · {cancer}',fontsize=8.7,weight='bold');foot(f,'Fixed 80 enrolled bootstrap slots; tuning uses training-only validation.\n40/200 IDs selected for display; every completed pair enters summaries.\nWhite box: median/IQR; whiskers: 2.5–97.5%. Failures reported separately.');save(f,4,f'G4_R2_predictive_{cancer}',q,'R27','Left sparse rain, narrow white box with dark median and whiskers, outlined right half-KDE, frame and right legend. Complete paired outcomes include negative effects; no parameter recovery claim.')

def g5():
 d=read(5,'G5_R2_summary');raw=read(5,'G5_R2_raw');aliases=read(5,'G5_R2_aliases');assert aliases.all_four_equal.all(),'Do not merge unequal strategies';methods=['RANDOM_FEASIBLE','COUNT_ONLY','INVARIANT_ROBUST'];cols=[C[0],C[1],C[5]]
 for cancer in ['LUAD','COAD']:
  q=d[d.cancer.eq(cancer)];f,a,ins=line_template(f'G5 | Equal actual acquisition cost · {cancer}','Source-record log-score gain ↑');lim=max(.05,float(q.gain_hi.max()));low=min(-.02,float(q.gain_lo.min()));span=lim-low;a.set_ylim(low-.05*span,low+1.85*span);a.spines['left'].set_bounds(low-.05*span,lim+.08*span);a.set_xlim(9,39);a.set_yticks([v for v in matplotlib.ticker.MaxNLocator(nbins=4).tick_values(low,lim) if low-.05*span<=v<=lim+.08*span]);a.set_xticks([12,24,36]);a.set_xlabel('Prespecified acquisition cap (cost proxy)');a.axhline(0,c='#999999',lw=.7)
  for j,(s,c) in enumerate(zip(methods,cols)):
   z=q[q.strategy.eq(s)].sort_values('cap');a.errorbar(z.cap,z.gain_mean,yerr=[z.gain_mean-z.gain_lo,z.gain_hi-z.gain_mean],color=c,lw=1.8,marker='o',ms=5,mec='#222222',mew=.8,ecolor='#222222',capsize=2.5,elinewidth=.7)
   t=z[z.cap.eq(36)].iloc[0];ins[0].barh(j,t.mean_cost,height=.5,fc=c,ec='#222222',lw=.6)
   pair=read(5,'G5_R2_paired');v=pair[pair.cancer.eq(cancer)&pair.cap.eq(36)&pair.reference.eq(s)].gain if s!='INVARIANT_ROBUST' else pd.Series([0.]*50);m=v.mean();err=1.96*v.std(ddof=1)/np.sqrt(len(v));ins[1].barh(j,m,height=.5,fc=c,ec='#222222',lw=.6);ins[1].errorbar(m,j,xerr=err,fmt='none',ecolor='#222222',capsize=1.5,lw=.6)
  for ax in ins:ax.set_ylim(2.6,-.6);ax.set_yticks(range(3),['Random','Count','4 aliases'])
  ins[0].set_title('Actual cost | cap 36',fontsize=6.5,pad=5);ins[0].set_xlim(0,40);ins[0].set_xticks([0,20,40]);ins[1].set_title('Aliases minus comparator',fontsize=6.5,pad=5);ins[1].axvline(0,c='#777777',lw=.5);legend(f,['Random','Count only','Max. panel = D = c = robust'],cols,y=.91);foot(f,'50 allocation seeds; mean ± 1.96 seed-MCSE, conditional on the same patient pool.\nEach replicate uses a common feasible actual cost, chosen before reveal.\nFour action-equivalent strategies shown once; no independent algorithm advantage.');save(f,5,f'G5_R2_costmatched_{cancer}',raw[raw.cancer.eq(cancer)],'R04','Black-edged circles, colored lines, capped MCSE; integrated actual-cost and paired-gain insets; top legend. No artificial phase shading: no meaningful phase change in this budget range.')
def g4_audit():
 d=read(4,'G4_R2_joint_audit_raw');methods=['SEPARATE_SHARP','JOINT_EXACT','INDEPENDENCE_IMPUTATION'];cols=[C[0],C[2],C[4]];from run_simulations import wilson
 for cancer in ['LUAD','COAD']:
  q=d[d.cancer.eq(cancer)];f,a,ins=line_template(f'G4 | Joint information at equal assay cost · {cancer}','Mean joint-frequency interval width ↓');maxy=q.groupby(['budget','method']).width.mean().max();a.set_ylim(0,maxy*1.9);a.spines['left'].set_bounds(0,maxy*1.10);a.set_yticks([v for v in matplotlib.ticker.MaxNLocator(nbins=4).tick_values(0,maxy*1.05) if v<=maxy*1.10]);a.set_xlim(150,850);a.set_xticks([200,400,800]);a.set_xlabel('Actual event-measurement cost')
  for j,(s,c) in enumerate(zip(methods,cols)):
   means=[];errs=[]
   for budget in [200,400,800]:
    vals=q[q.method.eq(s)&q.budget.eq(budget)].width;means.append(vals.mean());errs.append(1.96*vals.std(ddof=1)/np.sqrt(len(vals)))
   a.errorbar([200,400,800],means,yerr=errs,color=c,lw=1.8,marker='o',ms=5,mec='#222222',mew=.8,ecolor='#222222',capsize=2.5,elinewidth=.7)
   for ax,budget in zip(ins,[200,800]):
    z=q[q.method.eq(s)&q.budget.eq(budget)];v=z.covered.mean();lo,hi=wilson(z.covered.sum(),len(z));ax.barh(j,v,.5,fc=c,ec='#222222',lw=.6);ax.errorbar(v,j,xerr=[[max(0,v-lo)],[max(0,hi-v)]],fmt='none',ecolor='#222222',capsize=1.5,lw=.6);ax.text(1.1,j,f'{v:.1%}',fontsize=5.6,ha='left',va='center',clip_on=False)
  for ax,budget in zip(ins,[200,800]):ax.set_xlim(0,1.02);ax.set_ylim(2.6,-.6);ax.set_yticks(range(3),['Marginal','Joint','Indep.*']);ax.set_xticks([0,.5,1],['0','50','100%']);ax.set_title(f'Coverage | cost {budget}',fontsize=6.5,pad=5);ax.axvline(.95,c='#666666',ls=(0,(2,2)),lw=.5)
  legend(f,['Marginal bounds','Joint audit','Independence*'],cols,y=.91);foot(f,'500 random audits per cost; mean width ± 1.96 MCSE; inset Wilson intervals.\nTarget: finite-cohort reported joint frequency, not a natural evolution parameter.\n* Independence is an assumption ablation. More unpaired patients do not reveal linkage.');save(f,4,f'G4_R2_joint_audit_{cancer}',q,'R04','Actual-cost trajectories with filled black-edged circles, capped MCSE, horizontal top legend and two integrated coverage insets with separate value columns. No fabricated phase or clinical significance.')

if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('groups',nargs='+',type=int);args=a.parse_args()
 for g in args.groups:
  globals()[f'g{g}']();
  if g==4:g4_audit()
  (P/f'g{g}/revision_r2/configs/FIGURE_MANIFEST.json').write_text(json.dumps(MAN[g],ensure_ascii=False,indent=2));print('Plotted',g,len(MAN[g]),flush=True)
