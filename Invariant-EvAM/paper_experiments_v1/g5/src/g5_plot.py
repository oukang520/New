"""Data-selected template implementation: independent vector figures, all statistics retained."""
from pathlib import Path
import sys,json,hashlib,argparse
import numpy as np,pandas as pd
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from scipy.stats import gaussian_kde
R=Path(__file__).resolve().parents[3];G=R/'paper_experiments_v1/g5';sys.path.insert(0,str(R/'src'))
from invariant_evam.figure_style import apply_publication_style
apply_publication_style({'svg.fonttype':'none','pdf.fonttype':42,'font.size':8,'axes.labelsize':8,'xtick.labelsize':7,'ytick.labelsize':7,'axes.linewidth':.75})
C=['#0072B2','#56B4E9','#009E73','#E69F00','#D55E00','#CC79A7'];CORE=['IDENTITY_SHARED','BURDEN_SELECTION_SHARED','PRESPECIFIED_GENE_SELECTION','BOUNDED_BIOLOGY'];NAT=CORE+['HIERARCHICAL','TWO_STEP_FUSION','INDEPENDENT'];NL=dict(zip(NAT,['Identity','Burden','Gene selection','Bounded biology','Hierarchical','Two-step fusion','Independent']));STR=['RANDOM_FEASIBLE','COUNT_ONLY','MAX_PANEL','D_OPT','C_OPT','INVARIANT_ROBUST'];SL=dict(zip(STR,['Random','Count only','Max. panel','D-optimal','c-optimal','Robust target']));SC=dict(zip(STR,C));CAT=json.loads((R/'configs/figure_reference_catalog.json').read_text())['records'];OUT=G/'FINAL_FIGURES';DATA=G/'figure_data';PRE=Path('/tmp/invariant_evam_g5_previews');MAN=[]
for p in [OUT,DATA,PRE]:p.mkdir(exist_ok=True)
DISPLAY={k:set(np.random.default_rng(seed).choice(n,num,replace=False).tolist()) for k,n,num,seed in [('scatter',200,20,2026092501),('rain',200,40,2026092502),('bars',50,15,2026092503)]}
(G/'configs/DISPLAY_SUBSAMPLING.json').write_text(json.dumps(dict(purpose='DISPLAY_ONLY; no filtering in summary, limits or inferential data',ids={k:sorted(v) for k,v in DISPLAY.items()}),indent=2))
def read(n):return pd.read_csv(G/'results'/f'{n}.tsv',sep='\t')
def canvas(w,h,left=.15,right=.96,bottom=.19,top=.83):
 f,a=plt.subplots(figsize=(w/25.4,h/25.4));f.subplots_adjust(left=left,right=right,bottom=bottom,top=top);a.spines[['top','right']].set_visible(False);return f,a

def save(fig,name,df,ref,endpoint,adapt):
 df.to_csv(DATA/f'{name}.tsv',sep='\t',index=False)
 for ext in ['pdf','svg']:fig.savefig(OUT/f'{name}.{ext}',dpi=600)
 fig.savefig(PRE/f'{name}.png',dpi=180)
 rr=next(r for r in CAT if r['id']==ref);MAN.append(dict(panel_id=name,primary_reference_id=ref,reference_file=rr['source_file_absolute'],reference_sha256=rr['sha256'],endpoint=endpoint,adaptations=adapt,physical_size_mm=(fig.get_size_inches()*25.4).tolist(),dpi=600,palette=C,data=str(DATA/f'{name}.tsv'),data_sha256=hashlib.sha256((DATA/f'{name}.tsv').read_bytes()).hexdigest(),review_status='awaiting_visual_review'));plt.close(fig)

def frame(ax):
 for sp in ax.spines.values():sp.set_visible(True);sp.set_linewidth(.65);sp.set_color('#999999' if sp.spine_type in ['top','right'] else '#222222')

def triptych(n,height=115,labelwidth=.27):
 f=plt.figure(figsize=(160/25.4,height/25.4));bottom=.20;top=.83
 widths=[.155,.155,.16];starts=[labelwidth,labelwidth+.185,labelwidth+.37];axes=[f.add_axes([x,bottom,w,top-bottom]) for x,w in zip(starts,widths)]
 for j,a in enumerate(axes):
  a.set_ylim(n-.5,-.5);a.set_yticks(range(n));a.tick_params(axis='y',length=2,labelsize=6.3);frame(a);a.grid(axis='x',color='#dddddd',lw=.35);a.set_axisbelow(True)
  if j<2:a.set_xlim(0,1);a.set_xticks([0,.25,.5,.75,1]);a.tick_params(axis='x',rotation=90,labelsize=6)
  else:a.tick_params(axis='x',rotation=90,labelsize=6)
  if j>0:a.set_yticklabels([])
 return f,axes

def stacked(a,i,counts,colors,n):
 left=0
 for val,col in zip(counts,colors):a.barh(i,val/n,left=left,height=.86,color=col,edgecolor='white',linewidth=.25);left+=val/n

def legends(f,labels,colors,y,title):
 f.legend(handles=[Patch(fc=c,ec='#555555',lw=.35,label=t) for c,t in zip(colors,labels)],loc='upper left',bbox_to_anchor=(.835,y),frameon=False,fontsize=6.2,title=title,title_fontsize=7,handlelength=.85,handletextpad=.4,labelspacing=.35)

def audit():
 records=json.loads((G/'data_private/natural_records.json').read_text());d=read('G5_natural_denominator');clinical=read('G5_clinical_composition');f,axes=triptych(3,75,labelwidth=.21);tab=[]
 for ax in axes:
  pos=ax.get_position();ax.set_position([pos.x0,.34,pos.width,.43])
 for i,e in enumerate(['MSK-IMPACT468','MSK-IMPACT410','MSK-IMPACT505']):
  q=[z for z in records if z['environment']==e];burden=[min(bin(z['reference_state']).count('1'),3) for z in q];bc=np.bincount(burden,minlength=4);n=len(q);se=clinical[clinical.environment.eq(e)&clinical.field.eq('SEX')].set_index('value').n;sex=[int(se.get(k,0)) for k in ['Female','Male','Unknown']];stacked(axes[0],i,bc,C[:4],n);stacked(axes[1],i,sex,[C[5],C[4],'white'],n);axes[2].plot(n/1000,i,'o',ms=3,c='#222222');axes[2].annotate(f'{n:,}',(n/1000,i),xytext=(5,0),textcoords='offset points',va='center',fontsize=6.4);tab.append(dict(environment=e,n=n,nominal_target_coverage='4/4 IDENTICAL',strict_callable_negative_n=0,**dict(zip(['burden0','burden1','burden2','burden3plus'],bc)),**dict(zip(['female','male','unknown_sex'],sex))))
 axes[0].set_yticklabels(['IMPACT468','IMPACT410','IMPACT505']);axes[0].set_title('Module burden',fontsize=8);axes[1].set_title('Recorded sex',fontsize=8);axes[2].set_title('Unique patients',fontsize=8);axes[0].set_xlabel('Proportion',fontsize=7);axes[1].set_xlabel('Proportion',fontsize=7);axes[2].set_xlabel('Thousands',fontsize=7);axes[2].set_xlim(0,2.85);axes[2].set_xticks([0,1,2]);legends(f,['0','1','2','3 or 4'],C[:4],.81,'Altered events');legends(f,['Female','Male','Unknown'],[C[5],C[4],'white'],.49,'Recorded sex')
 for ax in axes[:2]:
  for patch in ax.patches:patch.set_y(patch.get_y()+.12);patch.set_height(.62);patch.set_edgecolor('#555555');patch.set_linewidth(.2)
 f.text(.045,.95,'Natural cohort | identical four-target coverage across panels',fontsize=9,weight='bold');f.text(.045,.055,'TP53 / KRAS / STK11 / KEAP1: same nominal measurement operator.\n4,920 unique patients; proxy negatives only; no sample-level callability.',fontsize=7,linespacing=1.5)
 save(f,'G5_01_natural_audit',pd.DataFrame(tab),'R28','Panel composition and patient denominators; panel names do not establish assay complementarity','R28 two narrow stacked columns, aligned count-dot column, vertical proportion ticks, frames and two right legends. Three real environments rather than 20 biological labels; no fictitious sampling dates or callable negatives.')

def tradeoff():
 b=read('G5_patient_bootstrap');b=b[b.target_id.eq('D4_SPARSE10__interaction_0_to_1')].copy();b['displayed_point']=b.replicate.isin(DISPLAY['scatter']);f,a=canvas(117,108,left=.15,top=.85,bottom=.16)
 for j,s in enumerate(CORE):
  q=b[b.spec_id.eq(s)&b.fit_success];shown=q[q.displayed_point];a.scatter(shown.training_deviance,shown.target_information,s=9,c=C[j],alpha=.6,edgecolors='none',label=NL[s]);a.errorbar(q.training_deviance.mean(),q.target_information.mean(),xerr=q.training_deviance.std(),yerr=q.target_information.std(),fmt='D',ms=4.2,color='black',mec='black',mfc='black',capsize=2.5,elinewidth=.8,zorder=5)
 a.axhline(0,c='#333333',ls=(0,(4,4)),lw=.85,label='No local target information');a.plot([],[],'D',c='black',ms=4,label='Mean ± bootstrap SD');a.set_xlabel('Training saturated deviance ↓');a.set_ylabel('Effective information for TP53 → KRAS ↑');a.set_ylim(bottom=-1);a.tick_params(direction='in',top=True,right=True);frame(a);a.spines[['top','right']].set_color('#222222');a.minorticks_on();a.tick_params(which='minor',direction='in',top=True,right=True,length=2);a.legend(loc='lower right',bbox_to_anchor=(.99,.035),frameon=False,fontsize=6.4,handlelength=1.7,labelspacing=.35);f.text(.15,.95,'Natural cohort | compatibility–information trade-off',fontsize=8.5,weight='bold');f.text(.15,.035,'20 / 200 patient resamples shown per family; summaries use all 200.',fontsize=6.5)
 save(f,'G5_02_compatibility_information',b,'R23','Observed deviance and target information, retaining full bootstrap uncertainty','R23 grouped colored points, black mean diamonds, horizontal AND vertical capped SD, inward four-sided major/minor ticks, inside legend. Dashed zero-information benchmark replaces unrelated theory curve; no artificial regression. Reference phi=0 anchor remains explicit in report.')

def heldout():
 b=read('G5_heldout_score_bootstrap');models=NAT[1:];col=dict(zip(models,C))
 for e in ['MSK-IMPACT410','MSK-IMPACT505']:
  q=b[b.heldout.eq(e)];identity=q[q.spec_id.eq('IDENTITY_SHARED')].set_index('replicate').log_score;f,a=canvas(150,108,left=.13,right=.78,top=.82,bottom=.21);tab=[];rand=np.random.default_rng(5201)
  for j,s in enumerate(models):
   z=q[q.spec_id.eq(s)].copy();z['gain_vs_identity']=identity.loc[z.replicate].to_numpy()-z.log_score.to_numpy();z['displayed_point']=z.replicate.isin(DISPLAY['rain']);v=z.gain_vs_identity.to_numpy();ys=np.linspace(v.min(),v.max(),240)
   if v.std()>1e-12:
    den=gaussian_kde(v)(ys);den*=.32/den.max();a.fill_betweenx(ys,j+.035,j+.035+den,fc=col[s],ec='#333333',lw=.7,alpha=.9)
   dx=j-.07-rand.uniform(.035,.24,len(v));a.scatter(dx[z.displayed_point],v[z.displayed_point],s=3,c=col[s],alpha=.7,linewidths=0)
   a.boxplot(v,positions=[j],widths=.115,whis=(2.5,97.5),showfliers=False,patch_artist=True,boxprops=dict(fc='white',ec='#333333',lw=.8),medianprops=dict(c='#222222',lw=.9),whiskerprops=dict(c='#333333',lw=.8),capprops=dict(c='#333333',lw=.7),manage_ticks=False,zorder=4);tab.extend(z.assign(plot_group=j).to_dict('records'))
  frame(a);a.axhline(0,c='#444444',ls=(0,(4,3)),lw=.8,zorder=0);a.set_xlim(-.55,5.55);a.set_xticks(range(6),['Burden','Genes','Biology','Hier.','Fusion','Indep.']);a.set_ylabel('Paired log-score gain over identity ↑');a.legend(handles=[Patch(fc=col[s],ec='#333333',label=NL[s]) for s in models],loc='center left',bbox_to_anchor=(1.015,.55),frameon=False,fontsize=6.4,handlelength=1,handletextpad=.4);f.text(.13,.96,f'Natural cohort | held-out {e.replace("MSK-","")}',fontsize=9,weight='bold');f.text(.13,.90,'Reference IMPACT468 retained; 20% nuisance calibration / 80% evaluation.',fontsize=6.7);f.text(.13,.055,'40 / 200 paired evaluation-patient resamples shown; full KDE, median + IQR.\nWhiskers: 2.5–97.5 percentiles; conditional on fixed training/calibration fits.',fontsize=6.5,linespacing=1.5)
  save(f,'G5_03_'+e.replace('MSK-',''),pd.DataFrame(tab),'R27','Foreground actual held-out predictive gains against common identity; positive and negative results included','R27 left colored rain, narrow white box with dark median and whiskers, right half-KDE outlined in dark stroke, framed axes, right legend. Six prespecified comparisons; only display dots thinned with paired outcome-independent IDs. No natural-theta coverage claim.')

def target_name(t):
 genes=['TP53','KRAS','STK11','KEAP1'];s=t.split('__')[1]
 if 'contrast' in s:return 'Contrast (edge 1 − 2)'
 u,v=s.replace('interaction_','').split('_to_');return genes[int(u)]+' → '+genes[int(v)]

def targets():
 b=read('G5_patient_bootstrap');t=read('G5_all_target_interpretations');registry=pd.read_csv(G/'configs/target_registry.tsv',sep='\t');ids=registry[registry.model_id.eq('D4_SPARSE10')].target_id.tolist();f,axes=triptych(28,168,labelwidth=.29);tab=[];labels=[]
 for i,(s,tid) in enumerate((s,t) for s in CORE for t in ids):
  q=b[b.spec_id.eq(s)&b.target_id.eq(tid)];n=len(q);complete=q.status.eq('COMPLETE');counts=[sum(complete&q.sign.eq('POSITIVE')),sum(complete&q.sign.eq('NEGATIVE')),sum(complete&q.sign.eq('UNKNOWN')),sum(~complete)];finite=complete&np.isfinite(q.lo)&np.isfinite(q.hi);extent=[sum(finite),sum(complete&~finite),sum(~complete)];stacked(axes[0],i,counts,[C[2],C[0],C[3],C[4]],n);stacked(axes[1],i,extent,[C[1],C[5],C[4]],n);z=t[t.spec_id.eq(s)&t.target_id.eq(tid)].iloc[0];axes[2].plot(z.estimate,i,'o',ms=2.5,c='#222222');labels.append(target_name(tid));tab.append(dict(spec_id=s,target_id=tid,n=n,positive=counts[0],negative=counts[1],unknown=counts[2],failed=counts[3],finite=extent[0],unbounded=extent[1],estimate=z.estimate,interpretation_level=z.interpretation_level,biological_claim_allowed=False))
 for j,a in enumerate(axes):
  for k in [6.5,13.5,20.5]:a.axhline(k,c='#555555',lw=.7)
 axes[0].set_yticklabels(labels,fontsize=6.1);axes[0].set_title('Bootstrap sign',fontsize=8);axes[1].set_title('Profile extent',fontsize=8);axes[2].set_title('Main estimate',fontsize=8);axes[0].set_xlabel('Proportion',fontsize=7);axes[1].set_xlabel('Proportion',fontsize=7);axes[2].set_xlabel('Log-hazard scale',fontsize=7);axes[2].set_xlim(-3.7,3.5);axes[2].set_xticks([-3,0,3]);axes[2].axvline(0,c='#bbbbbb',lw=.5)
 legends(f,['Positive','Negative','Unknown','Failed'],[C[2],C[0],C[3],C[4]],.81,'Sign');legends(f,['Finite','Unbounded','Failed'],[C[1],C[5],C[4]],.60,'Extent')
 for k,s in enumerate(CORE):f.text(.025,.83-(k*7+3.5)*.63/28,NL[s],fontsize=7,rotation=90,va='center',ha='center',weight='bold')
 f.text(.06,.95,'Natural cohort | all seven sparse-model targets, all four families',fontsize=9,weight='bold');f.text(.06,.895,'200 patient refits / row; local Bonferroni profiles. All four primary families are incompatible\nwith observed counts (parametric-bootstrap p = 0.005); sign stability does not override misfit.',fontsize=6.7,linespacing=1.5);f.text(.06,.07,'Full bidirectional topology: all 13 targets structurally unresolved in both tested families.\nAll 75 main target / specification entries are retained in the accompanying interpretation table.',fontsize=7,linespacing=1.5)
 save(f,'G5_04_complete_targets',pd.DataFrame(tab),'R28','All seven main targets including contrast, unknown and computational failures; no historical-positive selection','R28 aligned stacked sign and extent columns, third point-estimate column, compact repeated rows, vertical numeric ticks and separate right legends. Repeated seven-row blocks identify families; numerical failure not collapsed into scientific unknown. Full-topology13-target results remain complete in source table, not relabeled as same sparse contrast.')

def design_outcomes():
 d=read('G5_design_outer_evaluation');names=['Correct selection family','Common omitted pair selection','Biology shift (+0.2)']
 for scenario in range(3):
  f,axes=triptych(18,132,labelwidth=.245);tab=[];labels=[]
  for i,(budget,s) in enumerate((b,s) for b in [30,75,150] for s in STR):
   q=d[d.scenario.eq(scenario)&d.budget.eq(budget)&d.strategy.eq(s)];n=len(q);complete=q.profile_status.eq('COMPLETE');counts=[int(q.correct_call.sum()),int(q.wrong_sign.sum()),int((complete&~q.called).sum()),int((~complete).sum())];extent=[int(q.finite_interval.sum()),int((complete&~q.finite_interval).sum()),int((~complete).sum())];stacked(axes[0],i,counts,[C[2],C[4],C[3],C[5]],n);stacked(axes[1],i,extent,[C[1],C[0],C[5]],n);v=q.source_log_score_gain.dropna();mean=v.mean();err=1.96*v.std(ddof=1)/np.sqrt(len(v));axes[2].errorbar(mean,i,xerr=err,fmt='o',ms=2.4,c='#222222',capsize=1.5,elinewidth=.65);labels.append(f'{int(budget/3)}% · {SL[s]}');good=q[complete];tab.append(dict(scenario=scenario,budget=budget,strategy=s,n=n,correct=counts[0],wrong=counts[1],unknown=counts[2],failed=counts[3],finite=extent[0],unbounded=extent[1],source_gain=mean,mc_lo=mean-err,mc_hi=mean+err,coverage=good.covers_truth.mean(),actual_cost_mean=q.cost.mean()))
  for a in axes:
   for k in [5.5,11.5]:a.axhline(k,c='#555555',lw=.75)
  axes[0].set_yticklabels(labels);axes[0].set_title('Actual sign answer',fontsize=8);axes[1].set_title('Profile extent',fontsize=8);axes[2].set_title('Prediction gain',fontsize=8);axes[0].set_xlabel('Proportion',fontsize=7);axes[1].set_xlabel('Proportion',fontsize=7);axes[2].set_xlabel('Log score ↑',fontsize=7);axes[2].axvline(0,c='#777777',lw=.7,ls=(0,(3,3)));legends(f,['Correct','Wrong','Unknown','Failed'],[C[2],C[4],C[3],C[5]],.81,'Answer');legends(f,['Finite','Unbounded','Failed'],[C[1],C[0],C[5]],.55,'Extent');f.text(.05,.95,f'Independent simulation | {names[scenario]}',fontsize=9,weight='bold');f.text(.05,.90,('200 independent datasets; max-panel / D / c / robust coverage at 50% budget: '+ ('78.0%' if scenario==1 else '95.5%')+'.'),fontsize=6.8);f.text(.05,.07,'10 / 25 / 50% of base measurement cost; dots = mean gain ± 1.96 Monte Carlo SE.\nPrediction is scored on independent future records; unknown is not correct sign recovery.',fontsize=6.7,linespacing=1.5)
  save(f,f'G5_05_design_scenario_{scenario}',pd.DataFrame(tab),'R28','All three budgets and all six strategies: correct/wrong/unknown, finite intervals, actual independent prediction gains','R28 two narrow outcome compositions and aligned dot-metric column. Six strategy rows per budget, both legends, frames, vertical quarter ticks; small horizontal MC-error bars are added to avoid false precision. No theoretical rank substituted for empirical gains.')

def direct_gain():
 d=read('G5_design_direct_evaluation');f,a=canvas(150,108,left=.13,right=.97,top=.78,bottom=.25);tab=[];rand=np.random.default_rng(5206)
 # Collapse only exact action/result aliases, not unfavorable results or near ties.
 alias=['MAX_PANEL','D_OPT','C_OPT','INVARIANT_ROBUST'];base=d[d.strategy.eq('INVARIANT_ROBUST')].set_index(['budget','replicate'])
 for s in alias:
  z=d[d.strategy.eq(s)].set_index(['budget','replicate']);assert z.candidate_id.equals(base.candidate_id);assert np.allclose(z.source_log_score_gain,base.source_log_score_gain,equal_nan=True)
 plotted=['RANDOM_FEASIBLE','COUNT_ONLY','INVARIANT_ROBUST'];displaylabels=['Random','Count only','Max. panel = D = c = robust']
 for k,budget in enumerate([30,75,150]):
  for j,s in enumerate(plotted):
   z=d[d.budget.eq(budget)&d.strategy.eq(s)].copy();z['exact_alias_members']='|'.join(alias) if s=='INVARIANT_ROBUST' else s;v=z.source_log_score_gain.to_numpy();v=v[np.isfinite(v)];x=k+(j-1)*.225;mean=v.mean();sd=v.std(ddof=1);a.bar(x,mean,width=.197,fc=SC[s],ec='#222222',lw=.7,zorder=2);z['displayed_point']=z.replicate.isin(DISPLAY['bars']);shown=z[z.displayed_point];a.scatter(x+rand.uniform(-.060,.060,len(shown)),shown.source_log_score_gain,s=5,fc='white',ec='#222222',lw=.35,alpha=.65,zorder=3);a.errorbar(x,mean,yerr=sd,fmt='none',ecolor='#222222',elinewidth=.75,capsize=2,zorder=4);tab.extend(z.assign(plot_x=x,summary_mean=mean,summary_sd=sd).to_dict('records'));a.text(x+.09,mean+sd+.04,f'{mean:+.3f}',ha='center',va='bottom',rotation=90,fontsize=6.7)
 vals=d.source_log_score_gain.dropna();a.set_ylim(min(vals.min()-.02,-.03),max(vals.max(),max(z['summary_mean']+z['summary_sd'] for z in tab))+.18);a.axhline(0,c='#333333',lw=.8,zorder=1);a.set_xticks(range(3),['10% budget cap','25% budget cap','50% budget cap']);a.set_ylabel('Source-record log-score gain after acquisition ↑');a.legend(handles=[Patch(fc=SC[s],ec='#222222',lw=.6,label=label) for s,label in zip(plotted,displaylabels)],loc='lower center',bbox_to_anchor=(.5,1.025),ncol=3,frameon=True,fancybox=False,edgecolor='#bbbbbb',fontsize=6.3,handlelength=1.1,columnspacing=1);f.text(.08,.97,'Direct records | measured benefit, identical strategies shown once',fontsize=8.8,weight='bold');f.text(.13,.045,'15 / 50 seeds shown; mean ± SD from all 50. Four exact aliases share one bar, not fourfold N.\nSame 449 evaluation patients; disjoint acquisition pools. Actual spending is reported separately.',fontsize=6.3,linespacing=1.5)
 save(f,'G5_06_direct_record_gain',pd.DataFrame(tab),'R21','Actual source prediction gains and exact equivalence of maximum-panel and three information strategies','R21 narrow filled outlined bars, open raw markers, capped errors, boxed top legend and vertical endpoint labels. Four strategies collapsed ONLY after asserting every chosen action and every loss identical across all150 decisions; original900 rows retained. Means replace invalid P values; seed SD is not patient uncertainty. 15 outcome-independent IDs shown; summaries use all50.')

def direct_outcomes():
 d=read('G5_design_direct_evaluation');f,axes=triptych(18,132,labelwidth=.245);tab=[];labels=[]
 for i,(budget,s) in enumerate((b,s) for b in [30,75,150] for s in STR):
  q=d[d.budget.eq(budget)&d.strategy.eq(s)];n=len(q);ok=q.profile_status.eq('COMPLETE');counts=[sum(ok&q.sign.eq('POSITIVE')),sum(ok&q.sign.eq('NEGATIVE')),sum(ok&q.sign.eq('UNKNOWN')),sum(~ok)];extent=[sum(q.finite_interval),sum(ok&~q.finite_interval),sum(~ok)];stacked(axes[0],i,counts,[C[2],C[0],C[3],C[4]],n);stacked(axes[1],i,extent,[C[1],C[5],C[4]],n);axes[2].plot(q.cost.mean(),i,'o',ms=2.7,c='#222222');labels.append(f'{int(budget/3)}% · {SL[s]}');tab.append(dict(budget=budget,strategy=s,n=n,positive=counts[0],negative=counts[1],unknown=counts[2],failed=counts[3],finite=extent[0],unbounded=extent[1],cost_mean=q.cost.mean(),new_patients_mean=q.new_unique_patients.mean(),remeasured_mean=q.remeasured_existing.mean(),natural_theta_truth_available=False))
 for a in axes:
  for k in [5.5,11.5]:a.axhline(k,c='#555555',lw=.7)
 axes[0].set_yticklabels(labels);axes[0].set_title('Conditional sign',fontsize=8);axes[1].set_title('Profile extent',fontsize=8);axes[2].set_title('Actual spending',fontsize=8);axes[0].set_xlabel('Proportion',fontsize=7);axes[1].set_xlabel('Proportion',fontsize=7);axes[2].set_xlabel('Measurement proxy',fontsize=7);axes[2].set_xlim(0,160);axes[2].set_xticks([0,50,100,150]);legends(f,['Positive','Negative','Unknown','Failed'],[C[2],C[0],C[3],C[4]],.81,'Sign');legends(f,['Finite','Unbounded','Failed'],[C[1],C[5],C[4]],.55,'Extent');f.text(.05,.95,'Direct records | honest answers, interval limits and acquisition cost',fontsize=9,weight='bold');f.text(.05,.90,'All six strategies × three budget caps; no natural parameter truth is assumed.',fontsize=7);f.text(.05,.07,'Repeated allocations of one held-out patient pool; signs are model-conditional, not validated biology.\nUnknown and unbounded answers remain visible; remeasurement does not create new patients.',fontsize=6.7,linespacing=1.5)
 save(f,'G5_07_direct_uncertainty_cost',pd.DataFrame(tab),'R28','Retain unknown/unbounded conditional natural-record inference and actual proxy spending','R28 two aligned stacked columns, adjacent mean-cost dots, both legends, vertical ticks and budget-block separators; no invented natural coverage. Capacity restriction and integer slack remain visible in spend column and underlying data.')

def main():
 a=argparse.ArgumentParser();a.add_argument('--natural-only',action='store_true');args=a.parse_args();status=json.loads((G/'results/RUN_STATUS.json').read_text())
 if not args.natural_only:assert status['all_attempted'],'Do not publish figures from incomplete formal runs'
 audit();tradeoff();heldout();targets()
 if not args.natural_only:design_outcomes();direct_gain();direct_outcomes()
 (G/'configs/FIGURE_MANIFEST.json').write_text(json.dumps(MAN,ensure_ascii=False,indent=2));print('Wrote',len(MAN),'independent PDF/SVG pairs. Actual visual review pending.')
if __name__=='__main__':main()
