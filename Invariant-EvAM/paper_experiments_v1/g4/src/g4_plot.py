from pathlib import Path
import sys,json,hashlib
import numpy as np,pandas as pd
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch,FancyBboxPatch
from matplotlib.lines import Line2D
from scipy.stats import gaussian_kde
R=Path(__file__).resolve().parents[3];G=R/'paper_experiments_v1/g4';sys.path.insert(0,str(R/'src'))
from invariant_evam.figure_style import apply_publication_style
apply_publication_style({'svg.fonttype':'none','pdf.fonttype':42,'font.size':8,'axes.labelsize':8.5,'xtick.labelsize':7.5,'ytick.labelsize':7.5,'axes.linewidth':.85,'xtick.major.width':.8,'ytick.major.width':.8})
C=['#0072B2','#56B4E9','#009E73','#E69F00','#D55E00','#CC79A7'];METHODS=['M','B1','B3','B4'];MC=dict(zip(METHODS,[C[0],C[3],C[5],C[2]]));LABEL={'M':'Invariant-EvAM','B1':'Pooled (ablation)','B3':'Hierarchical','B4':'Two-step fusion'}
OUT=G/'FINAL_FIGURES';PRE=Path('/tmp/invariant_evam_g4_previews');OUT.mkdir(exist_ok=True);PRE.mkdir(exist_ok=True);DATA=G/'figure_data';MAN=[]
D=pd.read_csv(G/'results/G4_all_method_results.tsv',sep='\t');PAIR=pd.read_csv(G/'results/G4_paired_losses.tsv',sep='\t');AUX=pd.read_csv(G/'results/G4_model_anchored_auxiliary.tsv',sep='\t');CAT=json.loads((R/'configs/figure_reference_catalog.json').read_text())['records']
assert json.loads((G/'results/RUN_STATUS.json').read_text())['run_status']=='COMPLETE'
# Display-only IDs, sampled independently of values; shared across paired methods and cancers.
DISPLAY_IDS={kind:set(np.random.default_rng(seed).choice(total,n,replace=False).tolist()) for kind,total,n,seed in [('bars',50,15,2026092401),('rain',200,40,2026092402),('calibration',200,20,2026092403)]}
(G/'configs/DISPLAY_SUBSAMPLING.json').write_text(json.dumps({'purpose':'DISPLAY_ONLY_ALL_STATISTICS_USE_FULL_DATA','replicate_ids':{k:sorted(v) for k,v in DISPLAY_IDS.items()}},indent=2))
def shown(z,kind):return z.replicate.isin(DISPLAY_IDS[kind]).to_numpy()
def figax(w,h,left=.15,right=.97,bottom=.22,top=.82):
 fig,ax=plt.subplots(figsize=(w/25.4,h/25.4));fig.subplots_adjust(left=left,right=right,bottom=bottom,top=top);ax.spines[['top','right']].set_visible(False);return fig,ax

def save(fig,name,df,ref,endpoint,adapt):
 if 'displayed_point' in df.columns:adapt += ' Scatter layer uses fixed outcome-independent replicate IDs; all summary statistics and scales use full data.'
 df.to_csv(DATA/f'{name}.tsv',sep='\t',index=False)
 for ext in ['pdf','svg']:fig.savefig(OUT/f'{name}.{ext}',dpi=600)
 fig.savefig(PRE/f'{name}.png',dpi=170)
 MAN.append(dict(panel_id=name,primary_reference_id=ref,reference_file=next((r['source_file_absolute'] for r in CAT if r['id']==ref),None),endpoint=endpoint,adaptations=adapt,physical_size_mm=(fig.get_size_inches()*25.4).tolist(),palette=C,dpi=600,data=str(DATA/f'{name}.tsv'),data_sha256=hashlib.sha256((DATA/f'{name}.tsv').read_bytes()).hexdigest(),review_status='awaiting_visual_review'));plt.close(fig)
def legend(ax,handles,ncol=4,y=1.20,fs=7):
 leg=ax.legend(handles=handles,ncol=ncol,loc='lower center',bbox_to_anchor=(.5,y),frameon=True,fancybox=False,edgecolor='#aaaaaa',fontsize=fs,handlelength=1.1,columnspacing=1.1,borderpad=.35);leg.get_frame().set_linewidth(.5)

# 01: schematic is the protocol/flow concept, not an invented statistical chart.
F=pd.read_csv(G/'results/G4_patient_flow.tsv',sep='\t');fig=plt.figure(figsize=(170/25.4,105/25.4));ax=fig.add_axes([.02,.02,.96,.92]);ax.set(xlim=(0,10),ylim=(0,6));ax.axis('off');fig.text(.04,.975,'Direct-record | patient flow and information boundary',fontsize=10,weight='bold',va='top')
def box(x,y,w,h,text,color):
 ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.045,rounding_size=.06',fc='white',ec=color,lw=1.0));ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=7.5,linespacing=1.45)
def arrow(x,y,xx,yy):ax.annotate('',xy=(xx,yy),xytext=(x,y),arrowprops=dict(arrowstyle='-|>',color='#555555',lw=.8))
for k,c in enumerate(['LUAD','COAD']):
 q=F[F.cancer==c].set_index('stage');y=4.2-1.2*k;col=C[k*2];ax.text(.1,y+.42,c,color=col,weight='bold',fontsize=9)
 labels=[f"Clinical manifest\n{q.loc['cancer_manifest','n_patients']:,} patients",f"Primary\n{q.loc['primary','n_patients']:,} patients",f"IMPACT468 + dedup\n{q.loc['uniform_panel','n_patients']:,} patients",('Train 1,345 · Cal 448\nEvaluation 449' if k==0 else 'Train 857 · Cal 285\nEvaluation 287')]
 for i,t in enumerate(labels):box(1.0+i*2.22,y,2.02,.84,t,col)
 for i in range(3):arrow(3.05+i*2.22,y+.42,3.18+i*2.22,y+.42)
ax.text(.12,5.44,'One sample per patient; keep patients with no variant rows (LUAD 77; COAD 10).',fontsize=8)
box(.25,1.35,2.45,.82,'Train: environment → selection\n→ three-event observed record',C[0]);box(3.3,1.35,2.25,.82,'Calibration: 80 X + S\nSame information for baselines',C[2]);box(6.15,1.35,3.48,.82,'Estimator sees observed inputs only\nFrozen methods → held-out prediction',C[0]);ax.plot([1.47,1.47,7.85],[2.24,2.55,2.55],color='#555555',lw=.8);arrow(7.85,2.55,7.85,2.24);arrow(5.63,1.76,6.08,1.76)
box(.25,.10,4.05,.78,'Private evaluator: full reference → proper scores\nHidden X and selection truth never passed to K1',C[4]);box(4.9,.10,4.73,.78,'Proxy reference negatives, not callable biological negatives\nGENIE18 previously explored; no external replication',C[4]);ax.plot([6.7,6.7,2.3],[1.30,1.10,1.10],color='#555555',lw=.8);arrow(2.3,1.10,2.3,.96)
save(fig,'G4_01',F,'CONCEPT_SCHEMA_EXEMPT','True denominator and train/cal/evaluation information boundary','Concept schematic inherits reference-library typography; not a quantitative template substitute. Counts are audited manifests, not simulated patients.')

# 02: R21 exact mark structure: narrow filled outlined bars, small open raw dots, black cap errors, boxed upper legend, rotated values.
q=D[(D.purpose=='perturbation')&~D.fixed_n&D.method.isin(METHODS)];fig,ax=figax(174,100,left=.10,top=.83,bottom=.22);tab=[];rand=np.random.default_rng(841)
for k,(c,des) in enumerate([(c,d) for c in ['LUAD','COAD'] for d in ['REDUNDANT','HETEROGENEOUS','CROSSVIEW_FAILURE']]):
 for j,m in enumerate(METHODS):
  z=q[(q.cancer==c)&(q.design==des)&(q.method==m)&q.success];v=z.source_tv.to_numpy();x=k+(j-1.5)*.185;mean=v.mean();sd=v.std(ddof=1);ax.bar(x,mean,width=.162,fc=MC[m],ec='#222222',lw=.7,zorder=2);ax.scatter((x+rand.uniform(-.055,.055,len(v)))[shown(z,'bars')],v[shown(z,'bars')],s=4.0,facecolors='white',edgecolors='#222222',linewidths=.3,alpha=.55,zorder=3);ax.errorbar(x,mean,yerr=sd,fmt='none',ecolor='#222222',elinewidth=.75,capsize=1.5,zorder=4);ax.text(x,max(v.max(),mean+sd)+.01,f'{mean:.3f}',ha='center',va='bottom',rotation=90,fontsize=6.5);tab.extend(z.assign(plot_x=x,summary_mean=mean,summary_sd=sd,displayed_point=shown(z,'bars')).to_dict('records'))
ax.set_xticks(range(6),['Redundant','Heterogeneous','Cross-view']*2);ax.set_ylabel('Source-distribution total variation ↓');ax.set_ylim(0,max(z['source_tv'] for z in tab)*1.30);ax.axvline(2.5,c='#bbbbbb',lw=.7,ls=(0,(2,3)));legend(ax,[Patch(fc=MC[m],ec='#222222',label=LABEL[m],lw=.6) for m in METHODS],y=1.03)
ax.text(1,-.20,'LUAD',transform=ax.get_xaxis_transform(),ha='center',weight='bold');ax.text(4,-.20,'COAD',transform=ax.get_xaxis_transform(),ha='center',weight='bold');fig.text(.10,.96,'Direct-record | fixed candidate pool',fontsize=9,weight='bold');fig.text(.10,.035,'15 / 50 perturbations shown per bar; mean ± SD from all 50. Same patient pool.',fontsize=7)
save(fig,'G4_02',pd.DataFrame(tab),'R21','Show absolute source-recovery losses across all three designs, with full distribution and equal candidate cap','All four same-information/ablation methods retained. Numeric mean labels replace invalid seed-as-patient P values. Palette overridden by user. SD is mechanism variation, not an independent-cohort confidence interval.')

# 03/04: R27 vertical rain LEFT, white narrow box, half-density RIGHT, black outline, full framed axis, exact paired endpoint.
for name,metric,label in [('G4_03','hidden_brier','Paired Brier gain: comparator − M'),('G4_04','source_tv','Paired TV gain: comparator − M')]:
 q=PAIR[(PAIR.purpose=='bootstrap')&(PAIR.metric==metric)&(PAIR.reference_design=='HETEROGENEOUS')&PAIR.reference_method.isin(['B1','B3','B4'])];fig,ax=figax(154,108,left=.14,right=.80,top=.83,bottom=.22);tab=[];rand=np.random.default_rng(843)
 for k,(c,m) in enumerate([(c,m) for c in ['LUAD','COAD'] for m in ['B1','B3','B4']]):
  z=q[(q.cancer==c)&(q.reference_method==m)];v=z.gain.to_numpy();col=MC[m];y=np.linspace(v.min(),v.max(),240);den=gaussian_kde(v)(y);den=.33*den/den.max();ax.fill_betweenx(y,k+.03,k+.03+den,fc=col,ec='#333333',lw=.65,alpha=.9,zorder=2);ax.scatter((k-.06-rand.uniform(.035,.27,len(v)))[shown(z,'rain')],v[shown(z,'rain')],s=2.5,c=col,alpha=.65,linewidths=0,zorder=2)
  ax.boxplot(v,positions=[k],widths=.115,whis=(2.5,97.5),showfliers=False,patch_artist=True,boxprops=dict(facecolor='white',edgecolor='#333333',linewidth=.8),medianprops=dict(color='#222222',linewidth=.95),whiskerprops=dict(color='#333333',linewidth=.8),capprops=dict(color='#333333',linewidth=.7),manage_ticks=False,zorder=4);tab.extend(z.assign(plot_group=k,displayed_point=shown(z,'rain')).to_dict('records'))
 for sp in ax.spines.values():sp.set_visible(True);sp.set_color('#bbbbbb' if sp.spine_type in ['top','right'] else '#222222');sp.set_linewidth(.65)
 ax.axhline(0,c='#444444',lw=.8,ls=(0,(4,3)),zorder=0);ax.set_xlim(-.60,5.58);ax.set_xticks(range(6),['Pooled','Hier.','Fusion']*2);ax.set_ylabel(label);ax.text(1,-.16,'LUAD',transform=ax.get_xaxis_transform(),ha='center',weight='bold');ax.text(4,-.16,'COAD',transform=ax.get_xaxis_transform(),ha='center',weight='bold');ax.legend(handles=[Patch(fc=MC[m],ec='#333333',label=LABEL[m]) for m in ['B1','B3','B4']],loc='center left',bbox_to_anchor=(1.02,.55),frameon=False,fontsize=6.5,handlelength=1,handletextpad=.5)
 fig.text(.14,.96,'Direct-record | heterogeneous observations',fontsize=9,weight='bold');fig.text(.14,.905,'Above zero: M has lower loss. 40 / 200 paired resamples shown per group.',fontsize=7);fig.text(.14,.035,'200 patient bootstraps / group; box: median + IQR; whiskers: 2.5–97.5 percentiles.',fontsize=7)
 save(fig,name,pd.DataFrame(tab),'R27','Foreground actual paired gain and its sign, without hiding near-zero overlap or unfavorable tails','Vertical framed cloud-rain replicates half KDE, narrow white box, black median/whiskers and left raw points. Six genuine groups; original biological labels replaced. Percentile whiskers describe joint patient+perturbation variation, not formal confidence.')

# 05: fixed-enrolled sensitivity combined in one new dataset; do not merge old panels.
fig,ax=figax(144,98,left=.13,right=.98,top=.79,bottom=.23);tab=[];cols=[C[1],C[0],C[4]];deslist=['REDUNDANT','HETEROGENEOUS','CROSSVIEW_FAILURE'];rand=np.random.default_rng(845)
for k,(c,fixed) in enumerate([(c,b) for c in ['LUAD','COAD'] for b in [False,True]]):
 for j,des in enumerate(deslist):
  z=D[(D.purpose=='perturbation')&D.fixed_n.eq(fixed)&D.cancer.eq(c)&D.design.eq(des)&D.method.eq('M')&D.success];v=z.source_tv.to_numpy();x=k+(j-1)*.23;mean=v.mean();sd=v.std(ddof=1);ax.bar(x,mean,width=.20,fc=cols[j],ec='#222222',lw=.75);ax.scatter((x+rand.uniform(-.065,.065,len(v)))[shown(z,'bars')],v[shown(z,'bars')],s=5,facecolors='white',edgecolors='#222222',linewidths=.3,alpha=.55,zorder=3);ax.errorbar(x,mean,yerr=sd,fmt='none',ecolor='#222222',elinewidth=.7,capsize=2,zorder=4);ax.text(x,max(v.max(),mean+sd)+.009,f'n={len(v)}',rotation=90,ha='center',va='bottom',fontsize=6.7);tab.extend(z.assign(plot_x=x,summary_mean=mean,summary_sd=sd,displayed_point=shown(z,'bars')).to_dict('records'))
ax.set_xticks(range(4),['Candidate cap','80 enrolled']*2);ax.set_ylabel('M source-distribution total variation ↓');ax.set_ylim(0,max(z['source_tv'] for z in tab)*1.26);legend(ax,[Patch(fc=c,ec='#222222',label=t,lw=.6) for c,t in zip(cols,['Redundant','Heterogeneous','Cross-view'])],ncol=3,y=1.04);ax.text(.5,-.20,'LUAD',ha='center',transform=ax.get_xaxis_transform(),weight='bold');ax.text(2.5,-.20,'COAD',ha='center',transform=ax.get_xaxis_transform(),weight='bold');fig.text(.13,.96,'Direct-record | candidate cap versus matched enrollment',fontsize=9,weight='bold');fig.text(.13,.04,'15 / 50 points shown per bar; mean ± SD from all 50. n = completed runs.',fontsize=7)
save(fig,'G4_05',pd.DataFrame(tab),'R21','Determine whether heterogeneous source recovery persists when enrolled sample size is equal','Same narrow bars/open raw dots/capped errors/upper boxed legend as R21. Fixed 40 patients per environment, no resampling to inflate N. Vertical n labels replace unsupported significance annotations.')

# 06: R23 raw grouped scatter, black diamonds + x/y capped uncertainty, actual theoretical perfect-calibration line.
B=pd.read_csv(G/'results/G4_prediction_calibration_bins.tsv',sep='\t');B=B[(B.purpose=='bootstrap')&B.design.eq('HETEROGENEOUS')&B.method.eq('M')];B=B.groupby(['cancer','replicate','bin'],as_index=False)[['n','prediction_sum','event_sum']].sum();B['predicted']=B.prediction_sum/B.n;B['observed']=B.event_sum/B.n
for c in ['LUAD','COAD']:
 q=B[B.cancer.eq(c)&B.n.gt(0)].copy();q['displayed_point']=shown(q,'calibration');fig,ax=figax(99,99,left=.17,right=.965,bottom=.17,top=.85)
 ax.plot([0,1],[0,1],c='#222222',lw=1.15,ls=(0,(4,4)),label='Perfect calibration',zorder=0)
 for k,col in enumerate(C[:5]):
  z=q[q.bin==k];ax.scatter(z.loc[z.displayed_point,'predicted'],z.loc[z.displayed_point,'observed'],s=7,c=col,alpha=.45,edgecolors='none',label=f'{k/5:.1f}–{(k+1)/5:.1f}')
  if len(z):ax.errorbar(z.predicted.mean(),z.observed.mean(),xerr=z.predicted.std(),yerr=z.observed.std(),fmt='D',markersize=4.1,c='black',mfc='black',mec='black',elinewidth=.75,capsize=2.5,zorder=5)
 ax.scatter([],[],c='black',marker='D',s=18,label='Mean ± bootstrap SD');ax.set(xlim=(-.025,1.025),ylim=(-.035,1.05),xlabel='Predicted hidden-event probability',ylabel='Observed hidden-event fraction');ax.set_xticks([0,.25,.5,.75,1]);ax.set_yticks([0,.25,.5,.75,1]);ax.tick_params(direction='in',top=True,right=True);ax.spines[['top','right']].set_visible(True);ax.minorticks_on();ax.tick_params(which='minor',direction='in',top=True,right=True,length=2);ax.legend(loc='upper left',frameon=False,fontsize=5.7,handlelength=1.9,labelspacing=.28,handletextpad=.45);fig.text(.17,.94,f'Direct-record | {c} hidden-record calibration',fontsize=8.6,weight='bold');fig.text(.17,.025,'20 / 200 resamples shown; summaries use all nonempty bins.',fontsize=6)
 save(fig,'G4_06_'+c,q,'R23','Show probability calibration, raw bootstrap heterogeneity and a real diagonal benchmark','Reproduces colored raw groups, black mean diamonds, both directional capped SD bars, dashed benchmark, four-sided inward ticks and inset legend. Colors encode five predeclared prediction bins. Empty bins omitted only from that resample; denominators retained. No fitted trend masquerades as theory.')

# 07: source-fit uncertainty is a cluster of ten, not 100 natural cohorts.
fig,ax=figax(140,98,left=.12,right=.97,top=.78,bottom=.22);tab=[];rand=np.random.default_rng(847)
for k,(c,b) in enumerate([(c,b) for c in ['LUAD','COAD'] for b in [3,6]]):
 for j,m in enumerate(['M','K0']):
  z=AUX[(AUX.cancer==c)&(AUX.bound==b)&(AUX.method==m)];zz=z.groupby('source_draw').agg(completed=('covers_truth','count'),covered=('covers_truth','sum'),unknown=('sign',lambda v:int(v.eq('UNKNOWN').sum())),n=('method','size'));v=100*(1-zz.covered/zz.completed);x=k+(j-.5)*.30;col=C[0] if m=='M' else C[2];ax.bar(x,v.mean(),width=.265,fc=col,ec='#222222',lw=.8);ax.scatter(x+rand.uniform(-.075,.075,len(v)),v,s=10,facecolors='white',edgecolors='#222222',linewidths=.5,zorder=3);ax.errorbar(x,v.mean(),yerr=v.std(),fmt='none',ecolor='#222222',capsize=2,elinewidth=.8);tab.extend(zz.assign(cancer=c,bound=b,method=m,coverage_percent=100-v,noncoverage_percent=v).reset_index().to_dict('records'))
ax.axhline(5,c='#444444',lw=.85,ls=(0,(4,3)));ax.set(ylim=(0,30),ylabel='Uncovered / completed local profiles (%)');ax.set_xticks(range(4),['Source ±3','Source ±6']*2);legend(ax,[Patch(fc=C[0],ec='#222222',label='M: unknown selection'),Patch(fc=C[2],ec='#222222',label='K0: known selection')],ncol=2,y=1.07);ax.text(.5,-.20,'LUAD',ha='center',transform=ax.get_xaxis_transform(),weight='bold');ax.text(2.5,-.20,'COAD',ha='center',transform=ax.get_xaxis_transform(),weight='bold');fig.text(.12,.96,'Model-anchored | source-bootstrap profile noncoverage',fontsize=9,weight='bold');fig.text(.12,.04,'10 source fits × 10 replicates; dots = source-fit noncoverage; bars = mean ± SD; dashed = 5%.',fontsize=6.7)
save(fig,'G4_07',pd.DataFrame(tab),'R21','Conditional model coverage across source bootstrap and expanded source bounds, separate from direct records','Source-fit points supply real replication for grouped bars. SD is across ten source fits. Unbounded profiles count in coverage and are disclosed in companion tables; dashed nominal level is a reference, not a validation declaration.')

# 08: R28 paired composition columns and aligned side dots; retain unknown and unbounded intervals.
fig=plt.figure(figsize=(133/25.4,112/25.4));axes=[fig.add_axes([.245,.25,.19,.55]),fig.add_axes([.465,.25,.19,.55]),fig.add_axes([.685,.25,.13,.55])];tab=[];rr=pd.read_csv(G/'results/G4_reference_agreement.tsv',sep='\t');q=D[(D.purpose=='perturbation')&~D.fixed_n&D.method.isin(['M','M_NO_CAL'])]
for i,(c,des,m) in enumerate([(c,d,m) for c in ['LUAD','COAD'] for d in ['REDUNDANT','HETEROGENEOUS','CROSSVIEW_FAILURE'] for m in ['M','M_NO_CAL']]):
 z=q[(q.cancer==c)&(q.design==des)&(q.method==m)];n=len(z);decision=[int(z.sign_call.eq('POSITIVE').sum()),int(z.sign_call.eq('NEGATIVE').sum()),int(z.sign_call.eq('UNKNOWN').sum()),int(n-z.interval_status.eq('COMPLETE').sum())];finite=np.isfinite(z.ci_lower)&np.isfinite(z.ci_upper)&z.interval_status.eq('COMPLETE');extent=[int(finite.sum()),int(z.interval_status.eq('COMPLETE').sum()-finite.sum()),int(n-z.interval_status.eq('COMPLETE').sum())]
 for ax,counts,colors in [(axes[0],decision,[C[2],C[0],C[3],C[4]]),(axes[1],extent,[C[1],C[5],C[4]])]:
  left=0
  for v,col in zip(counts,colors):ax.barh(i,v/n,left=left,height=.88,color=col,edgecolor='white',linewidth=.2);left+=v/n
 ref=rr[(rr.purpose=='perturbation')&~rr.fixed_n&rr.cancer.eq(c)&rr.design.eq(des)&rr.method.eq(m)];overlap=ref.profile_overlap.mean();axes[2].plot(overlap,i,'o',ms=2.6,color='#222222');tab.append(dict(cancer=c,design=des,method=m,n=n,positive=decision[0],negative=decision[1],unknown=decision[2],failed=decision[3],finite=extent[0],unbounded=extent[1],profile_overlap=overlap,overlap_denominator=len(ref)))
labels=[f"{r['cancer']} · { {'REDUNDANT':'R','HETEROGENEOUS':'H','CROSSVIEW_FAILURE':'X'}[r['design']] } · {'M' if r['method']=='M' else 'no cal'}" for r in tab]
for j,ax in enumerate(axes):
 ax.set(xlim=(0,1.02),ylim=(11.6,-.6));ax.set_xticks([0,.25,.5,.75,1]);ax.tick_params(axis='x',labelrotation=90,labelsize=6);ax.set_yticks(range(12),labels if j==0 else []);ax.tick_params(axis='y',length=2,labelsize=6.4);ax.grid(axis='x',color='#dddddd',lw=.35,zorder=0);ax.set_axisbelow(True)
 for sp in ax.spines.values():sp.set_linewidth(.55)
 ax.axhline(5.5,c='#555555',lw=.75)
axes[0].set_title('Sign call',fontsize=8);axes[1].set_title('Profile extent',fontsize=8);axes[2].set_title('Reference\noverlap',fontsize=7.5);axes[0].set_xlabel('Proportion',fontsize=7);axes[1].set_xlabel('Proportion',fontsize=7);axes[2].set_xlabel('Fraction',fontsize=7)
fig.legend(handles=[Patch(fc=c,label=t) for c,t in zip([C[2],C[0],C[3],C[4]],['Positive','Negative','Unknown','Failed'])],loc='upper left',bbox_to_anchor=(.835,.77),frameon=False,fontsize=6.1,title='Sign',title_fontsize=6.7,handlelength=.9,handletextpad=.4)
fig.legend(handles=[Patch(fc=c,label=t) for c,t in zip([C[1],C[5],C[4]],['Finite','Unbounded','Failed'])],loc='upper left',bbox_to_anchor=(.835,.48),frameon=False,fontsize=6.1,title='Extent',title_fontsize=6.7,handlelength=.9,handletextpad=.4)
fig.text(.05,.955,'Direct-record | local profile uncertainty and reference agreement',fontsize=8.6,weight='bold');fig.text(.05,.06,'R / H / X: redundant / heterogeneous / cross-view; 50 perturbations per row.\nOverlap with a broad full-record profile is not evidence of natural parameter recovery.',fontsize=6.8,linespacing=1.5)
save(fig,'G4_08',pd.DataFrame(tab),'R28','Expose uncertain and unbounded answers alongside reference overlap rather than presenting point agreement as recovery','Replicates two narrow tightly aligned stacked columns, third dotted metric column, rotated quarter ticks, row labels, light grid and two compact right legends. 12 actual rows; no fake categories or precision. Full-record reference is not biological truth.')
(G/'configs/FIGURE_MANIFEST.json').write_text(json.dumps(MAN,ensure_ascii=False,indent=2));print('Wrote',len(MAN),'independent PDF/SVG pairs; visual review pending')
