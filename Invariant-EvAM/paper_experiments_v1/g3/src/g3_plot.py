from pathlib import Path
import sys,json,hashlib
import numpy as np,pandas as pd
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import gaussian_kde
from g3_design import G,R
from g3_summarize import wilson
sys.path.insert(0,str(R/'src'))
from invariant_evam.figure_style import apply_publication_style
apply_publication_style({'svg.fonttype':'none','pdf.fonttype':42,'font.size':8,'axes.labelsize':8,'xtick.labelsize':7,'ytick.labelsize':7})
COL=['#0072B2','#56B4E9','#009E73','#E69F00','#D55E00','#CC79A7']
OUT=G/'FINAL_FIGURES_01_07';PRE=Path('/tmp/invariant_evam_g3_previews');DATA=G/'figure_data'
for p in [OUT,PRE,DATA]:p.mkdir(exist_ok=True)
MAN=[]
S=pd.read_csv(G/'results/G3_summary.tsv',sep='\t');D=pd.read_csv(G/'results/G3_replicate_results.tsv.gz',sep='\t',low_memory=False)
assert json.loads((G/'runs/SUMMARY_STATUS.json').read_text())['run_status']=='COMPLETE','Do not render incomplete core results as final figures'
def canvas(w=165,h=108):
 fig,ax=plt.subplots(figsize=(w/25.4,h/25.4));fig.subplots_adjust(left=.14,right=.975,bottom=.24,top=.79);ax.spines[['top','right']].set_visible(False);return fig,ax

def save(fig,number,d,ref,endpoint,adapt):
 name=f'G3_{number:02d}'
 for ext in ['pdf','svg']:fig.savefig(OUT/f'{name}.{ext}',dpi=600)
 fig.savefig(PRE/f'{name}.png',dpi=150);d.to_csv(DATA/f'{name}.tsv',sep='\t',index=False)
 MAN.append(dict(panel_id=name,primary_reference_id=ref,endpoint=endpoint,adaptations=adapt,palette=COL,dpi=600,size_mm=(fig.get_size_inches()*25.4).tolist(),source_data=str(DATA/f'{name}.tsv'),data_sha256=hashlib.sha256((DATA/f'{name}.tsv').read_bytes()).hexdigest(),outputs=[str(OUT/f'{name}.{e}') for e in ['pdf','svg']],review_status='awaiting_visual_review'));plt.close(fig)

A_METHODS=['SHARED_K1','BOUNDED_BIOLOGY','FLEXIBLE_SELECTION','ORACLE_STRATIFIED'];A_LABELS=['Shared','Bounded biology','Flexible selection','Known selection + strata'];A_C=[COL[0],COL[2],COL[1],COL[5]]
for number,metric in [(1,'noncoverage'),(2,'correct')]:
 fig,ax=canvas(180,112);tab=[]
 for j,(m,label,c) in enumerate(zip(A_METHODS,A_LABELS,A_C)):
  f=S[(S.cell_id<6)&(S.method_id==m)].sort_values('cell_id');x=f.cell_id.to_numpy()+(j-1.5)*.19
  if metric=='noncoverage':y=1-f.coverage.to_numpy();lo=1-f.coverage_high.to_numpy();hi=1-f.coverage_low.to_numpy()
  else:y=f.correct_call_fraction.to_numpy();lo=f.correct_low.to_numpy();hi=f.correct_high.to_numpy()
  ax.bar(x,100*y,width=.175,fc=c,ec='#333333',lw=.6,label=label,zorder=2);ax.errorbar(x,100*y,yerr=np.maximum(0,[100*(y-lo),100*(hi-y)]),fmt='none',ecolor='#222222',elinewidth=.6,capsize=1.5,zorder=3)
  for (_,r),v,l,u in zip(f.iterrows(),y,lo,hi):tab.append({**r.to_dict(),'plotted_rate':v,'plotted_low':l,'plotted_high':u})
 ax.set_xticks(range(6),['Same bio\nSame obs','Same bio\nDifferent obs','Different bio\nSame obs','Different bio\nDifferent obs','D4 local\nalignment','D4 outside\nnuisance'])
 ax.set_ylabel('Uncovered / completed intervals (%)' if metric=='noncoverage' else 'Correct sign calls / all replicates (%)')
 top=max(z['plotted_high'] for z in tab)*100;ax.set_ylim(0,min(105,max(15,top*1.12)))
 if metric=='noncoverage':ax.axhline(5,color='#666666',lw=.8,ls=(0,(3,3)),zorder=0)
 ax.legend(ncol=2,loc='lower center',bbox_to_anchor=(.5,1.04),frameon=False,fontsize=7)
 fig.text(.14,.04,'200 independent replicates per condition; Wilson 95% Monte Carlo intervals.\nUnbounded intervals remain in coverage; useful answers and numerical failures are reported separately.',fontsize=7,linespacing=1.5)
 save(fig,number,pd.DataFrame(tab),'R21','Noncoverage versus nominal 5%' if metric=='noncoverage' else 'Correct sign-call yield across crossed mechanisms','Narrow grouped bars, dark outlines, black capped errors and upper horizontal legend. All six frozen conditions and four specifications; no fabricated binary replicate clouds or P values.')
# Budget: plot paired change, because absolute rates cluster and obscure small differences.
P=pd.read_csv(G/'results/G3_paired_comparisons.tsv',sep='\t');f=P[(P.cell_id.isin([6,7]))&(P.metric=='correct_call')&(P.reference=='ORDINARY_ONLY')];order=['RANDOM_X_0.1','CALIBRATION_XS_0.1','RANDOM_X_0.25','CALIBRATION_XS_0.25','KNOWN_OPERATOR_ORACLE']
fig,ax=canvas(160,110)
for j,(cell,c,label) in enumerate([(6,COL[1],'D3'),(7,COL[2],'D4')]):
 v=f[f.cell_id==cell].set_index('method').loc[order].reset_index();x=np.arange(5)+(j-.5)*.28;y=100*v.paired_difference.to_numpy();lo=100*v.normal95_low.to_numpy();hi=100*v.normal95_high.to_numpy()
 ax.bar(x,y,width=.25,fc=c,ec='#333333',lw=.65,label=label);ax.errorbar(x,y,yerr=np.maximum(0,[y-lo,hi-y]),fmt='none',ecolor='#222222',elinewidth=.7,capsize=2)
ax.axhline(0,color='#555555',lw=.8);ax.set_xticks(range(5),['Random X\n10% budget','X + S\n10% budget','Random X\n25% budget','X + S\n25% budget','Known-operator\noracle']);ax.set_ylabel('Correct-call change vs ordinary-only (pp)');ax.legend(ncol=2,loc='lower left',bbox_to_anchor=(0,1.04),frameon=False)
fig.text(.14,.035,'500 paired replicates per model; fixed budget = 1,000 proxy units.\nErrors: paired 95% Monte Carlo intervals. Oracle includes extra mechanism knowledge.',fontsize=7,linespacing=1.5)
save(fig,3,f,'R21','Budget-matched change in correct-call yield','Grouped narrow bars around true zero, black errors and top legend. Plot actual paired changes instead of visually indistinguishable absolute rates; preserve negative changes and known-operator comparison.')
# Common omitted bias: unconditional wrong-answer probability, not undefined no-call risk.
order=['OMITTED_U','MORE_ENVS_SAME_BUDGET','FLEXIBLE_U','CALIBRATED_FLEXIBLE_U'];labels=['Omitted U','More environments','Flexible U','Calibrated + flexible U'];fig,ax=canvas(168,111);tab=[]
for j,(m,label,c) in enumerate(zip(order,labels,[COL[0],COL[1],COL[5],COL[2]])):
 f=S[(S.cell_id.between(8,11))&(S.method_id==m)].sort_values('cell_id');x=np.arange(4)+(j-1.5)*.19;y=f.wrong_calls.to_numpy()/f.n_tasks.to_numpy();ci=np.array([wilson(int(k),int(n)) for k,n in zip(f.wrong_calls,f.n_tasks)])
 ax.bar(x,100*y,width=.175,fc=c,ec='#333333',lw=.6,label=label);ax.errorbar(x,100*y,yerr=np.maximum(0,[100*(y-ci[:,0]),100*(ci[:,1]-y)]),fmt='none',ecolor='#222222',elinewidth=.6,capsize=1.5)
 for (_,r),v,(lo,hi) in zip(f.iterrows(),y,ci):tab.append({**r.to_dict(),'unconditional_wrong_rate':v,'wrong_low':lo,'wrong_high':hi})
ax.set_xticks(range(4),['U = 0','U = 0.5','U = 1.0','Exact observational\nequivalence']);ax.set_ylabel('Wrong sign calls / all replicates (%)');ax.set_ylim(0,105);ax.legend(ncol=2,loc='lower center',bbox_to_anchor=(.5,1.04),fontsize=7,frameon=False)
fig.text(.14,.035,'500 replicates per condition; Wilson 95% Monte Carlo intervals.\nAll targets = −0.4. A zero wrong-call rate may reflect abstention; see source call counts.',fontsize=7,linespacing=1.5)
save(fig,4,pd.DataFrame(tab),'R21','Wrong stable signs under common omitted bias','All four prespecified common-bias conditions; same-budget environment control retained. Flexible and sensitivity core sets are identical, shown once rather than pretending they are independent methods.')
# Forward sensitivity: actual coefficient scale; errors derived from all 20 fixed replicate IDs.
Z=pd.read_csv(G/'results/G3_sensitivity_regions.tsv',sep='\t');fig,ax=canvas(160,115);tab=[]
grouped_curves={}
for cell,ls,marker,label in [(3,'-','o','Biology + obs'),(10,'--','s','Common U')]:
 for eps,c in zip([0,.2,.5],COL[:3]):
  points=[]
  for eta in [0,.5,1.]:
   f=Z[(Z.cell==cell)&(Z.epsilon_log_rate==eps)&(Z.eta_selection_coefficient==eta)];n=len(f);assert n==20;wrong=int(f.sign.eq('POSITIVE').sum());called=int(f.sign.isin(['POSITIVE','NEGATIVE']).sum());lo,hi=wilson(wrong,n);points.append((eta,wrong/n,lo,hi));tab.append(dict(cell=cell,epsilon=eps,eta=eta,n=n,wrong=wrong,called=called,coverage=f.covers_truth.mean(),n_completed=int(f.status.eq('COMPLETE').sum()),wrong_rate=wrong/n,wrong_low=lo,wrong_high=hi,inside_allowed_set=bool(f.true_mechanism_in_allowed_set.iloc[0])))
  key=(cell,tuple(points))
  if key not in grouped_curves:grouped_curves[key]=dict(points=points,color=c,ls=ls,marker=marker,label=label,epsilon=[])
  grouped_curves[key]['epsilon'].append(eps)
for curve in grouped_curves.values():
 p=np.array(curve['points']);c=curve['color'];label=curve['label']+'; ε='+','.join(f'{v:g}' for v in curve['epsilon'])
 ax.plot(p[:,0],100*p[:,1],color=c,ls=curve['ls'],lw=1.3,label=label);ax.errorbar(p[:,0],100*p[:,1],yerr=np.maximum(0,[100*(p[:,1]-p[:,2]),100*(p[:,3]-p[:,1])]),fmt=curve['marker'],mfc=c,mec='#333333',ms=4,ecolor='#333333',elinewidth=.5,capsize=1.5)

ax.set(xlim=(-.03,1.03),ylim=(-2.5,105),yticks=[0,20,40,60,80,100],xticks=[0,.5,1],xlabel='Allowed selection-feature coefficient |u| ≤ η',ylabel='Wrong sign calls / all replicates (%)');ax.legend(ncol=2,loc='lower center',bbox_to_anchor=(.5,1.04),fontsize=6.8,frameon=False)
fig.text(.14,.035,'20 prespecified replicates per curve: diagnostic precision, not full calibration.\nε: log-rate deviation; η: selection coefficient. Uncalled/failed counts retained in source data.',fontsize=7,linespacing=1.5)
save(fig,5,pd.DataFrame(tab),'R04','Wrong-answer reduction across explicit forward sensitivity sets','Filled dark-edge markers, black error bars, colored trajectories and upper legend. Actual three coefficient values; exactly coincident trajectories are drawn once with all epsilon labels, without pooling replicate counts. No invented phases/AUC. Distinguish scenarios by marker and line style; do not add ε and η across units.')

def raincloud(number,groups,labels,colors,width):
 fig,ax=canvas(width,120);fig.subplots_adjust(top=.93,bottom=.25);tab=[];rng=np.random.default_rng(713)
 for i,((cell,method),label,c) in enumerate(zip(groups,labels,colors)):
  f=D[(D.cell_id==cell)&(D.method_id==method)].copy();valid=f.fit_success.eq(True)&f.population_prediction_KL.notna();v=f.loc[valid,'population_prediction_KL'].to_numpy();v=np.maximum(v,1e-16);z=np.log10(v)
  jitter=rng.uniform(-.27,-.075,len(z));ax.scatter(i+jitter,z,s=3,c=c,alpha=.38,lw=0,zorder=2)
  if len(z)>1 and np.std(z)>1e-8:
   grid=np.linspace(z.min(),z.max(),180);density=gaussian_kde(z)(grid);density=.29*density/density.max();ax.fill_betweenx(grid,i+.06,i+.06+density,fc=c,alpha=.75,ec='#444444',lw=.65)
  ax.boxplot(z,positions=[i+.015],widths=.075,patch_artist=True,showfliers=False,whis=1.5,boxprops=dict(facecolor='white',edgecolor='#333333',lw=.65),medianprops=dict(color='#222222',lw=1),whiskerprops=dict(color='#333333',lw=.65),capprops=dict(color='#333333',lw=.65))
  for _,r in f.iterrows():tab.append({**r.to_dict(),'display_group':label,'display_log10_KL':np.log10(max(float(r.population_prediction_KL),1e-16)) if pd.notna(r.population_prediction_KL) and r.fit_success else None})
  ax.text(i,1.025,f'n={int(valid.sum())}/{len(f)}',transform=ax.get_xaxis_transform(),ha='center',fontsize=6.8)
 ax.set_xticks(range(len(groups)),labels);ax.set_xlim(-.5,len(groups)-.45);ax.set_ylabel('log10 KL(true observable distribution || fitted)');ax.tick_params(axis='x',length=0,pad=4)
 for sp in ax.spines.values():sp.set_visible(True);sp.set_linewidth(.6)
 fig.text(.14,.035,'All successful-fit replicates: raw points, white median/IQR box, right half-density.\nFailed fits excluded only from distribution; denominators shown. Display floor KL = 1e−16.',fontsize=7,linespacing=1.5)
 save(fig,number,pd.DataFrame(tab),'R27','Distribution of observable predictive discrepancy','Actual per-replicate points on left, slim white box in middle, colored half-violin on right, full fine frame. Density estimated on log10 KL with Scott bandwidth; no patient subsampling; jitter is display-only.')
raincloud(6,[(12,'MISSPECIFIED_SHARED'),(13,'MISSPECIFIED_SHARED'),(14,'MISSPECIFIED_SHARED'),(15,'MISSPECIFIED_SHARED')],['Gamma\ntime','Higher-order\ninteraction','Unlabelled\nmixture','Measurement\nerror'],[COL[0],COL[2],COL[5],COL[3]],155)
# Target coverage, rather than nearly identical predictive KL, is the endpoint of known-assay correction.
fig,ax=canvas(145,112);tab=[]
comparisons=[['MISSPECIFIED_SHARED','KNOWN_TIME_FAMILY','ESTIMATED_TIME_FAMILY'],['MISSPECIFIED_SHARED','KNOWN_MEASUREMENT','BIOLOGY_ENVELOPE']]
for j,(label,c) in enumerate(zip(['Wrong forward model','Known forward model','Additional nuisance'],[COL[0],COL[2],COL[5]])):
 for group,cell in enumerate([12,15]):
  r=S[(S.cell_id==cell)&(S.method_id==comparisons[group][j])].iloc[0];x=group+(j-1)*.23;y=100*(1-r.coverage);lo=100*(1-r.coverage_high);hi=100*(1-r.coverage_low)
  ax.bar(x,y,width=.21,fc=c,ec='#333333',lw=.6,label=label if group==0 else None);ax.errorbar(x,y,yerr=np.maximum(0,[[y-lo],[hi-y]]),fmt='none',ecolor='#222222',elinewidth=.7,capsize=2)
  ax.text(x,hi+1.1,f'{y:.1f}',ha='center',va='bottom',fontsize=7);tab.append({**r.to_dict(),'noncoverage':y/100,'noncoverage_low':lo/100,'noncoverage_high':hi/100})
ax.axhline(5,color='#666666',lw=.8,ls=(0,(3,3)));ax.set(ylim=(0,82),xticks=[0,1],xticklabels=['Gamma time','Measurement error'],ylabel='Uncovered / completed intervals (%)');ax.legend(ncol=1,loc='lower left',bbox_to_anchor=(0,1.02),fontsize=7,frameon=False)
fig.text(.14,.035,'200 replicates per condition; Wilson 95% Monte Carlo intervals.\nAdditional nuisance: estimated time CV / bounded biological deviation, respectively.',fontsize=7,linespacing=1.5)
save(fig,7,pd.DataFrame(tab),'R21','Target interval noncoverage under specified forward-model corrections','Grouped dark-edge bars, black capped intervals and direct endpoint values. Data audit found nearly identical wrong/known assay predictive KL, so use the scientifically relevant target coverage endpoint instead; all three frozen specifications retained per condition.')
(G/'configs/FIGURE_MANIFEST.json').write_text(json.dumps(MAN,ensure_ascii=False,indent=2));print('Generated',len(MAN),'independent PDF/SVG figures; awaiting visual review')
