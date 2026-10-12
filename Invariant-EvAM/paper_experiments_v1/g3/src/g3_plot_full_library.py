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
apply_publication_style({'svg.fonttype':'none','pdf.fonttype':42,'font.size':8,'axes.labelsize':8,'xtick.labelsize':7,'ytick.labelsize':7,'axes.linewidth':1.0,'xtick.major.width':1.0,'ytick.major.width':1.0})
COL=['#0072B2','#56B4E9','#009E73','#E69F00','#D55E00','#CC79A7']
OUT=G/'FINAL_FIGURES_01_07';PRE=Path('/tmp/invariant_evam_g3_previews');DATA=G/'figure_data'
for p in [OUT,PRE,DATA]:p.mkdir(exist_ok=True)
MAN=[]
S=pd.read_csv(G/'results/G3_summary.tsv',sep='\t');D=pd.read_csv(G/'results/G3_replicate_results.tsv.gz',sep='\t',low_memory=False)
assert json.loads((G/'runs/SUMMARY_STATUS.json').read_text())['run_status']=='COMPLETE','Do not render incomplete core results as final figures'
def canvas(w=165,h=108):
 fig,ax=plt.subplots(figsize=(w/25.4,h/25.4));fig.subplots_adjust(left=.14,right=.975,bottom=.20,top=.83);ax.spines[['top','right']].set_visible(False);return fig,ax

def save(fig,number,d,ref,endpoint,adapt):
 name=f'G3_{number:02d}'
 for artist in list(fig.texts):artist.remove()
 if number in [1,7]:annotate_pairs(ax,number)
 if number==3:ax.tick_params(axis='x',labelsize=7)
 for ext in ['pdf','svg']:fig.savefig(OUT/f'{name}.{ext}',dpi=600)
 fig.savefig(PRE/f'{name}.png',dpi=150);d.to_csv(DATA/f'{name}.tsv',sep='\t',index=False)
 MAN.append(dict(panel_id=name,primary_reference_id=ref,endpoint=endpoint,adaptations=adapt,palette=COL,dpi=600,size_mm=(fig.get_size_inches()*25.4).tolist(),source_data=str(DATA/f'{name}.tsv'),data_sha256=hashlib.sha256((DATA/f'{name}.tsv').read_bytes()).hexdigest(),outputs=[str(OUT/f'{name}.{e}') for e in ['pdf','svg']],review_status='awaiting_visual_review'));plt.close(fig)

# Exploratory paired annotations: defined contrasts, all cells, one Holm family.
from scipy.stats import binomtest
COMPS=[]
for num,metric,cells,m0,m1 in [(1,'covers_truth',range(6),'SHARED_K1','BOUNDED_BIOLOGY'),(2,'correct_call',range(6),'SHARED_K1','BOUNDED_BIOLOGY'),(4,'wrong_call',range(8,12),'OMITTED_U','CALIBRATED_FLEXIBLE_U'),(7,'covers_truth',[12],'MISSPECIFIED_SHARED','KNOWN_TIME_FAMILY'),(7,'covers_truth',[15],'MISSPECIFIED_SHARED','KNOWN_MEASUREMENT')]:
 for cell in cells:
  l=D[(D.cell_id==cell)&(D.method_id==m0)].set_index('replicate_id');r=D[(D.cell_id==cell)&(D.method_id==m1)].set_index('replicate_id');ids=l.index.intersection(r.index)
  if metric=='covers_truth':ids=[i for i in ids if l.loc[i,'computational_status']=='COMPLETE' and r.loc[i,'computational_status']=='COMPLETE' and pd.notna(l.loc[i,metric]) and pd.notna(r.loc[i,metric])]
  av=l.loc[ids,metric].eq(True).to_numpy();bv=r.loc[ids,metric].eq(True).to_numpy();ab=int((av&~bv).sum());ba=int((~av&bv).sum());pv=binomtest(ab,ab+ba,.5).pvalue if ab+ba else 1.
  COMPS.append(dict(panel=num,cell=cell,metric=metric,method_a=m0,method_b=m1,n_pairs=len(ids),a_only=ab,b_only=ba,p_raw=pv))
ix=np.argsort([r['p_raw'] for r in COMPS]);prev=0
for rank,i in enumerate(ix):prev=max(prev,min(1.,COMPS[i]['p_raw']*(len(COMPS)-rank)));COMPS[i]['p_holm']=prev
pd.DataFrame(COMPS).to_csv(DATA/'G3_exploratory_paired_comparisons.tsv',sep='\t',index=False)
def annotate_pairs(ax,num):
 rows=[r for r in COMPS if r['panel']==num];bottom,top=ax.get_ylim();step=(top-bottom)*.04
 for k,r in enumerate(rows):
  if num in [1,2]:x1=k-1.5*.19;x2=k-.5*.19
  elif num==4:x1=k-1.5*.19;x2=k+1.5*.19
  else:x1=k-.23;x2=k
  height=top+step*(1 if num!=7 else 1)
  ax.plot([x1,x1,x2,x2],[height,height+step*.35,height+step*.35,height],lw=.85,c='#222222',clip_on=False)
  q=r['p_holm'];label='ns' if q>=.05 else ('***' if q<.001 else '**' if q<.01 else '*')
  ax.text((x1+x2)/2,height+step*.5,label,ha='center',va='bottom',fontsize=8)
 ax.set_ylim(bottom,top+step*3.2)

A_METHODS=['SHARED_K1','BOUNDED_BIOLOGY','FLEXIBLE_SELECTION','ORACLE_STRATIFIED'];A_LABELS=['Shared','Bounded biology','Flexible selection','Known selection + strata'];A_C=[COL[0],COL[2],COL[1],COL[5]]
for number,metric in [(1,'noncoverage')]:
 fig,ax=canvas(180,105);tab=[]
 for j,(m,label,c) in enumerate(zip(A_METHODS,A_LABELS,A_C)):
  f=S[(S.cell_id<6)&(S.method_id==m)].sort_values('cell_id');x=f.cell_id.to_numpy()+(j-1.5)*.19
  if metric=='noncoverage':y=1-f.coverage.to_numpy();lo=1-f.coverage_high.to_numpy();hi=1-f.coverage_low.to_numpy()
  else:y=f.correct_call_fraction.to_numpy();lo=f.correct_low.to_numpy();hi=f.correct_high.to_numpy()
  ax.bar(x,100*y,width=.175,fc=c,ec='#222222',lw=.85,label=label,zorder=2);ax.errorbar(x,100*y,yerr=np.maximum(0,[100*(y-lo),100*(hi-y)]),fmt='none',ecolor='#222222',elinewidth=.6,capsize=1.5,zorder=3)
  for (_,r),v,l,u in zip(f.iterrows(),y,lo,hi):tab.append({**r.to_dict(),'plotted_rate':v,'plotted_low':l,'plotted_high':u})
 ax.set_xticks(range(6),['Same bio\nSame obs','Same bio\nDifferent obs','Different bio\nSame obs','Different bio\nDifferent obs','D4 local\nalignment','D4 outside\nnuisance'])
 ax.set_ylabel('Uncovered / completed intervals (%)' if metric=='noncoverage' else 'Correct sign calls / all replicates (%)')
 top=max(z['plotted_high'] for z in tab)*100;ax.set_ylim(0,min(105,max(15,top*1.12)))
 if metric=='noncoverage':ax.axhline(5,color='#666666',lw=.8,ls=(0,(3,3)),zorder=0)
 ax.legend(ncol=4,loc='lower center',bbox_to_anchor=(.5,1.04),frameon=False,fontsize=6.8)
 fig.text(.14,.04,'200 independent replicates per condition; Wilson 95% Monte Carlo intervals.\nUnbounded intervals remain in coverage; useful answers and numerical failures are reported separately.',fontsize=7,linespacing=1.5)
 save(fig,number,pd.DataFrame(tab),'R21','Noncoverage versus nominal 5%' if metric=='noncoverage' else 'Correct sign-call yield across crossed mechanisms','Narrow grouped bars, dark outlines, black capped errors and upper horizontal legend. All six frozen conditions and four specifications; no fabricated binary replicate clouds or P values.')
# R20: rectangular condition x method matrix, two true design blocks, top color key.
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch
fig=plt.figure(figsize=(112/25.4,116/25.4));ax=fig.add_axes([.35,.25,.60,.55]);cax=fig.add_axes([.35,.87,.60,.035]);tab=[]
ys=[0,1,2,3,4.5,5.5];cmap=LinearSegmentedColormap.from_list('yield',['white',COL[1],COL[0]])
from matplotlib.colors import Normalize
norm=Normalize(0,50)
for cell,y in enumerate(ys):
 for j,m in enumerate(A_METHODS):
  r=S[(S.cell_id==cell)&(S.method_id==m)].iloc[0];v=100*r.correct_call_fraction
  ax.add_patch(plt.Rectangle((j-.5,y-.5),1,1,fc=cmap(norm(v)),ec='#444444',lw=.55));ax.text(j,y,f'{v:.1f}',ha='center',va='center',fontsize=8,color='white' if v>32 else '#222222');tab.append({**r.to_dict(),'plotted_correct_percent':v,'display_row':y})
ax.set(xlim=(-.5,3.5),ylim=(6.1,-.6));ax.set_yticks(ys,['Same bio / same obs','Same bio / diff. obs','Diff. bio / same obs','Diff. bio / diff. obs','Local alignment','Outside nuisance']);ax.set_xticks(range(4),['Shared','Bounded biology','Flexible selection','Known + strata'],rotation=45,ha='right');ax.tick_params(length=0)
for sp in ax.spines.values():sp.set_visible(False)
cb=fig.colorbar(plt.cm.ScalarMappable(norm=norm,cmap=cmap),cax=cax,orientation='horizontal',ticks=[0,10,20,30,40,50]);cb.solids.set_rasterized(False);cax.set_title('Correct sign calls (%)',fontsize=8,pad=7)
ax.text(-2.35,1.5,'D3',rotation=90,ha='center',va='center',fontsize=8);ax.text(-2.35,5,'D4',rotation=90,ha='center',va='center',fontsize=8)
save(fig,2,pd.DataFrame(tab),'R20','Correct-call yield across the full condition-method matrix','Two design blocks, thin rectangular grid, top horizontal color key, rotated column labels; numeric cell values replace ungrounded significance symbols. Linear 0–50% color scale; confidence intervals retained in source data.')
# R25: each strategy linked to its own ordinary-only zero, with acquisition composition.
P=pd.read_csv(G/'results/G3_paired_comparisons.tsv',sep='\t');f=P[(P.cell_id.isin([6,7]))&(P.metric=='correct_call')&(P.reference=='ORDINARY_ONLY')].copy();order=['RANDOM_X_0.1','CALIBRATION_XS_0.1','RANDOM_X_0.25','CALIBRATION_XS_0.25','KNOWN_OPERATOR_ORACLE'];labels=['Random X 10%','X + S 10%','Random X 25%','X + S 25%','Known operator']
fig=plt.figure(figsize=(158/25.4,125/25.4));side=fig.add_axes([.27,.23,.065,.63]);ax=fig.add_axes([.38,.23,.59,.63]);tab=[]
for block,(cell,c,model) in enumerate([(6,COL[0],'D3'),(7,COL[2],'D4')]):
 for k,method in enumerate(order):
  r=f[(f.cell_id==cell)&(f.method==method)].iloc[0];y=block*6+k;delta=100*r.paired_difference;lo=100*r.normal95_low;hi=100*r.normal95_high
  ax.axhline(y,color='#eeeeee',lw=.6,zorder=0);ax.plot([0,delta],[y,y],c='#888888',lw=1,zorder=2);ax.errorbar(delta,y,xerr=[[delta-lo],[hi-delta]],fmt='none',ecolor=c,alpha=.5,elinewidth=2,capsize=2,zorder=1);ax.scatter([0],[y],s=22,fc='white',ec='#555555',lw=.8,zorder=3);ax.scatter([delta],[y],s=22,fc=c,ec='white',lw=.4,zorder=4)
  budget=[1.,0.,0.] if k==4 else [1-([.1,.1,.25,.25][k]),[.1,0,.25,0][k],[0,.1,0,.25][k]];left=0
  for v,col in zip(budget,[COL[1],COL[0],COL[2]]):side.barh(y,v,left=left,height=.58,color=col,edgecolor='white',lw=.2);left+=v
  tab.append({**r.to_dict(),'model_label':model,'display_row':y,'ordinary_fraction':budget[0],'random_X_fraction':budget[1],'random_XS_fraction':budget[2],'oracle_has_extra_knowledge':k==4})
for a in [side,ax]:a.set_ylim(10.7,-.8)
side.set(xlim=(0,1),xticks=[],yticks=[b*6+k for b in range(2) for k in range(5)],yticklabels=[f'{m}: {label}' for m in ['D3','D4'] for label in labels]);side.tick_params(axis='y',length=0,labelsize=7)
for sp in side.spines.values():sp.set_visible(False)
ax.set(yticks=[],xlim=(-11,7),xticks=[-10,-5,0,5],xlabel='Correct-call change vs ordinary-only (pp)');ax.spines[['left','top','right']].set_visible(False);ax.axvline(0,color='#aaaaaa',lw=.5,ls='--');side.set_title('Budget',fontsize=7);ax.set_title('Reference ○ — ● strategy; paired 95% interval',fontsize=8)
fig.legend(handles=[Patch(fc=c,label=l) for c,l in zip([COL[1],COL[0],COL[2]],['Ordinary records','Random X','Random X + S'])],loc='lower center',bbox_to_anchor=(.55,.03),ncol=3,frameon=False,fontsize=7)
save(fig,3,pd.DataFrame(tab),'R25','Paired budget-strategy effect relative to ordinary-only','Preserve row connectors, hollow reference, filled strategy endpoint, side composition bars and bottom legend. Fixed design order instead of favorable-effect sorting; zero represents each model-specific ordinary comparator. Added paired intervals; known operator has extra knowledge, not extra budget.')
# R28: all outcomes, interval structure, then conditional error risk on the same rows.
order=['OMITTED_U','MORE_ENVS_SAME_BUDGET','FLEXIBLE_U','CALIBRATED_FLEXIBLE_U'];names=['Omitted U','More env.','Flexible U','Calibrated U'];fig=plt.figure(figsize=(180/25.4,170/25.4));axes=[fig.add_axes(v) for v in [[.25,.18,.24,.68],[.52,.18,.18,.68],[.74,.18,.22,.68]]];tab=[];yl=[];yt=[]
for group,cell in enumerate(range(8,12)):
 for j,(method,name) in enumerate(zip(order,names)):
  r=S[(S.cell_id==cell)&(S.method_id==method)].iloc[0];y=group*4.7+j;n=int(r.n_tasks);failed=int(r.numerical_failures);correct=int(r.correct_calls);wrong=int(r.wrong_calls);unknown=n-correct-wrong-failed;finite=int(r.n_finite_width);unbounded=int(r.n_complete)-finite
  outcome=[correct,wrong,unknown,failed];widths=[finite,unbounded,failed];assert min(outcome+widths)>=0 and sum(widths)==n and sum(outcome)==n
  for ax,counts,colors in [(axes[0],outcome,[COL[2],COL[4],COL[1],COL[5]]),(axes[1],widths,[COL[0],COL[3],COL[5]])]:
   left=0
   for count,c in zip(counts,colors):ax.barh(y,100*count/n,left=left,height=.78,fc=c,ec='white',lw=.25);left+=100*count/n
  if r.called>0:
   risk=100*wrong/r.called;axes[2].errorbar(risk,y,xerr=[[risk-100*r.risk_low],[100*r.risk_high-risk]],fmt='o',ms=3,c='#222222',elinewidth=.6,capsize=1)
  else:axes[2].text(50,y,'NA',ha='center',va='center',fontsize=6.5,color='#777777')
  yt.append(y);yl.append(['U=0','U=0.5','U=1','Exact alias'][group]+' · '+name+(' [1 failure]' if failed else ''));tab.append({**r.to_dict(),'unknown_count':unknown,'finite_count':finite,'unbounded_count':unbounded,'failed_count':failed,'display_row':y,'wrong_given_called':wrong/r.called if r.called else float('nan')})
for i,(ax,title) in enumerate(zip(axes,['Sign decision','Interval extent','Wrong / called'])):
 ax.set(xlim=((-2,102) if i==2 else (-1,101)),ylim=(17.9,-.8),xticks=[0,50,100],xlabel='Percent');ax.set_title(title,fontsize=8,pad=10);ax.set_yticks(yt,yl if i==0 else []);ax.tick_params(axis='y',length=0,labelsize=6.7);ax.grid(axis='x',lw=.4,color='#eeeeee');ax.set_axisbelow(True)
 for sp in ax.spines.values():sp.set_linewidth(.55)
fig.legend(handles=[Patch(fc=c,label=l) for c,l in zip([COL[2],COL[4],COL[1],COL[5],COL[0],COL[3]],['Correct','Wrong','Unknown','Numerical failure','Finite interval','Unbounded interval'])],loc='lower center',bbox_to_anchor=(.57,.015),ncol=3,frameon=False,fontsize=7)
save(fig,4,pd.DataFrame(tab),'R28','Common-bias decision composition, interval extent and conditional error risk','Preserve two horizontally stacked composition columns and a narrow aligned point column; combine raw outcomes, not old images. Include all four bias conditions, all four distinct specifications, UNKNOWN and numerical failures. Conditional risk undefined for no calls shown NA; added Wilson intervals.')
# Forward sensitivity: actual coefficient scale; errors derived from all 20 fixed replicate IDs.
Z=pd.read_csv(G/'results/G3_sensitivity_regions.tsv',sep='\t');fig,ax=canvas(118,118);tab=[]
grouped_curves={}
for cell,ls,marker,label in [(3,'-','o','Biology + obs'),(10,'--','s','Common U')]:
 for eps,c in zip([0,.2,.5],[COL[0],COL[2],COL[5]]):
  points=[]
  for eta in [0,.5,1.]:
   f=Z[(Z.cell==cell)&(Z.epsilon_log_rate==eps)&(Z.eta_selection_coefficient==eta)];n=len(f);assert n==20;wrong=int(f.sign.eq('POSITIVE').sum());called=int(f.sign.isin(['POSITIVE','NEGATIVE']).sum());lo,hi=wilson(wrong,n);points.append((eta,wrong/n,lo,hi));tab.append(dict(cell=cell,epsilon=eps,eta=eta,n=n,wrong=wrong,called=called,coverage=f.covers_truth.mean(),n_completed=int(f.status.eq('COMPLETE').sum()),wrong_rate=wrong/n,wrong_low=lo,wrong_high=hi,inside_allowed_set=bool(f.true_mechanism_in_allowed_set.iloc[0])))
  if cell==3:c=COL[1]
  key=(cell,tuple(points))
  if key not in grouped_curves:grouped_curves[key]=dict(points=points,color=c,ls=ls,marker=marker,label=label,epsilon=[])
  grouped_curves[key]['epsilon'].append(eps)
for curve in grouped_curves.values():
 p=np.array(curve['points']);c=curve['color'];label=curve['label']+'; ε='+','.join(f'{v:g}' for v in curve['epsilon'])
 ax.plot(p[:,0],100*p[:,1],color=c,ls=curve['ls'],lw=2.2,label=label);ax.errorbar(p[:,0],100*p[:,1],yerr=np.maximum(0,[100*(p[:,1]-p[:,2]),100*(p[:,3]-p[:,1])]),fmt=curve['marker'],mfc=c,mec='#333333',ms=5.5,ecolor='#333333',elinewidth=.5,capsize=1.5)

ax.set(xlim=(-.03,1.03),ylim=(-2.5,105),yticks=[0,20,40,60,80,100],xticks=[0,.5,1],xlabel='Allowed selection-feature coefficient |u| ≤ η',ylabel='Wrong sign calls / all replicates (%)');ax.legend(ncol=1,loc='lower left',bbox_to_anchor=(0,1.02),fontsize=6.8,frameon=False)
fig.text(.14,.035,'20 prespecified replicates per curve: diagnostic precision, not full calibration.\nε: log-rate deviation; η: selection coefficient. Uncalled/failed counts retained in source data.',fontsize=7,linespacing=1.5)
save(fig,5,pd.DataFrame(tab),'R04','Wrong-answer reduction across explicit forward sensitivity sets','Filled dark-edge markers, black error bars, colored trajectories and upper legend. Actual three coefficient values; exactly coincident trajectories are drawn once with all epsilon labels, without pooling replicate counts. No invented phases/AUC. Distinguish scenarios by marker and line style; do not add ε and η across units.')

def raincloud(number,groups,labels,colors,width):
 fig,ax=canvas(width,120);fig.subplots_adjust(top=.93,bottom=.21,right=.77);tab=[];rng=np.random.default_rng(713)
 for i,((cell,method),label,c) in enumerate(zip(groups,labels,colors)):
  f=D[(D.cell_id==cell)&(D.method_id==method)].copy();valid=f.fit_success.eq(True)&f.population_prediction_KL.notna();v=f.loc[valid,'population_prediction_KL'].to_numpy();v=np.maximum(v,1e-16);z=v*1000
  jitter=rng.uniform(-.27,-.075,len(z));ax.scatter(i+jitter,z,s=3,c=c,alpha=.38,lw=0,zorder=2)
  if len(z)>1 and np.std(z)>1e-8:
   grid=np.linspace(z.min(),z.max(),180);density=gaussian_kde(z)(grid);density=.29*density/density.max();ax.fill_betweenx(grid,i+.06,i+.06+density,fc=c,alpha=.75,ec='#444444',lw=.65)
  ax.boxplot(z,positions=[i+.015],widths=.075,patch_artist=True,showfliers=False,whis=1.5,boxprops=dict(facecolor='white',edgecolor='#333333',lw=.65),medianprops=dict(color='#222222',lw=1),whiskerprops=dict(color='#333333',lw=.65),capprops=dict(color='#333333',lw=.65))
  for _,r in f.iterrows():tab.append({**r.to_dict(),'display_group':label,'display_KL_x1000':float(r.population_prediction_KL)*1000 if pd.notna(r.population_prediction_KL) and r.fit_success else None})
  ax.text(i,1.025,f'n={int(valid.sum())}/{len(f)}',transform=ax.get_xaxis_transform(),ha='center',fontsize=6.8)
 ax.set_xticks(range(len(groups)),labels,fontstyle='italic');ax.set_xlim(-.5,len(groups)-.45);ax.set_ylabel('Predictive KL (×0.001)');ax.set_ylim(bottom=0);ax.tick_params(axis='x',length=0,pad=4)
 ax.legend(handles=[Patch(fc=c,ec='#444444',lw=.5,label=l.replace('\n',' ')) for c,l in zip(colors,labels)],loc='center left',bbox_to_anchor=(1.01,.5),frameon=False,fontsize=6.5)
 for sp in ax.spines.values():sp.set_visible(True);sp.set_linewidth(.6)
 fig.text(.14,.035,'All successful-fit replicates: raw points, white median/IQR box, right half-density.\nFailed fits excluded only from distribution; denominators shown. Display floor KL = 1e−16.',fontsize=7,linespacing=1.5)
 save(fig,number,pd.DataFrame(tab),'R27','Distribution of observable predictive discrepancy','Actual per-replicate points on left, slim white box in middle, colored half-violin on right, full fine frame. Density estimated on linear KL with Scott bandwidth; no patient subsampling; jitter is display-only.')
raincloud(6,[(12,'MISSPECIFIED_SHARED'),(13,'MISSPECIFIED_SHARED'),(14,'MISSPECIFIED_SHARED'),(15,'MISSPECIFIED_SHARED')],['Gamma\ntime','Higher-order\ninteraction','Unlabelled\nmixture','Measurement\nerror'],[COL[0],COL[2],COL[5],COL[3]],150)
# Target coverage, rather than nearly identical predictive KL, is the endpoint of known-assay correction.
fig,ax=canvas(120,110);tab=[]
comparisons=[['MISSPECIFIED_SHARED','KNOWN_TIME_FAMILY','ESTIMATED_TIME_FAMILY'],['MISSPECIFIED_SHARED','KNOWN_MEASUREMENT','BIOLOGY_ENVELOPE']]
for j,(label,c) in enumerate(zip(['Wrong forward model','Known forward model','Additional nuisance'],[COL[0],COL[2],COL[5]])):
 for group,cell in enumerate([12,15]):
  r=S[(S.cell_id==cell)&(S.method_id==comparisons[group][j])].iloc[0];x=group+(j-1)*.23;y=100*(1-r.coverage);lo=100*(1-r.coverage_high);hi=100*(1-r.coverage_low)
  ax.bar(x,y,width=.21,fc=c,ec='#222222',lw=.85,label=label if group==0 else None);ax.errorbar(x,y,yerr=np.maximum(0,[[y-lo],[hi-y]]),fmt='none',ecolor='#222222',elinewidth=.7,capsize=2)
  ax.text(x,hi+1.1,f'{y:.1f}',ha='center',va='bottom',fontsize=7);tab.append({**r.to_dict(),'noncoverage':y/100,'noncoverage_low':lo/100,'noncoverage_high':hi/100})
ax.axhline(5,color='#666666',lw=.8,ls=(0,(3,3)));ax.set(ylim=(0,82),xticks=[0,1],xticklabels=['Gamma time','Measurement error'],ylabel='Uncovered / completed intervals (%)');ax.legend(ncol=1,loc='lower left',bbox_to_anchor=(0,1.02),fontsize=7,frameon=False)
fig.text(.14,.035,'200 replicates per condition; Wilson 95% Monte Carlo intervals.\nAdditional nuisance: estimated time CV / bounded biological deviation, respectively.',fontsize=7,linespacing=1.5)
save(fig,7,pd.DataFrame(tab),'R21','Target interval noncoverage under specified forward-model corrections','Grouped dark-edge bars, black capped intervals and direct endpoint values. Data audit found nearly identical wrong/known assay predictive KL, so use the scientifically relevant target coverage endpoint instead; all three frozen specifications retained per condition.')
(G/'configs/FIGURE_MANIFEST.json').write_text(json.dumps(MAN,ensure_ascii=False,indent=2));print('Generated',len(MAN),'independent PDF/SVG figures; awaiting visual review')
