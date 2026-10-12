from pathlib import Path
import sys,json,hashlib,argparse
import numpy as np,pandas as pd
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
P=Path(__file__).resolve().parents[2];R=P.parent;D=Path(__file__).resolve().parent
sys.path[:0]=[str(D.parent/'redesign_r2'),str(R/'src')]
from invariant_evam.figure_style import apply_publication_style
from run_simulations import wilson
apply_publication_style({'svg.fonttype':'none','pdf.fonttype':42,'font.size':8,'axes.labelsize':8,'xtick.labelsize':7,'ytick.labelsize':7,'axes.linewidth':.75})
C=['#0072B2','#56B4E9','#009E73','#E69F00','#D55E00','#CC79A7'];MAN={g:[] for g in [3,4,5]}
CAT=json.loads((R/'configs/figure_reference_catalog.json').read_text())['records']
def read(g,name):return pd.read_csv(P/f'g{g}/revision_r3/results/{name}.tsv',sep='\t')
def canvas(title,ylabel,width=150,height=108):
 f,a=plt.subplots(figsize=(width/25.4,height/25.4));f.subplots_adjust(left=.15,right=.965,bottom=.25,top=.76);a.spines[['top','right']].set_visible(False);a.set_ylabel(ylabel);f.text(.15,.95,title,fontsize=9,weight='bold');return f,a
def legend(f,labels,colors):
 f.legend(handles=[Patch(fc=c,ec='#222222',lw=.7,label=l) for l,c in zip(labels,colors)],loc='upper left',bbox_to_anchor=(.14,.895),ncol=len(labels),frameon=True,fancybox=False,edgecolor='#bbbbbb',fontsize=6.4,handlelength=1,columnspacing=.85,handletextpad=.4)
def foot(f,s):f.text(.15,.055,s,fontsize=6.1,linespacing=1.5)
def save(f,g,name,df,ref,adapt):
 G=P/f'g{g}/revision_r3';data=G/f'figure_data/{name}.tsv';df.to_csv(data,sep='\t',index=False)
 for ext in ['pdf','svg']:f.savefig(G/f'figures/{name}.{ext}',dpi=600)
 tmp=Path('/tmp/invariant_r3_previews');tmp.mkdir(exist_ok=True);f.savefig(tmp/f'{name}.png',dpi=150)
 reference=next(z for z in CAT if z['id']==ref);source=Path(reference['source_file_absolute'])
 MAN[g].append(dict(panel_id=name,reference=ref,reference_file=str(source),reference_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),data_sha256=hashlib.sha256(data.read_bytes()).hexdigest(),size_mm=(f.get_size_inches()*25.4).tolist(),dpi=600,palette=C,adaptations=adapt,review_status='AWAITING_ACTUAL_VISUAL_REVIEW',user_approved=False));plt.close(f)
def cap(a,x,m,lo,hi,c,width=.22):
 a.bar(x,m,width,fc=c,ec='#222222',lw=.7,zorder=2);a.errorbar(x,m,yerr=[[max(0,m-lo)],[max(0,hi-m)]],fmt='none',ecolor='#222222',capsize=2.4,lw=.7,zorder=3)
def g3():
 d=read(3,'evaluation_raw');methods=['XS_SOURCE','XS_JOINT','RANDOM_X'];labels=['XS: X only','XS: joint fit','Random X'];cols=[C[1],C[0],C[2]]
 for budget in [1000,4000]:
  q=d[d.budget.eq(budget)].copy();f=plt.figure(figsize=(150/25.4,125/25.4));a=f.add_axes([.15,.25,.81,.55]);a.spines[['top','right']].set_visible(False);a.set_ylabel('Correct sign answers (%)');a.set_xlabel('Nominal budget fraction allocated to calibration');a.set_xticks([.1,.25,.5],['10%','25%','50%']);a.set_xlim(.075,.525);a.set_ylim(0,130);a.set_yticks([0,20,40,60,80]);a.spines['left'].set_bounds(0,85)
  peak=100*q[q.method.isin(methods)&q.effect.ne(0)].groupby(['method','fraction']).correct.mean().max();upper=10*np.ceil((peak+3)/10);a.set_ylim(0,upper*1.6);a.spines['left'].set_bounds(0,upper);a.set_yticks(np.arange(0,upper+1,10 if upper<=50 else 20))
  ins=[a.inset_axes([.06,.73,.34,.22]),a.inset_axes([.64,.73,.31,.22])]
  for j,m in enumerate(methods):
   vals=[];errs=[]
   for frac in [.1,.25,.5]:
    z=q[q.method.eq(m)&q.fraction.eq(frac)&q.effect.ne(0)];cells=z.groupby(['effect','bias']).correct.agg(['mean','var','size']);v=cells['mean'].mean();se=np.sqrt((cells['var']/cells['size']).sum())/len(cells);vals.append(100*v);errs.append(196*se)
   a.errorbar([.1,.25,.5],vals,yerr=errs,c=cols[j],lw=1.8,marker='o',ms=5,mec='#222222',mew=.8,ecolor='#222222',capsize=2.5,elinewidth=.7)
   z=q[q.method.eq(m)&q.fraction.eq(.5)];wr=z.groupby(['effect','bias']).wrong.mean().max();ins[0].barh(j,100*wr,.5,fc=cols[j],ec='#222222',lw=.6)
   coverage=z.groupby(['effect','bias']).coverage.mean().min()
   if np.isfinite(coverage):ins[1].barh(j,100*coverage,.5,fc=cols[j],ec='#222222',lw=.6);ins[1].text(102,j,f'{coverage:.1%}',fontsize=5.5,va='center')
   else:ins[1].text(4,j,'sign test only',fontsize=5.5,va='center')
  for ax in ins:ax.spines[['top','right']].set_visible(False);ax.set_ylim(2.6,-.6);ax.set_yticks(range(3),['X only','Joint','Random X']);ax.tick_params(labelsize=5.5,length=2,pad=2)
  ins[0].set_xlim(0,8);ins[0].set_xticks([0,4,8],['0','4','8%']);ins[0].set_title('Worst-cell wrong calls',fontsize=6.2,pad=5);ins[1].set_xlim(0,101);ins[1].set_xticks([0,50,100],['0','50','100%']);ins[1].set_title('Minimum coverage',fontsize=6.2,pad=5)
  f.text(.15,.96,f'G3 | Two-phase calibration · total cost {budget:,}',fontsize=8.8,weight='bold');legend(f,labels,cols);foot(f,'Main: equal-weight mean of 6 nonzero-effect cells; 500 datasets per cell.\nCaps: 1.96 Monte Carlo SE. Insets: all 9 effect/bias cells at 50% allocation.\nMagnitude intervals concern source log-OR; they are not generator intervals.')
  save(f,3,f'G3_R3_joint_cost{budget}',q,'R04','Actual allocation-grid lines with black-edged circles and capped errors, two upper horizontal-bar insets. No smooth fit, phase shading, AUC or invented P values. Null effects retained in safety insets and all raw data.')
 if (P/'g3/revision_r3/results/anchor_evaluation_raw.tsv').exists():
  d=read(3,'anchor_evaluation_raw');f,a=canvas('G3 | Reference-target calibration','Correct reference-sign answers (%)');methods=['NO_ANCHOR','BALANCED_X','REFERENCE_X'];cols=[C[1],C[0],C[2]]
  for k,cell in enumerate([3,10]):
   for j,m in enumerate(methods):
    q=d[d.cell.eq(cell)&d.method.eq(m)];correct=q.called&~q.wrong;v=correct.mean();lo,hi=wilson(correct.sum(),len(q));x=k+(j-1)*.245;cap(a,x,100*v,100*lo,100*hi,cols[j],.215);a.annotate(f'{v:.1%}',(x,100*hi),xytext=(0,5),textcoords='offset points',ha='center',fontsize=6.5)
  a.set_ylim(0,60);a.set_xticks([0,1],['Biology + observation','Common omitted bias']);legend(f,['No anchor','Balanced X','Reference X'],cols);foot(f,'500 new datasets per scenario; Wilson intervals; actual cost 4,000 in every arm.\nOriginal reference interaction target; biology bound 0.5, pair-selection bound 1.\nNo-anchor unresolved sets remain visible; coverage and failures in source tables.');save(f,3,'G3_R3_reference_anchor',d,'R21','Narrow grouped outlined bars, black capped intervals, top boxed legend, numeric endpoints. Binary outcomes are summarized without redundant 0/1 scatter.')
def g4():
 d=read(4,'evaluation_raw');designs=['REDUNDANT','HETEROGENEOUS','CROSSVIEW_FAILURE'];cols=[C[0],C[1],C[3]];refs=['REG_FUSION','UNREG_FUSION','POOLED']
 for cancer in ['LUAD','COAD']:
  q=d[d.cancer.eq(cancer)].copy();q['display_batch']=q.replicate//5;f,a=canvas(f'G4 | Source prediction · {cancer}','Log-score gain over CTMC (nats / record)')
  for k,design in enumerate(designs):
   w=q[q.design.eq(design)].pivot(index='replicate',columns='method',values='source_log_score')
   for j,m in enumerate(refs):
    v=w.CTMC_M-w[m];mean=v.mean();err=1.96*v.std(ddof=1)/np.sqrt(len(v));x=k+(j-1)*.245;cap(a,x,mean,mean-err,mean+err,cols[j],.215);batch=v.groupby(v.index//5).mean();a.scatter(x+np.linspace(-.045,.045,len(batch)),batch,s=10,facecolor='white',edgecolor='#333333',lw=.55,zorder=4)
  a.axhline(0,c='#555555',lw=.75);a.set_xticks(range(3),['Redundant','Complementary','Missing joint path']);legend(f,['Shrinkage fusion','Raw fusion','Pooled'],cols);foot(f,'50 paired perturbations per design; caps: 1.96 seed-MCSE. Higher is better.\nOpen points: 10 fixed batches of 5 seeds; every seed contributes once.\nSame existing patient pool, cost and allowed information; failures retained.');save(f,4,f'G4_R3_source_{cancer}',q,'R21','Grouped outlined bars, open fixed-batch points and capped paired errors; zero baseline supports signed negative results, no offset bars or omitted conditions.')
 if (P/'g4/revision_r3/full_calibration/results/evaluation_raw.tsv').exists():
  full=pd.read_csv(P/'g4/revision_r3/full_calibration/results/evaluation_raw.tsv',sep='\t');small=d.copy();small['calibration_design']='80 calibration records';full['calibration_design']='All eligible calibration records';all_data=pd.concat([small,full],ignore_index=True)
  for cancer in ['LUAD','COAD']:
   q=all_data[all_data.cancer.eq(cancer)].copy();sample=json.loads(next((P/'g4/revision_r3/full_calibration/runs').glob(f'evaluation_{cancer}_*.json')).read_text());bigcost=sample['stats']['actual_gene_assays'];q['cost']=np.where(q.calibration_design.eq('80 calibration records'),560,bigcost)
   f,a=canvas(f'G4 | Calibration-data allocation · {cancer}','Source log score (nats / record) ↓',150,128);f.subplots_adjust(top=.78,bottom=.26);ins=[a.inset_axes([.08,.73,.31,.23]),a.inset_axes([.65,.73,.31,.23])];extrema=[]
   for j,m in enumerate(['CTMC_M','REG_FUSION','UNREG_FUSION']):
    mean=[];err=[]
    for cost in [560,bigcost]:
     z=q[q.method.eq(m)&q.cost.eq(cost)].groupby('replicate').source_log_score.mean();mean.append(z.mean());err.append(1.96*z.std(ddof=1)/np.sqrt(len(z)))
    extrema.extend(np.array(mean)-np.array(err));extrema.extend(np.array(mean)+np.array(err));a.errorbar([560,bigcost],mean,yerr=err,c=[C[0],C[2],C[3]][j],marker='o',ms=5,mec='#222222',mew=.8,lw=1.8,ecolor='#222222',capsize=2.5,elinewidth=.7)
   low=min(extrema);high=max(extrema);span=high-low;a.set_ylim(low-.10*span,low+2*span);a.spines['left'].set_bounds(low-.1*span,high+.1*span);a.set_yticks([v for v in matplotlib.ticker.MaxNLocator(nbins=4).tick_values(low,high) if low-.1*span<=v<=high+.1*span])
   for ax,cost in zip(ins,[560,bigcost]):
    w=q[q.cost.eq(cost)].groupby(['replicate','method']).source_log_score.mean().unstack()
    for j,m in enumerate(['REG_FUSION','UNREG_FUSION']):
     v=w.CTMC_M-w[m];err=1.96*v.std(ddof=1)/np.sqrt(len(v));ax.barh(j,v.mean(),.5,fc=[C[2],C[3]][j],ec='#222222',lw=.6);ax.errorbar(v.mean(),j,xerr=err,fmt='none',ecolor='#222222',capsize=1.5,lw=.6)
    ax.axvline(0,c='#777777',lw=.5);ax.spines[['top','right']].set_visible(False);ax.set_yticks([0,1],['Shrinkage','Raw']);ax.set_ylim(1.6,-.6);ax.tick_params(labelsize=5.5,length=2,pad=2);ax.set_title(f'Gain vs CTMC | cost {cost}',fontsize=6.1,pad=5)
   a.set_xticks([560,bigcost],[f'560\n80 calibration records',f'{bigcost:,}\n{int((bigcost-240)/4)} calibration records']);a.get_xticklabels()[0].set_ha('left');a.get_xticklabels()[1].set_ha('right');a.set_xlabel('Actual event-measurement cost');legend(f,['CTMC','Shrinkage fusion','Raw fusion'],[C[0],C[2],C[3]]);foot(f,'Each point: 50 perturbation seeds, equal-weight mean of all three designs.\nCaps: 1.96 seed-MCSE; within each budget, all methods receive the same data.\nTwo measured budgets only; connecting segments are not fitted learning curves.');save(f,4,f'G4_R3_calibration_budget_{cancer}',q,'R04','Colored actual-budget trajectories, filled black-edged circular marks, black capped errors and two upper paired-gain insets. No phase or AUC exists for two budget points; no fabricated phase decoration.')
 if (P/'g4/revision_r3/results/calibrate_evaluation_raw.tsv').exists():
  d=read(4,'calibrate_evaluation_raw')
  for cancer in ['LUAD','COAD']:
   q=d[d.cancer.eq(cancer)].copy();q['display_batch']=q.replicate//5;f,a=canvas(f'G4 | Cross-fitted probability calibration · {cancer}','Expected calibration error ↓');top=0
   for k,design in enumerate(designs):
    for j,m in enumerate(['CTMC','CALIBRATED']):
     z=q[q.design.eq(design)&q.method.eq(m)];v=z.ece;mean=v.mean();err=1.96*v.std(ddof=1)/np.sqrt(len(v));x=k+(j-.5)*.25;cap(a,x,mean,mean-err,mean+err,[C[0],C[2]][j],.215);batch=z.groupby('display_batch').ece.mean();a.scatter(x+np.linspace(-.045,.045,len(batch)),batch,s=10,facecolor='white',edgecolor='#333333',lw=.55,zorder=4);marktop=max(mean+err,batch.max());a.annotate(f'{mean:.3f}',(x,marktop),xytext=(0,5),textcoords='offset points',ha='center',fontsize=6.3);top=max(top,marktop)
   a.set_ylim(0,top*1.27);a.set_xticks(range(3),['Redundant','Complementary','Missing joint path']);legend(f,['CTMC','Cross-fitted adjustment'],[C[0],C[2]]);foot(f,'50 perturbation seeds; open points: 10 fixed batches of 5 (all seeds retained).\nFive probability bins; caps: 1.96 seed-MCSE, conditional on the same patient pool.\nCalibration uses training/calibration only; Brier and log scores also retained.');save(f,4,f'G4_R3_calibration_{cancer}',q,'R21','Narrow outlined grouped bars with open fixed-batch points, capped errors and numeric means; top legend and all three designs retained.')
def g5():
 d=read(5,'natural_paired_summary');refs=['LEGACY_IDENTITY_SHARED','LEGACY_BURDEN_SELECTION_SHARED','LEGACY_PRESPECIFIED_GENE_SELECTION','TARGET_ONLY_DIRICHLET','POOLED_TRAINING'];cols=[C[0],C[1],C[5],C[3],C[2]];f,a=canvas('G5 | Natural-panel predictive transfer','Transfer log-score gain (millinats / record)',160,110)
 for k,held in enumerate(['MSK-IMPACT410','MSK-IMPACT505']):
  for j,m in enumerate(refs):
   z=d[d.heldout.eq(held)&d.reference.eq(m)].iloc[0];cap(a,k+(j-2)*.17,1000*z.gain,1000*z.lo,1000*z.hi,cols[j],.145)
 a.axhline(0,c='#555555',lw=.75);a.set_xticks([0,1],['IMPACT410 · n=548','IMPACT505 · n=1,595']);legend(f,['Shared CTMC','Burden CTMC','Gene CTMC','Target only','Pooled'],cols);foot(f,'Paired patient-score differences; caps: 1.96 patient SE, conditional on fitted models.\nTransfer prior strength selected using training panels only. CTMC predictions are legacy.\nSame previously explored evaluation records; not an external replication.');save(f,5,'G5_R3_natural_transfer',d,'R21','Five matched bars per held-out panel, black outlines/caps, boxed legend, zero baseline; millinats expose small effects without changing their sign or uncertainty. Burden and gene selection baselines both visible.')
 if (P/'g5/revision_r3/results/anchor_baseline_raw.tsv').exists():
  d=read(5,'anchor_interval_evaluation_raw');d['method']='ANCHOR_'+d.method;d=pd.concat([d,read(5,'anchor_baseline_raw')],ignore_index=True)
  for metric,label in [('covered','Target interval coverage (%)'),('called','Sign answers (%)')]:
   f,a=canvas('G5 | Source anchoring at equal acquisition cost',label);methods=['COUNT_EXPANSION_CLASSIC','ANCHOR_CLASSIC','ANCHOR_CALIBRATED'];cols=[C[1],C[0],C[2]];top=0
   for k,s in enumerate([0,1,2]):
    for j,m in enumerate(methods):
     q=d[d.scenario.eq(s)&d.method.eq(m)];v=q[metric].mean();lo,hi=wilson(q[metric].sum(),len(q));x=k+(j-1)*.23;cap(a,x,100*v,100*lo,100*hi,cols[j],.20);a.annotate(f'{v:.1%}',(x,100*hi),xytext=(0,5),textcoords='offset points',ha='center',fontsize=6.2);top=max(top,100*hi)
   a.set_ylim(0,112 if metric=='covered' else max(20,top*1.3))
   if metric=='covered':a.axhline(95,c='#777777',lw=.7,ls=(0,(3,3)))
   a.set_xticks(range(3),['Model matched','Pair-selection bias','Biology shift']);legend(f,['More selected patients','Source anchor: LR','Source anchor: calibrated'],cols);foot(f,'500 new draws per scenario; matched added cost 1,500; Wilson intervals.\n350 source records versus 350 selected records; same 100 completed baseline records.\nExpanded nuisance family; original cost-74 results remain separately reported.');save(f,5,f'G5_R3_anchor_{metric}',d,'R21','Three narrow outlined bars, capped intervals, top boxed legend and endpoint values. Coverage and answerability separated to expose vacuous coverage; actual cost explicitly reported.')
 if (P/'g5/revision_r3/results/interval_evaluation_raw.tsv').exists():
  d=read(5,'interval_evaluation_raw')
  for metric,label in [('covered','Target interval coverage (%)'),('called','Sign answers (%)')]:
   f,a=canvas('G5 | Independent post-acquisition interval calibration',label);top=0
   for k,s in enumerate([0,1,2]):
    for j,m in enumerate(['CLASSIC','CALIBRATED']):
     q=d[d.scenario.eq(s)&d.method.eq(m)];v=q[metric].mean();lo,hi=wilson(q[metric].sum(),len(q));x=k+(j-.5)*.25;cap(a,x,100*v,100*lo,100*hi,[C[0],C[2]][j],.215);a.annotate(f'{v:.1%}',(x,100*hi),xytext=(0,5),textcoords='offset points',ha='center',fontsize=6.4);top=max(top,100*hi)
   if metric=='covered':a.axhline(95,c='#777777',lw=.7,ls=(0,(3,3)));a.set_ylim(0,112)
   else:a.set_ylim(0,max(15,top*1.3))
   a.set_xticks(range(3),['Model matched','Pair-selection bias','Biology shift']);legend(f,['Classic LR','Development-calibrated LR'],[C[0],C[2]]);foot(f,'500 new parameter/dataset draws per scenario; Wilson intervals; actual added cost 74.\nCutoff selected on 300 separate development draws; finite widths/failures reported.\nEmpirical simulator-family calibration; no universal misspecification guarantee.');save(f,5,f'G5_R3_interval_{metric}',d,'R21','Two narrow outlined bars per scenario, black caps and numeric endpoints; full baseline for coverage, separate answerability figure. No failure exclusions or invented significance.')
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('groups',nargs='+',type=int);args=a.parse_args()
 for g in args.groups:
  globals()[f'g{g}']();(P/f'g{g}/revision_r3/configs/FIGURE_MANIFEST.json').write_text(json.dumps(MAN[g],indent=2))
