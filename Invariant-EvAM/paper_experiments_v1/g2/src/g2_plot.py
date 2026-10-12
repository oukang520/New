from pathlib import Path
import sys,json,hashlib
R=Path(__file__).resolve().parents[3];G=R/'paper_experiments_v1/g2'
sys.path.insert(0,str(R/'src'))
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from invariant_evam.figure_style import apply_publication_style
from g2_summarize import wilson
C={'M':'#0072B2','B5':'#56B4E9','B4':'#009E73','B3':'#E69F00','B2':'#D55E00','B1':'#CC79A7'}
ORDER=['B1','B2','B3','B4','B5','M'];MARK={'B1':'v','B2':'s','B3':'D','B4':'^','B5':'o','M':'o'}
EFFECTS=[-.8,0,.15,.8];DESIGNS=['strong','weak','nonidentifying'];LABEL={'strong':'Complementary operators','weak':'Repeated operator family','nonidentifying':'Separate marginal panels'}
OUT=G/'figures';DATA=G/'figure_data';PRE=G/'reviewer_audit/previews';MAN=[]
apply_publication_style({'font.size':8,'axes.labelsize':8,'axes.titlesize':8,'xtick.labelsize':7,'ytick.labelsize':7,'legend.fontsize':7,'lines.markersize':3.5,'figure.dpi':150})

def canvas(title,ylabel,legend=True):
 fig,ax=plt.subplots(figsize=(100/25.4,82/25.4));fig.subplots_adjust(left=.18,right=.95,bottom=.18,top=.70 if legend else .87)
 ax.spines[['top','right']].set_visible(False);ax.set_title(title,loc='left',pad=9);ax.set_ylabel(ylabel);ax.tick_params(direction='out');return fig,ax

def effects(ax):
 ax.set_xticks(range(4),['−0.8','0','+0.15','+0.8']);ax.set_xlim(-.32,3.32);ax.set_xlabel('True interaction (ordered conditions)')
def legend(ax):ax.legend(ncol=3,loc='lower left',bbox_to_anchor=(-.03,1.12),borderaxespad=0,columnspacing=1.1,handlelength=1.6,handletextpad=.4)
def enrich_metadata(entry):
 name=entry['panel_id'];entry['N']={'observations_per_replicate':1000,'replicates_per_cell':500};entry['uncertainty_definition']='Wilson 95% Monte Carlo intervals; endpoint denominator explicitly in data/caption';entry['failure_handling']='All planned tasks retained; failures separate from scientific UNKNOWN; conditional and end-to-end coverage both exported'
 if name.startswith('G2A_paired'):entry['uncertainty_definition']='Paired normal Monte Carlo interval for difference; M rate Wilson band; zero discordance labeled without exact-equality claim'
 if name.startswith('G2C'):entry['N']['replicates_in_fixed_effect_mixture']=2000;entry['uncertainty_definition']='Wilson 95% conditional error interval; zero-call risk undefined'
 if name.startswith('G2D') and 'certified' not in name:
  entry['N']={'observations':1000,'independent_replicates':1,'paired_search_bounds':3};entry['palette']={'bound3':'#0072B2','bound4.5':'#E69F00','bound6':'#009E73','failed_grid_point':'#D55E00'};entry['uncertainty_definition']='Likelihood profile and chi-square1 candidate threshold; not a Monte Carlo interval'
 if name.startswith('G2E'):entry['N']['replicates_per_cell']=200;entry['palette']={'B6_OFFICIAL':'#56B4E9','M_MATCHED':'#0072B2'}
 if 'certified_sign' in name:entry['N']['independent_replicates_total']=6000
 return entry

def save(fig,name,df,ref,endpoint,notes):
 OUT.mkdir(parents=True,exist_ok=True);DATA.mkdir(parents=True,exist_ok=True);PRE.mkdir(parents=True,exist_ok=True)
 for ext in ['pdf','svg']:fig.savefig(OUT/(name+'.'+ext),dpi=600)
 fig.savefig(PRE/(name+'.png'),dpi=180)
 df.to_csv(DATA/(name+'.tsv'),sep='\t',index=False)
 MAN.append({'panel_id':name,'primary_reference_id':ref,'endpoint':endpoint,'source_data':str(DATA/(name+'.tsv')),'output_pdf':str(OUT/(name+'.pdf')),'output_svg':str(OUT/(name+'.svg')),'dpi':600,'size_mm':[100,82],'palette':C,'adaptations':notes,'statistical_unit':'independent paired simulation replicate; fixed conditions','review_status':'awaiting_visual_review','data_sha256':hashlib.sha256((DATA/(name+'.tsv')).read_bytes()).hexdigest()});enrich_metadata(MAN[-1]);plt.close(fig)

def performance(summary):
 for design in DESIGNS:
  d=summary[(summary.design==design)&summary.method_id.isin(ORDER)].copy()
  for endpoint,ylabel,count in [('correct_call_fraction','Correct sign calls / all replicates','correct_calls'),('wrong_call_fraction','Wrong sign calls / all replicates','wrong_calls'),('conditional_coverage','Coverage / completed intervals','n_covered'),('end_to_end_covered_fraction','Covered intervals / all replicates','n_covered')]:
   group='G2A' if 'call' in endpoint else 'G2B';fig,ax=canvas(LABEL[design],ylabel);table=[]
   coverage=group=='G2B'
   if coverage:ax.axhline(.95,color='#666666',ls=(0,(3,2)),lw=.75,zorder=0)
   for m in ORDER:
    f=d[d.method_id==m].sort_values('truth');x=np.arange(len(f));den=f.n_interval_completed if endpoint=='conditional_coverage' else f.n_tasks
    y=f[count].to_numpy()/den.to_numpy();ci=np.array([wilson(int(k),int(n)) for k,n in zip(f[count],den)])
    # Categorical offsets make six interval estimates legible, without moving continuous coordinates.
    xx=x+(ORDER.index(m)-2.5)*.045
    ax.errorbar(xx,y,yerr=np.maximum(0,np.array([y-ci[:,0],ci[:,1]-y])),color=C[m],marker=MARK[m],mfc=C[m] if m=='M' else 'white',mec=C[m],ms=3.5,lw=1.15 if m=='M' else .8,elinewidth=.65,capsize=1.5,label=m,zorder=5 if m=='M' else 2)
    for (_,r),yy,l,u,n in zip(f.iterrows(),y,ci[:,0],ci[:,1],den):table.append({**r.to_dict(),'display_endpoint':endpoint,'display_value':yy,'wilson_low':l,'wilson_high':u,'display_denominator':n})
   effects(ax)
   if coverage:
    finite=[r['wilson_low'] for r in table if np.isfinite(r['wilson_low'])];bottom=max(0,min(finite+[.95])-.04);ax.set_ylim(bottom,1.015)
   elif endpoint=='wrong_call_fraction' and design!='nonidentifying':
    top=max(.02,max(r['wilson_high'] for r in table)*1.18);ax.set_ylim(-.025*top,top)
   else:ax.set_ylim(-.035,1.035);ax.set_yticks([0,.25,.5,.75,1])
   if design=='nonidentifying' and not coverage and int(d.called.sum())==0:
    mdata=d[d.method_id=='M'];annotation=f'All six methods: no sign calls\nM: {int(mdata.solver_failures.sum())} numerical failures / {int(mdata.n_tasks.sum()):,} tasks';ax.text(.5,.58,annotation,ha='center',va='center',transform=ax.transAxes,fontsize=7.2)
   legend(ax)
   save(fig,group+'_'+design+'_'+endpoint,pd.DataFrame(table),'R04',ylabel,'Thin connected estimates, hollow markers, capped intervals and white space from R04. Discrete condition axis with disclosed small category offsets; no smoothing. Correct-call utility uses full probability range; wrong-call rates use a zero-based axis spanning all Monte Carlo intervals; coverage axis spans every displayed interval and the nominal line, no broken axes. Wilson 95% Monte Carlo intervals.')

def risk():
 d=pd.read_csv(G/'results/G2_risk_answerability.tsv',sep='\t')
 for design in DESIGNS:
  f=d[d.design==design].copy();fig,ax=canvas(LABEL[design],'Wrong signs / called replicates',legend=False)
  for m in ['M','B4']:
   v=f[(f.method_id==m)&(f.called>0)].sort_values('confidence');x=v.call_rate.to_numpy();y=v.wrong_sign_among_called.to_numpy()
   if len(v):
    ax.plot(x,y,'o-',color=C[m],mfc='white',ms=4,lw=1,label=m)
    ax.errorbar(x,y,yerr=np.maximum(0,np.array([y-v.wrong_sign_wilson_low,v.wrong_sign_wilson_high-y])),fmt='none',color=C[m],elinewidth=.7,capsize=2)
    a=v[v.confidence==.95];ax.scatter(a.call_rate,a.wrong_sign_among_called,color=C[m],s=20,zorder=5)
  ax.set_xlabel('Called replicates / all replicates');ax.set_xlim(-.025,1.025);riskmax=float(f.wrong_sign_wilson_high.max()) if (f.called>0).any() else 1.;ax.set_ylim(-.025*max(.05,riskmax),min(1.02,max(.05,riskmax*1.25)))
  if (f.called>0).any():ax.legend(loc='upper right');ax.text(.03,.96,'Filled point: 95% candidate interval',transform=ax.transAxes,va='top',fontsize=6.6)
  if not (f.called>0).any():ax.text(.5,.5,'No sign calls\nConditional error is undefined',ha='center',va='center',transform=ax.transAxes,fontsize=8)
  save(fig,'G2C_'+design,f,'R04','Risk versus answerability','All five frozen confidence levels, piecewise joining only; risk is undefined at zero calls; Wilson conditional risk bars. Filled marker marks 95%, no test-set threshold choice.')

def profiles():
 from scipy.stats import chi2
 paths=json.loads((G/'results/G2_bound_profiles.json').read_text())
 for cell,design in [(3,'strong'),(7,'weak'),(11,'nonidentifying')]:
  f=[p for p in paths if p['cell']==cell and p['rep']==0];fig,ax=canvas(LABEL[design],'Profile likelihood ratio',legend=False);rows=[]
  for path,color in zip(f,['#0072B2','#E69F00','#009E73']):
   pts=path['points'];x=np.array([p['c'] for p in pts]);y=2*(np.array([p['nll'] for p in pts])-path['fit_nll']);order=np.argsort(x);x=x[order];y=y[order]
   ax.plot(x,y,'o-',ms=2.3,mfc='white',mew=.6,lw=.95,color=color,label='Bound ±'+str(path['bound']))
   failed=[p for p in pts if not p['success']]
   if failed:ax.scatter([p['c'] for p in failed],[2*(p['nll']-path['fit_nll']) for p in failed],marker='x',s=16,color='#D55E00',zorder=6)
   for p in pts:rows.append({'cell_id':cell,'replicate_id':0,'design':design,'bound':path['bound'],'target':p['c'],'lr':2*(p['nll']-path['fit_nll']),'success':p['success'],'fit_nll':path['fit_nll']})
  ax.axhline(chi2.ppf(.95,1),ls=(0,(3,2)),color='#666666',lw=.7);ax.set_xlim(-6.15,6.15);ax.set_xticks([-6,-3,0,3,6]);ax.set_xlabel('Target interaction');ax.set_ylim(-.2,8);ax.legend(loc='upper left',fontsize=6.8);threshold_label=ax.text(.02,chi2.ppf(.95,1)+.15,'95% candidate threshold',transform=ax.get_yaxis_transform(),fontsize=6.5,va='bottom')
  # Full range is retained as a separate panel. Main view centers the actual LR decision region.
  save(fig,'G2D_'+design+'_full_range',pd.DataFrame(rows),'R06','Full search-bound profile','Exact profile coordinates; full search range and common LR display window. Failed optimizer points remain in data. No smoothing.')
  if design!='nonidentifying':
   near=[r['target'] for r in rows if r['lr']<=8];lo=min(near);hi=max(near);pad=max(.1,(hi-lo)*.15);ax.set_xlim(lo-pad,hi+pad);ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(4));ax.legend(loc='upper center',fontsize=6.8);threshold_label.set_x(.5);threshold_label.set_ha('center')
  else:
   ax.text(.5,.3,'Both signs remain compatible',ha='center',transform=ax.transAxes,fontsize=8)
  save(fig,'G2D_'+design,pd.DataFrame(rows),'R06','Finite or boundary-limited profile confidence set','R06 open-circle exact coordinates with thin connecting lines; three prespecified search bounds, no smoothing. Common vertical window [−0.2, 8] centers the 3.84 decision threshold; all full LR values exported, curves beyond window clipped explicitly. Main x-window is union of LR≤8 points plus15% margin; full search-range companion exported. Independent purpose207 replicate0, effect+0.8.')

def matched():
 d=pd.concat([pd.read_csv(G/f'results/B6_matched_cell{i}.tsv',sep='\t') for i in range(4)]);rows=[]
 for (effect,m),f in d.groupby(['truth','method_id']):
  ok=f.success.astype(bool);covered=f.covers_truth.fillna(False).astype(bool);correct=f.called.astype(bool)&~f.wrong.astype(bool)
  for endpoint,vals,den in [('coverage',covered,int(ok.sum())),('correct_calls',correct,len(f))]:
   k=int(vals.sum());lo,hi=wilson(k,den);rows.append({'truth':effect,'method_id':m,'endpoint':endpoint,'count':k,'denominator':den,'all_replicates':len(f),'solver_failures':int((~ok).sum()),'value':k/den if den else np.nan,'wilson_low':lo,'wilson_high':hi})
 s=pd.DataFrame(rows);s.to_csv(G/'results/B6_matched_summary.tsv',sep='\t',index=False)
 for endpoint,ylabel in [('coverage','Coverage / completed intervals'),('correct_calls','Correct sign calls / all replicates')]:
  fig,ax=canvas('Matched clinical-observation endpoint',ylabel,legend=False)
  if endpoint=='coverage':ax.axhline(.95,color='#666666',ls=(0,(3,2)),lw=.75)
  for m,color,label in [('B6_OFFICIAL','#56B4E9','Official oMHN + Wald'),('M_MATCHED','#0072B2','Equivalent likelihood + profile')]:
   f=s[(s.method_id==m)&(s.endpoint==endpoint)].sort_values('truth');y=f.value.to_numpy();xx=np.arange(4)+(-.025 if m=='B6_OFFICIAL' else .025);ax.errorbar(xx,y,yerr=np.maximum(0,np.array([y-f.wilson_low,f.wilson_high-y])),color=color,marker='o',mfc='white' if m=='B6_OFFICIAL' else color,ms=3.5,lw=1,elinewidth=.7,capsize=2,label=label)
  effects(ax)
  if endpoint=='coverage':ax.set_ylim(max(0,float(s[s.endpoint==endpoint].wilson_low.min())-.04),1.015)
  else:ax.set_ylim(-.035,1.035);ax.set_yticks([0,.25,.5,.75,1])
  ax.legend(loc='lower left',bbox_to_anchor=(-.02,1.10),fontsize=6.8);fig.subplots_adjust(top=.72)
  save(fig,'G2E_'+endpoint,s[s.endpoint==endpoint],'R04',ylabel,'Official matched endpoint only. Open versus filled markers and Wilson intervals, no artificially magnified point-estimate differences. All four effect conditions retained.')

def main():
 summary=pd.read_csv(G/'results/G2_core_summary.tsv',sep='\t');performance(summary);risk();profiles();matched()
 sys.modules.setdefault('g2_plot',sys.modules[__name__])
 from g2_paired_plot import paired
 paired(summary)
 from g2_certified_sign_plot import draw
 draw()
 (G/'configs/figure_manifest.json').write_text(json.dumps(MAN,ensure_ascii=False,indent=2)+'\n');print('Standalone panels',len(MAN))
if __name__=='__main__':main()
