"""Display-only R28 compaction and data-supported R04 budget trajectories."""
import g5_plot as p
import json,hashlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
G=p.G; S=['RANDOM_FEASIBLE','COUNT_ONLY','INVARIANT_ROBUST']; COLORS=[p.C[0],p.C[1],p.C[5]]
LABELS=['Random','Count only','Max. panel = D = c = robust']
def compact():
 d=pd.read_csv(G/'figure_data/G5_01_natural_audit.tsv',sep='\t');f=plt.figure(figsize=(137/25.4,54/25.4)); axes=[f.add_axes([x,.33,w,.39]) for x,w in [(.17,.18),(.395,.18),(.62,.17)]]
 for a in axes:
  p.frame(a);a.set_ylim(2.5,-.5);a.set_yticks(range(3));a.set_yticklabels([]);a.tick_params(length=2,labelsize=6);a.grid(axis='x',color='#dddddd',lw=.35);a.set_axisbelow(True)
 for i,z in d.iterrows():
  for a,keys,cs in [(axes[0],['burden0','burden1','burden2','burden3plus'],p.C[:4]),(axes[1],['female','male','unknown_sex'],[p.C[5],p.C[4],'white'])]:
   left=0
   for key,c in zip(keys,cs):a.barh(i,z[key]/z.n,left=left,height=.65,color=c,ec='#555555',lw=.25);left+=z[key]/z.n
  axes[2].plot(z.n/1000,i,'o',c='#222222',ms=3);axes[2].annotate(f'{z.n:,}',(z.n/1000,i),xytext=(5,0),textcoords='offset points',va='center',fontsize=6.4)
 axes[0].set_yticklabels(['IMPACT468','IMPACT410','IMPACT505'],fontsize=6.7)
 for a,title in zip(axes,['Module burden','Recorded sex','Patients (×1,000)']):a.set_title(title,fontsize=7.3,pad=5)
 for a in axes[:2]:a.set_xlim(0,1);a.set_xticks([0,.25,.5,.75,1]);a.tick_params(axis='x',rotation=90);a.set_xlabel('Proportion',fontsize=6.4,labelpad=2)
 axes[2].set_xlim(0,3.8);axes[2].set_xticks([0,1,2,3])
 for labels,cs,y,title in [(['0','1','2','3 or 4'],p.C[:4],.85,'Events'),(['Female','Male','Unknown'],[p.C[5],p.C[4],'white'],.46,'Sex')]:
  f.legend(handles=[Patch(fc=c,ec='#555555',lw=.3,label=l) for c,l in zip(cs,labels)],loc='upper left',bbox_to_anchor=(.815,y),fontsize=5.8,title=title,title_fontsize=6.4,frameon=False,handlelength=.8,labelspacing=.2,borderpad=.1)
 f.text(.045,.94,'Natural cohort | identical four-target coverage',fontsize=8.3,weight='bold');f.text(.045,.055,'4,920 patients · TP53 / KRAS / STK11 / KEAP1 · proxy negatives; no sample-level callability.',fontsize=6)
 p.save(f,'G5_01_natural_audit',d,'R28','Compact natural cohort composition and denominator audit','137 × 54 mm; aligned stacked columns and count dots, framed axes, quarter ticks and two right legends retained. No data filtering.')
def template(title):
 f=plt.figure(figsize=(148/25.4,126/25.4));a=f.add_axes([.14,.23,.81,.60]);a.spines[['top','right']].set_visible(False);a.spines['left'].set_bounds(-.19,.70);a.set_xlim(0,54);a.set_ylim(-.19,1.20);a.set_xticks([0,10,25,50]);a.set_yticks([0,.2,.4,.6]);a.set_xlabel('Acquisition budget cap (% of base cost)',fontsize=8);a.set_ylabel('Independent log-score gain ↑',fontsize=8)
 a.axhline(0,c='#999999',lw=.6);a.axvline(100/3,c='#777777',ls=(0,(3,3)),lw=.7,ymax=.62);a.axvspan(100/3,54,ymax=.62,color='#f0f0f0',zorder=0)
 f.text(.14,.96,title,fontsize=8.4,weight='bold');f.legend(handles=[Patch(fc=c,ec='#222222',lw=.7,label=l) for c,l in zip(COLORS,LABELS)],loc='upper left',bbox_to_anchor=(.13,.925),ncol=3,frameon=False,fontsize=6.8,handlelength=1.1,columnspacing=1)
 # Insets are integrated within the upper main plotting rectangle, as in R04.
 ins=[a.inset_axes([x,.72,.30,.24]) for x in [.075,.625]]
 for b in ins:
  b.spines[['top','right']].set_visible(False);b.tick_params(labelsize=5.8,length=2,pad=1);b.set_yticks(range(3));b.set_yticklabels(['Random','Count','4 aliases'],fontsize=5.7);b.set_ylim(2.6,-.6)
 return f,a,ins

def gainlines(a,d):
 for s,c in zip(S,COLORS):
  vals=[];errs=[]
  for budget in [30,75,150]:
   v=d[d.strategy.eq(s)&d.budget.eq(budget)].source_log_score_gain.dropna();vals.append(v.mean());errs.append(1.96*v.std(ddof=1)/np.sqrt(len(v)))
  a.errorbar([0,10,25,50],[0]+vals,yerr=[0]+errs,color=c,lw=2.2,marker='o',ms=5.8,mec='#222222',mew=1,ecolor='#222222',elinewidth=.9,capsize=3,zorder=4)

def wilson(k,n):
 z=1.96;q=k/n;den=1+z*z/n;mid=(q+z*z/(2*n))/den;half=z*np.sqrt(q*(1-q)/n+z*z/(4*n*n))/den;return mid-half,mid+half

def outer():
 raw=p.read('G5_design_outer_evaluation');names=['Correct selection family','Common omitted pair selection','Biology shift (+0.2)']
 for sc in range(3):
  d=raw[raw.scenario.eq(sc)].copy();f,a,ins=template('Independent simulation | '+names[sc]);gainlines(a,d);tab=[]
  for j,(s,c) in enumerate(zip(S,COLORS)):
   q=d[d.strategy.eq(s)&d.budget.eq(150)];complete=q[q.profile_status.eq('COMPLETE')]
   for ax,key,k,n in [(ins[0],'finite',q.finite_interval.sum(),len(q)),(ins[1],'coverage',complete.covers_truth.sum(),len(complete))]:
    v=k/n;lo,hi=wilson(k,n);ax.barh(j,v,height=.52,color=c,ec='#222222',lw=.6);ax.errorbar(v,j,xerr=[[v-lo],[hi-v]],fmt='none',ecolor='#222222',capsize=2,lw=.7);ax.text(1.10,j,f'{v:.1%}',fontsize=5.6,va='center',ha='left',color='black',clip_on=False);tab.append(dict(scenario=sc,strategy=s,budget=150,metric=key,n=n,k=int(k),value=v,lo=lo,hi=hi))
  for ax,title in zip(ins,['Finite interval | 50% budget','Coverage | 50% budget']):ax.set_xlim(0,1.0);ax.set_xticks([0,.5,1],['0','50','100%']);ax.set_title(title,fontsize=6.5,pad=4)
  ins[1].axvline(.95,c='#555555',lw=.6,ls=(0,(2,2)))
  coverage='78.0%' if sc==1 else '95.5%';wrong='16 / 66' if sc==1 else ('2 / 34' if sc==0 else '3 / 30')
  f.text(.14,.135,f'Four-alias coverage at 50%: {coverage}; wrong / called: {wrong}.',fontsize=7,weight='bold',color=p.C[4] if sc==1 else '#222222')
  f.text(.14,.065,'200 datasets per point; mean ± 1.96 MCSE. Insets: Wilson intervals.\n0% is the exact no-acquisition contrast. Dashed boundary: remeasurement capacity.\nFour aliases have identical actions; no extra replicates. All outcome counts in source tables.',fontsize=6.1,linespacing=1.4)
  pd.DataFrame(tab).to_csv(G/'figure_data'/f'G5_05_design_scenario_{sc}_insets.tsv',sep='\t',index=False)
  p.save(f,f'G5_05_design_scenario_{sc}',d,'R04','Budget-dependent independent prediction gain, with endpoint finite fraction and coverage','R04 solid colored trajectories, black-edged filled circles, black capped errors, horizontal legend and two integrated horizontal-bar insets. No smoothed curves, invented phases or significance. Real capacity boundary at 33 1/3%; zero budget is algebraic. All six strategies and failures retained in plot data; four exact action aliases shown once.')

def direct():
 raw=p.read('G5_design_direct_evaluation');d=raw[raw.strategy.isin(S)].copy();d['displayed_point']=d.replicate.isin(p.DISPLAY['bars']);d['exact_alias_members']=d.strategy.map(lambda s:'MAX_PANEL|D_OPT|C_OPT|INVARIANT_ROBUST' if s==S[2] else s);f,a,ins=template('Direct records | acquisition budget and prediction gain');gainlines(a,d);a.set_ylabel('Source-record log-score gain ↑',fontsize=8);rows=[]
 for ax,budget in zip(ins,[30,150]):
  base=d[d.budget.eq(budget)&d.strategy.eq('COUNT_ONLY')].set_index('replicate').source_log_score_gain
  for j,(s,c) in enumerate(zip(S,COLORS)):
   q=d[d.budget.eq(budget)&d.strategy.eq(s)].sort_values('replicate');v=q.source_log_score_gain.to_numpy()-base.loc[q.replicate].to_numpy();m=v.mean();err=1.96*v.std(ddof=1)/np.sqrt(len(v));ax.barh(j,m,height=.52,color=c,ec='#222222',lw=.6);ax.errorbar(m,j,xerr=err,fmt='none',ecolor='#222222',capsize=2,lw=.7,zorder=5);mask=q.displayed_point.to_numpy();jitter=np.random.default_rng(5206).uniform(-.15,.15,len(v));ax.scatter(v[mask],j+jitter[mask],s=4,facecolor='white',edgecolor='#333333',lw=.3,zorder=4)
   rows.extend(dict(budget=budget,strategy=s,replicate=int(r),paired_gain=float(x),displayed_point=bool(show)) for r,x,show in zip(q.replicate,v,mask))
  ax.set_title(f'Gain vs count | {budget/3:.0f}% budget',fontsize=6.5,pad=4);ax.set_xlim(-.18,.69);ax.set_xticks([0,.3,.6]);ax.axvline(0,c='#777777',lw=.5)
 f.text(.14,.135,'At 50% budget: four aliases +0.428; random +0.422.',fontsize=7,weight='bold');f.text(.14,.065,'50 allocation seeds; mean ± 1.96 seed-MCSE; 15 paired dots shown per inset row.\nSame 449 evaluation patients: uncertainty is conditional, not a patient confidence interval.\n0% contrast = 0. Dashed boundary: remeasurement capacity; cap is not actual spending.',fontsize=6.1,linespacing=1.4)
 pd.DataFrame(rows).to_csv(G/'figure_data/G5_06_direct_record_gain_insets.tsv',sep='\t',index=False)
 p.save(f,'G5_06_direct_record_gain',d,'R04','Direct-record budget trajectory and paired gain over count-only acquisition','R04 colored lines, black-edged circles, capped seed-MCSE, two integrated bar-and-raw-dot insets, top legend. Fixed 15/50 display seeds; all 50 used in means. Exact aliases collapsed without multiplying N. Zero budget is defined contrast, no interpolation beyond straight segments.')

def main():
 old=json.loads((G/'configs/FIGURE_MANIFEST.json').read_text());compact();outer();direct();updates={x['panel_id']:x for x in p.MAN};
 for x in updates.values():
  x['catalog_reference_sha256']=x['reference_sha256'];x['reference_sha256']=hashlib.sha256(p.Path(x['reference_file']).read_bytes()).hexdigest()
 (G/'configs/FIGURE_MANIFEST.json').write_text(json.dumps([updates.get(x['panel_id'],x) for x in old],ensure_ascii=False,indent=2));print('Revised five pairs; other five unchanged.')
if __name__=='__main__':main()
