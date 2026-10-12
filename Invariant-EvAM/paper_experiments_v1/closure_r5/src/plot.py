from pathlib import Path
import sys,json,hashlib
import numpy as np,pandas as pd
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib.colors import ListedColormap,BoundaryNorm
from scipy.stats import gaussian_kde
P=Path(__file__).resolve().parents[2];sys.path.insert(0,str(P.parent/'src'));from invariant_evam.figure_style import apply_publication_style
apply_publication_style({'font.family':'Arial','font.size':8,'axes.labelsize':8,'xtick.labelsize':7,'ytick.labelsize':7,'axes.linewidth':.7,'svg.fonttype':'none','pdf.fonttype':42})
C=['#0072B2','#56B4E9','#009E73','#E69F00','#D55E00','#CC79A7'];MAN=[];CAT=json.loads((P.parent/'configs/figure_reference_catalog.json').read_text())['records'];PRE=Path('/tmp/evam_r5_previews');PRE.mkdir(exist_ok=True)
def read(g,name):return pd.read_csv(P/f'g{g}/revision_r5/results/{name}.tsv',sep='\t')
def save(f,g,name,d,ref,scope):
 dp=P/f'g{g}/revision_r5/figure_data/{name}.tsv';d.to_csv(dp,sep='\t',index=False)
 for e in ['pdf','svg']:f.savefig(P/f'g{g}/final_figures/{name}.{e}',dpi=600)
 f.savefig(PRE/(name+'.png'),dpi=160);r=next(r for r in CAT if r['id']==ref)
 MAN.append(dict(panel_id=name,group=g,primary_reference_id=ref,reference_file=r['source_file_absolute'],scope=scope,data=str(dp.relative_to(P)),data_sha256=hashlib.sha256(dp.read_bytes()).hexdigest(),size_mm=(f.get_size_inches()*25.4).tolist(),dpi=600,palette=C,review_status='awaiting_actual_review',user_approved=False));plt.close(f)
def canvas(title,ylabel,w=160,h=114):
 f,a=plt.subplots(figsize=(w/25.4,h/25.4));f.subplots_adjust(left=.14,right=.97,bottom=.25,top=.76);f.text(.14,.95,title,fontsize=9,weight='bold');a.set_ylabel(ylabel);a.spines[['top','right']].set_visible(False);return f,a
def foot(f,txt):f.text(.14,.045,txt,fontsize=6.0,linespacing=1.5)
def g4():
 d=read(4,'external_paired_summary');bt=read(4,'external_bootstrap');order=['B','JOINT_VS_TWO_STEP','JOINT_VS_SHUFFLE','JOINT_VS_POOLED'];f,a=canvas('G4 | Frozen GENIE models on external TCGA records','Paired log-score gain (millinats / record)',168,118);ext=[0]
 for i,comp in enumerate(order):
  for j,(cancer,col) in enumerate([('LUAD',C[0]),('COAD',C[3])]):
   r=d[d.cancer.eq(cancer)&d.comparison.eq(comp)].iloc[0];x=i+(j-.5)*.26;m,lo,hi=1000*r[['gain','lo','hi']].to_numpy(dtype=float);a.bar(x,m,.22,fc=col,ec='#222222',lw=.7,zorder=2);a.errorbar(x,m,yerr=[[m-lo],[hi-m]],fmt='none',ecolor='#222222',lw=.7,capsize=2,zorder=3);z=1000*bt[bt.cancer.eq(cancer)&bt.comparison.eq(comp)&bt.replicate.isin([0,400,800,1200,1600])].gain.to_numpy();a.scatter(x+np.linspace(-.04,.04,len(z)),z,s=8,facecolor='white',edgecolor='#333333',lw=.5,zorder=4);ext.extend([lo,hi,*z])
 a.axhline(0,c='#555555',lw=.65);a.set_xticks(range(4),['Bridge vs more\ninformation value','Bridge vs two-step\nstrong baseline','Bridge vs shuffle\nrecord pairing','Bridge vs pooled\nstrong baseline']);a.set_ylabel('Log-score improvement (millinats / record)');span=max(ext)-min(ext);a.set_ylim(min(ext)-.08*span,max(ext)+.12*span)
 f.legend(handles=[Patch(fc=C[0],ec='#222222',label='TCGA LUAD · n=566'),Patch(fc=C[3],ec='#222222',label='TCGA colorectal · n=534')],loc='upper left',bbox_to_anchor=(.135,.89),ncol=2,fontsize=7,frameon=True,fancybox=False)
 foot(f,'Positive = candidate improves on named comparator; four retained B / baseline comparisons.\nCaps: 95% paired patient/seed bootstrap; points: 5 fixed draws of 2,000.\nGENIE fits frozen; no TCGA tuning. Recorded variants, not verified callable wild type.')
 save(f,4,'G4_R5_external_transfer',d[d.comparison.isin(order)],'R21','Grouped outlined bars, black capped paired bootstrap intervals, five fixed bootstrap dots, top boxed legend. No seed-as-patient significance test; intervals disclose uncertainty.')
 if (P/'g4/revision_r5/results/nested_outer_means.tsv').exists():
  d=read(4,'nested_outer_means');d=d[d.metric.eq('log_score') & d.comparison.eq('B')];f,a=canvas('G4 | Patient uncertainty of repeated-experiment means','Log-score gain (millinats / record)',160,116);f.subplots_adjust(right=.81,top=.84);labels=[]
  for k,(c,comp) in enumerate([(c,comp) for c in ['LUAD','COAD'] for comp in ['B']]):
   q=d[d.cancer.eq(c)&d.comparison.eq(comp)].sort_values('outer');v=q.gain.to_numpy()*100;y=np.linspace(v.min(),v.max(),240);den=gaussian_kde(v)(y);den=.28*den/den.max();col=C[[0,1][k]]
   # Display in millinats, including all paired outer means.
   v=q.gain.to_numpy()*1000;y=np.linspace(v.min(),v.max(),240);den=gaussian_kde(v)(y);den=.28*den/den.max();a.fill_betweenx(y,k,k+den,fc=col,alpha=.8,ec='#333333',lw=.6);a.boxplot([v],positions=[k],widths=.075,patch_artist=True,showfliers=False,boxprops=dict(fc='white',ec='#333333',lw=.6),medianprops=dict(c='#333333',lw=.7),whiskerprops=dict(c='#333333',lw=.6),capprops=dict(c='#333333',lw=.6));z=v[q.outer.to_numpy()%5==0];a.scatter(k-np.linspace(.10,.27,len(z)),z,s=9,c=col,alpha=.75,lw=0);labels.append(f'{comp} · {c}\nn={len(v)}')
  a.axhline(0,c='#666666',ls=(0,(3,3)),lw=.7);a.set_xticks(range(2),labels);a.set_xlim(-.4,1.4);a.spines[['top','right']].set_visible(True);a.spines[['top','right']].set_color('#bbbbbb')
  f.legend(handles=[Patch(fc=C[col],label=label) for col,label in [(0,'B · LUAD'),(1,'B · COAD')]],loc='center left',bbox_to_anchor=(.80,.58),fontsize=6.5,frameon=False,handlelength=.8)
  foot(f,'B: bridge vs more. 50 patient resamples × 8 seeds × 3 designs.\nEach value averages feasible seeds, then gives designs equal weight.\nBox/density: all valid outer blocks; dots: every fifth block. Capacity failures reported.')
  save(f,4,'G4_R5_nested_patient_means',d,'R27','Left sampled points, narrow central white box, right half-density, light closed frame, right legend. Unit is outer patient resample; no treating inner seeds as additional patients.')

def g5():
 d=read(5,'policy_raw');s=read(5,'policy_summary');tests=read(5,'figure_paired_tests');methods=['COUNT_ONLY','RANDOM','MAX_SOURCE','C_OPT','D_OPT','ROBUST_TARGET'];labs=['Count only','Random','Max. source','c-opt','D-opt','Robust target'];cols=[C[1],C[3],C[5],C[2],C[4],C[0]]
 for cost in [4,8]:
  f,a=canvas(f'G5 | Frozen closed-loop decisions · source cost {cost}','Correct sign answers / all datasets (%)',176,116);top=0
  for scenario in range(4):
   for j,(m,col) in enumerate(zip(methods,cols)):
    q=s[s.scenario.eq(scenario)&s.reference_source_cost.eq(cost)&s.policy.eq(m)].iloc[0];x=scenario+(j-2.5)*.13;v=q.correct*100;lo=q.correct_lo*100;hi=q.correct_hi*100;a.bar(x,v,.11,fc=col,ec='#222222',lw=.6,zorder=2);a.errorbar(x,v,yerr=[[v-lo],[hi-v]],fmt='none',ecolor='#222222',lw=.65,capsize=1.4,zorder=3)
    raw=d[d.scenario.eq(scenario)&d.reference_source_cost.eq(cost)&d.policy.eq(m)];batch=raw.groupby(raw.replicate//25).correct.mean()*100;a.scatter(x+np.linspace(-.025,.025,len(batch)),batch,s=4.5,fc='white',ec='#333333',lw=.35,zorder=4);top=max(top,hi,batch.max())
  top=10*np.ceil((top+33)/10);a.set_ylim(0,top);a.axvspan(2.52,3.48,fc='#eeeeee',zorder=0);a.set_xticks(range(4),['Model matched','Pair-selection bias','Biology shift','Outside declared\nselection bound']);a.set_xlim(-.5,3.5)
  for scenario in range(4):
   q=d[d.reference_source_cost.eq(cost)&d.scenario.eq(scenario)&d.policy.isin(methods)]
   peaks=q.groupby(['policy',q.replicate//25]).correct.mean()*100
   upper=s[s.reference_source_cost.eq(cost)&s.scenario.eq(scenario)&s.policy.isin(methods)].correct_hi.max()*100
   floor=max(peaks.max(),upper)+3
   for level,j in enumerate(range(4,-1,-1)):
    m=methods[j];mark=tests[tests.reference_source_cost.eq(cost)&tests.scenario.eq(scenario)&tests.policy.eq(m)].iloc[0]['mark'];x1=scenario+(j-2.5)*.13;x2=scenario+2.5*.13;y=floor+level*5
    a.plot([x1,x1,x2,x2],[y,y+1,y+1,y],lw=.65,c='#222222',clip_on=False)
    a.text((x1+x2)/2,y+1.25,mark,ha='center',va='bottom',fontsize=5.7)
  f.legend(handles=[Line2D([],[],marker='o',ms=4,c='none',mfc=col,mec='#333333',mew=.4,label=lab) for lab,col in zip(labs,cols)],loc='upper left',bbox_to_anchor=(.13,.89),ncol=6,fontsize=6.4,frameon=True,fancybox=False,handlelength=.8,columnspacing=.8,handletextpad=.3)
  foot(f,'150 new datasets per cell; same reference interaction and added cost 1,500 in every arm.\nCaps: Wilson 95% MC intervals; dots: 6 fixed batches of 25, all outcomes retained.\nMarks: paired McNemar vs robust target, Holm per figure; ns >= .05, * < .05, ** < .01, *** < .001.')
  save(f,5,f'G5_R5_closed_loop_cost{cost}',d[d.reference_source_cost.eq(cost)],'R21','Six narrow colored outlined bars, fixed batch points, capped intervals, paired significance above, boxed top legend. Outside-family stress cell remains shaded and explicit.')
 # Reliability scatter, using actual condition-level summaries (8 conditions x robust / c-opt / count).
 keep=['COUNT_ONLY','C_OPT','ROBUST_TARGET'];q=s[s.policy.isin(keep)].copy();f=plt.figure(figsize=(142/25.4,150/25.4));a=f.add_axes([.16,.22,.65,.59]);top=f.add_axes([.16,.83,.65,.145]);cax=f.add_axes([.835,.22,.025,.59]);colors=[C[1],C[2],C[0]]
 left=min(90,float(q.coverage_end_lo.min()*100)-1);right=100.4;top.hist(q.coverage*100,bins=np.linspace(left,right,12),color=C[5],alpha=.55,ec='white',lw=.3);tw=top.twinx();x=np.linspace(left,right,250)
 if q.coverage.std()>1e-10:tw.plot(x,gaussian_kde(q.coverage*100)(x),c='#333333',lw=.8)
 tw.set_ylabel('Density',fontsize=6.4);tw.tick_params(labelsize=6,direction='in');top.set_ylabel('Conditions',fontsize=6.4);top.set_xlim(left,right);top.tick_params(labelbottom=False,direction='in',top=True,labelsize=6);top.text(.02,.82,'a',transform=top.transAxes,weight='bold')
 for j,m in enumerate(keep):
  for row in q[q.policy.eq(m)].itertuples():
   a.errorbar(row.coverage*100,row.wrong*100,xerr=[[100*(row.coverage-row.coverage_end_lo)],[100*(row.coverage_end_hi-row.coverage)]],yerr=[[100*(row.wrong-row.wrong_lo)],[100*(row.wrong_hi-row.wrong)]],fmt='none',ecolor=colors[j],alpha=.5,capsize=1.5,lw=.6,zorder=1);a.scatter(row.coverage*100,row.wrong*100,marker='^' if row.scenario==3 else 'o',s=20,fc='white',ec=colors[j],lw=.8,zorder=2)
 a.axvline(95,c='#888888',ls=':',lw=.7);a.axhline(5,c='#888888',ls=':',lw=.7);a.set(xlim=(left,right),ylim=(-.25,max(6,float(q.wrong_hi.max()*100)+.5)),xlabel='End-to-end interval coverage (%)',ylabel='Wrong sign answers / all datasets (%)');a.tick_params(direction='in',top=True,right=True);a.text(.02,.94,'b',transform=a.transAxes,weight='bold')
 a.legend(handles=[Line2D([],[],marker='o',mfc='white',mec='#333333',c='none',label='Within family'),Line2D([],[],marker='^',mfc='white',mec='#333333',c='none',label='Outside family')],loc='upper right',fontsize=6,frameon=False)
 cb=matplotlib.colorbar.ColorbarBase(cax,cmap=ListedColormap(colors),norm=BoundaryNorm(np.arange(4)-.5,3),boundaries=np.arange(4)-.5,ticks=range(3));cb.ax.set_yticklabels(['Count only','c-opt','Robust target'],rotation=90,va='center',fontsize=6.5);cb.ax.tick_params(length=0)
 f.text(.16,.12,'8 frozen conditions × 3 displayed methods; all 7 policies in source tables.',fontsize=6)
 f.text(.16,.092,'Crossbars: Wilson 95% MC intervals; overlaps retained without jitter.',fontsize=6)
 f.text(.16,.064,'Numerical failures count as uncovered; no-action and other baselines retained.',fontsize=6)
 f.text(.16,.036,'Dashed guides: 95% coverage and 5% unconditional wrong answers.',fontsize=6)
 save(f,5,'G5_R5_reliability',q,'R12','Top count/density axes, bottom both-axis uncertainty scatter, right discrete color strip, closed frame, a/b labels. Main utility figures retain all six active methods; reliability foregrounds diagnostic vs point-optimal and count controls.')
if __name__=='__main__':
 for arg in sys.argv[1:]:globals()[arg]()
 mp=P/'closure_r5/configs/FIGURE_MANIFEST.json'
 old=json.loads(mp.read_text()) if mp.exists() else []
 changed={r['panel_id'] for r in MAN}
 mp.write_text(json.dumps([r for r in old if r['panel_id'] not in changed]+MAN,indent=2))
