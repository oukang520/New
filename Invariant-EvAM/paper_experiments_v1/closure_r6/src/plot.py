from pathlib import Path
import sys,json,hashlib
import numpy as np,pandas as pd
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import binomtest
P=Path(__file__).resolve().parents[2];R=P.parent;O=P/'closure_r6';sys.path.insert(0,str(R/'src'))
from invariant_evam.figure_style import apply_publication_style
apply_publication_style({'font.family':'Arial','font.size':7,'axes.labelsize':8,'xtick.labelsize':6.5,'ytick.labelsize':6.5,'axes.linewidth':.7,'svg.fonttype':'none','pdf.fonttype':42})
C=['#0072B2','#56B4E9','#009E73','#E69F00','#D55E00','#CC79A7'];CAT=json.loads((R/'configs/figure_reference_catalog.json').read_text())['records'];manifest=[]
sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
def holm(ps):
 p=np.array(ps);out=np.zeros(len(p));prev=0
 for j,i in enumerate(np.argsort(p)):prev=max(prev,min(1,p[i]*(len(p)-j)));out[i]=prev
 return out
def stars(p):return '***' if p<.001 else '**' if p<.01 else '*' if p<.05 else 'ns'
def wilson(x,n):
 ph=x/n;z=1.959964;mid=(ph+z*z/(2*n))/(1+z*z/n);half=z*np.sqrt(ph*(1-ph)/n+z*z/(4*n*n))/(1+z*z/n);return mid-half,mid+half
def save(f,name,ref,d,notes):
 g=name[:2].lower();path=P/g/'revision_r6/figure_data'/f'{name}.tsv';d.to_csv(path,sep='\t',index=False)
 for ext in ['pdf','svg']:f.savefig(P/g/'final_figures'/f'{name}.{ext}',dpi=600)
 rr=next(x for x in CAT if x['id']==ref);manifest.append(dict(panel_id=name,group=g,plot_data=str(path.relative_to(P)),data_sha256=sha(path),primary_reference_id=ref,reference_file=rr['source_file_absolute'],reference_sha256=sha(Path(rr['source_file_absolute'])),size_mm=(f.get_size_inches()*25.4).tolist(),dpi=600,palette=C,notes=notes,review_status='awaiting_actual_export_review',user_approved=False));plt.close(f)
def bracket(a,x1,x2,y,h,label):a.plot([x1,x1,x2,x2],[y,y+h,y+h,y],c='#333333',lw=.6,clip_on=False);a.text((x1+x2)/2,y+h+.35,label,ha='center',va='bottom',fontsize=5.5)

def g4():
 G=P/'g4/revision_r6/results';ext=pd.read_csv(G/'attribution_summary.tsv',sep='\t');inte=pd.read_csv(G/'internal_patient_summary.tsv',sep='\t');inte['cohort']='GENIE';ext['cohort']='TCGA';inte.contrast=inte.contrast.replace({'deassociated_gain':'margin'});d=pd.concat([inte,ext],ignore_index=True);order=['total','margin','pairing','optimizer','model'];d=d[d.contrast.isin(order)].copy();boots=pd.concat([pd.read_csv(G/'internal_patient_bootstrap.tsv',sep='\t').assign(cohort='GENIE'),pd.read_csv(G/'attribution_bootstrap.tsv',sep='\t').assign(cohort='TCGA')]);boots.contrast=boots.contrast.replace({'deassociated_gain':'margin'})
 ps=[]
 for r in d.itertuples():
  v=boots[(boots.cancer==r.cancer)&(boots.cohort==r.cohort)&(boots.contrast==r.contrast)].gain.to_numpy();ps.append((1+np.sum(abs(v-r.gain)>=abs(r.gain)))/(len(v)+1))
 d['p_bootstrap']=ps;d['p_holm']=holm(ps);d.to_csv(G/'figure_tests.tsv',sep='\t',index=False)
 cells=pd.read_csv(G/'attribution_by_design.tsv',sep='\t').assign(cohort='TCGA');v=pd.read_csv(G/'internal_raw.tsv',sep='\t').pivot_table(index=['cancer','design','replicate'],columns='method',values='loss');rows=[]
 for (cancer,design),q in v.groupby(level=[0,1]):
  for contrast,a,b in [('total','MORE','ALIGNED'),('margin','MORE','DEASSOCIATED'),('pairing','DEASSOCIATED','ALIGNED'),('optimizer','TWO_STEP','ALIGNED'),('model','POOLED','ALIGNED')]:rows.append(dict(cancer=cancer,design=design,contrast=contrast,gain=float((q[a]-q[b]).mean()),cohort='GENIE'))
 cells=pd.concat([cells,pd.DataFrame(rows)]);cells.to_csv(P/'g4/revision_r6/figure_data/G4_R6_attribution_design_points.tsv',sep='\t',index=False)
 f=plt.figure(figsize=(184/25.4,119/25.4));a=f.add_axes([.105,.25,.865,.61]);groups=[('LUAD','GENIE',C[0]),('LUAD','TCGA',C[1]),('COAD','GENIE',C[3]),('COAD','TCGA',C[4])]
 for j,(ca,co,col) in enumerate(groups):
  q=d[(d.cancer==ca)&(d.cohort==co)].set_index('contrast').loc[order];x=np.arange(5)+(j-1.5)*.18;y=q.gain.to_numpy()*1000
  a.bar(x,y,width=.155,color=col,alpha=.8,edgecolor='#333333',lw=.6,label=ca+' · '+co,zorder=2);a.errorbar(x,y,yerr=[y-q.lo.to_numpy()*1000,q.hi.to_numpy()*1000-y],fmt='none',ecolor='#333333',lw=.65,capsize=2,zorder=4)
  for k,name in enumerate(order):
   pts=cells[(cells.cancer==ca)&(cells.cohort==co)&(cells.contrast==name)].gain.to_numpy()*1000;a.scatter(x[k]+np.linspace(-.035,.035,len(pts)),pts,s=9,facecolors='white',edgecolors='#333333',lw=.4,zorder=5)
   top=max(q.loc[name,'hi']*1000,pts.max());yy=top+3+(j%2)*1.4;a.plot([x[k],x[k]],[top+.7,yy],color='#333333',lw=.5);a.annotate(stars(q.loc[name,'p_holm']),(x[k],yy),xytext=(0,3),textcoords='offset points',va='bottom',ha='center',fontsize=5.6)
 a.axhline(0,color='#666666',lw=.65);a.set(xticks=np.arange(5),xticklabels=['Total\nbridge gain','Gain after\ndeassociation','Pairing-dependent\ncontribution','Joint vs\ntwo-step','Joint vs\npooled'],ylabel='Hidden-event log-score gain (×10⁻³)',ylim=(-23,45));a.spines[['top','right']].set_visible(False);a.legend(loc='lower left',bbox_to_anchor=(-.01,1.015),ncol=4,fontsize=6.5,frameon=True,edgecolor='#bbbbbb',handlelength=1.1,columnspacing=1)
 f.text(.105,.125,'Total = gain after deassociation + pairing-dependent contribution (identical draws).',fontsize=6.5)
 f.text(.105,.088,'Bars: equal-design means; intervals: paired patient bootstrap; dots: all three design means.',fontsize=6.2)
 f.text(.105,.052,'Elbow labels: gain vs 0, 20-test Holm family. GENIE 449 / 287; TCGA 566 / 534 patients.',fontsize=6.2)
 save(f,'G4_R6_attribution','R21',d,['Grouped narrow colored bars, open points, capped uncertainty, top boxed legend, upright comparison labels with scalar-null leaders.','Negative contributions retained. Points are design means, not bootstrap patients.','Deassociation preserves both measured margins; contribution is intervention-dependent, not a unique causal decomposition.'])

def g5():
 G=P/'g5/revision_r6/results';d=pd.read_csv(G/'policy_raw.tsv',sep='\t');s=pd.read_csv(G/'policy_summary.tsv',sep='\t');paired=pd.read_csv(G/'paired_summary.tsv',sep='\t');labs=['A-rich · no bias','B-rich · no bias','A-rich · positive bias','B-rich · positive bias','A-rich · negative bias','B-rich · negative bias','A-rich · outside family','B-rich · outside family']
 # R28: two aligned horizontal compositions and a third aligned point/interval axis.
 f=plt.figure(figsize=(172/25.4,120/25.4));a=f.add_axes([.29,.25,.21,.59]);b=f.add_axes([.54,.25,.21,.59]);c=f.add_axes([.79,.25,.17,.59]);plot=[]
 acts=['COMPLETE_A','COMPLETE_B','NEW_SOURCE','MORE_SELECTED'];outcome=['correct','wrong','unknown'];q=d[d.policy.eq('DIAGNOSTIC')]
 for case in range(8):
  v=q[q.case==case];y=7-case;left=0
  for j,ac in enumerate(acts):
   prop=float(v.action.eq(ac).mean());a.barh(y,prop,left=left,height=.78,color=C[j],edgecolor='white',lw=.4);left+=prop;plot.append(dict(case=case,component=ac,value=prop))
  probs=[v.correct.mean(),v.wrong.mean(),1-v.called.mean()];left=0
  for j,(n,prop) in enumerate(zip(outcome,probs)):b.barh(y,prop,left=left,height=.78,color=[C[2],C[4],C[1]][j],edgecolor='white',lw=.4);left+=prop;plot.append(dict(case=case,component=n,value=prop))
  val=v.covered.mean()*100;lo,hi=np.array(wilson(v.covered.sum(),len(v)))*100;c.errorbar(val,y,xerr=[[val-lo],[hi-val]],fmt='o',ms=3.2,mfc='#333333',mec='#333333',ecolor='#333333',lw=.6,capsize=1.6);plot.append(dict(case=case,component='coverage',value=val,lo=lo,hi=hi))
 for ax in [a,b,c]:ax.set_ylim(-.7,7.7);ax.set_yticks(range(8));ax.tick_params(axis='y',length=0);ax.grid(axis='x',color='#dddddd',lw=.45,zorder=0)
 a.set_yticklabels(labs[::-1]);b.set_yticklabels([]);c.set_yticklabels([])
 for ax in [a,b]:ax.set_xlim(0,1);ax.set_xticks([0,.5,1]);ax.set_xlabel('Proportion')
 c.set_xlim(70,102);c.set_xticks([75,85,95]);c.set_xlabel('Coverage (%)');c.axvline(95,color='#777777',lw=.7,ls='--');a.set_title('Acquired action',fontsize=8,pad=8);b.set_title('Inference outcome',fontsize=8,pad=8);c.set_title('95% interval',fontsize=8,pad=8)
 from matplotlib.patches import Patch
 f.legend([Patch(fc=C[j]) for j in range(4)],['Complete A','Complete B','New source','More selected'],loc='upper center',bbox_to_anchor=(.59,.965),ncol=4,fontsize=6.1,frameon=False,handlelength=1.1,columnspacing=1)
 f.legend([Patch(fc=x) for x in [C[2],C[4],C[1]]],['Correct','Wrong','Unknown'],loc='lower center',bbox_to_anchor=(.64,.115),ncol=3,fontsize=6.5,frameon=False,handlelength=1.2)
 f.text(.06,.092,'150 independent datasets per row; 600 added assays; theta[4] is unchanged throughout.',fontsize=6.4);f.text(.06,.055,'Coverage whiskers: Wilson intervals. Outside-family controls retained; unknown includes invalid answers.',fontsize=6.1)
 save(f,'G5_R6_diagnostic_actions','R28',pd.DataFrame(plot),['Two aligned stacked-composition blocks and one aligned point/interval block; framed axes and light grid.','Public mask/cost diagnosis; includes two outside-family rows. No mechanism truth supplied to policy.'])
 # R29: hollow gain bars, all six condition means, paired uncertainty, scalar-null elbow labels.
 refs=['BEST_FIXED','COMPLETE_A','RANDOM','SHUFFLED_DIAGNOSIS','NEW_SOURCE','D_OPT','LARGEST_PANEL'];labels=['Fixed B','Fixed A','Random','Shuffled\ndiagnosis','New\nsource','D-opt','Largest\npanel'];q=paired[(paired.metric=='squared_error')].set_index('reference').loc[refs].reset_index();f=plt.figure(figsize=(177/25.4,119/25.4));a=f.add_axes([.105,.24,.865,.61]);base=s[s.policy=='DIAGNOSTIC'].set_index('case');points=[]
 for j,r in enumerate(q.itertuples()):
  col=C[j%6];a.bar(j,r.gain,width=.62,facecolor='white',edgecolor=col,lw=1.6,zorder=2);a.errorbar(j,r.gain,yerr=[[r.gain-r.lo],[r.hi-r.gain]],fmt='none',ecolor='#333333',capsize=3,lw=.8,zorder=4)
  vals=(s[(s.policy==r.reference)&(s.case<6)].set_index('case').mse-base.loc[range(6),'mse']).values;a.scatter(j+np.linspace(-.15,.15,6),vals,s=12,c=col,ec='white',lw=.3,zorder=5);points.extend(dict(reference=r.reference,case=k,gain=v) for k,v in enumerate(vals))
  yy=max(r.hi,max(vals))+.06;a.text(j,yy,stars(r.p_holm),ha='center',va='bottom',fontsize=6.4)
 a.axhline(0,c='#888888',lw=.65);a.set(xticks=range(len(refs)),xticklabels=labels,ylabel='Target squared-error reduction',ylim=(-.12,max(max(q.hi),max(x['gain'] for x in points))+.27));a.spines[['top','right']].set_visible(False)
 a.text(.99,.99,'vs zero; Holm-adjusted',transform=a.transAxes,ha='right',va='top',fontsize=6.5,color='#333333')
 f.text(.105,.94,'Diagnostic acquisition relative to each comparator',fontsize=8,weight='bold');f.text(.105,.125,'Positive = lower error; 900 paired datasets. Dots: all six predeclared condition means.',fontsize=6.3);f.text(.105,.085,'Whiskers: stratified paired 95% bootstrap intervals; Centered labels: paired sign-flip tests against zero, Holm-adjusted.',fontsize=6.1);f.text(.105,.047,'C-opt equals diagnostic. Expansion control: correct-answer figure and full result table.',fontsize=6.2)
 pd.DataFrame(points).to_csv(P/'g5/revision_r6/figure_data/G5_R6_diagnostic_value_points.tsv',sep='\t',index=False);save(f,'G5_R6_diagnostic_value','R29',q,['Open colored bars, six actual condition-mean points, capped paired intervals, centered scalar-null significance.','No assertion of novel c-opt criterion; largest-panel control explicitly shown.'])
 # R21: each cell's correctness, with raw nonoverlapping block proportions and Wilson intervals.
 policies=['DIAGNOSTIC','BEST_FIXED','SHUFFLED_DIAGNOSIS','RANDOM','D_OPT','COUNT_ONLY'];names=['Diagnostic / c-opt','Fixed B','Shuffled diagnosis','Random','D-opt','More selected'];tests=[]
 for case in range(8):
  pivot=d[d.case==case].pivot(index='replicate',columns='policy',values='correct').astype(int)
  for pol in policies[1:]:
   delta=pivot.DIAGNOSTIC-pivot[pol];aa=int((delta>0).sum());bb=int((delta<0).sum());pv=binomtest(aa,aa+bb,.5).pvalue if aa+bb else 1.;tests.append(dict(case=case,policy=pol,p_raw=pv))
 tests=pd.DataFrame(tests);tests['p_holm']=holm(tests.p_raw);tests.to_csv(G/'figure_tests.tsv',sep='\t',index=False)
 f=plt.figure(figsize=(195/25.4,129/25.4));a=f.add_axes([.09,.27,.885,.58]);points=[];plots=[]
 for case in range(8):
  tops=[]
  for j,pol in enumerate(policies):
   v=d[(d.case==case)&(d.policy==pol)].sort_values('replicate');mean=v.correct.mean()*100;lo,hi=np.array(wilson(v.correct.sum(),len(v)))*100;x=case+(j-2.5)*.125;tops.append(hi)
   a.bar(x,mean,width=.112,color=C[j],alpha=.8,edgecolor='#333333',lw=.45,label=names[j] if case==0 else None);a.errorbar(x,mean,yerr=[[mean-lo],[hi-mean]],fmt='none',ecolor='#333333',lw=.55,capsize=1.3)
   for k,part in enumerate(np.array_split(v.correct.to_numpy(),3)):pt=float(part.mean()*100);a.scatter(x+(k-1)*.022,pt,s=6,fc='white',ec='#333333',lw=.3,zorder=5);points.append(dict(case=case,policy=pol,block=k,value=pt))
   plots.append(dict(case=case,policy=pol,correct=mean,lo=lo,hi=hi,n=len(v)))
  top=max(max(tops),max(z['value'] for z in points if z['case']==case))+2
  for j,pol in enumerate(policies[1:],1):bracket(a,case-2.5*.125,case+(j-2.5)*.125,top+(j-1)*4,1.2,stars(tests[(tests.case==case)&(tests.policy==pol)].p_holm.iloc[0]))
 a.set(xticks=range(8),xticklabels=['A-rich\nNo bias','B-rich\nNo bias','A-rich\nPositive','B-rich\nPositive','A-rich\nNegative','B-rich\nNegative','A-rich\nOutside','B-rich\nOutside'],ylabel='Correct sign answers (%)',ylim=(0,max(z['value'] for z in points)+39));a.spines[['top','right']].set_visible(False);a.axvspan(5.5,7.5,color='#eeeeee',alpha=.45,zorder=-1);a.legend(loc='lower left',bbox_to_anchor=(-.01,1.02),ncol=3,frameon=True,fontsize=6.3,edgecolor='#bbbbbb',handlelength=1,columnspacing=1.2)
 f.text(.09,.13,'150 independent datasets / condition; dots: three fixed, disjoint blocks of 50 datasets.',fontsize=6.3);f.text(.09,.087,'Bars: correct-answer rates; whiskers: Wilson 95% intervals; brackets: paired McNemar, 40-test Holm.',fontsize=6.1);f.text(.09,.047,'Grey: outside the fitted selection family. Reliability, wrong answers and unknowns shown in companion figure.',fontsize=6.1)
 pd.DataFrame(points).to_csv(P/'g5/revision_r6/figure_data/G5_R6_correct_answers_points.tsv',sep='\t',index=False);save(f,'G5_R6_correct_answers','R21',pd.DataFrame(plots),['Dense grouped narrow bars, open points, capped uncertainty, boxed legend, all pairwise-to-diagnostic brackets.','No patient-style scatter for binary data: predeclared three disjoint block means shown; tests use all150 paired datasets.'])

if __name__=='__main__':
 if 'g4' in sys.argv:g4()
 if 'g5' in sys.argv:g5()
 mf=O/'configs/FIGURE_MANIFEST.json';old=json.loads(mf.read_text()) if mf.exists() else [];names={r['panel_id'] for r in manifest};old=[r for r in old if r['panel_id'] not in names];mf.write_text(json.dumps(old+manifest,ensure_ascii=False,indent=2));print('Exported',len(manifest),'PDF/SVG pairs')
