from pathlib import Path
import json,sys,hashlib
import numpy as np,pandas as pd
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
O=Path(__file__).resolve().parents[1];P=O.parent;R=P.parent
sys.path.insert(0,str(R/'src'))
from invariant_evam.figure_style import apply_publication_style
apply_publication_style({'font.family':'Arial','font.size':7.5,'svg.fonttype':'none','pdf.fonttype':42})
C=['#0072B2','#56B4E9','#009E73','#E69F00','#D55E00','#CC79A7'];K='#333333'
S=O/'staging';D=O/'publication_data';S.mkdir(exist_ok=True);D.mkdir(exist_ok=True)
Q=pd.read_csv(O/'research_traceability/confirmation_transport.tsv',sep='\t');V=pd.read_csv(O/'research_traceability/confirmation_transport_by_design.tsv',sep='\t')
names={'luad_oncosg_2020':'OncoSG','luad_cas_2020':'CAMS','luad_cptac_2020':'CPTAC-LUAD','luad_broad':'Broad-LUAD','lung_nci_2022':'NCI','coad_silu_2022':'SIDRA-LUMC','coadread_dfci_2016':'NHS/HPFS','coadread_genentech':'Genentech','coadread_cass_2020':'CASS'}
manifest=[]
def tidy(a):
 a.spines[['top','right']].set_visible(False);a.tick_params(direction='out',length=3,pad=2);a.axhline(0,color='#999999',lw=.7,zorder=1)
def save(f,name,ref,data,adapt):
 data.to_csv(D/f'{name}.tsv',sep='\t',index=False);f.canvas.draw();bbox=f.get_tightbbox(f.canvas.get_renderer()).padded(.04)
 for ext in ['pdf','svg']:f.savefig(S/f'{name}.{ext}',dpi=600,bbox_inches=bbox,pad_inches=0,facecolor='white')
 manifest.append(dict(panel=name,group='g4',primary_reference_id=ref,font='Arial',dpi=600,palette=C,adaptations=adapt,review_status='awaiting_actual_visual_review',plot_data=f'publication_data/{name}.tsv',plotting_code='src/plot_publication.py'));plt.close(f)

# Primary family: 9 selected cohorts, paired patient mean sign-flip tests.
T=[];plot=Q[Q.contrast=='total'].copy()
for ix,r in enumerate(plot.itertuples()):
 patients=pd.read_csv(O/'records'/f'{r.study}.tsv',sep='\t');states=patients[patients.split=='confirmation'].state.to_numpy();freq=np.bincount(states,minlength=16)/len(states);vec=[]
 for des in ['REDUNDANT','HETEROGENEOUS','CROSSVIEW_FAILURE']:
  z=np.load(O/f'research_traceability/confirmation_{r.study}_{des}_patient_losses.npz');vec.append((z['MORE']-z['ALIGNED']).mean(axis=0)/(z['selection_mass']@freq))
 gain=np.mean(vec,axis=0)[states];assert np.isclose(gain.mean(),r.gain);rng=np.random.default_rng([202610096,ix]);hits=0;B=49999
 for offset in range(0,B,1000):
  flips=rng.integers(0,2,(min(1000,B-offset),len(gain)),dtype=np.int8)*2-1;hits+=int((abs((flips*gain).mean(axis=1))>=abs(gain.mean())-1e-14).sum())
 T.append(dict(panel='G4_E1_multicohort_value',study=r.study,n_patients=len(gain),p_raw=(1+hits)/(B+1),gain=r.gain,test='two-sided paired mean sign-flip; symmetric-null assumption',family_size=len(plot),unit='independent confirmation patient; frozen ensemble averaged first'))
p=np.array([t['p_raw'] for t in T]);order=np.argsort(p);adj=np.maximum.accumulate(p[order]*(len(p)-np.arange(len(p))));q=np.empty(len(p));q[order]=np.minimum(adj,1)
for t,v in zip(T,q):t['p_holm']=float(v);t['label']='***' if v<.001 else '**' if v<.01 else '*' if v<.05 else 'ns'
pd.DataFrame(T).to_csv(D/'SIGNIFICANCE_TESTS.tsv',sep='\t',index=False)
plot=plot.merge(pd.DataFrame(T)[['study','p_raw','p_holm','label']],on='study')
f,a=plt.subplots(figsize=(7.1,3.1));f.subplots_adjust(left=.105,right=.99,bottom=.26,top=.88);tops=[]
for ix,r in enumerate(plot.itertuples()):
 col=C[0] if r.cancer=='LUAD' else C[2];y=r.gain*1000;a.bar(ix,y,.58,facecolor='white',edgecolor=col,lw=1.6,zorder=2);a.errorbar(ix,y,yerr=[[y-r.lo*1000],[r.hi*1000-y]],fmt='none',ecolor=K,capsize=2.8,lw=.85,zorder=4)
 vals=V[(V.study==r.study)&(V.contrast=='total')].gain.to_numpy()*1000;a.scatter(ix+np.linspace(-.13,.13,len(vals)),vals,s=15,c=col,ec='white',lw=.35,zorder=5)
 top=max(r.hi*1000,max(vals));tops.append(top);a.plot([ix,ix],[top+1.5,top+4],c=K,lw=.6);a.text(ix,top+4.7,r.label,ha='center',va='bottom',fontsize=7)
a.set_xticks(range(len(plot)),[f'{names[r.study]}\nn={r.n_patients}' for r in plot.itertuples()],rotation=32,ha='right');a.set_ylabel('Hidden-record log-score gain (×10³)');a.set_ylim(min(-5,plot.lo.min()*1000-3),max(tops)+12);tidy(a)
f.legend([Patch(fc='white',ec=C[0],lw=1.3),Patch(fc='white',ec=C[2],lw=1.3),Line2D([],[],marker='o',lw=0,color=K,ms=3)],['Lung adenocarcinoma','Colorectal cancer','Three predefined operators'],loc='upper center',bbox_to_anchor=(.53,1.01),ncol=3,fontsize=7,columnspacing=1.2)
a.text(1,.98,'vs zero; Holm-adjusted',transform=a.transAxes,ha='right',va='top',fontsize=6.6,color=K)
save(f,'G4_E1_multicohort_value','R29',plot,['Nine hollow bars replace three reference groups; all nine pilot-selected confirmation cohorts retained','Three dots per cohort are prescribed operator means, not extra patients','Capped patient+fit intervals; vertical scalar-null leaders do not imply a second-bar comparison'])

for cancer in ['LUAD','COAD']:
 studies=plot[plot.cancer==cancer].study.tolist();contrasts=['total','margin','pairing','model','optimizer'];labels=['Total','Deassociated','Pairing','vs pooled','vs two-step'];f,a=plt.subplots(figsize=(7.1,3.2));f.subplots_adjust(left=.10,right=.99,bottom=.20,top=.90)
 for j,(contrast,label,col) in enumerate(zip(contrasts,labels,C[:5])):
  for ix,study in enumerate(studies):
   r=Q[(Q.study==study)&(Q.contrast==contrast)].iloc[0];x=ix+(j-2)*.155;y=r.gain*1000;a.bar(x,y,.135,color=col,edgecolor=K,lw=.6,zorder=2);a.errorbar(x,y,yerr=[[y-r.lo*1000],[r.hi*1000-y]],fmt='none',ecolor=K,capsize=1.7,lw=.65,zorder=4)
   vals=V[(V.study==study)&(V.contrast==contrast)].gain.to_numpy()*1000;a.scatter(x+np.linspace(-.036,.036,len(vals)),vals,s=10,fc='white',ec=K,lw=.45,zorder=5)
 a.set_xticks(range(len(studies)),[names[s] for s in studies],fontsize=7);a.set_ylabel('Controlled log-score gain (×10³)');tidy(a)
 f.legend([Patch(fc=c,ec=K,lw=.5) for c in C[:5]],labels,loc='upper center',bbox_to_anchor=(.535,1),ncol=5,fontsize=7,columnspacing=1.1,handlelength=1)
 data=Q[Q.study.isin(studies)&Q.contrast.isin(contrasts)].copy();save(f,f'G4_E1_attribution_{cancer}','R21',data,['Five controlled contrasts per new cohort, narrow colored bars and hollow operator means','Two-sided values retained; total equals deassociated plus pairing exactly','Intervals are primary statistical inference for these secondary contrasts; no new unregistered star tests'])
(O/'configs/FIGURE_MANIFEST.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2));print('3 panels generated; awaiting visual review')
