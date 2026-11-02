import sys, numpy as np, pandas as pd
from scipy import stats as st
sys.path.insert(0,'/Users/deepanshumehta/ai/cv')
from research.malaria.stats import metrics, load_preds, paired_diff, cluster_bootstrap
from research.experiments.E6b_five_checkpoint_external_rescore.score_external import scored_table, SCORES
R='/Users/deepanshumehta/ai/cv/research/'
cells=pd.read_csv(R+'splits/cells.csv'); cells['date']=cells.rel_path.str.extract(r'IMG_(\d{8})')[0]
pinf=cells.groupby('patient').label.sum()>0
healthy=set(pinf.index[~pinf])
print('=== P2 E7 healthy-only control')
fl=pd.read_csv(R+'experiments/E7_label_noise_audit/flagged_cells.csv'); au=pd.read_csv(R+'experiments/E7_label_noise_audit/audit_top100.csv')
for name,d in (('all flagged',fl),('top100',au)):
    u=d[d.label==0]; print(name,'Uninfected-labelled flagged',len(u),'in healthy-only pts',u.patient.isin(healthy).sum(),'=%.1f%%'%(100*u.patient.isin(healthy).mean()))
un=cells[cells.label==0]; print('base rate: healthy-only share of Uninfected cells %.1f%%'%(100*un.patient.isin(healthy).mean()))
print('=== P5 operating points, DANet E1 OOF r0, clean negatives = healthy-only patients')
e=load_preds('E1_danet','grouped'); e=e[e.tag.str.startswith('grouped_r0')]
h=e[e.patient.isin(healthy)&(e.label==0)]; pos=e[e.label==1]
print('healthy cells',len(h),'patients',h.patient.nunique())
for thr in (.5,.9,.99,.999):
    fp=(h.p>=thr).mean(); se=(pos.p>=thr).mean()
    pp=h.groupby('patient').p.apply(lambda s:(s>=thr).mean())
    # min parasitemia m where SE*m = FP (signal equals false-positive background)
    print('thr',thr,'FP %.3f%% SE %.1f%% per-patient FP p95 %.2f%% | signal=background parasitemia %.2f%% | PPV@1%% %.1f%% PPV@0.1%% %.1f%%'%(100*fp,100*se,100*pp.quantile(.95),100*fp/se,100*se*.01/(se*.01+fp*.99),100*se*.001/(se*.001+fp*.999)))
print('=== P3 MP-IDB thresholds and ensembles')
t=scored_table(); print(t.columns.tolist(),len(t))
y=(t.label.astype(str).str.startswith('Para')|(t.label.astype(str)=='1')).astype(int).values
L=lambda p: np.log(np.clip(p,1e-6,1-1e-6)/(1-np.clip(p,1e-6,1-1e-6)))
def auc(y,s):
    n1=y.sum();n0=len(y)-n1;return (st.rankdata(s)[y==1].sum()-n1*(n1+1)/2)/(n1*n0)
def bacc(y,s,thr=.5):
    q=s>=thr;return 50*(q[y==1].mean()+(~q[y==0]).mean())
def oracle(y,s):
    qs=np.unique(np.quantile(s,np.linspace(0,1,801)));return max(bacc(y,s,q) for q in qs)
S={}
rows=[]
for a in ('E1','E8b','E8c'):
    for f in range(5):
        S[a,f]=np.load(SCORES/f'A_{a}_f{f}.npy')
        s=S[a,f]; rows.append((a,f,bacc(y,s),oracle(y,s),auc(y,s),(s>=.5).mean()))
r=pd.DataFrame(rows,columns='arm fold bacc oracle auc posrate'.split()); print(r.round(3).to_string())
print(r.groupby('arm')[['bacc','oracle','auc']].agg(['mean','std']).round(3))
print('all-15 mean bacc %.2f oracle %.2f auc %.4f'%(r.bacc.mean(),r.oracle.mean(),r.auc.mean()))
print('corr bacc~posrate %.3f, oracle SD %.2f vs bacc SD %.2f'%(np.corrcoef(r.bacc,r.posrate)[0,1],r.oracle.std(),r.bacc.std()))
for a in ('E1','E8b','E8c'):
    ens=np.mean([L(S[a,f]) for f in range(5)],axis=0); ep=1/(1+np.exp(-ens))
    print(a,'ensemble logit bacc@.5 %.2f auc %.4f oracle %.2f'%(bacc(y,ep),auc(y,ep),oracle(y,ep)))
ens=np.mean([L(v) for v in S.values()],axis=0); ep=1/(1+np.exp(-ens)); print('15-ensemble bacc %.2f auc %.4f oracle %.2f'%(bacc(y,ep),auc(y,ep),oracle(y,ep)))
print('=== P4 date clustering')
cells2=cells[['idx','date']]
pr=pd.read_csv(R+'experiments/E9_foundation_model_probes/oof_predictions_reddino_s.csv')
d=e[['idx','patient','label','p']]
A=d.merge(cells2,on='idx'); 
o=paired_diff(d,pr,B=1000)
da=A.assign(patient=A.date); db=pr.merge(cells2,on='idx').assign(patient=lambda x:x.date)
o2=paired_diff(da[['idx','patient','label','p']],db[['idx','patient','label','p']],B=1000)
print('DANet-RedDino patient %s'%{k:round(v,3) for k,v in o.items()}); print('DANet-RedDino date    %s'%{k:round(v,3) for k,v in o2.items()}, 'SE ratio %.2f'%(o2['se']/o['se']))
f=lambda g:metrics(g.label,g.p)['BACC']
c1=cluster_bootstrap(A,f,B=800); c2=cluster_bootstrap(A.assign(patient=A.date),f,B=800)
print('E1 r0 BACC CI patient',round(c1['lo'],2),round(c1['hi'],2),'date',round(c2['lo'],2),round(c2['hi'],2),'SE ratio %.2f'%(c2['se']/c1['se']))
print('cells per date: top5 share',(cells.groupby('date').size().sort_values(ascending=False).head(5).sum()/len(cells)).round(3))
