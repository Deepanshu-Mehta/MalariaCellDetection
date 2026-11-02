import sys, numpy as np, pandas as pd
sys.path.insert(0,'/Users/deepanshumehta/ai/cv')
from research.malaria.stats import metrics, load_preds, paired_diff
R='/Users/deepanshumehta/ai/cv/research/'
t=pd.read_csv(R+'splits/cells.csv'); print(t.columns.tolist(), len(t)); print(t.head(2).to_string())
fc=[c for c in t.columns if t[c].astype(str).str.contains('IMG_').any()][0]
lc='label' if 'label' in t else [c for c in t.columns if 'lab' in c.lower()][0]
y=t[lc]; y=(y.astype(str).str.lower().str.startswith('para')|(y.astype(str)=='1')).astype(int)
t=t.assign(y=y,date=t[fc].astype(str).str.extract(r'IMG_(\d{8})')[0])
p=t.groupby('patient').agg(n=('y','size'),par=('y','sum')); p['unf']=p.n-p.par; p['inf']=(p.par>0).astype(int)
print('patients',len(p),'infected',p.inf.sum()); print(p.groupby('inf')[['n','unf']].agg(['min','median','max']))
def auc(a,s):
    r=pd.Series(np.asarray(s)).rank().values;a=np.asarray(a);n1=a.sum();n0=len(a)-n1
    return (r[a==1].sum()-n1*(n1+1)/2)/(n1*n0)
print('AUC cell count %.3f | uninfected count %.3f'%(auc(p.inf,p.n),auc(p.inf,p.unf)))
print('dates',t.date.nunique(),'missing',t.date.isna().sum())
ds=t.groupby('date').patient.apply(set); inf=set(p.index[p.inf==1]); hea=set(p.index[p.inf==0])
sh=[d for d,s in ds.items() if s&inf and s&hea]
print('shared dates',len(sh),'healthy',len(set().union(*[ds[d]&hea for d in sh])),'infected',len(set().union(*[ds[d]&inf for d in sh])))
gf=pd.read_csv(R+'splits/grouped_folds.csv'); g=gf[(gf.repeat==0)&(gf.role=='test')]
m=t[['patient','date']].drop_duplicates().merge(g[['patient','fold']],on='patient')
print('dates spanning >1 test fold (r0):',(m.groupby('date').fold.nunique()>1).sum(),'of',m.date.nunique())
print('share of patients whose date also appears in another fold:', m.groupby('patient').date.first().map(m.groupby('date').fold.nunique()).gt(1).mean().round(3))
# patient-level DANet AUC vs cell count, mean p
e=load_preds('E1_danet','grouped'); e=e[e.tag.str.startswith('grouped_r0')]
q=e.groupby('patient').p.agg(['mean',lambda s:(s>=.5).mean()]); q.columns=['meanp','frac']; q=q.join(p)
print('patient AUC DANet mean-p %.3f frac-pos %.3f'%(auc(q.inf,q.meanp),auc(q.inf,q.frac)))
# ResNet vs DANet SE/AUC, all repeats
r=load_preds('E1r_resnet18','grouped')
for tag in ('r0','r1','r2'):
    a=e if tag=='r0' else None
ea=load_preds('E1_danet','grouped'); ea['rep']=ea.tag.str[9]; r['rep']=r.tag.str[9]
for rep in '012':
    A=ea[ea.rep==rep][['idx','patient','label','p']];B=r[r.rep==rep][['idx','patient','label','p']]
    for mtr in ('BACC','SE','AUC'):
        o=paired_diff(A,B,metric=mtr,B=300); print('r'+rep,'DANet-ResNet',mtr,'%.4f [%.4f,%.4f]'%(o['diff'],o['lo'],o['hi']))
