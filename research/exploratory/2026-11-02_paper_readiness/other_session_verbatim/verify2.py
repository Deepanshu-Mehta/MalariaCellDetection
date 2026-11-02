import re, ast, numpy as np, pandas as pd
from pathlib import Path
from PIL import Image
from scipy import ndimage as ndi
D=Path('/Users/deepanshumehta/ai/cv/milestone2-baseline/data')
rows=[]
for lab,fn in ((1,'patientid_cellmapping_parasitized.csv'),(0,'patientid_cellmapping_uninfected.csv')):
    for line in open(D/fn):
        pid,rest=line.split(',',1)
        files=re.findall(r"[\w\-\.]+\.png",rest)
        for f in files: rows.append((f.split('P')[0],pid,lab,f))
t=pd.DataFrame(rows,columns=['pat','pid','lab','f'])
t['date']=t.f.str.extract(r'IMG_(\d{8})')[0]
print('cells',len(t),'patients',t.pat.nunique(),'files with date',t.date.notna().mean(),'dates',t.date.nunique())
p=t.groupby('pat').agg(n=('f','size'),par=('lab','sum'),unf=('lab',lambda s:(s==0).sum()))
p['infected']=(p.par>0).astype(int)
print(p.groupby('infected')[['n','par','unf']].describe().T.loc[(['n','unf'],['count','min','50%','max']),:])
def auc(y,s):
    r=pd.Series(s).rank().values;y=np.asarray(y);n1=y.sum();n0=len(y)-n1
    return (r[y==1].sum()-n1*(n1+1)/2)/(n1*n0)
print('cell count alone patient AUC %.3f'%auc(p.infected,p.n))
print('uninfected count alone AUC %.3f (low=no info)'%auc(p.infected,p.unf))
pd_=t[t.date.notna()].groupby('date').pat.apply(set)
inf=set(p.index[p.infected==1]); hea=set(p.index[p.infected==0])
sh=[d for d,s in pd_.items() if (s&inf) and (s&hea)]
print('dates shared',len(sh),'healthy in shared',len(set().union(*[pd_[d]&hea for d in sh])),'infected in shared',len(set().union(*[pd_[d]&inf for d in sh])))
print('patients per date: median',pd_.map(len).median(),'max',pd_.map(len).max())
gf=pd.read_csv('/Users/deepanshumehta/ai/cv/research/splits/grouped_folds.csv');print(gf.columns.tolist()); print(gf.head(2))
# ---- E5 trivial baselines
C=Path('/Users/deepanshumehta/ai/cv/research/data_external/mpidb_pf_cells')
for arm in ['E1','E8b','E8c']:
    pass
res=pd.read_csv('/Users/deepanshumehta/ai/cv/research/experiments/E5_explanation_validity/results_E1.csv')
tp=res  # all rows; orig_pred present
print(res.orig_pred.mean(),len(res))
def r224(a,m):
    return np.array(Image.fromarray(a).resize((224,224),m))
out=[]
for r in res.itertuples():
    img=np.array(Image.open(C/r.file).convert('RGB')); mask=np.array(Image.open(C/'parasite_masks'/r.file))>0
    img224=r224(img,Image.BILINEAR); m=r224((mask*255).astype(np.uint8),Image.NEAREST)>127
    dm=ndi.binary_dilation(m,iterations=6)
    fg=img224.max(2)>0
    # centre
    ctr=dm[112,112]
    # centroid of fg
    ys,xs=np.nonzero(fg); cen=dm[int(ys.mean()),int(xs.mean())]
    # darkest smoothed foreground pixel (simple image-only heuristic, no model)
    fgi=ndi.binary_erosion(fg,iterations=8)
    g=ndi.gaussian_filter(img224.min(2).astype(float),3); g2=np.where(fgi,g,1e9)
    py,px=np.unravel_index(g2.argmin(),g2.shape); dark=dm[py,px]
    # same but on green channel saturation-style: max-min
    out.append((r.file,ctr,cen,dark,dm.mean(),m.mean(),fg.mean(),dm.sum()/max(fg.sum(),1)))
o=pd.DataFrame(out,columns='file ctr cen dark dilfrac maskfrac fgfrac dil_in_fg'.split())
print(o.mean(numeric_only=True).round(3).to_dict())
for arm in ['E1','E8b','E8c']:
    x=pd.read_csv(f'/Users/deepanshumehta/ai/cv/research/experiments/E5_explanation_validity/results_{arm}.csv').merge(o,on='file')
    t_=x[x.orig_pred==1]
    print(arm,'TP n',len(t_),'CAM pg %.3f | image centre %.3f | fg centroid %.3f | darkest-pixel %.3f | dilated-mask area/whole %.3f, /foreground %.3f'%(t_.pointing_game.mean(),t_.ctr.mean(),t_.cen.mean(),t_.dark.mean(),t_.dilfrac.mean(),t_.dil_in_fg.mean()))
