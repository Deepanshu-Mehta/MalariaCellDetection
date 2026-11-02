# One hand-made feature, no training: how dark is the darkest spot compared with the rest of the cell?
import sys, numpy as np, pandas as pd
from pathlib import Path
from PIL import Image
from scipy import ndimage as ndi, stats as st
from concurrent.futures import ProcessPoolExecutor
R=Path('/Users/deepanshumehta/ai/cv/research'); NIH=R.parent/'milestone2-baseline/data/cell_images'; MP=R/'data_external/mpidb_pf_cells'
def feat(path):
    a=np.array(Image.open(path).convert('RGB').resize((128,128),Image.BILINEAR)).astype(float)
    fg=a.max(2)>0
    fg=ndi.binary_fill_holes(fg)
    er=ndi.binary_erosion(fg,iterations=4)
    if er.sum()<50: return np.nan
    g=ndi.gaussian_filter(a[...,1],1.5)     # green channel: Giemsa chromatin absorbs here
    med=np.median(g[er]); return (med-g[er].min())/max(med,1)
def auc(y,s):
    n1=y.sum();n0=len(y)-n1;return (st.rankdata(s)[y==1].sum()-n1*(n1+1)/2)/(n1*n0)
def bacc(y,s,t):
    q=s>=t;return 50*(q[y==1].mean()+(~q[y==0]).mean())
if __name__=='__main__':
    c=pd.read_csv(R/'splits/cells.csv')
    with ProcessPoolExecutor(6) as ex: c['f']=list(ex.map(feat,[NIH/p for p in c.rel_path],chunksize=200))
    print('NIH nan',c.f.isna().sum()); c=c.dropna(subset=['f'])
    y=c.label.values; print('NIH AUC %.4f'%auc(y,c.f.values))
    gf=pd.read_csv(R/'splits/grouped_folds.csv'); t=gf[(gf.repeat==0)&(gf.role=='test')][['patient','fold']]; c=c.merge(t,on='patient')
    qs=np.quantile(c.f,np.linspace(.02,.98,193)); pred=[]; ths=[]
    for k in range(5):
        tr=c[c.fold!=k]; te=c[c.fold==k]; b=[bacc(tr.label.values,tr.f.values,q) for q in qs]; th=qs[int(np.argmax(b))]; ths.append(th)
        pred.append(pd.DataFrame(dict(y=te.label.values,s=te.f.values-th)))
    pp=pd.concat(pred); print('NIH grouped OOF BACC %.2f thresholds'%bacc(pp.y.values,pp.s.values,0),np.round(ths,3))
    th=float(np.mean(ths))
    m=pd.read_csv(MP/'mpidb_cells.csv'); m=m[m.label.isin(['Parasitized','Uninfected'])&m.file.notna()]
    if 'excluded' in m: m=m[m.excluded.isna()]
    with ProcessPoolExecutor(6) as ex: m['f']=list(ex.map(feat,[MP/p for p in m.file],chunksize=200))
    m=m.dropna(subset=['f']); y=(m.label=='Parasitized').astype(int).values
    print('MP-IDB n',len(m),'par',y.sum(),'AUC %.4f BACC@NIH-threshold %.2f SE %.1f SP %.1f'%(auc(y,m.f.values),bacc(y,m.f.values,th),100*(m.f.values[y==1]>=th).mean(),100*(m.f.values[y==0]<th).mean()))
    rng=np.random.default_rng(0); im=[np.where(m.image.values==i)[0] for i in m.image.unique()]; v=[]
    for b in range(1000):
        ix=np.concatenate([im[j] for j in rng.integers(0,len(im),len(im))]); v.append(auc(y[ix],m.f.values[ix]))
    print('MP-IDB AUC image-bootstrap CI',np.round(np.percentile(v,[2.5,97.5]),4))
