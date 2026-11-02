import sys, numpy as np, pandas as pd
from pathlib import Path
from PIL import Image
from scipy import ndimage as ndi, stats as st
from concurrent.futures import ProcessPoolExecutor
R=Path('/Users/deepanshumehta/ai/cv/research'); NIH=R.parent/'milestone2-baseline/data/cell_images'; MP=R/'data_external/mpidb_pf_cells'
VARS={'base':(1,1.5,4,True),'nofill':(1,1.5,4,False),'nofill_er8':(1,1.5,8,False),'red':(0,1.5,4,False),'blue':(2,1.5,4,False),'sig3':(1,3,4,False),'sig0.7':(1,.7,4,False),'minch':('min',1.5,4,False)}
def feat(path):
    im=np.array(Image.open(path).convert('RGB').resize((128,128),Image.BILINEAR)).astype(float)
    raw=im.max(2)>0; out={}
    out['holes']=int((ndi.binary_fill_holes(raw)&~raw).sum())
    for k,(ch,s,e,fill) in VARS.items():
        fg=ndi.binary_fill_holes(raw) if fill else raw
        er=ndi.binary_erosion(fg,iterations=e)
        if er.sum()<30: out[k]=np.nan; continue
        c=im.min(2) if ch=='min' else im[...,ch]
        # normalised convolution so black background never bleeds into the smoothed value
        w=ndi.gaussian_filter(raw.astype(float),s); g=ndi.gaussian_filter(c*raw,s)/np.maximum(w,1e-3)
        med=np.median(g[er]); out[k]=(med-g[er].min())/max(med,1)
    return out
def auc(y,s):
    y=np.asarray(y);n1=y.sum();n0=len(y)-n1;return (st.rankdata(s)[y==1].sum()-n1*(n1+1)/2)/(n1*n0)
def bacc(y,s,t):
    y=np.asarray(y);q=np.asarray(s)>=t;return 50*(q[y==1].mean()+(~q[y==0]).mean())
def run(paths):
    with ProcessPoolExecutor(6) as ex: return pd.DataFrame(list(ex.map(feat,paths,chunksize=200)))
if __name__=='__main__':
    c=pd.read_csv(R/'splits/cells.csv'); cf=run([NIH/p for p in c.rel_path]); c=pd.concat([c,cf],axis=1)
    all_=pd.read_csv(MP/'mpidb_cells.csv'); print(all_.excluded.value_counts(dropna=False).to_dict())
    m=all_[all_.label.notna()].reset_index(drop=True); mf=run([MP/p for p in m.file]); m=pd.concat([m,mf],axis=1); y=(m.label=='Parasitized').astype(int)
    print('holes>0 rate NIH par %.3f unf %.3f | MP par %.3f unf %.3f'%((c[c.label==1].holes>0).mean(),(c[c.label==0].holes>0).mean(),(m[y==1].holes>0).mean(),(m[y==0].holes>0).mean()))
    print('variant | NIH AUC | NIH-fit thr | NIH BACC | MP AUC | MP BACC@NIHthr SE SP')
    for k in VARS:
        n=c.dropna(subset=[k]); qs=np.quantile(n[k],np.linspace(.02,.98,97)); th=qs[int(np.argmax([bacc(n.label,n[k],q) for q in qs]))]
        v=m[k].fillna(0).values
        print(k.ljust(11),'%.4f %.3f %.2f | %.4f %.2f SE %.1f SP %.1f'%(auc(n.label,n[k]),th,bacc(n.label,n[k],th),auc(y,v),bacc(y,v,th),100*(v[y==1]>=th).mean(),100*(v[y==0]<th).mean()))
    k='nofill'; n=c; qs=np.quantile(n[k].dropna(),np.linspace(.02,.98,97)); th=qs[int(np.argmax([bacc(n.label,n[k].fillna(0),q) for q in qs]))]
    print('--- label-dependent exclusion stress test, variant nofill, thr %.3f'%th)
    from glob import glob
    # parasite-size strata
    import PIL
    area=[]
    for r in m[y==1].itertuples():
        a=np.array(Image.open(MP/'parasite_masks'/r.file))>0; area.append(a.mean())
    p=m[y==1].assign(a=area); p['q']=pd.qcut(p.a,4,labels=False)
    print('SE by parasite-size quartile',p.groupby('q')[k].apply(lambda s:round(100*(s>=th).mean(),1)).to_dict())
    print('n per image Parasitized: median',m[y==1].groupby('image').size().median())
    # per-image AUC: cells from same field only, removes between-field brightness effects
    A=[auc(g.label=='Parasitized',g[k].fillna(0)) for _,g in m.groupby('image') if 2<=(g.label=='Parasitized').sum()<len(g)-1]
    print('within-image AUC median %.4f min %.4f n %d'%(np.median(A),np.min(A),len(A)))
    # NIH: patient-level and healthy-only FP, disagreement with DANet
    sys.path.insert(0,str(R.parent)); from research.malaria.stats import load_preds
    e=load_preds('E1_danet','grouped'); e=e[e.tag.str.startswith('grouped_r0')][['idx','p']]; d=c.merge(e,on='idx')
    pinf=d.groupby('patient').label.sum()>0; hl=set(pinf.index[~pinf])
    h=d[d.patient.isin(hl)]; print('NIH healthy-only FP: feature %.2f%% DANet %.2f%%'%(100*(h[k]>=th).mean(),100*(h.p>=.5).mean()))
    print('NIH corr feature vs DANet logit (spearman) %.3f'%st.spearmanr(d[k].fillna(0),d.p).statistic)
    fl=pd.read_csv(R/'experiments/E7_label_noise_audit/flagged_cells.csv'); u=fl[fl.label==0].merge(c[['idx',k]],on='idx')
    print('E7 flagged Uninfected: feature fires in %.1f%% overall; in healthy-only %.1f%%, infected-pt %.1f%%'%(100*(u[k]>=th).mean(),100*(u[u.patient.isin(hl)][k]>=th).mean(),100*(u[~u.patient.isin(hl)][k]>=th).mean()))
