import sys, re, glob, json
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0,'/Users/deepanshumehta/ai/cv')
from research.malaria.stats import metrics, load_preds, paired_diff, cluster_bootstrap
R=Path('/Users/deepanshumehta/ai/cv/research')
RK=R/'runs_kaggle'
def lk(cfg,s): 
    return load_preds(cfg,f'leaky_s{s}')
print('=== A. E4b: paired leaky s42 E8b baseline vs colour-transfer (same test cells)')
a=load_preds('E4b_e8b_colourtransfer','leaky_s42'); b=lk('E8_paper_faithful',42)
print(paired_diff(a,b,B=2000))
print('  same contrast for jitter on E1 s42 for scale:'); 
print(paired_diff(load_preds('E4_colourjitter','leaky_s42'),lk('E1_danet',42),B=2000))
print('=== A2. A/A-like scale on leaky: across split seeds the arm SDs')
for c in ['E1_danet','E8_paper_faithful','E8c_paper_instancenorm','E1r_resnet18']:
    v=[metrics(*(lambda d:(d.label,d.p))(lk(c,s)))['BACC'] for s in (42,43,44)]
    print(c,np.round(v,3),'mean',round(np.mean(v),3),'sd',round(np.std(v,ddof=1),3))
print('=== B. E3 training logs')
for f in sorted(glob.glob(str(RK/'e3-random-patient-label-v2/runs/E3_random_patient/*/train_log.csv'))):
    l=pd.read_csv(f); m=json.load(open(Path(f).parent/'metrics.json')) if (Path(f).parent/'metrics.json').exists() else {}
    be=l.val_loss.idxmin()
    print(Path(f).parent.name[:32].ljust(33),'ep',len(l),'bestep',int(l.epoch[be]),'train_acc@best %.3f final %.3f val_acc@best %.3f'%(l.train_acc[be],l.train_acc.iloc[-1],l.val_acc[be]))
print('=== B2. E3 null SD if a model outputs one arbitrary score per patient (patient-level random agreement)')
gf=pd.read_csv(R/'splits/grouped_folds.csv'); print(gf.columns.tolist()[:12])
print('=== C. E9 vs DANet repeats (pooled OOF, same cells; r0 was the only one used)')
e=load_preds('E1_danet','grouped')
e['rep']=e.tag.str.extract(r'grouped_r(\d)f')[0].astype(int)
print(e.groupby('rep').size().to_dict())
for bb in ['dinov2_s','dinobloom_s','reddino_s']:
    pr=pd.read_csv(R/f'experiments/E9_foundation_model_probes/oof_predictions_{bb}.csv')
    for rep in (0,1,2):
        d=e[e.rep==rep][['idx','patient','label','p']]
        if len(d)!=len(pr): print('len mismatch',len(d),len(pr)); continue
        o=paired_diff(d,pr,B=400)
        print(bb,'r',rep,'DANet-probe %.3f [%.3f, %.3f]'%(o['diff'],o['lo'],o['hi']))
print('=== D. MDE from all A/A pairs and DANet vs ResNet')
for x,y in ((0,1),(0,2),(1,2)):
    o=paired_diff(e[e.rep==x][['idx','patient','label','p']],e[e.rep==y][['idx','patient','label','p']],B=400)
    print('A/A',x,y,'diff %.3f se %.3f -> 2.8se %.3f'%(o['diff'],o['se'],2.8*o['se']))
