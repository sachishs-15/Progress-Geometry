"""Endpoint-controlled parent-step order and paired retrieval checks."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT=Path(__file__).parent
OUT=ROOT/'results'
SOURCE=ROOT.parent/'long_procedure_raw/results/val_raw_dinov2'
rng=np.random.default_rng(20261010)
meta=pd.read_csv(SOURCE/'parent_steps.csv')
payload=np.load(SOURCE/'parent_step_embeddings.npz')
lookup={(str(u),int(i)):v for u,i,v in zip(payload['video_uid'],payload['parent_step_idx'],payload['embeddings'])}

def boot(x):
    x=np.asarray(x,float);x=x[np.isfinite(x)]
    if not len(x):return {}
    b=rng.choice(x,(4000,len(x)),replace=True).mean(1)
    return dict(n_videos=len(x),mean=float(x.mean()),ci95=np.quantile(b,[.025,.975]).tolist())

records=[]
for uid,g in meta.groupby('video_uid'):
    g=g.sort_values(['start_time','parent_step_idx'])
    for scope,gg in [('all_parent_events',g),('essential_only',g[g.relevance=='essential'])]:
        if len(gg)<5:continue
        z=np.stack([lookup[(uid,int(i))] for i in gg.parent_step_idx])
        c=z[-1]-z[0]
        p=(z-z[0])@c/max(float(c@c),1e-12)
        score=spearmanr(np.arange(len(z)-2),p[1:-1]).statistic
        # Equivalent to shuffling only interior events, holding start and end fixed.
        sh=np.array([spearmanr(np.arange(len(z)-2),rng.permutation(p[1:-1])).statistic for _ in range(500)])
        v=np.diff(z,axis=0)
        efficiency=np.linalg.norm(c)/max(np.linalg.norm(v,axis=1).sum(),1e-12)
        row=dict(video_uid=uid,scope=scope,n_steps=len(z),interior_rho=float(score),
                 shuffled_mean=float(np.mean(sh)),beat_shuffle_fraction=float(np.mean(score>sh)),
                 shuffle_p_upper=float((np.sum(sh>=score)+1)/501),sparse_path_efficiency=float(efficiency))
        records.append(row)
order=pd.DataFrame(records);order.to_csv(OUT/'parent_endpoint_controls.csv',index=False)
os={}
for scope,g in order.groupby('scope'):
    os[scope]={c:boot(g[c]) for c in ['interior_rho','beat_shuffle_fraction','sparse_path_efficiency']}
    os[scope]['fraction_p_below_05']=float(np.mean(g.shuffle_p_upper<.05))

queries=pd.read_csv(SOURCE/'next_step_retrieval_queries.csv')
rs={}
for scope,g in [('all_queries',queries),('common_extrapolation_queries',queries.dropna(subset=['rr_linear_extrapolation']))]:
    row=dict(n_queries=len(g))
    for metric in ['rr','r1','r5']:
        names=[metric+'_cosine_current',metric+'_earliest_future_baseline',metric+'_random_expected']
        if scope=='common_extrapolation_queries':names.append(metric+'_linear_extrapolation')
        row[metric]={c:boot(g.groupby('video_uid')[c].mean()) for c in names}
        for c in names:
            if c.endswith('earliest_future_baseline'):continue
            d=g[c]-g[metric+'_earliest_future_baseline']
            row[metric][c+'_minus_earliest']=boot(d.groupby(g.video_uid).mean())
    rs[scope]=row
json.dump(dict(order=os,retrieval=rs),open(OUT/'parent_controls_summary.json','w'),indent=2)
print(json.dumps(dict(order=os,retrieval=rs),indent=2))
