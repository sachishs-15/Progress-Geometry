"""Join manually reviewed videos to matched frozen latents and draw examples."""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import mannwhitneyu

from analyze_features import DATA, OUT, load, metrics


def review(kind):
    with open(OUT/f'{kind}_review.csv',newline='') as f:return list(csv.DictReader(f))


def get_rows(path,reviewed):
    x,paths=load(path)
    lookup={p:i for i,p in enumerate(paths)}
    found=[r for r in reviewed if r['path'] in lookup]
    xx=np.stack([x[lookup[r['path']]].numpy() for r in found]).astype(np.float32)
    return found,xx


def group_metrics(rows,xx):
    m=metrics(xx);out={}
    cats=np.array([r['category'] for r in rows])
    for c in sorted(set(cats)):
        mask=cats==c
        out[c]={'n':int(mask.sum()), **{k+'_median':float(np.median(v[mask])) for k,v in m.items() if k in ['straight','time_rho','velocity_cos','interpolation_r2','goal_rho']}}
    if np.any(cats=='S') and np.any(cats=='R'):
        out['S_vs_R_straight_mannwhitney_p']=float(mannwhitneyu(m['straight'][cats=='S'],m['straight'][cats=='R'],alternative='two-sided').pvalue)
    return m,out


def example_plot(found_by_model):
    targets=[('kinetics',44,'Peeling apples: persistent object change'),
             ('kinetics',65,'Push-ups: repeated pose'),
             ('ego_leaf_uniform',25,'Dicing celery: pieces accumulate'),
             ('ego_leaf_uniform',11,'Stirring sauce: cycle around stable bowl')]
    fig,axs=plt.subplots(4,2,figsize=(11,12),sharex=True)
    for row,(dataset,index,title) in enumerate(targets):
        for model,(records,xx) in found_by_model[dataset].items():
            ids=[i for i,r in enumerate(records) if int(r['index'])==index]
            if not ids:continue
            x=xx[ids[0]]
            u=x[-1]-x[0]
            progress=((x-x[0])@u)/max(float(u@u),1e-8)
            dist=np.linalg.norm(x-x[-1],axis=1)
            dist=dist/max(float(dist[0]),1e-8)
            axs[row,0].plot(np.arange(16),progress,marker='.',label=model)
            axs[row,1].plot(np.arange(16),dist,marker='.',label=model)
        axs[row,0].set_ylabel(title+'\nchord position')
        axs[row,1].set_ylabel('distance to final\nnormalized to t=0')
        for ax in axs[row]:ax.grid(alpha=.25)
    axs[0,0].set_title('Position along start-to-end chord')
    axs[0,1].set_title('Goal-distance curve')
    for ax in axs[-1]:ax.set_xlabel('sampled frame (0–15)')
    axs[0,0].legend(fontsize=8);axs[0,1].legend(fontsize=8)
    fig.tight_layout();fig.savefig(OUT/'annotated_latent_examples.png',dpi=180);plt.close(fig)


if __name__=='__main__':
    examples={};summary={}
    for kind,sources in {
      'kinetics':{'DINOv2 CLS':DATA/'kinetics-dataset/precomputed_features/dinov2/k400_train_full.pt',
                  'V-JEPA 2.1 mean patch':DATA/'kinetics-dataset/precomputed_features/vjepa2.1/k400_train_full.pt'},
      'ego_leaf_uniform':{'DINOv2 CLS':DATA/'ego4dgoal/precomputed_features/dinov2/ego4dgoal_val.pt',
                          'V-JEPA 2.1 mean patch':OUT/'vjepa_features/vjepa2.1/ego_uniform_vjepa_manifest.pt'},
    }.items():
        reviewed=review(kind);examples[kind]={};summary[kind]={}
        for model,path in sources.items():
            if not path.exists():continue
            rows,xx=get_rows(path,reviewed)
            m,out=group_metrics(rows,xx)
            summary[kind][model]={'matched_n':len(rows),'groups':out,
               'overall':{k:float(np.median(v)) for k,v in m.items() if k in ['straight','time_rho','velocity_cos','interpolation_r2']}}
            examples[kind][model]=(rows,xx)
    with open(OUT/'reviewed_feature_analysis.json','w') as f:json.dump(summary,f,indent=2)
    example_plot(examples)
    print(json.dumps(summary,indent=2))
