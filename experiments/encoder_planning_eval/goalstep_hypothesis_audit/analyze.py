"""Frozen-feature diagnostics, preserving original caches and training outputs.

Run with the lift Python environment. All numeric metrics use original feature space.
Dense caches use a different backbone/preprocessing from the legacy 16-frame cache;
compare lags within a cache, never attribute differences between caches to sampling.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
from scipy.stats import rankdata, spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from decord import VideoReader, cpu

ROOT = Path('/home/sachish/proc-planning')
DATA = Path('/mnt/rdata7/proprogress/data/ego4dgoal')
OUT = Path(__file__).parent / 'results'
OUT.mkdir(exist_ok=True)
RNG = np.random.default_rng(20261010)

def row_rho(x):
    r = rankdata(x, axis=1)
    r -= r.mean(1, keepdims=True)
    t = np.arange(x.shape[1], dtype=float)
    t -= t.mean()
    return (r @ t) / np.maximum(np.linalg.norm(r, axis=1)*np.linalg.norm(t), 1e-12)

def metrics(z):
    v = np.diff(z, axis=1)
    norms = np.linalg.norm(v, axis=2)
    c = z[:,-1]-z[:,0]
    cn = np.linalg.norm(c, axis=1)
    den = np.maximum(cn**2,1e-12)
    p = np.sum((z-z[:,0,None])*c[:,None], axis=2)/den[:,None]
    d = np.linalg.norm(z-z[:,-1,None],axis=2)
    cos = np.sum(v[:,:-1]*v[:,1:],axis=2)/np.maximum(norms[:,:-1]*norms[:,1:],1e-12)
    valid=(norms[:,:-1]>1e-6)&(norms[:,1:]>1e-6)
    cos=np.where(valid,cos,np.nan)
    alpha=np.linspace(0,1,z.shape[1],dtype=np.float32)[None,:,None]
    interp=(1-alpha)*z[:,0,None]+alpha*z[:,-1,None]
    err=np.mean(np.sum((z[:,1:-1]-interp[:,1:-1])**2,axis=2),axis=1)
    midpoint=(z[:,0]+z[:,-1])/2
    miderr=np.mean(np.sum((z[:,1:-1]-midpoint[:,None])**2,axis=2),axis=1)
    # Optimistic oracle: use correct endpoint direction, fit each frame's position
    # on its infinite line independently, permit reverse movement and overshoot.
    residual=z-z[:,0,None]-p[:,:,None]*c[:,None]
    total=np.mean(np.sum((z[:,1:-1]-z[:,1:-1].mean(1,keepdims=True))**2,axis=2),axis=1)
    return dict(efficiency=cn/np.maximum(norms.sum(1),1e-12),
                velocity_cosine=np.nanmean(cos,axis=1),
                progress_rho=row_rho(p), interior_progress_rho=row_rho(p[:,1:-1]),
                interior_goal_distance_rho=row_rho(-d[:,1:-1]),
                goal_approach_fraction=np.mean(np.diff(d,axis=1)<0,axis=1),
                interpolation_mse=err, midpoint_mse=miderr,
                interpolation_better_than_midpoint=(err<miderr).astype(float),
                oracle_line_residual_ratio=np.mean(np.sum(residual[:,1:-1]**2,axis=2),axis=1)/np.maximum(total,1e-12))

def bootstrap_video(df, col, seed=51):
    x=df.groupby('video_uid')[col].mean().dropna().to_numpy()
    if not len(x): return {}
    rng=np.random.default_rng(seed)
    b=rng.choice(x,(3000,len(x)),replace=True).mean(1)
    return dict(n_videos=len(x),mean=float(x.mean()),ci95=np.quantile(b,[.025,.975]).tolist())

def summary(df, cols):
    return {c:dict(clip_median=float(df[c].median()),video_macro=bootstrap_video(df,c)) for c in cols}

manifest=pd.read_csv(ROOT/'data/ego4dgoal/ego4dgoal_val.csv')
payload=torch.load(DATA/'precomputed_features/dinov2/ego4dgoal_val.pt',map_location='cpu',weights_only=False)
idx={p:i for i,p in enumerate(payload['video_paths'])}
assert len(idx)==len(payload['video_paths'])
assert manifest.video_path.is_unique
z=payload['features'][[idx[p] for p in manifest.video_path]].numpy()
assert z.shape==(len(manifest),16,384)
assert np.isfinite(z).all()
features_constant=(np.ptp(z,axis=1).max(1)<1e-6)
df=manifest.copy()
for k,v in metrics(z).items(): df[k]=v
df['sample_spacing_sec']=df.duration/15
keys=list(metrics(z[:2]))
print('raw metrics computed',flush=True)

# Same endpoints, shuffled interiors. Geometry with fixed end-goal remains comparable.
shuffle={k:np.zeros(len(z)) for k in keys}
for it in range(40):
    for begin in range(0,len(z),256):
        end=min(begin+256,len(z))
        perm=np.argsort(RNG.random((end-begin,14)),axis=1)+1
        order=np.concatenate([np.zeros((end-begin,1),int),perm,np.full((end-begin,1),15)],axis=1)
        zz=np.take_along_axis(z[begin:end],order[:,:,None],axis=1)
        for k,v in metrics(zz).items(): shuffle[k][begin:end]+=v/40
for k,v in shuffle.items():
    df['shuffled_'+k]=v
    df['delta_'+k]=df[k]-v
df.to_csv(OUT/'clip_diagnostics.csv',index=False)
raw_summary=dict(n_clips=len(df),n_videos=int(df.video_uid.nunique()),constant_clips=int(features_constant.sum()),
                 cache_has_extraction_metadata='extraction_metadata' in payload,
                 duration_quantiles=df.duration.quantile([0,.25,.5,.75,.95,.99,1]).to_dict(),
                 cooking_videos=int(manifest.groupby('video_uid').first().goal_category.str.startswith('COOKING:').sum()),
                 leaf_relevance=df.leaf_is_relevant.value_counts().to_dict(),
                 raw=summary(df,keys),shuffled=summary(df,['shuffled_'+k for k in keys]),
                 paired_delta=summary(df,['delta_'+k for k in keys]))
groups=[]
df['duration_bin']=pd.cut(df.duration,[0,2,5,15,60,np.inf],right=False).astype(str)
for by in ['duration_bin','leaf_is_relevant']:
    for group,sub in df.groupby(by):
        row=dict(grouping=by,group=str(group),n_clips=len(sub),n_videos=sub.video_uid.nunique())
        row.update({k:float(sub[k].median()) for k in keys})
        groups.append(row)
pd.DataFrame(groups).to_csv(OUT/'stratified_metrics.csv',index=False)
json.dump(raw_summary,open(OUT/'raw_summary.json','w'),indent=2)
print('shuffle and raw summaries complete',flush=True)

# Dense lag curves; triples must have all three times inside one annotated leaf.
dense=[]
dense_integrity=[]
for uid,m in manifest.groupby('video_uid'):
    for pool in ['cls','meanpatch']:
        path=DATA/'oad_4fps'/('dinov2_noreg_'+pool)/(uid+'.npy')
        if not path.exists(): continue
        a=np.asarray(np.load(path),dtype=np.float32)
        if a.ndim!=2 or a.shape[1]!=384 or not np.isfinite(a).all():
            raise ValueError(f'invalid dense features {path}')
        dense_integrity.append(dict(video_uid=uid,pooling=pool,n_frames=len(a),
                                    duplicate_adjacent_fraction=float(np.mean(np.all(a[1:]==a[:-1],axis=1)))))
        labels=np.full(len(a),-1,dtype=int)
        # Times inside intervals; overlapping leaves become ambiguous and are excluded.
        for i,r in m.iterrows():
            s=max(0,int(np.ceil(r.start_time*4))); e=min(len(a),int(np.ceil(r.end_time*4)))
            labels[s:e]=np.where(labels[s:e]==-1,i,-2)
        for lag in [1,2,4,8,16]:
            v1=a[lag:-lag]-a[:-2*lag]; v2=a[2*lag:]-a[lag:-lag]
            n1=np.linalg.norm(v1,axis=1);n2=np.linalg.norm(v2,axis=1)
            ok=(n1>1e-6)&(n2>1e-6)
            cos=np.sum(v1*v2,axis=1)/np.maximum(n1*n2,1e-12)
            same=(labels[:-2*lag]>=0)&(labels[:-2*lag]==labels[lag:-lag])&(labels[:-2*lag]==labels[2*lag:])
            boundary=(labels[:-2*lag]>=0)&(labels[lag:-lag]>=0)&(labels[2*lag:]>=0)&(~same)
            for scope,mask in [('within_leaf',same),('cross_leaf_boundary',boundary)]:
                mask=mask&ok
                if not mask.any():continue
                dense.append(dict(video_uid=uid,pooling=pool,lag_sec=lag/4,scope=scope,
                                  n_triples=int(mask.sum()),velocity_cosine_mean=float(cos[mask].mean()),
                                  velocity_cosine_median=float(np.median(cos[mask])),
                                  positive_fraction=float(np.mean(cos[mask]>0))))
        # Same 16 observations in each fixed-time window, avoiding length confound.
        for lag in [1,2,4,8,16]:
            span=15*lag
            starts=np.arange(0,len(a)-span,span)
            orders=starts[:,None]+np.arange(16)[None,:]*lag
            if not len(orders):continue
            ll=labels[orders]
            same=(ll[:,0]>=0)&np.all(ll==ll[:,0,None],axis=1)
            for scope,mask in [('within_leaf_16frames',same),('all_16frames',np.ones(len(starts),bool))]:
                if mask.any():
                    mm=metrics(a[orders[mask]])
                    dense.append(dict(video_uid=uid,pooling=pool,lag_sec=lag/4,scope=scope,
                                      n_triples=int(mask.sum()),velocity_cosine_mean=float(np.nanmean(mm['velocity_cosine'])),
                                      velocity_cosine_median=float(np.nanmedian(mm['velocity_cosine'])),
                                      efficiency=float(np.nanmean(mm['efficiency'])),
                                      interior_progress_rho=float(np.nanmean(mm['interior_progress_rho'])),
                                      interior_goal_distance_rho=float(np.nanmean(mm['interior_goal_distance_rho']))))
    print('dense',uid,flush=True)
dense=pd.DataFrame(dense)
dense.to_csv(OUT/'dense_per_video.csv',index=False)
pd.DataFrame(dense_integrity).to_csv(OUT/'dense_integrity.csv',index=False)
ds=[]
for group,sub in dense.groupby(['pooling','lag_sec','scope']):
    row=dict(zip(['pooling','lag_sec','scope'],group));row['n_observations']=int(sub.n_triples.sum())
    for c in ['velocity_cosine_mean','positive_fraction','efficiency','interior_progress_rho','interior_goal_distance_rho']:
        if c in sub and sub[c].notna().any():row[c]=bootstrap_video(sub,c)
    ds.append(row)
json.dump(ds,open(OUT/'dense_summary.json','w'),indent=2)
fig,ax=plt.subplots(1,2,figsize=(11,4),layout='constrained')
for j,pool in enumerate(['cls','meanpatch']):
    for scope in ['within_leaf','cross_leaf_boundary']:
        sub=dense[(dense.pooling==pool)&(dense.scope==scope)]
        grp=sub.groupby('lag_sec').velocity_cosine_mean
        ax[j].plot(grp.mean().index,grp.mean(),'-o',label=scope.replace('_',' '))
    ax[j].axhline(0,color='gray',lw=1);ax[j].set_xscale('log',base=2)
    ax[j].set(xlabel='Seconds between frames',ylabel='Mean adjacent velocity cosine',title='Dense DINOv2 '+pool)
    ax[j].legend()
fig.savefig(OUT/'dense_lag_curves.png',dpi=170);plt.close(fig)

# Illustrative clips: closest to category median efficiency, <=60s, essential leaves.
selected=[]
for term in ['stir','cut','pour','wash','open','peel']:
    sub=df[df.leaf_step_description.str.contains(term,case=False,na=False)&(df.duration>=3)&(df.duration<=60)&(df.leaf_is_relevant=='essential')]
    if sub.empty:continue
    i=(sub.efficiency-sub.efficiency.median()).abs().idxmin();selected.append(i)
canvas=Image.new('RGB',(1920,len(selected)*245),'white');draw=ImageDraw.Draw(canvas)
examples=[]
for row,i in enumerate(selected):
    r=df.loc[i]; vr=VideoReader(r.video_path,ctx=cpu(0),num_threads=1)
    sample=np.linspace(0,len(vr)-1,16).astype(int)
    keep=np.array([0,3,6,9,12,15])
    frames=vr.get_batch(sample[keep]).asnumpy()
    draw.text((10,row*245+5),f'{r.clip_id} | {r.leaf_step_description} | {r.duration:.1f}s | efficiency {r.efficiency:.3f}',fill='black')
    for col,f in enumerate(frames):
        im=Image.fromarray(f);im.thumbnail((310,190))
        canvas.paste(im,(col*320,row*245+30))
        draw.text((col*320+5,row*245+222),f'frame {keep[col]} / relative time {sample[keep[col]]/vr.get_avg_fps():.2f}s',fill='black')
    zz=z[i]; zn=zz/np.linalg.norm(zz,axis=1,keepdims=True)
    c=zz[-1]-zz[0]; p=(zz-zz[0])@c/np.dot(c,c)
    distance=np.linalg.norm(zz-zz[-1],axis=1)
    fig,axs=plt.subplots(1,3,figsize=(12,3.5),layout='constrained')
    axs[0].imshow(zn@zn.T,vmin=0,vmax=1);axs[0].set(title='Frame cosine similarity',xlabel='Sample index',ylabel='Sample index')
    axs[1].plot(p,'o-');axs[1].plot(np.linspace(0,1,16),'--');axs[1].set(title='Projection onto endpoint direction',xlabel='Sample index')
    axs[2].plot(distance,'o-');axs[2].set(title='Distance to final frame',xlabel='Sample index')
    fig.suptitle(r.leaf_step_description+' | '+r.clip_id,fontsize=9)
    fig.savefig(OUT/(r.clip_id+'_geometry.png'),dpi=160);plt.close(fig)
    examples.append(dict(clip_id=r.clip_id,description=r.leaf_step_description,duration=float(r.duration),efficiency=float(r.efficiency),velocity_cosine=float(r.velocity_cosine),interior_progress_rho=float(r.interior_progress_rho),video_path=r.video_path,source_n_frames=len(vr),source_fps=float(vr.get_avg_fps())))
canvas.save(OUT/'video_contact_sheet.jpg')
json.dump(examples,open(OUT/'examples.json','w'),indent=2)
print('complete',OUT,flush=True)
