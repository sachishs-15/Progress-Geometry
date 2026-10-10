"""Read-only analysis of existing feature stores; outputs are confined to this folder."""
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
import torch

OUT=Path(__file__).resolve().parent
PROJECT=OUT.parents[1]
DATA=Path('/mnt/rdata7/proprogress/data')
RNG=np.random.default_rng(20261009)


def readcsv(path):
    with open(path,newline='') as f:return list(csv.DictReader(f))


def load(path):
    d=torch.load(path,map_location='cpu',weights_only=False,mmap=True)
    return d['features'],[str(x) for x in d['video_paths']]


def metrics(x):
    x=np.asarray(x,dtype=np.float32)
    v=np.diff(x,axis=1)
    vn=np.linalg.norm(v,axis=2)
    chord=x[:,-1]-x[:,0]
    cn=np.linalg.norm(chord,axis=1)
    length=vn.sum(1)
    straight=cn/np.maximum(length,1e-8)
    velcos=np.sum(v[:,:-1]*v[:,1:],axis=2)/np.maximum(vn[:,:-1]*vn[:,1:],1e-8)
    unit=x/np.maximum(np.linalg.norm(x,axis=2,keepdims=True),1e-8)
    u=unit[:,-1]-unit[:,0]
    p=np.sum((unit-unit[:,0:1])*u[:,None],axis=2)/np.maximum(np.sum(u*u,axis=1,keepdims=True),1e-8)
    # Rank correlation with time (time ranks fixed 0...15).
    ranks=np.argsort(np.argsort(p,axis=1),axis=1).astype(np.float32)
    t=np.arange(x.shape[1],dtype=np.float32)
    ranks-=ranks.mean(1,keepdims=True);t-=t.mean()
    rho=(ranks*t).sum(1)/np.sqrt((ranks*ranks).sum(1)*(t*t).sum())
    dgoal=np.linalg.norm(x[:,:-1]-x[:,-1,None],axis=2)
    dr=np.argsort(np.argsort(dgoal,axis=1),axis=1).astype(np.float32)
    tt=np.arange(x.shape[1]-1,0,-1,dtype=np.float32)
    dr-=dr.mean(1,keepdims=True);tt-=tt.mean()
    goalrho=(dr*tt).sum(1)/np.sqrt((dr*dr).sum(1)*(tt*tt).sum())
    # Interpolation error normalized to within-clip variation. Static clips fail this guard.
    alpha=np.linspace(0,1,x.shape[1],dtype=np.float32)[None,:,None]
    interp=x[:,0:1]+alpha*chord[:,None,:]
    mse=np.sum((x-interp)**2,axis=(1,2))
    var=np.sum((x-x.mean(1,keepdims=True))**2,axis=(1,2))
    r2=1-mse/np.maximum(var,1e-8)
    return {'straight':straight,'velocity_cos':velcos.mean(1),'reversal':(velcos<0).mean(1),
            'time_rho':rho,'goal_rho':goalrho,'monotone_steps':(np.diff(p,axis=1)>=0).mean(1),
            'chord':cn,'path_length':length,'interpolation_r2':r2,
            'adjacent_l2':vn.mean(1),'static_exact':(length<1e-6)}


def summarize(m):
    ret={'n':len(next(iter(m.values())))}
    for k,v in m.items():
        v=v.astype(np.float32) if v.dtype==np.bool_ else v
        ret[k]={'median':float(np.median(v)),'mean':float(np.mean(v)),
                'p10':float(np.percentile(v,10)),'p90':float(np.percentile(v,90))}
    for k,condition in {
      'straight_ge_0.3':m['straight']>=0.3,
      'straight_ge_0.5':m['straight']>=0.5,
      'time_rho_ge_0.8':m['time_rho']>=0.8,
      'goal_rho_ge_0.7':m['goal_rho']>=0.7,
      'interpolation_r2_ge_0.5':m['interpolation_r2']>=0.5,
      'joint_order_and_direct':(m['time_rho']>=0.8)&(m['straight']>=0.3),
      'static_exact':m['static_exact'],
    }.items():ret[k]=float(np.mean(condition))
    return ret


def draw_metric_distributions(all_metrics):
    keys=['straight','time_rho','velocity_cos','interpolation_r2']
    fig,axs=plt.subplots(2,2,figsize=(12,8))
    for ax,key in zip(axs.ravel(),keys):
        for label,m in all_metrics.items():
            ax.hist(m[key],bins=45,density=True,histtype='step',lw=2,label=label)
        ax.set_title(key);ax.set_ylabel('density');ax.grid(alpha=.2)
    axs[0,0].legend(fontsize=8)
    fig.tight_layout();fig.savefig(OUT/'latent_metric_distributions.png',dpi=170);plt.close(fig)


def evaluate_kinetics():
    manifest=readcsv(PROJECT/'data/kinetics-400/k400_train_full.csv')
    label_by_path={r['video_path']:r['label'] for r in manifest}
    groups=defaultdict(list)
    for r in manifest:groups[r['label']].append(r['video_path'])
    selected=set()
    for label in sorted(groups):
        paths=groups[label]
        for ix in RNG.choice(len(paths),size=min(25,len(paths)),replace=False):selected.add(paths[int(ix)])
    results={};all_m={};descriptors={}
    for name,subdir in [('DINOv2_CLS','dinov2'),('VJEPA2.1_meanpatch','vjepa2.1')]:
        x,paths=load(DATA/f'kinetics-dataset/precomputed_features/{subdir}/k400_train_full.pt')
        indices=np.array([i for i,p in enumerate(paths) if p in selected and p in label_by_path],dtype=int)
        xx=x[indices].numpy().astype(np.float32)
        pp=[paths[i] for i in indices]
        m=metrics(xx)
        valid=~m['static_exact']
        excluded=int((~valid).sum())
        xx=xx[valid]
        pp=[p for p,keep in zip(pp,valid) if keep]
        m={k:v[valid] for k,v in m.items()}
        results[name]=summarize(m)
        results[name]['excluded_exact_static']=excluded
        all_m[name]=m
        descriptors[name]=(xx.mean(1),pp)
        # Unsupervised low-dimensional projection, fitted on clip centroids, then tested on trajectories.
        train=RNG.choice(len(xx),size=min(2500,len(xx)),replace=False)
        test=np.setdiff1d(np.arange(len(xx)),train)
        pca=PCA(n_components=64,svd_solver='randomized',random_state=20261009).fit(xx[train].mean(1))
        heldout=xx[test]
        projected=pca.transform(heldout.reshape(-1,heldout.shape[-1])).reshape(len(heldout),16,64)
        results[name+'_PCA64']={**summarize(metrics(projected)),'explained_variance_ratio':float(pca.explained_variance_ratio_.sum())}
        results[name+'_PCA64_paired_raw']=summarize(metrics(heldout))
        # Projected features retain frame order if geometry improves; shuffled intermediate order is a null.
        sh=xx.copy()
        for row in sh:
            row[1:-1]=row[1:-1][RNG.permutation(14)]
        results[name]['shuffled_time_rho_median']=float(np.median(metrics(sh)['time_rho']))
        results[name]['feature_rows']=len(paths)
        with open(OUT/f'kinetics_{subdir}_per_video.csv','w',newline='') as f:
            w=csv.writer(f);w.writerow(['path','label']+list(m))
            for j,p in enumerate(pp):w.writerow([p,label_by_path[p]]+[float(m[k][j]) for k in m])
    # Same-action retrieval without training, excludes self. Reporting macro mean over classes.
    for name,(z,paths) in descriptors.items():
        z=z/np.maximum(np.linalg.norm(z,axis=1,keepdims=True),1e-8)
        labs=np.array([label_by_path[p] for p in paths])
        hits=[]
        for start in range(0,len(z),500):
            sim=z[start:start+500]@z.T
            sim[np.arange(len(sim)),np.arange(start,start+len(sim))]=-np.inf
            pred=labs[np.argmax(sim,axis=1)]
            hits.extend(pred==labs[start:start+len(sim)])
        results[name]['same_class_1nn_accuracy']=float(np.mean(hits))
        results[name]['same_class_chance']=float(np.mean([np.mean(labs==a) for a in labs]))
    draw_metric_distributions(all_m)
    return results


def evaluate_ego():
    rng=np.random.default_rng(20261009)
    x,paths=load(DATA/'ego4dgoal/precomputed_features/dinov2/ego4dgoal_val.pt')
    manifest=readcsv(PROJECT/'data/ego4dgoal/ego4dgoal_val.csv')
    by_path={r['video_path']:r for r in manifest}
    ix=np.array([i for i,p in enumerate(paths) if p in by_path])
    xx=x[ix].numpy().astype(np.float32);pp=[paths[i] for i in ix]
    meta=[by_path[p] for p in pp]
    m=metrics(xx)
    ret={'clips':summarize(m),'n_recordings':len(set(r['video_uid'] for r in meta))}
    z=xx.mean(1)
    z=z/np.maximum(np.linalg.norm(z,axis=1,keepdims=True),1e-8)
    vids=np.array([r['video_uid'] for r in meta]);n=len(z)
    labels={k:np.array([r[k] for r in meta]) for k in ['leaf_step_category','parent_step_category','goal_category']}
    # One nearest neighbor with and without same-recording exclusion.
    nearest_same=[];nearest_cross=[]
    for a in range(0,n,256):
        s=z[a:a+256]@z.T
        s[np.arange(len(s)),np.arange(a,a+len(s))]=-np.inf
        nearest_same.extend(np.argmax(s,axis=1))
        s[vids[a:a+len(s),None]==vids[None,:]]=-np.inf
        nearest_cross.extend(np.argmax(s,axis=1))
    ret['nearest_neighbor_same_recording_fraction']=float(np.mean(vids[np.asarray(nearest_same)]==vids))
    for key,y in labels.items():
        valid=(y!='')
        pairs=[]
        for name,nn in [('unrestricted',nearest_same),('cross_recording',nearest_cross)]:
            hit=np.mean((y[np.asarray(nn)]==y)[valid])
            # Exact frequency baseline conditional on different-recording pool.
            chance=np.mean([np.mean(y[vids!=vids[i]]==y[i]) for i in range(n) if valid[i]])
            pairs.append((name,float(hit)))
        ret[f'{key}_nn']={**dict(pairs),'cross_recording_chance':float(chance),'n_categories':len(set(y))}
    # Balanced pair comparison: same leaf category, different recordings vs different categories.
    by_label=defaultdict(list)
    y=labels['leaf_step_category']
    for i,a in enumerate(y):by_label[a].append(i)
    positive=[];negative=[];positive_direction=[];negative_direction=[]
    direction=xx[:,-1]-xx[:,0]
    direction=direction/np.maximum(np.linalg.norm(direction,axis=1,keepdims=True),1e-8)
    for a,ids in by_label.items():
        if len(ids)<2 or not a:continue
        for _ in range(min(100,len(ids)*2)):
            i,j=rng.choice(ids,2,replace=False)
            if vids[i]!=vids[j]:
                positive.append(float(z[i]@z[j]))
                positive_direction.append(float(direction[i]@direction[j]))
            k=rng.integers(n)
            tries=0
            while (y[k]==a or vids[k]==vids[i]) and tries<20:
                k=rng.integers(n);tries+=1
            if y[k]!=a and vids[k]!=vids[i]:
                negative.append(float(z[i]@z[k]))
                negative_direction.append(float(direction[i]@direction[k]))
    ret['same_leaf_cross_video_cosine']={'n':len(positive),'median':float(np.median(positive)),'mean':float(np.mean(positive))}
    ret['different_leaf_cross_video_cosine']={'n':len(negative),'median':float(np.median(negative)),'mean':float(np.mean(negative))}
    ret['same_leaf_cross_video_direction_cosine']={'n':len(positive_direction),'median':float(np.median(positive_direction)),'mean':float(np.mean(positive_direction))}
    ret['different_leaf_cross_video_direction_cosine']={'n':len(negative_direction),'median':float(np.median(negative_direction)),'mean':float(np.mean(negative_direction))}
    fig,axs=plt.subplots(1,2,figsize=(12,4.5))
    for ax,pos,neg,title in [(axs[0],positive,negative,'Clip centroid cosine'),
                             (axs[1],positive_direction,negative_direction,'Start-to-end direction cosine')]:
        ax.hist(neg,bins=50,density=True,histtype='step',lw=2,label='different leaf action, different recording')
        ax.hist(pos,bins=50,density=True,histtype='step',lw=2,label='same leaf action, different recording')
        ax.set_title(title);ax.set_xlabel('cosine');ax.grid(alpha=.2)
    axs[0].set_ylabel('density');axs[0].legend(fontsize=7)
    fig.tight_layout();fig.savefig(OUT/'ego_cross_video_action_geometry.png',dpi=170);plt.close(fig)
    # Whole recording represented by chronological annotated clip centroids.
    by_video=defaultdict(list)
    for i,r in enumerate(meta):by_video[r['video_uid']].append(i)
    video_rows=[]
    ordered_above_null=[]
    for uid,ids in by_video.items():
        ids.sort(key=lambda i:float(meta[i]['start_time']))
        if len(ids)<3:continue
        zz=np.stack([xx[i].mean(0) for i in ids])
        step=np.linalg.norm(np.diff(zz,axis=0),axis=1)
        eff=float(np.linalg.norm(zz[-1]-zz[0])/max(step.sum(),1e-8))
        chord=zz[-1]-zz[0]
        progress=(zz-zz[0])@chord/max(float(chord@chord),1e-8)
        rho=float(spearmanr(np.arange(len(ids)),progress).statistic)
        null=[];null_eff=[]
        for _ in range(100):
            perm=np.r_[0,rng.permutation(np.arange(1,len(ids)-1)),len(ids)-1]
            null.append(float(spearmanr(np.arange(len(ids)),progress[perm]).statistic))
            shuffled_length=float(np.linalg.norm(np.diff(zz[perm],axis=0),axis=1).sum())
            null_eff.append(float(np.linalg.norm(chord)/max(shuffled_length,1e-8)))
        above=rho>np.percentile(null,95)
        ordered_above_null.append(above)
        video_rows.append({'video_uid':uid,'n_clips':len(ids),'goal':meta[ids[0]]['goal_category'],'sparse_chord_efficiency':eff,'shuffle_median_efficiency':float(np.median(null_eff)),'time_rho':rho,'beats_95pct_endpoint_preserving_shuffle':int(above),'n_parent_steps':len(set(meta[i]['parent_step_idx'] for i in ids))})
    ret['whole_video_sparse']={'n':len(video_rows),'median_efficiency':float(np.median([r['sparse_chord_efficiency'] for r in video_rows])),'fraction_efficiency_ge_0.2':float(np.mean([r['sparse_chord_efficiency']>=.2 for r in video_rows]))}
    ret['whole_video_sparse']['median_time_rho']=float(np.median([r['time_rho'] for r in video_rows]))
    ret['whole_video_sparse']['fraction_order_above_shuffle95']=float(np.mean(ordered_above_null))
    ret['whole_video_sparse']['median_shuffled_efficiency']=float(np.median([r['shuffle_median_efficiency'] for r in video_rows]))
    with open(OUT/'ego_whole_geometry.csv','w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=video_rows[0].keys());w.writeheader();w.writerows(video_rows)
    # PCA shows only global appearance structure; numeric claims above use full dimensions.
    pca_model=PCA(n_components=2,random_state=20261009).fit(z)
    pca=pca_model.transform(z)
    ret['pca2_explained_variance_fraction']=float(pca_model.explained_variance_ratio_.sum())
    biggest=[k for k,_ in Counter(labels['goal_category']).most_common(8)]
    fig,ax=plt.subplots(figsize=(10,8))
    for label in biggest:
        mask=labels['goal_category']==label
        ax.scatter(pca[mask,0],pca[mask,1],s=5,alpha=.35,label=label[:40])
    ax.legend(fontsize=7,markerscale=3);ax.set_xlabel('PC1');ax.set_ylabel('PC2')
    ax.set_title('Ego4D GoalStep clip centroids: top 8 goal categories (DINOv2 CLS)')
    fig.tight_layout();fig.savefig(OUT/'ego_goal_pca.png',dpi=170);plt.close(fig)
    with open(OUT/'ego_per_clip_geometry.csv','w',newline='') as f:
        w=csv.writer(f);w.writerow(['path','video_uid','leaf_step_category','goal_category']+list(m))
        for i,r in enumerate(meta):w.writerow([pp[i],r['video_uid'],r['leaf_step_category'],r['goal_category']]+[float(m[k][i]) for k in m])
    return ret


if __name__=='__main__':
    k=evaluate_kinetics()
    e=evaluate_ego()
    with open(OUT/'feature_analysis.json','w') as f:json.dump({'kinetics':k,'ego4d':e},f,indent=2)
    print(json.dumps({'kinetics':{key:{'n':v.get('n'),'straight':v.get('straight',{}).get('median'),'time_rho':v.get('time_rho',{}).get('median')} for key,v in k.items()},'ego4d':e['whole_video_sparse']},indent=2))
