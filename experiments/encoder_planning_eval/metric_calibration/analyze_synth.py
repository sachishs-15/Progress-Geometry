import sys, glob
import numpy as np, pandas as pd, torch
from metrics_batch import per_clip_metrics, dataset_metrics
import synth_videos as sv
KEYS=["time_progress_spearman","trajectory_straightness","mean_velocity_cosine_raw","direction_reversal_fraction","adjacent_temporal_nn_fraction","distance_to_goal_spearman","monotonic_progress_fraction"]
rows=[]
encs=[p.split("/")[-1][:-3] for p in sorted(glob.glob("results/synth/*.pt"))]
for e in encs:
    d=torch.load(f"results/synth/{e}.pt")
    for c in sv.CONDITIONS:
        z=d[c].numpy().astype(np.float64)
        pc,keep=per_clip_metrics(z)
        r={"encoder":e,"cond":c,"kept":int(keep.sum())}
        r.update({k:float(pc[k][keep].mean()) if keep.sum() else np.nan for k in KEYS})
        if keep.sum(): r["adjacent_l2"]=float(pc["adjacent_l2_distance"][keep].mean())
        rows.append(r)
df=pd.DataFrame(rows); df.to_csv("results/synth_metrics.csv",index=False)
pd.set_option("display.width",250); pd.set_option("display.max_columns",30)
for k in KEYS[:6]:
    print("\n##",k); print(df.pivot(index="cond",columns="encoder",values=k).reindex(sv.CONDITIONS).round(3).to_string())
