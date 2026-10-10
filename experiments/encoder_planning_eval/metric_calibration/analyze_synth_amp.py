import glob
import numpy as np, pandas as pd, torch
from metrics_batch import per_clip_metrics
import synth_videos as sv
KEYS=["time_progress_spearman","trajectory_straightness","mean_velocity_cosine_raw","direction_reversal_fraction","adjacent_temporal_nn_fraction","distance_to_goal_spearman"]
rows=[]
for p in sorted(glob.glob("results/synth_amp/*.pt")):
    e=p.split("/")[-1][:-3]; d=torch.load(p)
    for c in sv.AMPLITUDE_CONDITIONS:
        z=d[c].numpy().astype(np.float64); pc,keep=per_clip_metrics(z)
        rows.append({"encoder":e,"radius_px":int(c.split("_r")[1]),**{k:float(pc[k][keep].mean()) for k in KEYS}})
df=pd.DataFrame(rows); df.to_csv("results/synth_amp_metrics.csv",index=False)
pd.set_option("display.width",250)
for k in KEYS[:4]: print("\n##",k); print(df.pivot(index="encoder",columns="radius_px",values=k).round(3).to_string())
