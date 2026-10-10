"""Summarize dense duplicate sensitivity and existing trained-model evaluations."""
from pathlib import Path
import json
import pandas as pd

root = Path(__file__).parent
out = root / 'results'
integrity = pd.read_csv(out / 'dense_integrity.csv')
dense = pd.read_csv(out / 'dense_per_video.csv')
good = set(integrity.loc[(integrity.pooling == 'cls') &
                         (integrity.duplicate_adjacent_fraction < .01), 'video_uid'])
subset = dense[dense.video_uid.isin(good)]
subset.groupby(['pooling', 'lag_sec', 'scope']).agg(
    n_videos=('video_uid', 'nunique'),
    velocity_cosine_mean=('velocity_cosine_mean', 'mean')
).to_csv(out / 'dense_low_duplicate_sensitivity.csv')

saved = {}
base = root.parent / 'progress_metrics/results/ego4dgoal_val_patch'
for name in ['raw_dinov2_noreg_mean', 'pwm_ego4dgoal_16f_best']:
    data = json.load(open(base / (name + '.json')))
    saved[name] = {k: data[k] for k in [
        'checkpoint', 'checkpoint_epoch', 'n_clips', 'features', 'manifest',
        'effective_rank', 'effective_rank_fraction', 'summary_mean', 'summary_median'
    ]}
with open(out / 'saved_patch_model_comparison.json', 'w') as f:
    json.dump(saved, f, indent=2)

for path in out.glob('*.json'):
    json.load(open(path))
assert len(pd.read_csv(out / 'clip_diagnostics.csv')) == 6732
assert len(integrity) == 268
print('Artifacts verified: readable JSONs, 6,732 clips, 134 dense videos per pooling.')
