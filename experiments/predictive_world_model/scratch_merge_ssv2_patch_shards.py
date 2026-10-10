"""One-off: merge ssv2_train.pt.shard{0..3}.pt (dinov2_noreg patch features) into a
single ssv2_train.pt, matching the sibling ssv2_validation.pt already in that directory.

Not scripts/local/precompute_features/merge_features.py: that script requires
extraction_metadata/extraction_fingerprint on every shard for a checksum cross-check,
and these four shards have neither (confirmed: keys are only features/n_frames/
video_paths) -- pre-dates whatever extraction run added that field. Legacy shards per
merge_features.py's own error message. This is the no-validation equivalent for exactly
this shape, used once, not a replacement for that script.
"""
import glob
import torch

shard_paths = sorted(glob.glob(
    "/mnt/rdata7/proprogress/data/SSv2/precomputed_features/dinov2_noreg/ssv2_train.pt.shard*.pt"))
out_path = "/mnt/rdata7/proprogress/data/SSv2/precomputed_features/dinov2_noreg/ssv2_train.pt"
print(f"merging {len(shard_paths)} shards -> {out_path}")

shards = [torch.load(p, map_location="cpu") for p in shard_paths]
n_frames = shards[0]["n_frames"]
for p, s in zip(shard_paths, shards):
    assert s["n_frames"] == n_frames, f"{p}: n_frames {s['n_frames']} != {n_frames}"

features = torch.cat([s["features"] for s in shards], dim=0)
video_paths = [vp for s in shards for vp in s["video_paths"]]
print(f"merged: {len(video_paths)} clips, features shape {tuple(features.shape)}")

torch.save({"features": features, "video_paths": video_paths, "n_frames": n_frames}, out_path)
print(f"saved -> {out_path}")
