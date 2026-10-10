"""Main runner: pick which frozen encoder(s) to run over a video and compare.

Each model writes its own PNG (frame strip [+ attention strip] + temporal
dimred trajectory) plus raw embeddings to `models/<name>/outputs/`, so results
across models are directly comparable.

Usage:
    python run.py --video ../../external/chirality_in_action/sample_videos/sample1.mp4 \
        --models dinov2 vjepa2 --stride 4 --pooling mean_patch --method tsne
"""
import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import models  # noqa: E402  (import triggers registration of all plugins)
from common.base import REGISTRY, build  # noqa: E402
from common.video_utils import load_frames_stride  # noqa: E402
from common.viz import make_progression_video, ordinality_score, trajectory_curvature  # noqa: E402


def run_one(model_name, video_path, stride, max_frames, pooling, method, device, n_show, make_video, fps,
            pca_video, remove_background, pca_threshold, hf_id):
    frames, video_frame_indices = load_frames_stride(video_path, stride=stride, max_frames=max_frames)
    print(f"[{model_name}] loaded {len(frames)} frames (stride={stride}) from {video_path}")

    out_dir = Path(__file__).parent / "models" / model_name / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{Path(video_path).stem}_stride{stride}_{pooling}_{method}"

    encoder_kwargs = dict(device=device, pooling=pooling)
    if hf_id:
        encoder_kwargs["hf_id"] = hf_id
    encoder = build(model_name, **encoder_kwargs)
    patch_tokens, patch_grid_hw = None, None
    with encoder:
        result = encoder.encode(frames)

        if pca_video and make_video:
            if encoder.supports_patch_tokens:
                patch_tokens, patch_grid_hw = encoder.get_patch_tokens(frames)
            else:
                print(f"[{model_name}] does not expose raw patch tokens, skipping PCA patch-color column")

    print(f"[{model_name}] embeddings: {result.embeddings.shape}, "
          f"attention: {'yes' if result.attention_maps is not None else 'no'}")

    # make_summary_figure was computed here but its png/npz save was already
    # disabled below, making the whole call (full TSNE fit + rendering, ~9s on
    # a 32-frame run) pure waste — make_progression_video recomputes the same
    # dimred anyway, so just get the scores directly instead.
    score = ordinality_score(result.embeddings)
    curvature = trajectory_curvature(result.embeddings)
    # np.savez(
    #     out_dir / f"{tag}.npz",
    #     embeddings=result.embeddings,
    #     frame_indices=result.frame_indices,
    #     video_frame_indices=video_frame_indices,
    #     ordinality_score=score,
    #     curvature=curvature,
    # )
    print(f"[{model_name}] ordinality score = {score:.3f}, curvature = {curvature:.1f}° (lower = straighter)")

    metrics_path = out_dir / f"{tag}_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "model": model_name,
            "video": Path(video_path).name,
            "stride": stride,
            "max_frames": max_frames,
            "pooling": pooling,
            "method": method,
            "num_frames": len(frames),
            "ordinality_score": round(score, 4),
            "curvature_deg": None if np.isnan(curvature) else round(curvature, 2),
            "video_path": str(out_dir / f"{tag}.mp4") if make_video else None,
        }, f, indent=2)
    print(f"[{model_name}] metrics -> {metrics_path}")

    if make_video:
        make_progression_video(
            frames=frames,
            embeddings=result.embeddings,
            out_path=str(out_dir / f"{tag}.mp4"),
            model_name=model_name,
            attn_maps=result.attention_maps,
            attn_frame_indices=result.frame_indices if result.attention_maps is not None else None,
            attn_label=encoder.attention_label,
            embedding_frame_indices=result.frame_indices,
            patch_tokens=patch_tokens,
            patch_grid_hw=patch_grid_hw,
            remove_background=remove_background,
            pca_threshold=pca_threshold,
            method=method,
            fps=fps,
        )
        print(f"[{model_name}] progression video -> {out_dir / tag}.mp4")

    return {
        "score": score,
        "native_curvature": curvature,
        "embeddings": result.embeddings,
        "frame_indices": np.asarray(result.frame_indices),
        "metrics_path": metrics_path,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--video", required=True)
    parser.add_argument("--models", nargs="+", default=list(REGISTRY), choices=list(REGISTRY))
    parser.add_argument("--stride", type=int, default=4, help="take every Nth frame from the video")
    parser.add_argument("--max-frames", type=int, default=64, help="cap on number of sampled frames")
    parser.add_argument("--pooling", default="mean_patch", choices=["cls", "pooler", "mean_patch", "max_patch"])
    parser.add_argument("--method", default="tsne", choices=["tsne", "pca"])
    parser.add_argument("--n-show", type=int, default=8, help="frames shown in the strip")
    parser.add_argument("--device", default=None)
    parser.add_argument("--no-video", action="store_true", help="skip the mp4 progression video, PNG only")
    parser.add_argument("--fps", type=int, default=4, help="fps for the progression video")
    parser.add_argument("--no-pca-video", action="store_true",
                         help="skip the global-PCA patch-color column in the progression video (needs get_patch_tokens support)")
    parser.add_argument("--remove-background", dest="remove_background", action="store_true", default=True,
                         help="zero out background patches (PC1 <= --pca-threshold) in the PCA column (default: on)")
    parser.add_argument("--keep-background", dest="remove_background", action="store_false",
                         help="disable background removal in the PCA column")
    parser.add_argument("--pca-threshold", type=float, default=0.0,
                         help="PC1 cutoff (in PCA's own centered units) for foreground vs background")
    parser.add_argument("--hf-id", default=None,
                         help="override a model's default HF checkpoint, e.g. facebook/dinov2-with-registers-large for ViT-L/14")
    args = parser.parse_args()

    scores = {}
    for name in args.models:
        pooling = args.pooling
        if pooling not in REGISTRY[name].pooling_choices:
            pooling = REGISTRY[name].pooling_choices[0]
            print(f"[{name}] does not support pooling={args.pooling!r}, falling back to {pooling!r}")
        scores[name] = run_one(
            name, args.video, args.stride, args.max_frames, pooling, args.method, args.device, args.n_show,
            not args.no_video, args.fps,
            not args.no_pca_video, args.remove_background, args.pca_threshold, args.hf_id,
        )

    # Curvature changes with temporal sampling density. Compare every encoder only at
    # source-frame positions present in all outputs (notably V-JEPA tubelets emit fewer
    # positions than per-frame image encoders), while retaining the native value as a
    # diagnostic in each metrics file.
    common_indices = sorted(set.intersection(*(
        set(map(int, result["frame_indices"])) for result in scores.values()
    )))
    for result in scores.values():
        row_for_index = {int(index): row for row, index in enumerate(result["frame_indices"])}
        matched = result["embeddings"][[row_for_index[index] for index in common_indices]]
        result["curvature"] = trajectory_curvature(matched)
        with open(result["metrics_path"]) as f:
            metrics = json.load(f)
        native = result["native_curvature"]
        matched_curvature = result["curvature"]
        metrics.update({
            "curvature_deg_native": None if np.isnan(native) else round(native, 2),
            "curvature_deg": None if np.isnan(matched_curvature) else round(matched_curvature, 2),
            "curvature_frame_indices": common_indices,
        })
        with open(result["metrics_path"], "w") as f:
            json.dump(metrics, f, indent=2)

    print("\n=== Ordinality scores (|Spearman| of PC1 vs frame order, higher = more ordinal) ===")
    for name, result in sorted(scores.items(), key=lambda kv: -kv[1]["score"]):
        print(f"  {name:10s} {result['score']:.3f}")

    print("\n=== Trajectory curvature (deg, perceptual straightening, lower = straighter) ===")
    for name, result in sorted(scores.items(), key=lambda kv: kv[1]["curvature"]):
        print(f"  {name:10s} {result['curvature']:.1f}")
