"""Convert Something-Something V2 (raw short clips, 2-6s each) into this project's
unified feature cache (data/cache.py), reusing `experiments/frozen_encoders`'s existing
frozen ViT plugins (DINOv2/DINOv3/CLIP/SigLIP/...) to turn sampled frames into per-timestep
feature vectors — SSv2 ships raw video, not pre-extracted features like CrossTask.

NOT run yet: SSv2 requires a registered account on Qualcomm's AI-datasets portal (no
public direct-download URL like CrossTask), so getting the files on disk is a manual step.
Once downloaded and extracted, point --raw-dir at the directory containing the JSONs below
and --videos-dir at the directory of <id>.{webm,mp4} files.

Expected raw layout (verified against facebookresearch/SlowFast's Ssv2 dataset loader,
which documents the same official file/JSON schema — not by downloading SSv2 itself):
  <raw_dir>/something-something-v2-labels.json       {template_without_brackets: class_id}
  <raw_dir>/something-something-v2-train.json        [{"id","label","template","placeholders"}, ...]
  <raw_dir>/something-something-v2-validation.json   same schema as train
  <raw_dir>/something-something-v2-test.json         {"id"} only, unlabeled — not used here
  <videos_dir>/<id>.webm  or  <videos_dir>/<id>.mp4   one raw clip per id — the original
                                                       2017 TwentyBN release used .webm,
                                                       but current Qualcomm-hosted mirrors
                                                       appear to redistribute .mp4 instead
                                                       (per MMAction2's docs), so both
                                                       extensions are tried (find_video()).

There are no procedural sub-step annotations for SSv2 (each clip is one templated
action) — every video's `step_labels` in the cache is left None, so only
`dataset.progress_type: timestamp` is valid for this dataset, not `step`.

We pool train.json + validation.json ourselves and let this project's own
data/splits.py do the (video-level, stratified-by-template) train/val/test split,
exactly like prepare_crosstask.py — SSv2's own train/val assignment isn't used.
`template` (the "Doing [something] to [something]" string, brackets stripped) is used as
`task_id`, the closest analog to a CrossTask task: many videos share one template.

Note: decord's webm/VP9 support can be inconsistent across builds; if a clip fails to
decode, this script skips it and logs a count rather than aborting the whole run — if
that count is large, re-encoding the affected clips to mp4 with ffmpeg first is the usual
fix.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

TEMPORAL_ORDERING_ROOT = Path(__file__).resolve().parent.parent
FROZEN_ENCODERS_ROOT = TEMPORAL_ORDERING_ROOT.parent / "frozen_encoders"

sys.path.insert(0, str(TEMPORAL_ORDERING_ROOT))
from data.cache import write_manifest, write_video_cache  # noqa: E402
from utils.config import load_config  # noqa: E402

# Inserted after temporal_ordering's own root, so it's checked first for `import models` /
# `import common` — this script never needs temporal_ordering's own `models` package, only
# frozen_encoders', so there's no name collision within this process.
sys.path.insert(0, str(FROZEN_ENCODERS_ROOT))
import models as _frozen_encoder_plugins  # noqa: E402,F401  (import registers all plugins)
from common.base import REGISTRY as FROZEN_ENCODER_REGISTRY, build as build_frozen_encoder  # noqa: E402
from common.video_utils import load_frames_stride  # noqa: E402


VIDEO_EXTENSIONS = (".webm", ".mp4")


def _normalize_template(template: str) -> str:
    return template.replace("[", "").replace("]", "").strip()


def find_video(videos_dir: Path, video_id: str) -> Path:
    """The original TwentyBN release used .webm; some current re-hosted mirrors (e.g. the
    archive MMAction2's docs extract) use .mp4 instead — try both rather than assuming."""
    for ext in VIDEO_EXTENSIONS:
        path = videos_dir / f"{video_id}{ext}"
        if path.exists():
            return path
    return videos_dir / f"{video_id}{VIDEO_EXTENSIONS[0]}"  # doesn't exist; caller checks


def load_pool(raw_dir: Path, splits_json) -> list:
    entries = []
    for name in splits_json:
        path = raw_dir / f"something-something-v2-{name}.json"
        with open(path) as f:
            entries.extend(json.load(f))
    return entries


def subsample(entries: list, max_videos_per_template: int, max_templates, seed: int) -> list:
    rng = np.random.RandomState(seed)
    by_template = {}
    for e in entries:
        by_template.setdefault(_normalize_template(e["template"]), []).append(e)

    templates = sorted(by_template)
    if max_templates is not None and len(templates) > max_templates:
        templates = list(rng.choice(templates, size=max_templates, replace=False))

    kept = []
    for t in templates:
        vids = by_template[t]
        rng.shuffle(vids)
        kept.extend(vids[:max_videos_per_template] if max_videos_per_template else vids)
    return kept


def main(args):
    raw_dir = Path(args.raw_dir)
    videos_dir = Path(args.videos_dir) if args.videos_dir else raw_dir / "20bn-something-something-v2"

    entries = load_pool(raw_dir, ["train", "validation"])
    entries = subsample(entries, args.max_videos_per_template, args.max_templates, args.seed)
    print(f"processing {len(entries)} clips across "
          f"{len({_normalize_template(e['template']) for e in entries})} templates")

    if args.encoder not in FROZEN_ENCODER_REGISTRY:
        raise KeyError(f"Unknown --encoder {args.encoder!r}. Available: {sorted(FROZEN_ENCODER_REGISTRY)}")
    encoder_kwargs = dict(device=args.device, pooling=args.pooling)
    if args.hf_id:
        encoder_kwargs["hf_id"] = args.hf_id
    encoder = build_frozen_encoder(args.encoder, **encoder_kwargs)

    video_to_task = {}
    feature_dim = None
    n_missing, n_decode_failed, n_too_short = 0, 0, 0
    with encoder:
        for i, entry in enumerate(entries):
            video_id = entry["id"]
            task_id = _normalize_template(entry["template"])
            video_path = find_video(videos_dir, video_id)
            if not video_path.exists():
                n_missing += 1
                continue
            try:
                frames, _frame_indices = load_frames_stride(
                    str(video_path), stride=args.stride, max_frames=args.max_frames,
                )
            except Exception as e:  # decord can throw various backend-specific errors
                print(f"[{video_id}] failed to decode: {e}")
                n_decode_failed += 1
                continue
            if len(frames) < 2:
                n_too_short += 1
                continue

            result = encoder.encode(frames)
            feature_dim = result.embeddings.shape[1]
            write_video_cache(args.out, video_id, task_id, result.embeddings, step_labels=None)
            video_to_task[video_id] = task_id

            if (i + 1) % 100 == 0:
                print(f"  {i + 1}/{len(entries)} processed ({len(video_to_task)} written)")

    write_manifest(args.out, name="ssv2", feature_dim=feature_dim, video_to_task=video_to_task)
    print(f"wrote {len(video_to_task)} videos ({n_missing} missing file, {n_decode_failed} failed to "
          f"decode, {n_too_short} too short) -> {args.out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None, help="optional yaml overriding the CLI defaults below")
    parser.add_argument("--raw-dir", default="data_cache/_ssv2_raw", help="dir with the SSv2 JSON files")
    parser.add_argument("--videos-dir", default=None,
                         help="dir with <id>.webm files (default: <raw-dir>/20bn-something-something-v2)")
    parser.add_argument("--out", default="data_cache/ssv2", help="unified cache output dir")
    parser.add_argument("--encoder", default="dinov2", help=f"one of {sorted(FROZEN_ENCODER_REGISTRY)}")
    parser.add_argument("--hf-id", default=None, help="override the encoder's default HF checkpoint")
    parser.add_argument("--pooling", default="mean_patch")
    parser.add_argument("--stride", type=int, default=2, help="take every Nth frame (clips are short: 2-6s)")
    parser.add_argument("--max-frames", type=int, default=32)
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-videos-per-template", type=int, default=20,
                         help="subsample per template — SSv2 has ~220k clips total; 0 = keep all")
    parser.add_argument("--max-templates", type=int, default=None, help="cap number of distinct templates used")
    parser.add_argument("--seed", type=int, default=42)
    cli_args = parser.parse_args()
    if cli_args.config:
        for key, value in load_config(cli_args.config).items():
            setattr(cli_args, key, value)
    main(cli_args)
