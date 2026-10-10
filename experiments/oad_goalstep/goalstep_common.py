"""Shared constants for the Ego4D Goal-Step online step detection setup.

Feature timestamps follow the Goal-Step paper's Omnivore features: one vector every 16
frames of 30 fps video (0.533 s, 1.875 per second), so feature k sits at frame 16*k.
"""
import json
import subprocess
from pathlib import Path

DATA_ROOT = Path("/mnt/rdata7/proprogress/data/ego4dgoal")
ANNOTATIONS = DATA_ROOT / "annotations"
VIDEOS = DATA_ROOT / "videos"
STRIDE = 16
FPS = 30.0
FEATURE_FPS = FPS / STRIDE


def load_split(split):
    return json.load(open(ANNOTATIONS / f"goalstep_{split}.json"))["videos"]


def flatten_steps(segments):
    """Every step and substep segment under a goal, as (start, end, category). The paper
    trains and scores steps and substeps together ("SSteps"); they share one taxonomy."""
    out = []
    for seg in segments:
        out.append((seg["start_time"], seg["end_time"], seg["step_category"]))
        out += flatten_steps(seg.get("segments", []))
    return out


def num_frames(uid):
    out = subprocess.check_output(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=nb_frames", "-of", "csv=p=0", str(VIDEOS / f"{uid}.mp4")], text=True)
    return int(out.strip())


def num_features(n_frames):
    return -(-n_frames // STRIDE)


def num_features_at(n_frames, fps, video_fps=FPS):
    """Features when frames are taken at time k / fps (frame floor(k / fps * video_fps)); the
    same rule as predictive_world_model's FeatureProvider.get_dense."""
    return int((n_frames - 1) / video_fps * fps) + 1
