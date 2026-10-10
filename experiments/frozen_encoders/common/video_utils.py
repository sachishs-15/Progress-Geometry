"""Stride-based video frame loading, shared across all model plugins."""
from typing import List, Tuple

from decord import VideoReader, cpu
from PIL import Image


def load_frames_stride(
    video_path: str, stride: int = 8, max_frames: int = 64, max_size: int = 480,
) -> Tuple[List[Image.Image], List[int]]:
    """Load every `stride`-th frame from a video, up to `max_frames` frames.

    Args:
        max_size: cap the longer edge at this many pixels (aspect preserved).
            Every model here internally resizes to ~224-256px anyway, so full
            native resolution (e.g. 2560x1920 for some source videos) buys
            nothing for the embeddings and is the single biggest cost in the
            visualization pipeline — attention/PCA overlays get resized up to
            whatever size `frames` are, and that's what matplotlib/ffmpeg then
            have to push per animation frame. Downscaling once here speeds up
            everything downstream. Set to None to keep native resolution.

    Returns:
        frames: list of PIL Images (RGB).
        frame_indices: original frame index in the video for each returned frame.
    """
    # num_threads=1: decord's default multi-threaded decoder has a known race
    # condition (fails intermittently with "Error sending packet" on some
    # videos/codecs) — single-threaded avoids it entirely.
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    total = len(vr)
    indices = list(range(0, total, stride))[:max_frames]
    batch = vr.get_batch(indices).asnumpy()  # (T, H, W, 3)
    frames = [Image.fromarray(f) for f in batch]

    if max_size is not None and frames and max(frames[0].size) > max_size:
        scale = max_size / max(frames[0].size)
        new_size = (max(1, round(frames[0].width * scale)), max(1, round(frames[0].height * scale)))
        frames = [f.resize(new_size, Image.BILINEAR) for f in frames]

    return frames, indices
