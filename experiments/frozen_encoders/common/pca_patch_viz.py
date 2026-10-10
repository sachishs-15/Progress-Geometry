"""DINOv2-style global-PCA patch coloring, applied across an entire video.

Every sampled frame's patch tokens (CLS excluded) get concatenated into one
(T*P, D) matrix, a single PCA(3) is fit over that whole matrix (never per
frame), and each patch is colored by its PC1/PC2/PC3 as R/G/B — normalized
globally across the video, not per frame, so color is directly comparable
across time. Background patches (PC1 <= threshold, in the PCA's own centered
units) are optionally zeroed out.
"""
from typing import List, Tuple

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from sklearn.decomposition import PCA


def compute_pca_patch_colors(
    patch_tokens: torch.Tensor,
    grid_hw: Tuple[int, int],
    remove_background: bool = True,
    threshold: float = 0.0,
) -> np.ndarray:
    """
    Args:
        patch_tokens: (T, P, D) raw patch embeddings across all sampled frames.
        grid_hw: (grid_h, grid_w) spatial layout of the P patches — used to
            orient the foreground/background split (see below).
        remove_background: zero out patches classified as background.
        threshold: PC1 cutoff between the two sides of the split; 0.0 is the
            natural choice since PCA output is already mean-centered.

    Returns:
        (T, P, 3) uint8 RGB, one solid color per patch.
    """
    t, p, d = patch_tokens.shape
    gh, gw = grid_hw
    flat = patch_tokens.reshape(t * p, d).detach().cpu().numpy()

    pca = PCA(n_components=3)
    proj = pca.fit_transform(flat)  # (T*P, 3), single global fit — not per frame

    # PCA's sign is arbitrary (could point either way), so "PC1 > threshold"
    # isn't reliably "foreground" — it flips depending on the checkpoint/data.
    # A plain majority/minority vote isn't reliable either: it assumes the
    # subject is smaller than the background, which fails whenever the subject
    # fills most of the (typically center-cropped) frame. Instead, assume the
    # subject is roughly centered and the background sits at the edges — pick
    # whichever side of the split is more concentrated in the frame's center.
    side_a = proj[:, 0] > threshold
    side_a_grid = side_a.reshape(t, gh, gw)
    cy0, cy1 = gh // 4, gh - gh // 4
    cx0, cx1 = gw // 4, gw - gw // 4
    center_frac = side_a_grid[:, cy0:cy1, cx0:cx1].mean() if cy1 > cy0 and cx1 > cx0 else side_a.mean()
    foreground = side_a if center_frac >= 0.5 else ~side_a

    mins = proj.min(axis=0)
    maxs = proj.max(axis=0)
    norm = (proj - mins) / (maxs - mins + 1e-8)  # global [0, 1] per component, across the whole video

    colors = (norm * 255).astype(np.uint8)
    if remove_background:
        colors[~foreground] = 0

    return colors.reshape(t, p, 3)


def render_patch_color_frame(color_grid: np.ndarray, out_size: Tuple[int, int]) -> Image.Image:
    """color_grid: (grid_h, grid_w, 3) uint8 -> upscaled with NEAREST so each
    patch stays one flat solid-color block instead of being smoothed."""
    return Image.fromarray(color_grid, mode="RGB").resize(out_size, Image.NEAREST)


def make_pca_patch_video(
    frames: List[Image.Image],
    patch_tokens: torch.Tensor,
    grid_hw: Tuple[int, int],
    out_path: str,
    model_name: str,
    remove_background: bool = True,
    threshold: float = 0.0,
    fps: int = 4,
) -> None:
    gh, gw = grid_hw
    t = len(frames)
    colors = compute_pca_patch_colors(patch_tokens, grid_hw, remove_background=remove_background, threshold=threshold)
    color_grids = colors.reshape(t, gh, gw, 3)

    fig, (ax_frame, ax_pca) = plt.subplots(1, 2, figsize=(10, 5))
    ax_frame.axis("off")
    ax_pca.axis("off")
    frame_im = ax_frame.imshow(frames[0])
    ax_frame.set_title(f"{model_name} — frame 0/{t - 1}")
    pca_im = ax_pca.imshow(render_patch_color_frame(color_grids[0], frames[0].size))
    bg_note = "bg removed" if remove_background else "bg kept"
    ax_pca.set_title(f"global PCA patch colors ({bg_note})", fontsize=10)
    fig.tight_layout()

    def update(i):
        frame_im.set_data(np.array(frames[i]))
        ax_frame.set_title(f"{model_name} — frame {i}/{t - 1}")
        pca_im.set_data(np.array(render_patch_color_frame(color_grids[i], frames[i].size)))
        return [frame_im, pca_im]

    anim = animation.FuncAnimation(fig, update, frames=t, blit=False)
    anim.save(out_path, writer=animation.FFMpegWriter(fps=fps))
    plt.close(fig)
