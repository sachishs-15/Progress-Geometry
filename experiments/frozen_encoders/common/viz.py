"""Model-agnostic outputs: temporal dimred trajectory, an ordinality score, and
attention overlays. Every model plugin's output goes through this so results are
directly comparable regardless of architecture.
"""
from typing import List, Optional, Tuple

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def reduce_dim(x: np.ndarray, method: str = "tsne", n_components: int = 2, perplexity: float = None) -> np.ndarray:
    if method == "pca":
        return PCA(n_components=n_components).fit_transform(x)
    if method == "tsne":
        perplexity = perplexity or min(30.0, max(5.0, len(x) / 3))
        return TSNE(n_components=n_components, perplexity=perplexity, init="pca", random_state=0).fit_transform(x)
    raise ValueError(f"Unknown dimred method: {method}")


def ordinality_score(x: np.ndarray) -> float:
    """|Spearman correlation| between temporal order and position along the
    embedding trajectory's top principal component. A quick proxy for whether
    the raw embedding space is ordinal/linear in time (1.0 = perfectly ordinal).
    """
    pc1 = PCA(n_components=1).fit_transform(x)[:, 0]
    rho, _ = spearmanr(np.arange(len(x)), pc1)
    return float(abs(rho))


def trajectory_curvature(x: np.ndarray) -> float:
    """Mean angle (degrees) between consecutive step vectors along the raw
    (T, D) embedding trajectory — the actual metric from perceptual
    straightening (Henaff et al., Nature Neuroscience 2019), the concept the
    CiA/LiFT project this framework is inspired by is built around.

    Unlike `ordinality_score`, this needs no PCA/global-axis assumption — it
    works directly in the full embedding space, so it also catches
    curved-but-smooth trajectories (e.g. a spiral) that a single-axis score
    would misjudge as non-ordinal.

    Returns:
        Mean curvature in degrees, in [0, 180]. 0 = perfectly straight line
        (each step continues in the same direction as the last); larger =
        more curved; ~90 = consecutive steps are unrelated in direction.
        NaN if there are fewer than 3 points or no pair of consecutive nonzero steps.
    """
    if len(x) < 3:
        return float("nan")
    v = np.diff(x, axis=0)  # (T-1, D) step vectors
    norms = np.linalg.norm(v, axis=1)
    valid = (norms[:-1] > 1e-8) & (norms[1:] > 1e-8)
    if not np.any(valid):
        # A zero step has no direction. Treating its normalized value as the zero
        # vector makes arccos(0)=90 degrees and falsely describes collapse as curvature.
        return float("nan")
    denom = norms[:-1][valid] * norms[1:][valid]
    cos_theta = np.clip(np.sum(v[:-1][valid] * v[1:][valid], axis=1) / denom, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_theta)).mean())


def overlay_attention_on_frame(frame: Image.Image, attn_map: np.ndarray, alpha: float = 0.55, cmap: str = "jet") -> Image.Image:
    w, h = frame.size
    attn_map = attn_map - attn_map.min()
    attn_map = attn_map / (attn_map.max() + 1e-8)
    attn_resized = np.array(
        Image.fromarray((attn_map * 255).astype(np.uint8)).resize((w, h), Image.BICUBIC)
    ).astype(np.float32) / 255.0
    heatmap = plt.get_cmap(cmap)(attn_resized)[:, :, :3] * 255
    frame_np = np.array(frame.convert("RGB")).astype(np.float32)
    blended = (alpha * heatmap + (1 - alpha) * frame_np).clip(0, 255).astype(np.uint8)
    return Image.fromarray(blended)


def concat_images(images: List[Image.Image], border: int = 6, bg=(255, 255, 255)) -> Image.Image:
    h = max(im.height for im in images)
    resized = [im.resize((max(1, int(im.width * h / im.height)), h)) for im in images]
    total_w = sum(im.width for im in resized) + border * (len(resized) + 1)
    canvas = Image.new("RGB", (total_w, h + 2 * border), bg)
    x = border
    for im in resized:
        canvas.paste(im, (x, border))
        x += im.width + border
    return canvas


def make_summary_figure(
    frames: List[Image.Image],
    embeddings: np.ndarray,
    model_name: str,
    attn_maps: Optional[np.ndarray] = None,
    attn_frame_indices: Optional[List[int]] = None,
    attn_label: str = "attention",
    method: str = "tsne",
    n_show: int = 8,
) -> Tuple[Image.Image, float]:
    """One PNG per run: frame strip [+ attention strip] + temporal dimred trajectory.

    `attn_frame_indices[i]` gives the position in `frames` that `attn_maps[i]`
    corresponds to (needed because some encoders emit fewer attention maps than
    input frames, e.g. one per temporal tubelet instead of one per frame).
    """
    show_pos = np.linspace(0, len(frames) - 1, min(n_show, len(frames)), dtype=int)

    n_rows = 3 if attn_maps is not None else 2
    height_ratios = [0.18, 0.18, 0.64] if attn_maps is not None else [0.22, 0.78]
    fig, axes = plt.subplots(n_rows, 1, figsize=(7, 6.2), height_ratios=height_ratios)

    ax = axes[0]
    ax.axis("off")
    ax.imshow(concat_images([frames[i] for i in show_pos]))
    ax.set_title(f"{model_name} — input frames (stride sampled)", fontsize=10)

    row = 1
    if attn_maps is not None:
        # For each shown frame position, find the nearest attention map by frame index.
        attn_frame_indices = attn_frame_indices if attn_frame_indices is not None else list(range(len(attn_maps)))
        attn_frame_indices = np.array(attn_frame_indices)
        overlays = []
        for pos in show_pos:
            nearest = int(np.argmin(np.abs(attn_frame_indices - pos)))
            overlays.append(overlay_attention_on_frame(frames[pos], attn_maps[nearest]))
        ax = axes[1]
        ax.axis("off")
        ax.imshow(concat_images(overlays))
        ax.set_title(attn_label, fontsize=10)
        row = 2

    proj = reduce_dim(embeddings, method=method)
    score = ordinality_score(embeddings)

    ax = axes[row]
    order = np.arange(len(proj))
    sc = ax.scatter(proj[:, 0], proj[:, 1], c=order, cmap="viridis", s=25, zorder=1)
    ax.plot(proj[:, 0], proj[:, 1], c="gray", alpha=0.4, linewidth=1, zorder=0)
    fig.colorbar(sc, ax=ax, label="frame order")
    ax.set_title(f"{method.upper()} of embeddings — ordinality (|Spearman| of PC1 vs time) = {score:.3f}", fontsize=10)

    fig.tight_layout()
    fig.canvas.draw()
    img = Image.frombuffer(
        "RGBA", fig.canvas.get_width_height(), fig.canvas.buffer_rgba(), "raw", "RGBA", 0, 1
    ).convert("RGB")
    plt.close(fig)
    return img, score


def make_progression_video(
    frames: List[Image.Image],
    embeddings: np.ndarray,
    out_path: str,
    model_name: str,
    attn_maps: Optional[np.ndarray] = None,
    attn_frame_indices: Optional[List[int]] = None,
    attn_label: str = "attention",
    embedding_frame_indices: Optional[List[int]] = None,
    patch_tokens=None,
    patch_grid_hw: Optional[Tuple[int, int]] = None,
    remove_background: bool = True,
    pca_threshold: float = 0.0,
    method: str = "tsne",
    fps: int = 4,
) -> float:
    """Animate the video playing alongside its dimred trajectory being traced out
    point-by-point, with optional attention and global-PCA-patch-color columns,
    saved as an mp4.

    The dimred layout itself is computed once on the full trajectory (t-SNE needs
    all points to lay out sensibly) — what animates is how much of that fixed
    layout is *revealed* as time progresses, plus the other panels playing in
    sync next to it.

    `embedding_frame_indices[i]` gives the position in `frames` that
    `embeddings[i]` corresponds to — needed because some encoders (e.g. V-JEPA2)
    emit fewer embeddings than input frames (one per temporal tubelet instead of
    one per frame), so the trajectory reveal can't just assume row `i` lines up
    with played frame `i`; it has to reveal by frame-index, not by row-index.

    `patch_tokens` (T, P, D) / `patch_grid_hw`: if given, adds a global-PCA
    patch-color column (see `common/pca_patch_viz.py`) — one PCA(3) fit over
    every sampled frame's patch tokens at once, not per frame.
    """
    proj = reduce_dim(embeddings, method=method)
    score = ordinality_score(embeddings)
    curvature = trajectory_curvature(embeddings)
    t = len(frames)

    embedding_frame_indices = np.array(
        embedding_frame_indices if embedding_frame_indices is not None else list(range(len(embeddings)))
    )

    has_attn = attn_maps is not None
    if has_attn:
        attn_frame_indices = np.array(
            attn_frame_indices if attn_frame_indices is not None else list(range(len(attn_maps)))
        )

    has_pca = patch_tokens is not None
    if has_pca:
        from common.pca_patch_viz import compute_pca_patch_colors, render_patch_color_frame
        gh, gw = patch_grid_hw
        pca_colors = compute_pca_patch_colors(patch_tokens, patch_grid_hw, remove_background=remove_background, threshold=pca_threshold)
        pca_color_grids = pca_colors.reshape(t, gh, gw, 3)

    n_panels = 2 + int(has_attn) + int(has_pca)
    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 5))
    axes = list(axes)
    ax_frame = axes.pop(0)
    ax_attn = axes.pop(0) if has_attn else None
    ax_pca = axes.pop(0) if has_pca else None
    ax_proj = axes.pop(0)

    ax_frame.axis("off")
    frame_im = ax_frame.imshow(frames[0])
    ax_frame.set_title(f"{model_name} — frame 0/{t - 1}")

    if has_attn:
        ax_attn.axis("off")
        attn_im = ax_attn.imshow(overlay_attention_on_frame(frames[0], attn_maps[0]))
        ax_attn.set_title(attn_label, fontsize=10)

    if has_pca:
        ax_pca.axis("off")
        pca_im = ax_pca.imshow(render_patch_color_frame(pca_color_grids[0], frames[0].size))
        bg_note = "bg removed" if remove_background else "bg kept"
        ax_pca.set_title(f"global PCA patch colors ({bg_note})", fontsize=10)

    pad_x = 0.05 * (np.ptp(proj[:, 0]) + 1e-6)
    pad_y = 0.05 * (np.ptp(proj[:, 1]) + 1e-6)
    ax_proj.set_xlim(proj[:, 0].min() - pad_x, proj[:, 0].max() + pad_x)
    ax_proj.set_ylim(proj[:, 1].min() - pad_y, proj[:, 1].max() + pad_y)
    ax_proj.set_title(f"{method.upper()} — ordinality={score:.3f}, curvature={curvature:.1f}°", fontsize=10)
    scatter = ax_proj.scatter([], [], c=[], cmap="viridis", vmin=0, vmax=max(t - 1, 1), s=25)
    line, = ax_proj.plot([], [], c="gray", alpha=0.4, linewidth=1)
    fig.colorbar(scatter, ax=ax_proj, label="frame order")
    fig.tight_layout()

    def update(i):
        frame_im.set_data(np.array(frames[i]))
        ax_frame.set_title(f"{model_name} — frame {i}/{t - 1}")
        artists = [frame_im]
        if has_attn:
            nearest = int(np.argmin(np.abs(attn_frame_indices - i)))
            attn_im.set_data(np.array(overlay_attention_on_frame(frames[i], attn_maps[nearest])))
            artists.append(attn_im)
        if has_pca:
            pca_im.set_data(np.array(render_patch_color_frame(pca_color_grids[i], frames[i].size)))
            artists.append(pca_im)

        # Reveal by frame-index reached, not by row-index — embeddings can be
        # sparser than frames (e.g. one per V-JEPA2 tubelet instead of per frame).
        n_reveal = int(np.sum(embedding_frame_indices <= i))
        pts = proj[:n_reveal]
        scatter.set_offsets(pts if n_reveal > 0 else np.empty((0, 2)))
        scatter.set_array(embedding_frame_indices[:n_reveal])
        line.set_data(pts[:, 0], pts[:, 1])
        artists += [scatter, line]
        return artists

    anim = animation.FuncAnimation(fig, update, frames=t, blit=False)
    anim.save(out_path, writer=animation.FFMpegWriter(fps=fps))
    plt.close(fig)
    return score
