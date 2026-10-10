"""Pairwise rank prediction: for two frames of the same clip, which one is further along?

The purest relative-ordering objective in the comparison -- it never sees an absolute
progress value, only the SIGN of a difference, so anything ordinal it induces in the
latent space came from ordering alone. That makes it the control for every absolute
objective (progress_regression, ordinal_classification): if those do better, the extra
signal is doing work; if they do not, the absolute clock was never needed.

Two loss forms, both standard learning-to-rank:
  "logistic"  RankNet (Burges et al.): -log sigmoid(s_j - s_i) for i < j, i.e. a logistic
              loss on the score difference. Never satisfied -- it keeps pushing correctly
              ordered pairs further apart, which is what makes it spread the score axis.
  "margin"    hinge: relu(margin - (s_j - s_i)). Stops caring once a pair is ordered by
              `margin`, so it asks for ordering without asking for spread. The difference
              between these two on trajectory_straightness is itself informative.

All valid pairs within a clip are used rather than a random sample of them: at T=16 that
is at most 120 pairs per clip, which is cheaper than the sampling logic would be.

min_gap drops pairs whose frames are closer than that many steps apart. Adjacent frames of
a 16-frame clip can be near-identical, and asking the model to confidently order them is
asking it to fit noise -- the same reason Shuffle and Learn samples its tuples from
high-motion windows rather than uniformly.

weight_by_gap scales each pair's loss by |p_j - p_i|, so far-apart pairs (whose order is
unambiguous) dominate over near-ties. Off by default; it is the soft version of min_gap.

cross_clip additionally ranks frames against frames from OTHER clips in the batch. Within
a clip, the progress label is a within-video quantity, so cross-clip comparisons are only
meaningful under the assumption that "40% through video A" and "40% through video B" name
the same thing -- true by construction for the evenly-sampled clips here, and a strong
extra constraint (it asks for ONE globally shared progress axis rather than a per-clip
one), but an assumption worth being able to switch off.
"""
import torch
import torch.nn.functional as F

from heads import FrameHead

from .base import Objective


class PairwiseRank(Objective):
    def __init__(self, latent_dim, mode="logistic", margin=1.0, min_gap=1,
                 weight_by_gap=False, cross_clip=False, hidden_dim=None, depth=2,
                 dropout=0.0):
        super().__init__()
        if mode not in ("logistic", "margin"):
            raise ValueError(f"mode must be 'logistic' or 'margin', got {mode!r}")
        self.mode = mode
        self.margin = margin
        self.min_gap = min_gap
        self.weight_by_gap = weight_by_gap
        self.cross_clip = cross_clip
        self.head = FrameHead(latent_dim, out_dim=1, hidden_dim=hidden_dim, depth=depth,
                              dropout=dropout)

    def _pair_loss(self, diff, gap):
        """diff: score(later) - score(earlier), gap: |progress difference|, both (P,)."""
        if self.mode == "logistic":
            per_pair = F.softplus(-diff)
        else:
            per_pair = F.relu(self.margin - diff)
        if self.weight_by_gap:
            w = gap
            return (per_pair * w).sum() / w.sum().clamp(min=1e-8)
        return per_pair.mean()

    @staticmethod
    def _cross_clip_targets(s, progress):
        """One random target per frame, always from a different clip."""
        b, t = s.shape
        if b < 2:
            raise ValueError("cross_clip=True needs a batch containing at least two clips")
        shift = torch.randint(1, b, (), device=s.device)
        other_clip = (torch.arange(b, device=s.device) + shift) % b
        other_frame = torch.randint(t, (b, t), device=s.device)
        return (s[other_clip[:, None], other_frame],
                progress[other_clip[:, None], other_frame], other_clip)

    def forward(self, z, progress, frame_index=None, z_pair=None, progress_pair=None,
                frame_index_pair=None):
        s = self.head(z).squeeze(-1)  # (B, T)
        b, t = s.shape

        # Within-clip: every (i, j) with j - i >= min_gap. triu_indices gives them once
        # each, already oriented i < j, so `later` is unambiguous.
        i, j = torch.triu_indices(t, t, offset=max(self.min_gap, 1), device=s.device)
        if i.numel() == 0:
            # .mean() over an empty tensor is NaN, which would poison the whole run
            # silently rather than failing. Reachable whenever crop_size <= min_gap.
            raise ValueError(f"min_gap={self.min_gap} leaves no valid pair in a length-{t} "
                             f"clip; lower min_gap or raise crop_size")
        diff = s[:, j] - s[:, i]
        gap = (progress[:, j] - progress[:, i]).abs()
        # crop_label='local' keeps progress monotone in frame order, and so does
        # 'global'; but a future label convention need not, so orient by the LABEL rather
        # than assuming index order matches progress order.
        sign = torch.sign(progress[:, j] - progress[:, i])
        valid = sign != 0
        diff, gap = (diff * sign)[valid], gap[valid]
        loss = self._pair_loss(diff, gap)
        metrics = {"pair_acc": (diff > 0).float().mean().detach()}

        if self.cross_clip:
            # RankNet itself compares items within one query; this project's optional
            # global-progress extension compares videos. Preserve that declared meaning:
            # cyclically derange clip IDs, then independently draw a target frame from
            # the other clip. This stays O(B*T) and cannot produce a same-clip or self pair.
            target_s, target_p, _ = self._cross_clip_targets(s, progress)
            xd = (target_s - s).reshape(-1)
            xg = (target_p - progress).reshape(-1)
            keep = xg.abs() >= (self.min_gap / max(t - 1, 1))
            # The key is set unconditionally, even when no cross pair survives: run_epoch
            # accumulates metrics by name and divides by the batch count, so a key present
            # in only some batches would be averaged over the wrong denominator (and could
            # be missing from the CSV header, which is fixed from the first epoch's keys).
            cross = (self._pair_loss(xd[keep] * torch.sign(xg[keep]), xg[keep].abs())
                     if keep.any() else s.sum() * 0.0)
            loss = loss + cross
            metrics["cross_loss"] = cross.detach()
        return loss, metrics
