"""Sequence Contrastive Loss from CARL (Chen et al., "Frame-wise Action Representations
for Long Videos via Sequence Contrastive Learning", CVPR 2022, arXiv:2203.14957).

The efficient sibling of TCC: instead of a cycle through a second sequence, it directly
requires the similarity profile between two views of a video to match a GAUSSIAN PRIOR
over timestamp distance. A frame is most similar to the frame at the same time, and its
similarity should fall off smoothly and symmetrically from there.

    L_i = - sum_j  w_ij * log  exp(sim(z1_i, z2_j)/tau) / sum_k exp(sim(z1_i, z2_k)/tau)   (Eq. 1)
    w_ij = G(s1_i - s2_j) / sum_k G(s1_i - s2_k),   G(x) = N(x; 0, sigma^2)                (Eq. 2)
    L    = (L_1 + L_2) / 2                                                                 (Eq. 3)

with sim cosine similarity and s the frames' raw timestamps.

Where this differs from TCN, which it otherwise resembles: TCN's positive/negative split
is a hard threshold, so every "far" frame is pushed away equally hard. SCL's target is a
smooth function of temporal distance, so it encodes HOW far apart two frames are, not just
whether they are near. That graded target is the reason to expect it to order the space
rather than merely cluster it locally.

Views. CARL's two views come from two independent spatio-temporal augmentations of the
same video, and the temporal half of that augmentation -- randomly subsampling different
frames -- is directly reproducible on precomputed features, which is what view="temporal"
(the default) does.

disjoint=True (default, temporal view only) is load-bearing, not a tuning knob. CARL's
two views may contain the same timestamp because AUGMENTATION makes the two copies of
that frame differ. Here there is no augmentation -- the features are frozen -- so a frame
drawn into both views contributes a pair of bit-identical vectors with cosine similarity
exactly 1 at temporal distance 0, which is precisely where the Gaussian prior puts its
most mass. Two independent k-subsets of 16 frames at the k=8 default share 4 frames on
average, so half the highest-weighted targets would be solved by identity and teach the
encoder nothing. disjoint=True instead partitions the clip's frames between the two
views, so no pair is ever trivially matched. Set it False only to reproduce the
overlapping-view behaviour deliberately. view="pair" instead uses a different video of the same activity,
which is a harder and more TCC-like setting. Spatial augmentation is not reproducible on
frozen features at all and is simply absent, so this is CARL's loss on CARL's temporal
augmentation only.

sigma is in FRAMES of the source clip, not seconds. CARL's default of sigma^2=10 is
calibrated to raw second-scale timestamps on minutes-long videos; over a 16-frame clip it
would be almost flat (every frame nearly equally "positive"), so the default here is 1.0.
It is the single most important hyperparameter of this objective and the configs sweep it.
"""
import torch
import torch.nn.functional as F

from heads import maybe_projection

from .base import Objective, gather_frames, gather_scalar, sorted_subset_indices


class SCL(Objective):
    def __init__(self, latent_dim, sigma=1.0, temperature=0.1, view="temporal",
                 num_frames=0, disjoint=True, project=False, proj_dim=128, proj_hidden=None):
        super().__init__()
        if view not in ("temporal", "pair"):
            raise ValueError(f"view must be temporal/pair, got {view!r}")
        if sigma <= 0:
            raise ValueError(f"sigma must be > 0, got {sigma}")
        self.sigma = sigma
        self.temperature = temperature
        self.view = view
        self.num_frames = num_frames
        self.disjoint = disjoint
        self.proj = maybe_projection(latent_dim, project, proj_dim, proj_hidden)

    @property
    def requires_pair(self):
        return self.view == "pair"

    def _one_side(self, z1, s1, z2, s2):
        """Eq. 1 + Eq. 3 for view 1 against view 2."""
        z1 = F.normalize(z1, dim=-1)
        z2 = F.normalize(z2, dim=-1)
        logits = torch.bmm(z1, z2.transpose(1, 2)) / self.temperature      # (B, N, M)
        delta = s1.unsqueeze(-1) - s2.unsqueeze(-2)                        # (B, N, M)
        # G is normalized over j anyway, so its 1/(sigma*sqrt(2pi)) prefactor cancels and
        # a plain softmax of -delta^2/(2 sigma^2) is the same distribution.
        w = F.softmax(-delta.pow(2) / (2 * self.sigma ** 2), dim=-1)
        return -(w * F.log_softmax(logits, dim=-1)).sum(-1).mean()

    def forward(self, z, progress, frame_index=None, z_pair=None, progress_pair=None,
                frame_index_pair=None):
        v = self.proj(z)
        b, t, _ = v.shape
        # Timestamps are frame indices within the source clip, independent of the
        # crop_label convention -- SCL's prior is over elapsed TIME, which local
        # renormalization would distort (it would make a short crop's frames look as far
        # apart as a long one's).
        ts = frame_index.to(v.dtype) if frame_index is not None else \
            torch.arange(t, device=v.device, dtype=v.dtype).expand(b, t)

        if self.view == "pair":
            if z_pair is None:
                raise ValueError("SCL(view='pair') needs a paired clip -- pass --pair-col")
            v2 = self.proj(z_pair)
            ts2 = frame_index_pair.to(v.dtype) if frame_index_pair is not None else ts
            z1, s1, z2, s2 = v, ts, v2, ts2
        else:
            k = self.num_frames or max(t // 2, 3)
            if self.disjoint:
                if 2 * k > t:
                    raise ValueError(f"disjoint temporal views need 2*num_frames ({2 * k}) <= "
                                     f"clip length ({t}); lower num_frames or set disjoint=False")
                # One permutation split in two: the halves are disjoint by construction,
                # so no frame appears in both views (see the module docstring).
                perm = torch.rand(b, t, device=v.device).argsort(dim=1)
                i1 = perm[:, :k].sort(dim=1).values
                i2 = perm[:, k:2 * k].sort(dim=1).values
            else:
                i1 = sorted_subset_indices(b, t, k, v.device)
                i2 = sorted_subset_indices(b, t, k, v.device)
            z1, s1 = gather_frames(v, i1), gather_scalar(ts, i1)
            z2, s2 = gather_frames(v, i2), gather_scalar(ts, i2)

        loss = 0.5 * (self._one_side(z1, s1, z2, s2) + self._one_side(z2, s2, z1, s1))
        return loss, {}
