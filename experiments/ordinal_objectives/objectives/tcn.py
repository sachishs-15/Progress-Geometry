"""Single-view Time-Contrastive Networks (Sermanet et al., "Time-Contrastive Networks:
Self-Supervised Learning from Video", ICRA 2018).

A metric objective rather than a predictive or ranking one: frames close in time are
pulled together, frames far apart in time are pushed apart, by a triplet margin loss on
embedding distance.

    L = mean max(0, ||z_a - z_pos||^2 - ||z_a - z_neg||^2 + margin)

with the positive drawn from within `pos_window` frames of the anchor and the negative
from at least `neg_margin` frames away. The multi-view form of TCN uses simultaneous
views from different cameras as positives; single-view (used here, and also in the paper)
substitutes temporal proximity, which is all that is available for this data.

What distinguishes it from every ranking objective here: TCN constrains DISTANCES, not
order. It never learns which of two frames comes first, so any ordering it produces is a
by-product of locally smooth, globally separated structure. That makes it the natural
control for the question "how much of latent ordinality is just temporal smoothness?" --
if TCN matches the explicitly ordinal objectives on time_progress_spearman, then
smoothness was the whole story and direction never mattered.

pos_window/neg_margin are in frames of the (possibly cropped) clip. At T=16, the defaults
(1 / 4) mean "positives are immediate neighbours, negatives at least a quarter of the
clip away". The gap between them is deliberate: frames in between are neither, so the
objective is never asked to make a call it cannot support.
"""
import torch
import torch.nn.functional as F

from heads import maybe_projection

from .base import Objective


class TCN(Objective):
    def __init__(self, latent_dim, pos_window=1, neg_margin=4, margin=1.0, normalize=True,
                 project=False, proj_dim=128, proj_hidden=None):
        super().__init__()
        if pos_window < 1:
            raise ValueError(f"pos_window must be >= 1, got {pos_window}")
        if neg_margin <= pos_window:
            raise ValueError(f"neg_margin ({neg_margin}) must exceed pos_window ({pos_window}), "
                             f"or a frame could be both positive and negative for the same anchor")
        self.pos_window = pos_window
        self.neg_margin = neg_margin
        self.margin = margin
        # Normalizing bounds every distance into [0, 4], which makes `margin` mean the
        # same thing regardless of how the encoder's norm_type scales z -- without it,
        # margin=1.0 is a very different constraint under norm_type='none' than under
        # 'layernorm' (see predictive_world_model/models/norms.py on how much ||z|| drifts).
        self.normalize = normalize
        self.proj = maybe_projection(latent_dim, project, proj_dim, proj_hidden)

    def forward(self, z, progress, frame_index=None, z_pair=None, progress_pair=None,
                frame_index_pair=None):
        v = self.proj(z)
        if self.normalize:
            v = F.normalize(v, dim=-1)
        b, t, _ = v.shape
        device = v.device
        anchors = torch.arange(t, device=device).expand(b, t)     # every frame is an anchor

        offsets = (anchors.unsqueeze(-1) - anchors.unsqueeze(-2)).abs()  # (B, T, T)
        pos_mask = (offsets <= self.pos_window) & (offsets > 0)
        neg_mask = offsets >= self.neg_margin
        usable = pos_mask.any(-1) & neg_mask.any(-1)              # (B, T)
        if not usable.any():
            raise ValueError(f"no anchor in a length-{t} clip has both a positive within "
                             f"pos_window={self.pos_window} and a negative at "
                             f"neg_margin={self.neg_margin}; lower neg_margin or raise crop_size")

        # One positive and one negative per anchor per step. argmax over uniform noise
        # restricted to the legal set IS a uniform draw from that set, and it samples
        # every anchor in the batch in one shot -- no per-anchor host-side sampling.
        def sample(mask):
            scores = torch.rand_like(mask, dtype=v.dtype).masked_fill(~mask, -1.0)
            return scores.argmax(-1)

        pos_idx, neg_idx = sample(pos_mask), sample(neg_mask)
        d = lambda idx: (v - v.gather(1, idx.unsqueeze(-1).expand(-1, -1, v.size(-1)))).pow(2).sum(-1)
        d_pos, d_neg = d(pos_idx), d(neg_idx)
        triplet = F.relu(d_pos - d_neg + self.margin)
        loss = triplet[usable].mean()
        return loss, {"triplet_acc": (d_pos < d_neg)[usable].float().mean().detach(),
                      "d_pos": d_pos[usable].mean().detach(),
                      "d_neg": d_neg[usable].mean().detach()}
