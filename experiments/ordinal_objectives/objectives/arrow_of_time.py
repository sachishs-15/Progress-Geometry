"""Arrow of time: is this clip playing forwards or backwards? (Wei, Lim, Zisserman &
Freeman, "Learning and Using the Arrow of Time", CVPR 2018.)

The weakest temporal signal in the comparison -- one bit per clip, about DIRECTION only,
with nothing said about how far along any individual frame is. It is here as a floor.
Every objective in this project is supposed to be measured by how ordinal it makes the
latent space; if arrow-of-time training scores close to the explicitly ordinal objectives
on time_progress_spearman, that is evidence about the METRIC (that it is too easy to
satisfy) at least as much as about the objective, and that is worth knowing before
reading any of the other rows.

Mechanically: reverse a random half of the clips, then a SequenceHead over the (still
per-frame, still independently encoded) latents predicts which half this was. Note that
direction is only recoverable from the set of latents if their arrangement encodes
something asymmetric in time -- a perfectly symmetric embedding of a clip is
indistinguishable from its reverse, which is what makes this a real constraint on the
latent space despite the tiny label.
"""
import torch
import torch.nn.functional as F

from heads import SequenceHead

from .base import Objective, gather_frames, sorted_subset_indices


class ArrowOfTime(Objective):
    def __init__(self, latent_dim, num_frames=0, d_model=256, nhead=4, num_layers=2,
                 dropout=0.1, max_len=16):
        super().__init__()
        self.num_frames = num_frames
        self.head = SequenceHead(latent_dim, 2, d_model=d_model, nhead=nhead,
                                 num_layers=num_layers, dropout=dropout, max_len=max_len,
                                 output="pooled")

    def forward(self, z, progress, frame_index=None, z_pair=None, progress_pair=None,
                frame_index_pair=None):
        b, t, _ = z.shape
        if self.num_frames > 0:
            z = gather_frames(z, sorted_subset_indices(b, t, self.num_frames, z.device))
        labels = (torch.rand(b, device=z.device) < 0.5).long()   # 1 = forward, 0 = reversed
        seq = torch.where(labels.view(b, 1, 1).bool(), z, z.flip(1))
        logits = self.head(seq)
        loss = F.cross_entropy(logits, labels)
        return loss, {"acc": (logits.argmax(-1) == labels).float().mean().detach()}
