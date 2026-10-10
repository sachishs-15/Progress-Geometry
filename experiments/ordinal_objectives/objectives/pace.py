"""Pace / playback-rate prediction (Wang, Jiao & Liu, "Self-supervised Video
Representation Learning by Pace Prediction", ECCV 2020; also SpeedNet, PRP).

Classify the temporal STRIDE a sequence was sampled at. This asks the latent space about
the RATE of change rather than its direction or position -- to tell stride 1 from stride
4, the embedding must encode how much visual change a unit of time produces, which is a
different demand than any other objective here makes.

Included for the same diagnostic reason as arrow_of_time: it is temporally supervised but
not ordinally supervised, so it separates "this objective used time" from "this objective
imposed an order". Pace is also the objective most likely to be confounded by content
rather than progress (a fast-moving clip at stride 1 looks like a slow one at stride 4),
which is worth seeing directly in the numbers rather than assuming.

rates are strides into the already-sampled clip, so the largest one must satisfy
max(rates) * (num_frames - 1) < T. With the default T=16, num_frames=4 and rates
(1, 2, 3, 4) fit exactly.
"""
import torch
import torch.nn.functional as F

from heads import SequenceHead

from .base import Objective, gather_frames


class PacePrediction(Objective):
    def __init__(self, latent_dim, num_frames=4, rates=(1, 2, 3, 4), d_model=256, nhead=4,
                 num_layers=2, dropout=0.1):
        super().__init__()
        rates = tuple(int(r) for r in rates)
        if min(rates) < 1:
            raise ValueError(f"rates must all be >= 1, got {rates}")
        self.num_frames = num_frames
        self.rates = rates
        self.head = SequenceHead(latent_dim, len(rates), d_model=d_model, nhead=nhead,
                                 num_layers=num_layers, dropout=dropout, max_len=num_frames,
                                 output="pooled")

    def forward(self, z, progress, frame_index=None, z_pair=None, progress_pair=None,
                frame_index_pair=None):
        b, t, _ = z.shape
        k = self.num_frames
        rates = torch.tensor(self.rates, device=z.device)
        span = rates * (k - 1) + 1                      # frames spanned by each rate
        if int(span.max()) > t:
            raise ValueError(f"rate {int(rates[span.argmax()])} over {k} frames spans "
                             f"{int(span.max())} frames, but clips are only {t} long -- "
                             f"lower num_frames/rates or raise crop_size")
        cls = torch.randint(0, len(self.rates), (b,), device=z.device)
        rate = rates[cls]
        max_start = (t - span[cls])                      # (B,), inclusive
        start = (torch.rand(b, device=z.device) * (max_start + 1)).long().clamp(max=t - 1)
        idx = start.unsqueeze(1) + rate.unsqueeze(1) * torch.arange(k, device=z.device)
        logits = self.head(gather_frames(z, idx))
        loss = F.cross_entropy(logits, cls)
        return loss, {"acc": (logits.argmax(-1) == cls).float().mean().detach()}
