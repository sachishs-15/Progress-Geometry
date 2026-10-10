"""Progress as an ORDINAL CLASSIFICATION problem rather than a regression one.

This family is in the comparison because of a specific, testable claim from the
regression-representation literature: Zhang et al., "Improving Deep Regression with
Ordinal Entropy" (ICLR 2023, arXiv:2301.08915), argue that cross-entropy on a discretised
target learns HIGHER-ENTROPY (more spread out, better separated) feature spaces than MSE
on the same continuous target, and that this -- not the readout -- is where
classification's advantage on continuous tasks comes from. If that is right, then
`ordinal_classification` should produce a visibly more ordered latent space than
`progress_regression` while being supervised by exactly the same information, which is a
much sharper result than either number alone.

Four variants, in increasing order of how much ordinal structure the LOSS itself knows
about (as opposed to leaving it to the features):

  "ce"     plain softmax cross-entropy over num_bins. Knows nothing about bin order: bin 0
           and bin 15 are as different as bin 0 and bin 1. The pure form of the
           entropy-vs-order tradeoff the paper is about.
  "soft"   SORD-style soft labels: target mass exp(-((c_k - p)/sigma)^2) over bin centres
           c_k, normalised. Order enters through the target distribution, which is
           unimodal and centred on the true value, so adjacent-bin mistakes are cheap and
           distant ones are not.
  "coral"  CORAL (Cao et al.): num_bins-1 cumulative binary tasks "is p past threshold k",
           all reading the SAME scalar projection with per-threshold biases (see
           heads.CoralHead). Rank-monotonic by construction, and the variant that most
           directly asks the latent space for a single ordered axis.
  "corn"   CORN (Shi et al.): the same cumulative decomposition, but each task k is trained
           only on the samples that already passed threshold k-1, so the binary tasks
           estimate CONDITIONAL probabilities P(y>k | y>k-1) and need no weight sharing to
           stay consistent. Unlike CORAL it keeps an independent head per threshold, so it
           is the natural control for "is CORAL's ordering coming from the shared
           projection or from the cumulative decomposition?".

Every variant decodes to a scalar the same way (expected value under the implied
distribution) so `mae` is comparable across all four and against progress_regression.
"""
import torch
import torch.nn.functional as F

from heads import CoralHead, FrameHead

from .base import Objective


class OrdinalClassification(Objective):
    def __init__(self, latent_dim, num_bins=16, variant="ce", hidden_dim=None, depth=2,
                 dropout=0.0, soft_sigma=0.1):
        super().__init__()
        if variant not in ("ce", "soft", "coral", "corn"):
            raise ValueError(f"variant must be one of ce/soft/coral/corn, got {variant!r}")
        if num_bins < 2:
            raise ValueError(f"num_bins must be >= 2, got {num_bins}")
        self.variant = variant
        self.num_bins = num_bins
        self.soft_sigma = soft_sigma
        if variant == "coral":
            self.head = CoralHead(latent_dim, num_bins, hidden_dim=hidden_dim, depth=depth,
                                  dropout=dropout)
        else:
            out_dim = num_bins if variant in ("ce", "soft") else num_bins - 1
            self.head = FrameHead(latent_dim, out_dim=out_dim, hidden_dim=hidden_dim,
                                  depth=depth, dropout=dropout)
        # Bin centres, registered so .to(device) carries them and so the decode below
        # never rebuilds them per step.
        self.register_buffer("centres", (torch.arange(num_bins).float() + 0.5) / num_bins)
        self.register_buffer("thresholds", (torch.arange(num_bins - 1).float() + 1.0) / num_bins)

    def _bins(self, progress):
        """progress in [0,1] -> integer bin index, with 1.0 landing in the last bin rather
        than one past it."""
        return (progress * self.num_bins).long().clamp(max=self.num_bins - 1)

    def forward(self, z, progress, frame_index=None, z_pair=None, progress_pair=None,
                frame_index_pair=None):
        logits = self.head(z)
        flat_logits = logits.reshape(-1, logits.size(-1))
        flat_p = progress.reshape(-1)

        if self.variant in ("ce", "soft"):
            if self.variant == "ce":
                loss = F.cross_entropy(flat_logits, self._bins(flat_p))
            else:
                target = torch.exp(-((self.centres - flat_p.unsqueeze(-1)) / self.soft_sigma) ** 2)
                target = target / target.sum(dim=-1, keepdim=True)
                loss = -(target * F.log_softmax(flat_logits, dim=-1)).sum(-1).mean()
            probs = flat_logits.softmax(-1)
            p_hat = (probs * self.centres).sum(-1)
        else:
            # Cumulative decomposition: target_k = 1 iff p is past threshold k.
            past = (flat_p.unsqueeze(-1) > self.thresholds).float()
            if self.variant == "coral":
                loss = F.binary_cross_entropy_with_logits(flat_logits, past)
            else:
                # CORN: task k is only trained on samples that passed threshold k-1
                # (task 0 sees everything), so each head estimates a CONDITIONAL
                # probability. Implemented as a mask rather than by slicing, so all K-1
                # tasks stay one fused BCE call.
                mask = torch.ones_like(past)
                mask[:, 1:] = past[:, :-1]
                per_task = F.binary_cross_entropy_with_logits(flat_logits, past, reduction="none")
                loss = (per_task * mask).sum() / mask.sum().clamp(min=1.0)
            cum = torch.sigmoid(flat_logits)
            if self.variant == "corn":
                cum = torch.cumprod(cum, dim=-1)  # conditional -> marginal P(y > k)
            # E[y] for a non-negative ordinal variable is the sum of its survival
            # probabilities -- here, the expected NUMBER of thresholds passed, which is
            # the expected bin index. +0.5 and /num_bins turn that index into the same
            # bin CENTRE the ce/soft branch decodes to, so `mae` means the same thing for
            # all four variants (and against progress_regression).
            p_hat = (cum.sum(-1) + 0.5) / self.num_bins

        return loss, {"mae": (p_hat - flat_p).abs().mean().detach()}
