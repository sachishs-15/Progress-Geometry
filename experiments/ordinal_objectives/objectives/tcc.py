"""Temporal Cycle-Consistency (Dwibedi, Aytar, Tompson, Sermanet & Zisserman, CVPR 2019,
arXiv:1904.07846).

The canonical "learn an ordinal phase embedding" objective, and the one whose learned
space the alignment literature actually uses to read off where in a task a frame is. It
never sees a progress label at all: supervision comes from requiring that a frame's soft
nearest neighbour in ANOTHER video of the same activity map back to itself.

    alpha_j  = softmax_j( -||u_i - v_j||^2 / temperature )       (Eq. 1)
    v_tilde  = sum_j alpha_j v_j                                  soft nearest neighbour
    beta_k   = softmax_k( -||v_tilde - u_k||^2 / temperature )    (Eq. 3)

  variant="classification" (cycle-back classification, Eq. 2): cross-entropy of beta
      against the one-hot index i. Treats every wrong frame as equally wrong, so it gets
      no credit for landing NEXT to i -- the paper's own criticism of it, and the reason
      their main results use the regression form.
  variant="regression" (cycle-back regression, Eq. 4, default):
      mu = sum_k beta_k * k,  sigma^2 = sum_k beta_k * (k - mu)^2
      L  = |i - mu|^2 / sigma^2 + variance_lambda * log(sigma^2)
      i.e. fit a Gaussian over the time axis and penalise its mean being wrong, scaled by
      how confident it claims to be. This is the variant that knows time is ORDERED: being
      off by one frame costs less than being off by ten. (The paper writes the penalty as
      lambda*log(sigma); the official implementation uses log(sigma^2), which differs only
      by a factor of two folded into lambda. log(sigma^2) is used here.)

Pairing. The paper aligns two different videos of the same activity. Accordingly,
view="pair" is required and train.py's --pair-col supplies the second video (`verb` for
SSv2). Same-video temporal subsets are rejected because they are a different objective:
an encoder can cycle via clip-specific appearance rather than cross-video action phase.

normalize_indices divides frame indices by the sequence length before the regression, so
variance_lambda means the same thing across clip lengths -- as in the official
implementation.

normalize (default True) L2-normalizes the latents before any distance is taken, and is
NOT optional in practice on this project's encoder. The softmaxes above are over
-||.||^2 / temperature, so they are only meaningful if the distances are on a bounded
scale; this encoder's raw ||z|| is around 19 in 384 dimensions, which puts squared
distances in the hundreds and saturates both softmaxes into one-hot. A one-hot beta has
sigma^2 = 0, and the regression loss divides by it -- measured, that produced a loss of
6.7e4 and a gradient norm of 4.2e6 on the very first epoch. Normalizing bounds squared
distances to [0, 4] and makes temperature mean something. It also matches what the
evaluation does: clip_progress_metrics L2-normalizes per frame before computing any
metric, so the objective and the measurement then operate on the same geometry.
"""
import torch
import torch.nn.functional as F

from heads import maybe_projection

from .base import Objective, gather_frames, pairwise_sqdist, sorted_subset_indices


class TCC(Objective):
    def __init__(self, latent_dim, variant="regression", temperature=0.1,
                 variance_lambda=0.001, normalize_indices=True, normalize=True,
                 view="pair", num_frames=0, min_variance=1e-4,
                 project=False, proj_dim=128, proj_hidden=None):
        super().__init__()
        if variant not in ("regression", "classification"):
            raise ValueError(f"variant must be regression/classification, got {variant!r}")
        if view != "pair":
            raise ValueError(
                "TCC follows the paper's cross-video protocol and requires view='pair'; "
                "same-clip temporal subsets are not TCC"
            )
        self.variant = variant
        self.temperature = temperature
        self.variance_lambda = variance_lambda
        self.normalize_indices = normalize_indices
        self.normalize = normalize
        self.view = view
        # Floor on sigma^2 in the regression loss. In normalized index units the whole
        # clip spans [0, 1), so a uniform beta has variance ~1/12; 1e-4 is a standard
        # deviation of 1% of the clip, i.e. already absurdly confident. A much smaller
        # floor (or none) lets |i-mu|^2/sigma^2 diverge on exactly the samples the model
        # is most wrong about.
        self.min_variance = min_variance
        self.num_frames = num_frames
        self.proj = maybe_projection(latent_dim, project, proj_dim, proj_hidden)

    @property
    def requires_pair(self):
        return True

    def _views(self, z, z_pair):
        u, v = z, z_pair
        if self.num_frames > 0:
            b, t, _ = z.shape
            # The official implementation samples frames independently inside each of
            # two DIFFERENT videos. Overlap is not defined across distinct sequences.
            u = gather_frames(u, sorted_subset_indices(b, t, self.num_frames, z.device))
            v = gather_frames(v, sorted_subset_indices(
                b, z_pair.size(1), self.num_frames, z.device
            ))
        return u, v

    def _cycle(self, u, v):
        """u: (B, N, d) anchor sequence, v: (B, M, d) other sequence -> beta (B, N, N)."""
        alpha = F.softmax(-pairwise_sqdist(u, v) / self.temperature, dim=-1)  # (B, N, M)
        v_tilde = alpha @ v                                                    # (B, N, d)
        return F.softmax(-pairwise_sqdist(v_tilde, u) / self.temperature, dim=-1)  # (B, N, N)

    def _loss(self, beta):
        b, n, _ = beta.shape
        idx = torch.arange(n, device=beta.device, dtype=beta.dtype)
        target = idx.expand(b, n)
        if self.variant == "classification":
            # beta is already a distribution (softmax'd in _cycle), so this is an NLL on
            # log(beta) -- NOT cross_entropy, which would apply a second log_softmax.
            loss = F.nll_loss(torch.log(beta.reshape(b * n, n) + 1e-8),
                              target.reshape(b * n).long())
            return loss, {"cycle_acc": (beta.argmax(-1) == target.long()).float().mean().detach()}

        k = idx / n if self.normalize_indices else idx
        tgt = target / n if self.normalize_indices else target
        mu = (beta * k).sum(-1)
        sigma_sq = (beta * (k - mu.unsqueeze(-1)) ** 2).sum(-1).clamp(min=self.min_variance)
        loss = ((tgt - mu) ** 2 / sigma_sq + self.variance_lambda * torch.log(sigma_sq)).mean()
        return loss, {"cycle_mae": (tgt - mu).abs().mean().detach(),
                      "cycle_sigma": sigma_sq.sqrt().mean().detach()}

    def forward(self, z, progress, frame_index=None, z_pair=None, progress_pair=None,
                frame_index_pair=None):
        if z_pair is None:
            raise ValueError("TCC needs a paired clip -- pass --pair-col "
                             "(e.g. `verb` for SSv2) so the dataset yields one")
        u, v = self.proj(z), self.proj(z_pair) if z_pair is not None else None
        if self.normalize:
            u = F.normalize(u, dim=-1)
            v = F.normalize(v, dim=-1) if v is not None else None
        u, v = self._views(u, v)
        # Symmetric: cycle U->V->U and V->U->V. The paper cycles from one sequence only,
        # but both directions are free here (the pair is already encoded) and halve the
        # variance of the estimate.
        loss_uv, m_uv = self._loss(self._cycle(u, v))
        loss_vu, m_vu = self._loss(self._cycle(v, u))
        # Average the diagnostics over both directions too -- the loss does, and reporting
        # only the U->V half would make cycle_mae/cycle_sigma describe a different
        # quantity from the number being optimised.
        metrics = {k: 0.5 * (m_uv[k] + m_vu[k]) for k in m_uv}
        return 0.5 * (loss_uv + loss_vu), metrics
