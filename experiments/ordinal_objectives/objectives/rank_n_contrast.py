"""Rank-N-Contrast (Zha et al., NeurIPS 2023, arXiv:2210.01189) with progress as the
continuous label.

The most on-target objective in this comparison. RnC does not ask the latent space to
predict anything: it ranks every other sample by how far its LABEL is from the anchor's,
then requires feature similarity to respect that ranking. The paper proves that
minimising it yields features ordered according to the continuous label -- which is
verbatim the property `time_progress_spearman` and `pc1_time_spearman` measure. If any
objective in this project ought to win on those metrics by construction, it is this one,
and a result where it does not is worth more than one where it does.

    L_RnC = mean_i mean_{j != i} -log  exp(sim(v_i, v_j)/tau)
                                       ---------------------------------
                                       sum_{v_k in S_ij} exp(sim(v_i,v_k)/tau)

    S_ij = { v_k : k != i, d(y_i, y_k) >= d(y_i, y_j) }

with sim a negative L1/L2 norm (the paper's choice; cosine also offered here) and d the
label distance |p_i - p_k|.

Implementation note -- why this is not the naive triple loop. S_ij is a different set for
every (i, j), so the literal form needs an (N, N, N) mask; at batch_size=64 over T=16
frames N is already 1024, i.e. 10^9 entries. Instead, for each anchor the other samples
are SORTED by label distance: S_ij is then a contiguous SUFFIX of that sorted list, so
every denominator for that anchor comes from one reverse cumulative logsumexp
(torch.logcumsumexp on the flipped row). Cost drops to O(N^2 log N) time and O(N^2)
memory with no approximation.

Ties matter here and are handled exactly. Progress labels take only T distinct values, so
label distances tie constantly, and "d >= d_j" must include every tied sample, not just
the ones that happened to sort later. The suffix is therefore taken from the START of the
anchor's tie block (found with a cummax over block boundaries), not from the sample's own
sorted position.
"""
import torch

from heads import maybe_projection

from .base import Objective


class RankNContrast(Objective):
    def __init__(self, latent_dim, temperature=2.0, sim="l2", normalize=False,
                 project=False, proj_dim=128, proj_hidden=None, max_frames=0,
                 tie_tol=1e-6):
        super().__init__()
        if sim not in ("l1", "l2", "cosine"):
            raise ValueError(f"sim must be l1/l2/cosine, got {sim!r}")
        self.temperature = temperature
        self.sim = sim
        # normalize=False matches the reference, which applies F.normalize ONLY in its
        # cosine branch and feeds RAW feature distances to the l1/l2 branches. Defaulting
        # it to True (as this file first did) silently breaks the objective at the paper's
        # own temperature: measured on a real batch with this encoder, unit-normalized
        # features give a similarity range of [-1.49, 0], so tau=2 leaves a logit spread
        # of 0.74 nats over 2048 candidates -- a near-uniform softmax, a loss of 6.79
        # against a uniform ceiling of ln(2048)=7.63, and summed encoder gradient 4.8.
        # Unnormalized, the same batch spans [-32.3, 0]: 16.1 nats of spread, loss 8.18,
        # gradient 932. The `cosine` branch normalizes internally, as the reference does.
        self.normalize = normalize
        # max_frames > 0 subsamples the flattened frame set before building the (N, N)
        # matrices. N = batch_size * T, so batch_size=128 at T=16 gives N=2048 and four
        # 2048x2048 float32 matrices (~64MB) live through the backward pass; cap it here
        # rather than shrinking the batch, which would change every other objective's
        # effective batch too.
        self.max_frames = max_frames
        # Two label distances that are mathematically equal are NOT bitwise equal in
        # float32: progress labels are i/(T-1), and e.g. |5/15 - 3/15| differs from
        # |9/15 - 7/15| in the last ulp (measured: 4 distinct float values among the 15
        # gap-1 pairs of a 16-frame clip). An exact `!=` therefore splits tie blocks that
        # should be one, silently dropping genuinely-tied samples out of the denominator.
        # The smallest REAL gap between distinct label distances is 1/(T-1) ~ 0.067, so
        # any tolerance far below that separates real differences from float noise.
        self.tie_tol = tie_tol
        self.proj = maybe_projection(latent_dim, project, proj_dim, proj_hidden)

    def _similarity(self, v):
        if self.sim == "cosine":
            v = torch.nn.functional.normalize(v, dim=-1)
            return v @ v.t()
        # Negative Lp distance. cdist is the right call here (N is large and this is the
        # dominant cost); its near-zero-distance gradient weakness is not a concern
        # because RnC never drives a pair's distance to zero -- the denominator always
        # contains the numerator.
        return -torch.cdist(v, v, p=1.0 if self.sim == "l1" else 2.0)

    def forward(self, z, progress, frame_index=None, z_pair=None, progress_pair=None,
                frame_index_pair=None):
        v = self.proj(z)
        v = v.reshape(-1, v.size(-1))
        y = progress.reshape(-1)
        if self.normalize:
            v = torch.nn.functional.normalize(v, dim=-1)
        n = v.size(0)
        if self.max_frames > 0 and n > self.max_frames:
            keep = torch.randperm(n, device=v.device)[:self.max_frames]
            v, y, n = v[keep], y[keep], self.max_frames
        if n < 3:
            # Connected to the graph (see objectives/none.py) so a degenerate batch can
            # never make loss.backward() raise.
            return v.sum() * 0.0, {}

        logits = self._similarity(v) / self.temperature      # (N, N)
        label_dist = (y.unsqueeze(1) - y.unsqueeze(0)).abs()  # (N, N)

        # Drop the diagonal (k != i) by compressing each row to its N-1 off-diagonal
        # entries, so no masking with -inf is needed anywhere downstream.
        off = ~torch.eye(n, dtype=torch.bool, device=v.device)
        logits = logits[off].view(n, n - 1)
        label_dist = label_dist[off].view(n, n - 1)

        order = label_dist.argsort(dim=1)                     # ascending label distance
        d_sorted = label_dist.gather(1, order)
        l_sorted = logits.gather(1, order)

        # suffix_lse[:, r] = logsumexp(l_sorted[:, r:]) -- every candidate denominator.
        suffix_lse = torch.logcumsumexp(l_sorted.flip(1), dim=1).flip(1)

        # Start index of each tie block, so equal label distances share one denominator
        # (the "d >= d_j" in S_ij is inclusive).
        pos = torch.arange(n - 1, device=v.device).expand(n, -1)
        is_block_start = torch.ones_like(d_sorted, dtype=torch.bool)
        is_block_start[:, 1:] = (d_sorted[:, 1:] - d_sorted[:, :-1]) > self.tie_tol
        block_start = torch.cummax(torch.where(is_block_start, pos, torch.zeros_like(pos)),
                                   dim=1).values
        denom = suffix_lse.gather(1, block_start)

        loss = (denom - l_sorted).mean()
        return loss, {}
