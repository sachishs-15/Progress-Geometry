"""LAV: Learning by Aligning Videos in Time (Haresh et al., CVPR 2021, arXiv:2103.17260).

The third member of the alignment family, alongside TCC and SCL, and the one that aligns
two videos with an explicit MONOTONIC WARPING PATH rather than a frame-to-frame soft
nearest neighbour. That distinction is the point: TCC's cycle and SCL's Gaussian prior
both operate pairwise and can be satisfied by a space where order is only locally right;
soft-DTW's cost is minimised over paths that are monotone by construction, so it can only
be low if the two sequences are globally, consistently ordered.

    min^gamma{a_1..a_n} = -gamma * log sum_i exp(-a_i / gamma)               (Eq. 3)
    dtw^gamma(X, Y)     = min^gamma over alignment paths of <A, D>           (Eq. 4)
    D(i, j)             = ||f(x_i) - f(y_j)||^2

Soft-DTW alone collapses -- "optimizing solely for this term leads to trivial solutions,
particularly, one where all frames get mapped to a small cluster" -- so LAV adds
Contrastive-IDM (Eq. 7), a temporal regularizer applied to EACH sequence's own
self-distance matrix:

    I*(X) = sum_i sum_j [ y_ij * Wbar(i,j) * max(0, lambda - D_X(i,j))
                          + (1 - y_ij) * W(i,j) * D_X(i,j) ]
    y_ij     = 1 if |i - j| > sigma else 0
    Wbar(i,j) = (i-j)^2 + 1        (push temporally DISTANT frames apart, harder the
                                     further apart in time they are)
    W(i,j)    = 1 / ((i-j)^2 + 1)  (pull temporally CLOSE frames together, gently)

    L(X, Y) = dtw^gamma(X, Y) + alpha * (I*(X) + I*(Y))

normalize=True by default, for the same reason as objectives/tcc.py: gamma and lambda are
scale-dependent (they live in the same units as D), and this encoder's raw ||z|| ~ 19.5 in
384 dimensions puts squared distances in the hundreds, which makes the paper's own
hyperparameters meaningless. Normalizing bounds D to [0, 4] and it also matches the
evaluation, which L2-normalizes per frame before computing any metric.

idm_weight defaults to 0.2, not 1.0. Contrastive-IDM's Wbar(i,j) = (i-j)^2 + 1 grows
quadratically with temporal distance, so at T=16 the most distant pairs carry a weight of
226 and the term is numerically far larger than the alignment cost it exists to
regularize. Measured on a real batch with this encoder: dtw = 23.3, idm = 112.5 -- at
idm_weight=1.0 the "regularizer" is 4.8x the objective, which makes the run
Contrastive-IDM training with a soft-DTW side term rather than LAV. 0.2 puts it at roughly
a fifth of the total, which is the regularizing regime. Sweep it before reading the row;
the paper's own alpha is not stated in units that transfer to this feature scale.

Cost note: the DP is a T x T recurrence that cannot be parallelised across (i, j), so this
is the slowest objective here -- 256 sequential steps per forward at T=16, batched across
clips. Acceptable at this clip length; it is the reason nothing longer is attempted.
"""
import torch
import torch.nn.functional as F

from heads import maybe_projection

from .base import Objective, gather_frames, pairwise_sqdist, sorted_subset_indices

_BIG = 1e10  # stands in for +inf in the DP: inf - inf would give NaN under autograd


def _softmin(x, gamma):
    """min^gamma of Eq. 3, over the last dim."""
    return -gamma * torch.logsumexp(-x / gamma, dim=-1)


def soft_dtw(d, gamma):
    """d: (B, N, M) pairwise cost -> (B,) soft-DTW distance.

    Built as a Python list-of-lists rather than by writing into a preallocated tensor:
    in-place assignment into an autograd-tracked tensor breaks the graph, and the grid is
    small enough (16x16) that the overhead is irrelevant next to correctness.
    """
    b, n, m = d.shape
    r = [[None] * (m + 1) for _ in range(n + 1)]
    r[0][0] = d.new_zeros(b)
    for j in range(1, m + 1):
        r[0][j] = d.new_full((b,), _BIG)
    for i in range(1, n + 1):
        r[i][0] = d.new_full((b,), _BIG)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            prev = torch.stack([r[i - 1][j], r[i][j - 1], r[i - 1][j - 1]], dim=-1)
            r[i][j] = d[:, i - 1, j - 1] + _softmin(prev, gamma)
    return r[n][m]


def contrastive_idm(z, sigma, margin):
    """Eq. 7, on one sequence's own self-distance matrix. z: (B, N, D) -> (B,)."""
    n = z.size(1)
    idx = torch.arange(n, device=z.device, dtype=z.dtype)
    delta = (idx.unsqueeze(1) - idx.unsqueeze(0))            # (N, N), i - j
    far = (delta.abs() > sigma).to(z.dtype)                  # y_ij
    w_bar = delta.pow(2) + 1.0                               # Wbar(i,j)
    w = 1.0 / w_bar                                          # W(i,j)
    d_self = pairwise_sqdist(z, z)                           # D_X(i,j), (B, N, N)
    push = far * w_bar * F.relu(margin - d_self)
    pull = (1.0 - far) * w * d_self
    # Diagonal is i == j: D_X = 0 and |i-j| = 0, so it lands in `pull` contributing
    # nothing. Left in rather than masked, matching the paper's unrestricted double sum.
    return (push + pull).sum(dim=(1, 2)) / (n * n)


class SoftDTW(Objective):
    def __init__(self, latent_dim, gamma=0.1, idm_weight=0.2, idm_sigma=2.0,
                 idm_margin=2.0, view="pair", num_frames=0, normalize=True,
                 project=False, proj_dim=128, proj_hidden=None):
        super().__init__()
        if view not in ("pair", "temporal"):
            raise ValueError(f"view must be pair/temporal, got {view!r}")
        if gamma <= 0:
            raise ValueError(f"gamma must be > 0, got {gamma}")
        self.gamma = gamma
        self.idm_weight = idm_weight
        self.idm_sigma = idm_sigma
        self.idm_margin = idm_margin
        self.view = view
        self.num_frames = num_frames
        self.normalize = normalize
        self.proj = maybe_projection(latent_dim, project, proj_dim, proj_hidden)

    @property
    def requires_pair(self):
        return self.view == "pair"

    def forward(self, z, progress, frame_index=None, z_pair=None, progress_pair=None,
                frame_index_pair=None):
        if self.view == "pair" and z_pair is None:
            raise ValueError("SoftDTW(view='pair') needs a paired clip -- pass --pair-col "
                             "(e.g. `verb` for SSv2)")
        u = self.proj(z)
        v = self.proj(z_pair) if z_pair is not None else None
        if self.normalize:
            u = F.normalize(u, dim=-1)
            v = F.normalize(v, dim=-1) if v is not None else None
        if self.view == "temporal":
            b, t, _ = u.shape
            k = self.num_frames or max(t // 2, 3)
            v = gather_frames(u, sorted_subset_indices(b, t, k, u.device))
            u = gather_frames(u, sorted_subset_indices(b, t, k, u.device))
        elif self.num_frames > 0:
            b, t, _ = u.shape
            u = gather_frames(u, sorted_subset_indices(b, t, self.num_frames, u.device))
            v = gather_frames(v, sorted_subset_indices(b, v.size(1), self.num_frames, v.device))

        align = soft_dtw(pairwise_sqdist(u, v), self.gamma).mean()
        loss = align
        metrics = {"dtw": align.detach()}
        if self.idm_weight > 0:
            idm = (contrastive_idm(u, self.idm_sigma, self.idm_margin)
                   + contrastive_idm(v, self.idm_sigma, self.idm_margin)).mean()
            loss = loss + self.idm_weight * idm
            metrics["idm"] = idm.detach()
        return loss, metrics
