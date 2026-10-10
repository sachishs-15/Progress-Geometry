"""PROGRESSOR (Ayalew et al., ICCV 2025, arXiv:2411.17764): progress predicted from the
TRIPLET (initial, current, goal), not from the current frame alone.

The only goal-conditioned objective here besides VIP, and the one that makes the
conditioning explicit rather than implicit in a value function. Its premise is that "how
far along am I?" is not well posed without saying what you are heading toward -- the
answer depends on the goal, and a model given only the current frame has to guess it.

That premise is invisible on this data unless you look for it. Clips are 16 frames spread
evenly over a whole video, so the goal is ALWAYS frame 15 and progress is always i/15 --
which is exactly why plain progress_regression works here at all. Under --crop-size the
endpoint moves and the conditioning starts to earn its keep, so crop is the setting where
this objective should separate from progress_regression, not the default one.

Per clip: sample s < g (sub-trajectory start and goal) and s <= t <= g (current frame),
then predict progress (t - s) / (g - s) from [z_s ; z_t ; z_g].

variance=True (default, the paper's form): the head emits (mu, log sigma^2) and the loss
is the Gaussian negative log-likelihood, so the model can say how sure it is. The paper
minimises a KL between its predicted progress distribution and a ground-truth one; with a
point target that reduces to this NLL up to a constant.

variance=False: same triplet, same target, plain MSE on mu. Deliberately offered so that
"does conditioning on the goal help?" can be separated from "does modelling uncertainty
help?" -- against progress_regression, the three form a clean ladder:

    progress_regression        no conditioning, no uncertainty
    progressor(variance=False) start+goal conditioning, no uncertainty
    progressor                 start+goal conditioning, uncertainty
"""
import torch
import torch.nn.functional as F

from heads import mlp

from .base import Objective


class Progressor(Objective):
    def __init__(self, latent_dim, variance=True, hidden_dim=None, depth=2, dropout=0.0,
                 num_samples=4, min_span=2, min_logvar=-8.0, max_logvar=4.0):
        super().__init__()
        self.variance = variance
        # More than one (s, g, t) draw per clip per step: unlike the per-frame objectives,
        # which supervise all 16 frames of every clip every step, one triplet would use a
        # single frame's worth of signal from a whole clip.
        self.num_samples = num_samples
        if min_span < 2:
            raise ValueError(f"min_span must be >= 2 so that s < t <= g is possible, got {min_span}")
        self.min_span = min_span
        # Clamping log-variance is what keeps the NLL from running away early, when the
        # head can drive sigma^2 -> 0 on an easy triplet and make the loss -inf.
        self.min_logvar, self.max_logvar = min_logvar, max_logvar
        self.head = mlp(3 * latent_dim, 2 if variance else 1, hidden_dim=hidden_dim or latent_dim,
                        depth=depth, dropout=dropout)

    def _sample_indices(self, n, t, device):
        """Official loader semantics: uniform valid start, then goal, then current."""
        s = (torch.rand(n, device=device) * (t - self.min_span)).long()
        span = t - self.min_span - s
        g = s + self.min_span + (torch.rand(n, device=device) * span).long()
        cur = s + (torch.rand(n, device=device) * (g - s + 1).to(torch.float)).long()
        return s, cur, g

    def forward(self, z, progress, frame_index=None, z_pair=None, progress_pair=None,
                frame_index_pair=None):
        b, t, d = z.shape
        if t < self.min_span + 1:
            raise ValueError(f"progressor needs at least {self.min_span + 1} frames/clip "
                             f"to sample s < t <= g with a span of {self.min_span}, got {t}")
        z = z.repeat(self.num_samples, 1, 1)
        n = z.size(0)
        device = z.device

        # s in [0, T-1-min_span], g in [s+min_span, T-1], t in [s, g].
        # Match the official PROGRESSOR loader's randint half-open ranges. The previous
        # clamp-based draw made the last goal roughly twice as likely as every other one.
        s, cur, g = self._sample_indices(n, t, device)

        pick = lambda idx: z[torch.arange(n, device=device), idx]
        feats = torch.cat([pick(s), pick(cur), pick(g)], dim=-1)
        out = self.head(feats)
        target = (cur - s).to(z.dtype) / (g - s).to(z.dtype)

        mu = out[:, 0]
        if not self.variance:
            return F.mse_loss(mu, target), {"mae": (mu - target).abs().mean().detach()}
        logvar = out[:, 1].clamp(self.min_logvar, self.max_logvar)
        nll = 0.5 * (logvar + (target - mu) ** 2 / logvar.exp())
        return nll.mean(), {"mae": (mu - target).abs().mean().detach(),
                            "sigma": (0.5 * logvar).exp().mean().detach()}
