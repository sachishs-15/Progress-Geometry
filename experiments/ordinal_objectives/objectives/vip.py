"""VIP: Value-Implicit Pre-Training (Ma et al., ICLR 2023, arXiv:2210.00030).

The only objective here that comes at ordinality from control rather than from
representation learning. VIP casts a video as an offline goal-conditioned RL problem in
which the reward is -1 everywhere except the goal, and derives an action-free dual
objective whose optimum makes the EMBEDDING DISTANCE TO THE GOAL an optimal value
function:

    V*(o; g) := -||phi(o) - phi(g)||_2

    L(phi) = (1-gamma) * E_{o~mu_0}[ ||phi(o) - phi(g)||_2 ]                       (Eq. 6)
             + log E_{(o,o',g)}[ exp( ||phi(o)-phi(g)||_2 - r(o;g)
                                      - gamma*||phi(o')-phi(g)||_2 ) ]

with r(o; g) = 1(o == g) - 1, i.e. -1 for every non-goal frame.

Why it belongs in this comparison. "Distance to the goal decreases monotonically along
the video" is a strictly stronger statement than "frames are ordered": it fixes not just
the order but the metric structure along the trajectory, and it is exactly what
../encoder_planning_eval/interpolation_planning/ tests when it interpolates between z_1
and z_T. So VIP is the objective most likely to move that second evaluation, and the one
whose progress-metric and planning-metric results are most interesting to read against
each other.

Sampling follows the paper: per clip, a sub-trajectory start s, a goal g > s, a transition
(o, o') = (frame t, frame t+1) with t in [s, g). Because t < g always, r(o;g) is -1
throughout and the indicator never fires -- kept explicit below anyway, since it is the
term that makes the objective a value function rather than a smoothness penalty.

The second term is a log-mean-exp over the batch, which makes VIP the one objective here
whose loss genuinely couples clips to each other -- small batches make it noticeably
noisier, so batch_size is not a free knob for it the way it is elsewhere.
"""
import torch
import torch.nn.functional as F

from heads import maybe_projection

from .base import Objective


class VIP(Objective):
    def __init__(self, latent_dim, gamma=0.98, normalize=False, num_samples=1,
                 project=False, proj_dim=128, proj_hidden=None):
        super().__init__()
        if not 0.0 < gamma < 1.0:
            raise ValueError(f"gamma must be in (0, 1), got {gamma}")
        self.gamma = gamma
        self.normalize = normalize
        # num_samples > 1 draws several (s, g, t) triples per clip per step. The paper
        # uses one; more trades compute for a lower-variance log-mean-exp, which matters
        # because that term is a log of a mean of exponentials and is dominated by its
        # largest few entries.
        self.num_samples = num_samples
        self.proj = maybe_projection(latent_dim, project, proj_dim, proj_hidden)

    def forward(self, z, progress, frame_index=None, z_pair=None, progress_pair=None,
                frame_index_pair=None):
        v = self.proj(z)
        if self.normalize:
            v = F.normalize(v, dim=-1)
        b, t, d = v.shape
        if t < 3:
            raise ValueError(f"VIP needs at least 3 frames per clip to sample s < t < g, got {t}")
        device = v.device

        v = v.repeat(self.num_samples, 1, 1)
        n = v.size(0)
        # s in [0, T-3], g in [s+2, T-1], t in [s, g-1]. s+2 rather than s+1 so that the
        # transition (t, t+1) always exists strictly before the goal.
        s = (torch.rand(n, device=device) * (t - 2)).long()
        g = s + 2 + (torch.rand(n, device=device) * (t - 2 - s).clamp(min=1)).long()
        g = g.clamp(max=t - 1)
        cur = s + (torch.rand(n, device=device) * (g - s)).long().clamp(max=t - 2)
        nxt = (cur + 1).clamp(max=t - 1)

        pick = lambda idx: v[torch.arange(n, device=device), idx]
        z_init, z_goal = pick(s), pick(g)
        z_cur, z_next = pick(cur), pick(nxt)

        dist = lambda a, bb: (a - bb).pow(2).sum(-1).clamp(min=1e-12).sqrt()
        d_init, d_cur, d_next = dist(z_init, z_goal), dist(z_cur, z_goal), dist(z_next, z_goal)

        reward = (cur == g).to(v.dtype) - 1.0        # r(o; g), always -1 here (cur < g)
        initial_term = (1.0 - self.gamma) * d_init.mean()
        td = d_cur - reward - self.gamma * d_next
        # log E[exp(.)] over the batch, via logsumexp for numerical stability.
        td_term = torch.logsumexp(td, dim=0) - torch.log(torch.tensor(float(n), device=device))
        loss = initial_term + td_term
        return loss, {"v_cur": (-d_cur).mean().detach(), "v_goal_init": (-d_init).mean().detach(),
                      "td": td.mean().detach()}
