"""Direct progress regression -- the baseline the progress-prediction literature is built
on.

Becattini et al., "Am I Done? Predicting Action Progress in Videos" (arXiv:1705.01781),
defines the task and the target: p_i = (t_i - t_S)/(t_E - t_S), the fraction of the action
already elapsed, regressed per frame. Their ProgressNet gets the temporal context from an
LSTM; here the encoder is per-frame by construction, so this is the "static" variant of
their own ablation -- which is the right comparison for this project, since a recurrent
head would let the head, not the latent space, carry the ordering.

Two loss options beyond plain L1/L2/Huber:

`boundary_observant` implements their Boundary Observant loss (Eqs. 2-3), which is an L1
error weighted by how close prediction and target are to a PHASE BOUNDARY rather than a
phase centre:

    e_k(p, p_hat) = min[1, ((p - m_k)/(r_k*sqrt2))^2 + ((p_hat - m_k)/(r_k*sqrt2))^2]
    L_BO          = mean_i  min_k{e_k} * |p_i - p_hat_i|

with phase k spanning [l_k, u_k], centre m_k = (l_k+u_k)/2 and half-width r_k =
(u_k-l_k)/2. At a phase centre (p = p_hat = m_k) the weight is 0; at a boundary it is 1.
The paper's rationale is that boundary precision is what makes progress usable
downstream; for this project it is also a test of a sharper hypothesis -- whether
upweighting the ENDS of the trajectory is what actually straightens it, since
time_progress_spearman is driven by the start/end chord.

The paper does not fix the number of phases (it comes from the action's own linguistic
phase structure); bo_num_phases here divides [0,1] into that many equal phases, so
bo_num_phases=1 degenerates to a centre-weighted L1 over the whole clip and large values
approach uniform L1.
"""
import torch
import torch.nn.functional as F

from heads import FrameHead

from .base import Objective


class ProgressRegression(Objective):
    def __init__(self, latent_dim, loss="l2", hidden_dim=None, depth=2, dropout=0.0,
                 sigmoid=True, boundary_observant=False, bo_num_phases=3, huber_delta=0.1):
        super().__init__()
        if loss not in ("l1", "l2", "huber"):
            raise ValueError(f"loss must be 'l1', 'l2' or 'huber', got {loss!r}")
        if bo_num_phases < 1:
            raise ValueError(f"bo_num_phases must be >= 1, got {bo_num_phases}")
        self.loss = loss
        # NOT torch's default of 1.0. Targets are in [0,1] and sigmoid predictions are
        # too, so |error| <= 1 always -- at delta=1 the Huber loss never leaves its
        # quadratic regime and is bitwise identical to 0.5*MSE (verified), i.e. the
        # `huber` option would be an alias for `l2` rather than a third loss. 0.1 puts
        # the linear regime where errors actually are.
        self.huber_delta = huber_delta
        # sigmoid=True bounds the prediction to the target's own [0,1] range. Off by
        # default nowhere -- but worth being able to turn off, because a saturating
        # output is also a way for the head to flatten large latent differences into
        # identical predictions, which is exactly the kind of "the head solved it, the
        # latent space did not" outcome this comparison must be able to see.
        self.sigmoid = sigmoid
        self.boundary_observant = boundary_observant
        self.bo_num_phases = bo_num_phases
        self.head = FrameHead(latent_dim, out_dim=1, hidden_dim=hidden_dim, depth=depth,
                              dropout=dropout)

    def _bo_weight(self, p, p_hat):
        k = torch.arange(self.bo_num_phases, device=p.device, dtype=p.dtype)
        width = 1.0 / self.bo_num_phases
        centres = (k + 0.5) * width          # m_k
        half = width / 2.0                   # r_k
        scale = half * (2.0 ** 0.5)
        e = (((p.unsqueeze(-1) - centres) / scale) ** 2
             + ((p_hat.unsqueeze(-1) - centres) / scale) ** 2)
        return e.clamp(max=1.0).min(dim=-1).values

    def forward(self, z, progress, frame_index=None, z_pair=None, progress_pair=None,
                frame_index_pair=None):
        p_hat = self.head(z).squeeze(-1)
        if self.sigmoid:
            p_hat = torch.sigmoid(p_hat)
        if self.boundary_observant:
            # The weight depends on p_hat, and the paper treats it as a weight rather
            # than a second thing to optimise -- detached so gradient flows only through
            # the |p - p_hat| factor, otherwise the model could reduce the loss by moving
            # its prediction toward a phase centre instead of toward the target.
            w = self._bo_weight(progress, p_hat.detach())
            loss = (w * (progress - p_hat).abs()).mean()
        elif self.loss == "l1":
            loss = F.l1_loss(p_hat, progress)
        elif self.loss == "l2":
            loss = F.mse_loss(p_hat, progress)
        else:
            loss = F.huber_loss(p_hat, progress, delta=self.huber_delta)
        return loss, {"mae": (p_hat - progress).abs().mean().detach()}
