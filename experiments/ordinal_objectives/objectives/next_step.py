"""Next-step latent prediction -- the control, and the objective this whole project is a
comparison AGAINST.

This is experiments/predictive_world_model/'s own objective, reduced to its cls-token
path: a causal transformer predicts z_{t+1} from z_{<=t}, and the encoder is trained
through that prediction error. Straightness is not asked for; it is whatever emerges from
latents being one-step predictable (see ProceduralWorldModel's docstring for the full
argument).

Both the predictor (models/predictor.py's CausalPredictor) and the loss
(losses/world_model_loss.py's prediction_loss, including its stop-grad convention) are
IMPORTED from that project rather than reimplemented -- see pwm_bridge.py. A control that
is a reimplementation of the thing it controls for is not a control; any divergence
between the two would silently become part of the result.

What this does NOT reproduce is that project's auxiliary terms (curvature/chord/VICReg/
SIGReg). Those are available to every objective here through train.py's --aux-weights, so
`objective: next_step` with `aux_weights: {straighten: 0.1, vicreg_std: 1.0,
vicreg_cov: 0.04}` reconstructs configs/cls_mlp_k400.yaml's full recipe, and with empty
aux_weights it isolates pure prediction.
"""
from pwm_bridge import CausalPredictor, WorldModelLoss

from .base import Objective


class NextStepPrediction(Objective):
    def __init__(self, latent_dim, pred_loss="l1", nhead=8, num_layers=2, dropout=0.1,
                 max_len=16, stop_grad=True):
        super().__init__()
        self.predictor = CausalPredictor(latent_dim, nhead=nhead, num_layers=num_layers,
                                         dropout=dropout, max_len=max_len)
        # Instantiated only for its prediction_loss method (every regularizer weight left
        # at 0) -- train.py owns the auxiliary terms, via its own WorldModelLoss instance.
        self.criterion = WorldModelLoss(pred_loss=pred_loss, pred_weight=1.0,
                                        stop_grad=stop_grad)

    def forward(self, z, progress, frame_index=None, z_pair=None, progress_pair=None,
                frame_index_pair=None):
        z_pred = self.predictor(z)
        loss = self.criterion.prediction_loss(z_pred, z)
        return loss, {}
