"""No primary objective at all -- only whatever train.py's --aux-weights adds.

Exists so the geometric regularizers (curvature/chord/VICReg/SIGReg, all borrowed from
predictive_world_model/losses/world_model_loss.py) can be run as objectives in their own
right. `objective: none` with `aux_weights: {straighten: 1.0}` answers "what does asking
for straightness, and nothing else, do to the latent space?", which is a control worth
having before attributing any other objective's straightness to its own mechanism.

With empty aux_weights it trains nothing (zero loss, zero gradient) and the encoder stays
at its initialization -- occasionally useful as a sanity check that the evaluation path
is measuring the encoder and not something downstream of it.
"""
from .base import Objective


class NoObjective(Objective):
    def __init__(self, latent_dim):
        super().__init__()
        self.latent_dim = latent_dim

    def forward(self, z, progress, frame_index=None, z_pair=None, progress_pair=None,
                frame_index_pair=None):
        # z.sum() * 0, NOT z.new_zeros(()): the latter is a fresh leaf with no grad_fn, so
        # `objective: none` with empty aux_weights -- a use case this module's docstring
        # explicitly offers -- would die in loss.backward() with "element 0 of tensors does
        # not require grad". This is the same value with a (zero) gradient attached.
        return z.sum() * 0.0, {}
