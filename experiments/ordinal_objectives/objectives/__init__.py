"""OBJECTIVE_REGISTRY: every training objective this project compares, keyed by the name
that goes in a config's `objective:` field.

Same registry convention as predictive_world_model/models/norms.py's NORM_REGISTRY and
encoder_planning_eval/interpolation_planning/predictors/'s PREDICTOR_REGISTRY: adding an
objective from the literature is one new module plus one line here, with no change to
train.py, the configs schema, or the evaluation path.

Every entry is constructed as CLASS(latent_dim=..., **objective_kwargs), and train.py
validates objective_kwargs against the class's own signature, so a typo in a config is an
error naming the valid keys rather than a silently-ignored hyperparameter.

Grouped by what the objective is actually supervised BY, which is the axis the comparison
is really about:

  absolute progress   progress_regression, ordinal_classification
  goal-conditioned    progressor (and vip, below, via its value function)
  relative order      pairwise_rank, shuffle_sort, arrow_of_time
  temporal metric     rank_n_contrast, tcn, scl, tcc, soft_dtw, vip
  rate of change      pace
  prediction          next_step (the predictive_world_model control)
  nothing             none (auxiliary regularizers only)
"""
from .arrow_of_time import ArrowOfTime
from .soft_dtw import SoftDTW
from .none import NoObjective
from .ordinal_classification import OrdinalClassification
from .pace import PacePrediction
from .pairwise_rank import PairwiseRank
from .progress_regression import ProgressRegression
from .progressor import Progressor
from .rank_n_contrast import RankNContrast
from .scl import SCL
from .shuffle_sort import ShuffleSort
from .tcc import TCC
from .tcn import TCN
from .vip import VIP
from .next_step import NextStepPrediction

OBJECTIVE_REGISTRY = {
    "progress_regression": ProgressRegression,
    "ordinal_classification": OrdinalClassification,
    "pairwise_rank": PairwiseRank,
    "shuffle_sort": ShuffleSort,
    "rank_n_contrast": RankNContrast,
    "tcn": TCN,
    "tcc": TCC,
    "vip": VIP,
    "soft_dtw": SoftDTW,
    "progressor": Progressor,
    "scl": SCL,
    "arrow_of_time": ArrowOfTime,
    "pace": PacePrediction,
    "next_step": NextStepPrediction,
    "none": NoObjective,
}

__all__ = ["OBJECTIVE_REGISTRY"]
