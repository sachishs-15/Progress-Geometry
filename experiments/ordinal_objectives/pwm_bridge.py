"""The one place this project reaches into its siblings' code.

Nothing under experiments/ordinal_objectives/ is allowed to modify
experiments/predictive_world_model/ or experiments/encoder_planning_eval/, but three
things there are worth importing rather than re-implementing, because the whole point of
this project is that its numbers be directly comparable to theirs:

  ClsMlpEncoder      the encoder every objective here trains. Imported, not copied, so a
                     checkpoint written here has BYTE-IDENTICAL state_dict keys to one
                     written by predictive_world_model/train.py -- which is what lets
                     encoder_planning_eval/progress_metrics/compute_progress_metrics.py
                     score our checkpoints with no changes on its side (it rebuilds a
                     ProceduralWorldModel from config.yaml's ARCH_KEYS and loads
                     everything that isn't `predictor.*`; see train.py's save_checkpoint).
  CausalPredictor    only used by objectives/next_step.py, the world-model control. A
                     control that isn't the same code as the thing it's controlling for
                     isn't a control.
  WorldModelLoss     its straightening/chord/VICReg/SIGReg terms are this project's
                     auxiliary regularizers (train.py's --aux-weights). Same reasoning.
  FeatureProvider    the precomputed-or-live feature tiering, unchanged.

sys.path surgery is confined to this module so that no other file here has to know where
the siblings live. NOTE the name collision this implies: predictive_world_model/ is put
at the FRONT of sys.path, so its top-level module names -- `models`, `data`, `losses` and
`train` -- win over any same-named module in this directory. Hence `clip_data.py` and
`heads.py` here rather than `data/` and `models/`. `train.py` does exist here, but only
ever as the __main__ entry point, never imported by name; anything that needs to import it
as a module (tests/test_objectives.py) must load it by file path, because a plain
`import train` would silently get predictive_world_model's instead.

clip_progress_metrics is deliberately NOT imported here at module scope -- LiFT's
visualization/progress_metrics.py pulls in the `lift` package (lightning and friends) at
import time, and training itself has no need of it. train.py's optional in-training
ordinality evaluation calls load_clip_progress_metrics() to pay that cost only when
--eval-every is actually on.
"""
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
EXPERIMENTS_ROOT = _HERE.parent
REPO_ROOT = EXPERIMENTS_ROOT.parent

PWM_ROOT = EXPERIMENTS_ROOT / "predictive_world_model"
LIFT_ROOT = REPO_ROOT / "external" / "chirality_in_action" / "LiFT"

if str(PWM_ROOT) not in sys.path:
    sys.path.insert(0, str(PWM_ROOT))

from data.feature_provider import FeatureProvider  # noqa: E402
from losses.world_model_loss import WorldModelLoss  # noqa: E402
from models.encoder import ClsMlpEncoder  # noqa: E402
from models.predictor import CausalPredictor  # noqa: E402


def load_clip_progress_metrics():
    """LiFT's own clip_progress_metrics, imported lazily (see module docstring).

    Same function experiments/encoder_planning_eval/progress_metrics/ scores every
    encoder in this repo with, so the in-training monitor metric and the final offline
    evaluation are literally the same computation -- a checkpoint selected on
    time_progress_spearman here is selected on the number the comparison table reports.
    """
    viz = LIFT_ROOT / "visualization"
    if str(viz) not in sys.path:
        sys.path.insert(0, str(viz))
    from progress_metrics import clip_progress_metrics
    return clip_progress_metrics


__all__ = ["ClsMlpEncoder", "CausalPredictor", "FeatureProvider", "WorldModelLoss",
           "load_clip_progress_metrics", "PWM_ROOT", "LIFT_ROOT", "REPO_ROOT"]
