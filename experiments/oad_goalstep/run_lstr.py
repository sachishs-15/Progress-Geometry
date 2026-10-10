"""Run LSTR (external/lstr, unmodified) on Goal-Step with single-stream frozen features.

  python run_lstr.py train --config_file configs/lstr_goalstep_dinov2.yaml [--gpu 0] [KEY VAL ...]
  python run_lstr.py test  --config_file configs/lstr_goalstep_dinov2.yaml MODEL.CHECKPOINT <ckpt>

Registers the 'GoalStep' dataset with LSTR's registries, then hands over to LSTR's own
tools/train_net.py / tools/test_net.py.
"""
import runpy
import sys
from pathlib import Path

LSTR = Path(__file__).resolve().parents[2] / "external" / "lstr"
sys.path.insert(0, str(LSTR / "src"))

from rekognition_online_action_detection.datasets import datasets, perframe_data_layers as layers  # noqa: E402
from rekognition_online_action_detection.models import feature_head  # noqa: E402
from rekognition_online_action_detection.utils import parser  # noqa: E402
from rekognition_online_action_detection.config import defaults  # noqa: E402

import lstr_tokens  # noqa: E402

datasets.DATA_LAYERS.register("LSTRGoalStep")(layers.LSTRDataLayer)
datasets.DATA_LAYERS.register("LSTRBatchInferenceGoalStep")(layers.LSTRBatchInferenceDataLayer)
feature_head.FEATURE_SIZES["dinov2_cls"] = 384
feature_head.FEATURE_SIZES["dinov2_noreg_meanpatch"] = 384
feature_head.FEATURE_SIZES["dinov2_noreg_cls"] = 384
feature_head.FEATURE_SIZES["pwm_ep19_agg"] = 128
feature_head.FEATURE_SIZES["pwm_ep19_z"] = 196 * 8
feature_head.FEATURE_SIZES["pwm_best_agg"] = 128
feature_head.FEATURE_SIZES["pwm_best_z"] = 196 * 8
feature_head.FEATURE_SIZES["pwm_ep19_mean"] = 8
feature_head.FEATURE_SIZES["pwm_best_mean"] = 8
lstr_tokens.add_defaults(defaults)


def _goalstep_head(cfg):
    if cfg.INPUT.VISUAL_FEATURE in feature_head.FEATURE_SIZES:
        return feature_head.BaseFeatureHead(cfg)
    return lstr_tokens.TokenFeatureHead(cfg)


feature_head.FEATURE_HEADS["GoalStep"] = _goalstep_head

# 1.875 features/s makes LSTR's seconds-to-frames products floats; its slicing needs ints.
_infer = parser.assert_and_infer_cfg


def _infer_int(cfg, args):
    _infer(cfg, args)
    m = cfg.MODEL.LSTR
    for k in [k for k in m if k.endswith(("_LENGTH", "_NUM_SAMPLES"))]:
        m[k] = int(m[k])


parser.assert_and_infer_cfg = _infer_int

if __name__ == "__main__":
    mode = sys.argv.pop(1)
    sys.path.insert(0, str(LSTR / "tools"))
    runpy.run_path(str(LSTR / "tools" / f"{mode}_net.py"), run_name="__main__")
