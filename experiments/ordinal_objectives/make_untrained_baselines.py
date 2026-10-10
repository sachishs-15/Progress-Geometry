"""Emit the two UNTRAINED encoder baselines the comparison needs, as ordinary checkpoints.

baseline_raw_dinov2 (already in the eval spec) scores the frozen features directly, with
no ClsMlpEncoder in the path at all. That answers "is this encoder family better than the
features it sits on", which is a fine question but NOT "did this objective's training
help" -- every trained run starts from a RANDOMLY INITIALIZED 2-layer MLP, which is not
the identity and can destroy whatever ordinal structure DINOv2 already had before a single
gradient step. Comparing a trained run to raw DINOv2 therefore confounds "what the
objective did" with "what the random projection undid".

Two references fix that:

  untrained_random_init  the exact ClsMlpEncoder every run starts from, at seed 0, with
                         no training whatsoever. The honest zero point: any trained run
                         above this line learned something, any run below it made things
                         worse than doing nothing.
  untrained_identity     use_projector=False, i.e. ClipRMSNorm applied to raw DINOv2 and
                         nothing else. Isolates how much of any gap is the normalization
                         rather than the MLP.

Both are written in train.py's checkpoint layout so compute_progress_metrics.py scores
them through exactly the same path as a trained run -- that equivalence is the point, and
is why these are emitted as checkpoints rather than special-cased in the evaluator.
"""
import argparse
import importlib.util
import sys
from pathlib import Path

import torch
import yaml

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
from pwm_bridge import ClsMlpEncoder  # noqa: E402

# By PATH, not by name. Importing pwm_bridge puts both predictive_world_model/ and (via
# its feature_provider) LiFT's root ahead of this directory on sys.path, and BOTH have a
# train.py -- a plain `import train` silently gets one of theirs. See pwm_bridge's
# docstring; tests/test_objectives.py loads it the same way.
_spec = importlib.util.spec_from_file_location("ordinal_objectives_train", _HERE / "train.py")
_train = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_train)
EVAL_ARCH_DEFAULTS, save_checkpoint = _train.EVAL_ARCH_DEFAULTS, _train.save_checkpoint


class _Empty(torch.nn.Module):
    """Stands in for an objective head, so save_checkpoint's signature is satisfied and
    the written file is byte-comparable to a real run's."""


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoints-dir", default="checkpoints")
    p.add_argument("--feature-dim", type=int, default=384)
    p.add_argument("--latent-dim", type=int, default=384)
    p.add_argument("--hidden-dim", type=int, default=384)
    p.add_argument("--norm-type", default="clip_rmsnorm")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    root = Path(__file__).resolve().parent / args.checkpoints_dir
    for name, use_projector in [("untrained_random_init", True), ("untrained_identity", False)]:
        torch.manual_seed(args.seed)
        enc = ClsMlpEncoder(feature_dim=args.feature_dim, latent_dim=args.latent_dim,
                            hidden_dim=args.hidden_dim, use_projector=use_projector,
                            norm_type=args.norm_type)
        d = root / name
        d.mkdir(parents=True, exist_ok=True)
        save_checkpoint(d / "best.ckpt", enc, _Empty(), -1, "untrained")
        cfg = {"feature_dim": args.feature_dim, "latent_dim": args.latent_dim,
               "hidden_dim": args.hidden_dim, "use_projector": use_projector,
               "norm_type": args.norm_type, "epochs": 0, **EVAL_ARCH_DEFAULTS}
        (d / "config.yaml").write_text(yaml.safe_dump(cfg))
        # A one-row train_log.csv so make_eval_spec's completeness check (epoch >= epochs-1)
        # treats these as finished rather than skipping them as partial runs.
        (d / "train_log.csv").write_text("epoch\n-1\n")
        print(f"  wrote {d}  (use_projector={use_projector}, "
              f"{sum(q.numel() for q in enc.parameters())} params, untrained)")


if __name__ == "__main__":
    main()
