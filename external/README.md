# External / baseline repos

Everything else in this directory is gitignored — cloned repos have their own git history and
shouldn't be nested inside this one. Track what you've cloned here instead, so the setup is
reproducible.

| Repo | Path | URL | Commit | Purpose |
|------|------|-----|--------|---------|
| LiFT | `chirality_in_action/LiFT` | https://github.com/bpiyush/LiFT.git | `1d28901` | `PredictiveWorldModel` reference, DINOv2 feature precompute pipeline (`dinov2.py`, `precompute_features.py`) and progress metrics (`visualization/progress_metrics.py`) — `experiments/predictive_world_model` imports from this directly |
| temporal-straightening | `temporal-straightening` | https://github.com/agentic-learning-ai-lab/temporal-straightening.git | `9efb6fd` | source architecture (`models/dino.py`'s `ChannelProjector`/`GlobalProjector`, `models/vit.py`'s block-causal predictor) that `experiments/predictive_world_model`'s patch-token mode ports from |

To add one:

```bash
git clone <url> external/<name>
cd external/<name> && git rev-parse HEAD   # record this commit in the table above
```
