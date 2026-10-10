# proc-planning

Procedural planning and progress estimation for long-horizon tasks (computer vision).

## Setup

```bash
conda env create -f environment.yml
conda activate proc-planning
```

`environment.yml` pins the direct runtime dependencies used by the checked-in training,
feature-extraction, and evaluation code, including the CUDA 12.6 PyTorch build. Create a
separate environment if the host requires a different CUDA build. External baseline repos
under `external/` may additionally carry their own environment requirements.

For an exact Linux environment snapshot, including transitive packages and Conda build
strings, use `conda env create -f environment.lock.yml` and activate
`proc-planning-locked`.

The active implementation lives under `experiments/`, with launchers and cluster jobs under
`scripts/`. Run each experiment's documented entry point from its experiment directory; the
entry points add their local modules to `sys.path` where needed. `src/proc_planning/` is only a
reserved package skeleton and currently contains no implementation.

## Data

`data/` is a symlink to wherever the actual dataset lives (not stored in this repo):

```bash
ln -s /path/to/actual/data ~/proc-planning/data
```

See [docs/data_integrity.md](docs/data_integrity.md) before using legacy precomputed
features; several existing artifacts were confirmed to contain encoded replacement
black frames and must be regenerated.

## Layout

```
src/proc_planning/    reserved package skeleton (currently empty)
configs/              shared top-level configuration files
experiments/          implementations, experiment-specific configs, tests, and outputs
external/             cloned baseline repos, not tracked by git — see external/README.md
notebooks/            exploratory analysis, scratch
scripts/local/        run scripts for local/single-GPU use
scripts/slurm/        sbatch templates for cluster runs
results/              small, versioned tables/figures that feed the paper
paper/                writeup, figures, Overleaf sync
docs/                 design notes, literature notes, decisions log
```

## Baseline repos

See `external/README.md` for how baseline/comparison repos are tracked (they're gitignored,
so clone them fresh per the manifest rather than expecting them in git history).
