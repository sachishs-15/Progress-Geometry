"""Run a whole list of objective configs (and/or hyperparameter overrides of them) as
parallel subprocesses.

Each run is its own `train.py` process -- not imported and looped in-process -- so one
objective blowing up cannot corrupt torch/CUDA state for the rest, and "parallel" means
real OS-level parallelism. Same reasoning and same shape as
../encoder_planning_eval/progress_metrics/run_progress_metrics_batch.py.

Spec format (see sweeps/objectives.example.json):
{
  "defaults": {"epochs": 100},                  // merged into every run below
  "runs": [
    {"config": "configs/rank_n_contrast.yaml"}, // name/checkpoint_dir come from the config
    {"config": "configs/rank_n_contrast.yaml",  // ... unless overridden:
     "name": "rnc_t0.5",
     "objective_kwargs": {"temperature": 0.5}}  // nested dicts are passed as JSON flags
  ]
}

A run's "name" (when given) sets --checkpoint-dir to <checkpoints-dir>/<name>, which is
how a hyperparameter sweep over one config avoids every run auto-suffixing into
_1/_2/_3 and losing track of which was which.

Resources are handed out as SLOTS -- one (gpu, cpu-block) pair per concurrent run --
claimed from a pool when a run starts and returned when it exits. Deliberately not a
round-robin over the run index: with `i % n_gpus`, two runs that happen to be in flight
together can share an index parity (if an earlier run finishes early the pool refills out
of order), putting both on one GPU while another sits idle. Claiming from a pool is
correct whatever order runs finish in.

--gpus sets CUDA_VISIBLE_DEVICES per run, so each sees exactly one card as cuda:0 and
nothing downstream needs to know which physical device it got. Without it every
concurrent run opens a context on device 0 and contends for its SMs -- the models here
are small enough that memory never forces the issue, so the only symptom would be a
wall-clock that fails to improve with --workers.

--cpus-per-run wraps each run in `taskset -c`, giving it a disjoint block of cores, and
pins OMP/MKL thread counts (--omp-threads). Both matter on a shared box: without taskset
every run's dataloader workers compete across all cores, and without the thread cap torch
spawns one OMP thread per visible CPU inside a taskset-restricted process, so e.g. 64
threads thrash over 8 cores.

Blocks are carved out of os.sched_getaffinity(0) -- the cores this process may ACTUALLY
use -- not out of range(0, ncpus). Under PBS the job runs in a cpuset that is usually not
cores 0..n-1, so `taskset -c 0-7` would fail outright with "Invalid argument" or, worse,
silently fight the cgroup. Reading the affinity makes the same command correct whether it
runs inside a job or on a bare host.

Usage:
    python run_sweep.py --spec sweeps/objectives.example.json --workers 2
    python run_sweep.py --configs configs/*.yaml --workers 2     # one run per config
"""
import argparse
import json
import os
import queue
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent
TRAIN = HERE / "train.py"
#: Keys whose values are dicts and therefore go across as JSON on the command line;
#: everything else becomes a plain --flag value.
JSON_KEYS = {"objective_kwargs", "aux_weights"}


def visible_gpus():
    """Device indices to spread runs over. Honours an inherited CUDA_VISIBLE_DEVICES so
    this composes with an outer scheduler (PBS) rather than fighting it."""
    inherited = os.environ.get("CUDA_VISIBLE_DEVICES")
    if inherited:
        return [d.strip() for d in inherited.split(",") if d.strip()]
    try:
        import torch
        return [str(i) for i in range(torch.cuda.device_count())]
    except Exception:
        return []


def build_command(run: dict, python: str, checkpoints_dir: Path) -> list:
    config = run.get("config")
    if not config:
        raise SystemExit(f"run {run!r} has no 'config'")
    cmd = [python, str(TRAIN), "--config", str(HERE / config)]
    name = run.get("name")
    if name:
        cmd += ["--checkpoint-dir", str(checkpoints_dir / name)]
    for key, value in run.items():
        if key in ("config", "name"):
            continue
        flag = "--" + key.replace("_", "-")
        cmd += [flag, json.dumps(value) if key in JSON_KEYS else str(value)]
    return cmd


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--spec", help="JSON spec (see the docstring); mutually exclusive with --configs")
    p.add_argument("--configs", nargs="*", help="config paths, one run each, no overrides")
    p.add_argument("--workers", type=int, default=1,
                   help="concurrent training processes. Capped at the number of resource "
                        "slots, i.e. at len(--gpus).")
    p.add_argument("--gpus", default=None,
                   help="comma-separated device indices to distribute runs over "
                        "(default: every device torch can see, or an inherited "
                        "CUDA_VISIBLE_DEVICES). Pass '' to leave the environment alone.")
    p.add_argument("--cpus-per-run", type=int, default=0,
                   help="pin each run to this many cores with taskset; 0 (default) does not "
                        "pin. Total cores used is this times the number of slots.")
    p.add_argument("--omp-threads", type=int, default=2,
                   help="OMP_NUM_THREADS/MKL_NUM_THREADS per run, only applied when "
                        "--cpus-per-run pins cores (see the docstring on oversubscription)")
    p.add_argument("--checkpoints-dir", default="checkpoints")
    p.add_argument("--python", default="/home/sachish/.conda/envs/lift/bin/python")
    p.add_argument("--dry-run", action="store_true", help="print the commands and exit")
    args = p.parse_args()

    if bool(args.spec) == bool(args.configs):
        raise SystemExit("pass exactly one of --spec / --configs")

    if args.spec:
        spec = json.loads(Path(args.spec).read_text())
        defaults = spec.get("defaults", {})
        runs = [{**defaults, **r} for r in spec["runs"]]
    else:
        runs = [{"config": c} for c in args.configs]

    ckpt_root = HERE / args.checkpoints_dir
    gpus = visible_gpus() if args.gpus is None else [g for g in args.gpus.split(",") if g]
    affinity = sorted(os.sched_getaffinity(0))
    n_slots = len(gpus or [None])
    if args.cpus_per_run > 0 and args.cpus_per_run * n_slots > len(affinity):
        raise SystemExit(f"--cpus-per-run {args.cpus_per_run} x {n_slots} slot(s) needs "
                         f"{args.cpus_per_run * n_slots} cores, but this process is only "
                         f"allowed {len(affinity)} ({affinity[0]}..{affinity[-1]})")
    slots = []
    for k, gpu in enumerate(gpus or [None]):
        cpus = None
        if args.cpus_per_run > 0:
            block = affinity[k * args.cpus_per_run:(k + 1) * args.cpus_per_run]
            cpus = ",".join(str(c) for c in block)
        slots.append((gpu, cpus))
    workers = min(args.workers, len(slots))
    if workers < args.workers:
        print(f"note: --workers {args.workers} capped to {workers}, the number of "
              f"resource slots (one per --gpus entry)")

    commands = [(r, build_command(r, args.python, ckpt_root)) for r in runs]
    print(f"{len(commands)} run(s) over {len(slots)} slot(s), {workers} at a time:")
    for gpu, cpus in slots:
        print(f"  slot: gpu={gpu if gpu is not None else '-'} "
              f"cpus={cpus or 'unpinned'}"
              + (f" omp={args.omp_threads}" if cpus else ""))
    for run, cmd in commands:
        print(f"  {run.get('name') or Path(run['config']).stem}: {' '.join(cmd)}")
    if args.dry_run:
        return

    failures = []
    slot_pool = queue.Queue()
    for slot in slots:
        slot_pool.put(slot)

    def launch(item):
        run, cmd = item
        label = run.get("name") or Path(run["config"]).stem
        gpu, cpus = slot_pool.get()
        try:
            env = dict(os.environ)
            if gpu is not None:
                env["CUDA_VISIBLE_DEVICES"] = gpu
            if cpus is not None:
                cmd = ["taskset", "-c", cpus] + cmd
                env["OMP_NUM_THREADS"] = str(args.omp_threads)
                env["MKL_NUM_THREADS"] = str(args.omp_threads)
            print(f"[start] {label} (gpu {gpu}, cpus {cpus or 'unpinned'})", flush=True)
            # stdout/stderr are left to train.py's own <checkpoint-dir>/train.log rather
            # than captured here -- with several runs interleaving, a merged stream is
            # unreadable and the per-run log is where anyone actually looks.
            proc = subprocess.run(cmd, cwd=HERE, env=env)
            return label, proc.returncode
        finally:
            # Returned even if the run raised, or the slot leaks and the sweep deadlocks.
            slot_pool.put((gpu, cpus))

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(launch, item) for item in commands]
        for fut in as_completed(futures):
            label, code = fut.result()
            print(f"[{'ok' if code == 0 else 'FAILED'}] {label} (exit {code})", flush=True)
            if code != 0:
                failures.append(label)

    if failures:
        print(f"{len(failures)} run(s) failed: {failures}")
        sys.exit(1)
    print("all runs finished")


if __name__ == "__main__":
    main()
