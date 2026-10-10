"""Smoke + contract tests for every objective in the registry, runnable on CPU with no
dataset:

    /home/sachish/.conda/envs/lift/bin/python tests/test_objectives.py

The two contracts that actually matter here (everything else is shape checking):

  1. Every objective produces a finite loss with gradient reaching the ENCODER, not just
     its own head. An objective whose gradient stops at the head would train happily,
     log a falling loss, and leave the latent space -- the only thing this project
     measures -- untouched.
  2. A checkpoint written by train.py loads into a ProceduralWorldModel the way
     ../encoder_planning_eval/progress_metrics/compute_progress_metrics.py loads it, with
     no missing or unexpected keys outside `predictor.*`. That script is not allowed to
     change, so this is the test that keeps the evaluation path working.
"""
import importlib.util
import inspect
import sys
import tempfile
from pathlib import Path

import torch
import yaml
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from clip_data import ObjectiveClipDataset  # noqa: E402
from objectives import OBJECTIVE_REGISTRY  # noqa: E402
from heads import CoralHead  # noqa: E402
from objectives.pairwise_rank import PairwiseRank  # noqa: E402
from objectives.progressor import Progressor  # noqa: E402
from objectives.shuffle_sort import ShuffleSort  # noqa: E402
from objectives.tcc import TCC  # noqa: E402
from pwm_bridge import ClsMlpEncoder  # noqa: E402

# Loaded by PATH, not by name: pwm_bridge puts predictive_world_model/ at the front of
# sys.path, and it has a train.py of its own that a plain `import train` would find first
# (see pwm_bridge's docstring).
_spec = importlib.util.spec_from_file_location("ordinal_objectives_train", ROOT / "train.py")
train_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(train_mod)

B, T, D = 6, 16, 32
#: Per-objective constructor overrides needed to fit the tiny (B, T, D) fixture above --
#: everything not listed here runs at its real defaults.
FIXTURE_KWARGS = {
    "shuffle_sort": {"num_frames": 3, "d_model": 32, "num_layers": 1},
    "arrow_of_time": {"d_model": 32, "num_layers": 1},
    "pace": {"num_frames": 3, "rates": [1, 2, 3], "d_model": 32, "num_layers": 1},
    "next_step": {"nhead": 4, "num_layers": 1},
}
VARIANTS = {
    "progress_regression": [{"loss": "l1"}, {"loss": "l2"}, {"loss": "huber"},
                            {"boundary_observant": True, "bo_num_phases": 3}],
    "ordinal_classification": [{"variant": v, "num_bins": 8} for v in ("ce", "soft", "coral", "corn")],
    "pairwise_rank": [{"mode": "logistic"}, {"mode": "margin"}, {"cross_clip": True}],
    "shuffle_sort": [{"variant": "sort", "num_frames": 4, "d_model": 32, "num_layers": 1},
                     {"variant": "verify", "num_frames": 3, "d_model": 32, "num_layers": 1},
                     {"variant": "permutation", "num_frames": 4, "d_model": 32, "num_layers": 1}],
    "rank_n_contrast": [{"sim": s} for s in ("l1", "l2", "cosine")] + [{"project": True, "proj_dim": 16}],
    "tcc": [{"variant": "regression"}, {"variant": "classification"},
            {"view": "pair", "num_frames": 6}],
    "scl": [{"view": "temporal"}, {"view": "pair"}],
    "vip": [{"gamma": 0.9}, {"num_samples": 3}],
}
FAILED = []


def check(name, cond, detail=""):
    status = "ok  " if cond else "FAIL"
    print(f"  [{status}] {name}{(' -- ' + detail) if detail and not cond else ''}")
    if not cond:
        FAILED.append(name)


def make_batch(requires_pair):
    torch.manual_seed(0)
    x = torch.randn(B, T, D)
    progress = torch.arange(T, dtype=torch.float32).expand(B, T) / (T - 1)
    frame_index = torch.arange(T).expand(B, T)
    out = {"features": x, "progress": progress.contiguous(), "frame_index": frame_index.contiguous()}
    if requires_pair:
        out.update({"features_pair": torch.randn(B, T, D), "progress_pair": progress.contiguous(),
                    "frame_index_pair": frame_index.contiguous()})
    return out


def run_one(name, kwargs):
    label = f"{name}{kwargs if kwargs else ''}"
    torch.manual_seed(0)
    encoder = ClsMlpEncoder(feature_dim=D, latent_dim=D, hidden_dim=D)
    obj = OBJECTIVE_REGISTRY[name](latent_dim=D, **kwargs)
    batch = make_batch(getattr(obj, "requires_pair", False))
    z = encoder(batch["features"])
    z_pair = encoder(batch["features_pair"]) if "features_pair" in batch else None
    loss, metrics = obj(z, batch["progress"], frame_index=batch["frame_index"], z_pair=z_pair,
                        progress_pair=batch.get("progress_pair"),
                        frame_index_pair=batch.get("frame_index_pair"))
    check(f"{label}: scalar finite loss",
          loss.dim() == 0 and torch.isfinite(loss), f"got {loss}")
    check(f"{label}: metrics are detached scalars",
          all(v.dim() == 0 and not v.requires_grad for v in metrics.values()))
    # Even a zero loss must carry a grad_fn, or train.py's loss.backward() raises for any
    # run whose total loss is only this objective (`objective: none`, empty aux_weights).
    check(f"{label}: loss is connected to the graph", loss.requires_grad)
    if name == "none":
        loss.backward()
        return
    loss.backward()
    grads = [p.grad for p in encoder.parameters() if p.grad is not None]
    total = sum(float(g.abs().sum()) for g in grads)
    check(f"{label}: gradient reaches the ENCODER", total > 0, f"summed |grad| = {total}")


def test_registry_signatures():
    print("registry contract")
    for name, cls in OBJECTIVE_REGISTRY.items():
        params = inspect.signature(cls.__init__).parameters
        check(f"{name}: takes latent_dim", "latent_dim" in params)
        check(f"{name}: forward signature matches Objective",
              list(inspect.signature(cls.forward).parameters)
              == ["self", "z", "progress", "frame_index", "z_pair", "progress_pair",
                  "frame_index_pair"])


def test_all_objectives():
    print("every objective: forward + backward")
    for name in OBJECTIVE_REGISTRY:
        for kwargs in VARIANTS.get(name, [FIXTURE_KWARGS.get(name, {})]):
            run_one(name, kwargs)


def test_rank_n_contrast_orders():
    """RnC's whole claim is that its optimum orders features by the label. A latent space
    that is already perfectly ordered (a straight line in time) must therefore score a
    LOWER RnC loss than a shuffled version of the same points -- if it doesn't, the tie
    handling or the suffix-logsumexp denominator is wrong, and no shape test would catch
    that."""
    print("rank_n_contrast: ordered space scores better than shuffled")
    obj = OBJECTIVE_REGISTRY["rank_n_contrast"](latent_dim=D, normalize=False)
    t = torch.linspace(0, 1, T)
    line = t.unsqueeze(-1) * torch.ones(D)
    z_ordered = line.expand(B, T, D).contiguous()
    progress = t.expand(B, T).contiguous()
    perm = torch.randperm(T)
    z_shuffled = z_ordered[:, perm].contiguous()
    ordered, _ = obj(z_ordered, progress)
    shuffled, _ = obj(z_shuffled, progress)
    check("ordered < shuffled", float(ordered) < float(shuffled),
          f"ordered={float(ordered):.4f} shuffled={float(shuffled):.4f}")


def test_rank_n_contrast_matches_the_paper():
    """The vectorized loss against a LITERAL transcription of the paper's formula.

    rank_n_contrast.py does not compute S_ij = {k != i : d(y_i,y_k) >= d(y_i,y_j)} as
    stated -- an (N,N,N) mask is impossible at N=2048 -- it sorts each anchor's
    candidates by label distance, takes a reverse cumulative logsumexp, and indexes the
    start of each tie block. That is a real reformulation, and this is the only test that
    can tell whether it is still the same number. Run with heavy ties as well as distinct
    labels, because the tie-block indexing is the part most likely to rot: progress labels
    take only T distinct values, so at a realistic batch size ~1/16 of every denominator
    is a tie block.
    """
    print("rank_n_contrast: vectorized loss == the paper's formula")
    obj_cls = OBJECTIVE_REGISTRY["rank_n_contrast"]

    def brute_force(v, y, tau, tol):
        n = v.shape[0]
        sim = -torch.cdist(v, v, p=2.0) / tau
        terms = []
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                d_ij = abs(float(y[i]) - float(y[j]))
                sset = [k for k in range(n)
                        if k != i and abs(float(y[i]) - float(y[k])) >= d_ij - tol]
                terms.append(torch.logsumexp(sim[i, sset], 0) - sim[i, j])
        return torch.stack(terms).mean()

    n = 40
    cases = [("distinct labels", torch.rand(n)),
             ("progress labels (heavy ties)", (torch.arange(n) % 16).float() / 15)]
    for label_kind, y in cases:
        torch.manual_seed(0)
        v = torch.randn(n, 8)
        for tau in (0.5, 2.0):
            obj = obj_cls(latent_dim=8, temperature=tau, sim="l2", normalize=False)
            mine, _ = obj(v.unsqueeze(0), y.unsqueeze(0))
            ref = brute_force(v, y, tau, obj.tie_tol)
            check(f"{label_kind}, tau={tau}", torch.allclose(mine, ref, atol=1e-5),
                  f"mine={float(mine):.8f} reference={float(ref):.8f}")


def test_rank_n_contrast_actually_orders():
    """RnC's entire claim is that its optimum orders features by the continuous label.
    Optimize FREE embeddings (no encoder, nothing else in the way) against it and check
    the space really does end up ordered. This is what distinguishes "the loss runs" from
    "the loss does its job" -- and it is scale-sensitive, so it also fails if the
    normalize/temperature defaults drift back into a flat-softmax regime."""
    print("rank_n_contrast: minimizing it orders the space")
    torch.manual_seed(0)
    n, d = 128, 16
    y = torch.rand(n)
    v = torch.nn.Parameter(torch.randn(n, d) * 0.5)
    obj = OBJECTIVE_REGISTRY["rank_n_contrast"](latent_dim=d, temperature=2.0, sim="l2",
                                                normalize=False)
    opt = torch.optim.Adam([v], lr=0.05)

    def order_quality():
        vc = (v - v.mean(0)).detach()
        pc1 = torch.linalg.svd(vc, full_matrices=False)[2][0]
        return abs(spearmanr((vc @ pc1).numpy(), y.numpy()).statistic)

    before = order_quality()
    for _ in range(150):
        loss, _ = obj(v.unsqueeze(0), y.unsqueeze(0))
        opt.zero_grad()
        loss.backward()
        opt.step()
    after = order_quality()
    check("random init is unordered", before < 0.3, f"|spearman| = {before:.4f}")
    check("optimized space is ordered by the label", after > 0.95,
          f"|spearman| went {before:.4f} -> {after:.4f}")


def test_progress_labels():
    print("clip_data: progress labels and crops")

    class FakeProvider:
        def get(self, path):
            return torch.arange(T * D, dtype=torch.float32).reshape(T, D)

    ds = ObjectiveClipDataset(["a"], FakeProvider())
    item = ds[0]
    check("uncropped progress spans [0, 1]",
          float(item["progress"][0]) == 0.0 and float(item["progress"][-1]) == 1.0)
    ds_g = ObjectiveClipDataset(["a"], FakeProvider(), crop_size=8, crop_label="global")
    starts = set()
    for _ in range(40):
        it = ds_g[0]
        start = int(it["frame_index"][0])
        starts.add(start)
        if abs(float(it["progress"][0]) - start / (T - 1)) > 1e-6:
            break
    check("global crop keeps the video's clock",
          abs(float(it["progress"][0]) - start / (T - 1)) < 1e-6,
          f"start={start} progress0={float(it['progress'][0])}")
    check("global crop length", it["features"].shape[0] == 8)
    # Guards the worker-RNG bug: a dataset-owned generator cloned into every DataLoader
    # worker produced the SAME offset every time, which is what made `crop_size` not
    # actually random across epochs.
    check("crops actually vary between draws", len(starts) > 1, f"offsets seen: {sorted(starts)}")
    ds_l = ObjectiveClipDataset(["a"], FakeProvider(), crop_size=8, crop_label="local")
    it = ds_l[0]
    check("local crop renormalizes to [0, 1]",
          float(it["progress"][0]) == 0.0 and float(it["progress"][-1]) == 1.0)


def test_reference_method_invariants():
    print("paper/reference method invariants")

    # Official PROGRESSOR uses a uniform half-open randint for the valid goal range.
    torch.manual_seed(0)
    obj = Progressor(latent_dim=4, min_span=2)
    s, _, g = obj._sample_indices(200_000, 8, torch.device("cpu"))
    counts = torch.bincount(g[s == 0] - 2, minlength=6).float()
    check("PROGRESSOR goals are uniform conditional on start",
          float(counts.max() / counts.min()) < 1.08, f"counts={counts.tolist()}")

    # Official coral-pytorch initializes the shared-task biases in descending order.
    coral = CoralHead(latent_dim=4, num_bins=8)
    check("CORAL starts rank-monotonic",
          bool(torch.all(coral.bias[:-1] >= coral.bias[1:])), str(coral.bias.tolist()))

    try:
        TCC(latent_dim=4, view="temporal")
    except ValueError:
        temporal_rejected = True
    else:
        temporal_rejected = False
    check("TCC requires the paper's paired-video protocol", temporal_rejected)

    scores = torch.arange(4 * 5).reshape(4, 5).float()
    progress = torch.arange(5).expand(4, 5).float()
    _, _, other_clip = PairwiseRank._cross_clip_targets(scores, progress)
    check("cross-clip rank targets have different clip IDs",
          bool(torch.all(other_clip != torch.arange(4))))

    # Shuffle and Learn samples a<b<c<d<e, holds b/d fixed, and changes only the
    # middle frame: c for positives and a/e for negatives. Tuples may be reversed.
    torch.manual_seed(0)
    shuffle = ShuffleSort(latent_dim=4, variant="verify", num_frames=3,
                          positive_fraction=0.25, d_model=8, nhead=2, num_layers=1)
    order, labels = shuffle._verification_indices(400, 16, torch.device("cpu"))
    ascending = order[:, 0] < order[:, 2]
    canonical = torch.where(ascending.unsqueeze(1), order, order.flip(1))
    positive = labels.bool()
    check("Shuffle verification uses the paper's 25% positive ratio",
          int(labels.sum()) == 100)
    check("Shuffle positives are monotone triplets",
          bool(torch.all((canonical[positive, 0] < canonical[positive, 1])
                         & (canonical[positive, 1] < canonical[positive, 2]))))
    check("Shuffle negatives put the middle frame outside fixed endpoints",
          bool(torch.all((canonical[~positive, 1] < canonical[~positive, 0])
                         | (canonical[~positive, 1] > canonical[~positive, 2]))))


def test_checkpoint_is_evaluable():
    """Ordinal checkpoints load through the evaluator's real shared model loader."""
    print("checkpoint: loadable by encoder_planning_eval/progress_metrics")
    eval_root = ROOT.parent / "encoder_planning_eval"
    sys.path.insert(0, str(eval_root))
    from eval_common.loading import load_world_model

    encoder = ClsMlpEncoder(feature_dim=D, latent_dim=D, hidden_dim=D, norm_type="clip_rmsnorm")
    obj = OBJECTIVE_REGISTRY["shuffle_sort"](latent_dim=D, num_frames=4, d_model=32, num_layers=1)
    with tempfile.TemporaryDirectory() as tmp:
        ckpt_path = Path(tmp) / "best.ckpt"
        train_mod.save_checkpoint(ckpt_path, encoder, obj, 3, "shuffle_sort")
        cfg = {"feature_dim": D, "latent_dim": D, "hidden_dim": D, "use_projector": True,
               "norm_type": "clip_rmsnorm", **train_mod.EVAL_ARCH_DEFAULTS,
               "irrelevant_key": "ignored by the evaluator"}
        (Path(tmp) / "config.yaml").write_text(yaml.safe_dump(cfg))

        model, _, epoch = load_world_model(
            str(ckpt_path), str(Path(tmp) / "config.yaml"), "cpu", with_predictor=False,
        )
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        check("head kept separately for our own use",
              "head" in ckpt and ckpt["epoch"] == 3 and epoch == 3)
        # And the reloaded encoder must actually reproduce the trained one's outputs.
        x = torch.randn(2, T, D)
        check("reloaded encoder matches bit-for-bit",
              torch.allclose(model.encoder(x), encoder(x)))


if __name__ == "__main__":
    test_registry_signatures()
    test_all_objectives()
    test_rank_n_contrast_orders()
    test_rank_n_contrast_matches_the_paper()
    test_rank_n_contrast_actually_orders()
    test_progress_labels()
    test_reference_method_invariants()
    test_checkpoint_is_evaluable()
    print()
    if FAILED:
        print(f"{len(FAILED)} FAILED: {FAILED}")
        sys.exit(1)
    print("all checks passed")
