"""Regression tests for the planning-eval pipeline: interpolation math, predictor
shapes/registry, metric sanity, dataset indexing, and one end-to-end check that the
whole thing actually measures what it claims to (a genuinely straight synthetic
trajectory should score ~perfectly under interpolation alone).

Run:  python tests/test_planning.py     (plain asserts, no pytest dependency)
"""
import sys
import tempfile
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dataset import PlanningDataset  # noqa: E402
from interpolate import interpolate_sequence, linear_interpolate  # noqa: E402
from metrics import cosine_sim, r2_score, relative_l2, summarize  # noqa: E402
from predictors.predictor import PREDICTOR_REGISTRY, build_predictor  # noqa: E402


def test_linear_interpolate_endpoints_and_midpoint():
    z1 = torch.randn(8, 16)
    zT = torch.randn(8, 16)
    assert torch.allclose(linear_interpolate(z1, zT, torch.zeros(8)), z1, atol=1e-6)
    assert torch.allclose(linear_interpolate(z1, zT, torch.ones(8)), zT, atol=1e-6)
    mid = linear_interpolate(z1, zT, torch.full((8,), 0.5))
    assert torch.allclose(mid, (z1 + zT) / 2, atol=1e-6)


def test_interpolate_sequence_endpoints_and_shape():
    z1 = torch.randn(4, 12)
    zT = torch.randn(4, 12)
    t_len = 16
    seq = interpolate_sequence(z1, zT, t_len=t_len)
    assert seq.shape == (4, t_len, 12)
    assert torch.allclose(seq[:, 0], z1, atol=1e-6)
    assert torch.allclose(seq[:, -1], zT, atol=1e-6)
    alpha_8 = 8 / (t_len - 1)
    assert torch.allclose(seq[:, 8], z1 + alpha_8 * (zT - z1), atol=1e-5)


def test_metrics_identical_vectors_are_perfect():
    z = torch.randn(32, 16)
    assert torch.allclose(cosine_sim(z, z), torch.ones(32), atol=1e-5)
    assert torch.allclose(relative_l2(z, z), torch.zeros(32), atol=1e-5)
    assert abs(r2_score(z, z) - 1.0) < 1e-5


def test_metrics_opposite_vectors_are_worst_case_cosine():
    z = torch.randn(32, 16)
    cos = cosine_sim(-z, z)
    assert torch.allclose(cos, -torch.ones(32), atol=1e-5), "antipodal vectors must score cosine_sim == -1"


def test_r2_score_matches_predicting_the_mean():
    torch.manual_seed(0)
    z_true = torch.randn(100, 8)
    z_hat = z_true.mean(dim=0, keepdim=True).expand_as(z_true)
    assert abs(r2_score(z_hat, z_true)) < 1e-5, "predicting the batch mean everywhere should give r2 ~ 0"


def test_summarize_keys():
    z1, z2 = torch.randn(10, 4), torch.randn(10, 4)
    out = summarize(z1, z2)
    assert set(out) == {"cosine_sim_median", "cosine_sim_mean", "relative_l2_median",
                         "relative_l2_mean", "r2"}


def test_registry_has_mlp_and_transformer():
    assert {"mlp", "transformer"} <= set(PREDICTOR_REGISTRY)


def test_every_registered_predictor_shape_and_interface():
    """Every registry entry must implement the common forward(z1, zT, t_len) -> (B,T,D)
    interface -- registry-driven, so a variant added later is automatically covered."""
    dim, t_len, b = 24, 10, 5
    for name in PREDICTOR_REGISTRY:
        model = build_predictor(name, dim=dim, max_len=t_len).eval()
        z1 = torch.randn(b, dim)
        zT = torch.randn(b, dim)
        out = model(z1, zT, t_len)
        assert out.shape == (b, t_len, dim), f"{name}: {out.shape}"


def test_every_registered_predictor_fixed_architecture_scales_only_with_dim():
    """The point of the "fixed architecture" claim: hyperparameters identical across
    encoders, only d_model-derived widths track the encoder's own feature_dim."""
    for name in PREDICTOR_REGISTRY:
        small = build_predictor(name, dim=8, max_len=6)
        big = build_predictor(name, dim=384, max_len=6)
        small_params = sum(p.numel() for p in small.parameters())
        big_params = sum(p.numel() for p in big.parameters())
        assert big_params > small_params, f"{name}: bigger feature_dim should mean more params"


def test_mlp_endpoints_are_verbatim():
    """PlanningMLP's positions 0/T-1 are filled with z1/zT exactly (unused downstream,
    but should still be an honest passthrough, not e.g. leftover garbage)."""
    model = build_predictor("mlp", dim=8, max_len=6).eval()
    z1, zT = torch.randn(3, 8), torch.randn(3, 8)
    out = model(z1, zT, 6)
    assert torch.allclose(out[:, 0], z1) and torch.allclose(out[:, -1], zT)


def test_transformer_uses_both_start_and_goal():
    """Bidirectional attention: changing z_T must change the filled-in intermediate
    positions too (not just be ignored in favor of z1 alone) -- this is the whole reason
    to use a bidirectional transformer instead of, say, a causal one."""
    torch.manual_seed(0)
    dim, t_len = 16, 8
    model = build_predictor("transformer", dim=dim, max_len=t_len, nhead=4, num_layers=2).eval()
    z1 = torch.randn(1, dim)
    zT_a = torch.randn(1, dim)
    zT_b = zT_a + 5.0 * torch.randn(1, dim)
    with torch.no_grad():
        out_a = model(z1, zT_a, t_len)
        out_b = model(z1, zT_b, t_len)
    assert not torch.allclose(out_a[:, 1:-1], out_b[:, 1:-1], atol=1e-4), (
        "changing z_T should change the filled-in intermediate positions")


def test_mlp_intermediate_positions_are_independent():
    """Contrast with the transformer test above: PlanningMLP has no attention between
    intermediate positions, so perturbing z1/zT the same way should move every
    intermediate position by the SAME shared mechanism (position-independent weights) --
    verified indirectly by checking each position is a pure function of (z1, zT, its own
    alpha) alone, i.e. changing t_len (which changes OTHER positions' alphas) must not
    change an unrelated position's prediction as long as that position's own alpha is
    unchanged. Here: position 1 of a T=6 sequence (alpha=1/5) is recomputed as position 2
    of a T=11 sequence with the same alpha=1/5, and must match."""
    torch.manual_seed(0)
    dim = 12
    model = build_predictor("mlp", dim=dim, max_len=16).eval()
    z1, zT = torch.randn(1, dim), torch.randn(1, dim)
    with torch.no_grad():
        out_a = model(z1, zT, 6)   # position 1 has alpha = 1/5
        out_b = model(z1, zT, 11)  # position 2 has alpha = 2/10 = 1/5
    assert torch.allclose(out_a[:, 1], out_b[:, 2], atol=1e-5)


def _write_synthetic_features(tmpdir, z, video_paths):
    """z: (N, T, D) tensor. Writes the LiFT precompute_features.py-format .pt + a matching
    manifest CSV, returns (features_path, manifest_path)."""
    features_path = Path(tmpdir) / "features.pt"
    manifest_path = Path(tmpdir) / "manifest.csv"
    torch.save({"features": z, "video_paths": video_paths}, features_path)
    pd.DataFrame({"video_path": video_paths}).to_csv(manifest_path, index=False)
    return str(features_path), str(manifest_path)


def test_dataset_shapes():
    torch.manual_seed(0)
    n, t, d = 5, 6, 4
    z = torch.randn(n, t, d)
    paths = [f"clip_{i}.mp4" for i in range(n)]
    with tempfile.TemporaryDirectory() as tmp:
        feat_path, manifest_path = _write_synthetic_features(tmp, z, paths)
        ds = PlanningDataset(feat_path, manifest_path)
        assert ds.n_clips == n and ds.T == t and ds.D == d
        assert len(ds) == n, "one example per CLIP (whole sequence), not per intermediate frame"
        ex = ds[0]
        assert ex["z"].shape == (t, d)
        assert torch.allclose(ex["z"], z[0])


def test_dataset_rejects_too_short_clips():
    z = torch.randn(3, 2, 4)  # T=2, no intermediate frame possible
    paths = [f"clip_{i}.mp4" for i in range(3)]
    with tempfile.TemporaryDirectory() as tmp:
        feat_path, manifest_path = _write_synthetic_features(tmp, z, paths)
        try:
            PlanningDataset(feat_path, manifest_path)
            assert False, "expected ValueError for T < 3"
        except ValueError:
            pass


def test_dataset_rejects_patch_shaped_features():
    z = torch.randn(3, 6, 5, 4)  # (N, T, P, D) -- patch-shaped, unsupported without a checkpoint
    paths = [f"clip_{i}.mp4" for i in range(3)]
    with tempfile.TemporaryDirectory() as tmp:
        feat_path, manifest_path = _write_synthetic_features(tmp, z, paths)
        try:
            PlanningDataset(feat_path, manifest_path)
            assert False, "expected ValueError for a 4D (patch-shaped) features file with no --checkpoint"
        except ValueError:
            pass


def test_dataset_checkpoint_mode_runs_source_features_through_encoder():
    """--checkpoint/--config path (dataset.py): a trained ProceduralWorldModel's own
    encoder+agg_head, not the raw source features, should end up in self.features -- and
    it must accept patch-shaped (N,T,P,D) source features (the whole point of this path:
    cls-only PlanningDataset can't otherwise consume a token_mode=patch encoder)."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "predictive_world_model"))
    from models.world_model import ProceduralWorldModel  # noqa: E402

    torch.manual_seed(0)
    n, t, p, feat_dim = 3, 6, 4, 8  # N clips, T frames, P patches, source feature_dim
    arch = dict(token_mode="patch", projector_type="channel", feature_dim=feat_dim,
                channel_out_dim=5, channel_hidden=6, num_patches=p, agg_mlp_hidden_dim=6, agg_out_dim=7)
    model = ProceduralWorldModel(**arch)

    z_patch = torch.randn(n, t, p, feat_dim)
    paths = [f"clip_{i}.mp4" for i in range(n)]
    with tempfile.TemporaryDirectory() as tmp:
        feat_path, manifest_path = _write_synthetic_features(tmp, z_patch, paths)
        ckpt_path = Path(tmp) / "model.ckpt"
        config_path = Path(tmp) / "config.yaml"
        torch.save({"model": model.state_dict(), "epoch": 0}, ckpt_path)
        import yaml
        with open(config_path, "w") as f:
            yaml.safe_dump(arch, f)

        ds = PlanningDataset(feat_path, manifest_path, checkpoint=str(ckpt_path), config=str(config_path))
        assert ds.n_clips == n and ds.T == t and ds.D == arch["agg_out_dim"]

        with torch.no_grad():
            expected = model.aggregate(model.encoder(z_patch))
        assert torch.allclose(ds.features, expected, atol=1e-5)


def test_dataset_checkpoint_loader_preserves_residual_projector():
    """A behavioral constructor flag must survive loading even though it adds no state
    dict keys and therefore cannot be detected by missing/unexpected-key checks."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "predictive_world_model"))
    from models.world_model import ProceduralWorldModel  # noqa: E402

    torch.manual_seed(0)
    n, t, d = 2, 5, 8
    arch = dict(feature_dim=d, latent_dim=d, hidden_dim=d, norm_type="none",
                residual_projector=True, max_len=t)
    model = ProceduralWorldModel(**arch).eval()
    source = torch.randn(n, t, d)
    paths = [f"residual_{i}.mp4" for i in range(n)]
    with tempfile.TemporaryDirectory() as tmp:
        feat_path, manifest_path = _write_synthetic_features(tmp, source, paths)
        ckpt_path = Path(tmp) / "model.ckpt"
        config_path = Path(tmp) / "config.yaml"
        torch.save({"model": model.state_dict(), "epoch": 0}, ckpt_path)
        import yaml
        with open(config_path, "w") as f:
            yaml.safe_dump(arch, f)

        ds = PlanningDataset(feat_path, manifest_path, checkpoint=str(ckpt_path),
                             config=str(config_path))
        with torch.no_grad():
            expected = model.encoder(source)
            plain = model.encoder.projector(source)
        assert torch.allclose(ds.features, expected, atol=1e-5)
        assert not torch.allclose(ds.features, plain, atol=1e-5)


def test_straight_synthetic_trajectory_interpolation_is_near_perfect():
    """End-to-end sanity check: an encoder whose latent space is EXACTLY straight between
    start and goal should let linear interpolation reconstruct every intermediate frame
    almost exactly -- this is the whole premise the eval is built to detect."""
    torch.manual_seed(0)
    t_len, d = 16, 32
    z1 = torch.randn(d)
    direction = torch.randn(d)
    alphas = torch.linspace(0, 1, t_len)
    z_straight = torch.stack([z1 + a * direction for a in alphas]).unsqueeze(0)  # (1,T,D)

    z_hat = interpolate_sequence(z_straight[:, 0], z_straight[:, -1], t_len)
    out = summarize(z_hat[:, 1:-1].reshape(-1, d), z_straight[:, 1:-1].reshape(-1, d))
    assert out["cosine_sim_median"] > 0.9999, out
    assert out["relative_l2_median"] < 1e-3, out
    assert out["r2"] > 0.9999, out


def test_curved_synthetic_trajectory_interpolation_is_imperfect():
    """Contrast case: a trajectory that visibly bows away from the start-goal chord
    should NOT score near-perfect under interpolation -- otherwise the metric would be
    vacuous (always ~1 regardless of curvature)."""
    torch.manual_seed(0)
    t_len, d = 16, 32
    alphas = torch.linspace(0, 1, t_len)
    z1 = torch.randn(d)
    direction = torch.randn(d)
    perp = torch.randn(d)
    # quadratic bulge orthogonal to the chord -- zero at both endpoints, peaks at the middle
    bulge = 4 * alphas * (1 - alphas)
    z_curved = torch.stack([z1 + a * direction + b * perp for a, b in zip(alphas, bulge)]).unsqueeze(0)

    z_hat = interpolate_sequence(z_curved[:, 0], z_curved[:, -1], t_len)
    out = summarize(z_hat[:, 1:-1].reshape(-1, d), z_curved[:, 1:-1].reshape(-1, d))
    assert out["r2"] < 0.95, f"expected the chord-perpendicular bulge to hurt R^2, got {out}"


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in tests:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
        except Exception as e:
            failed += 1
            print(f"FAIL  {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
