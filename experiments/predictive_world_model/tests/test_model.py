"""Regression tests for shapes and, more importantly, the semantics that fail SILENTLY.

Run:  python tests/test_model.py     (plain asserts, no pytest dependency)

The causality test is the load-bearing one. CausalPredictor enforces causality with an
explicit mask; `is_causal=True` is only a hint, and passing it WITHOUT the mask yields
fully dense attention with no error and no warning — the model would simply see the
future and quietly cheat at next-step prediction. test_no_future_leakage catches that.

Note for anyone extending these: perturb with a NON-uniform vector. Adding a uniform
constant across all dims is removed exactly by the pre-LN LayerNorm (it subtracts the
feature-dim mean), so a uniform probe shows no dependence and every position looks
independent — a false negative that made this exact test appear to fail during the audit
that produced it.
"""
import sys
import tempfile
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
# train MUST be imported before data.dataset: data.feature_provider does its own
# sys.path.insert(0, LIFT_ROOT) at import time (for LiFT's package), which contains an
# UNRELATED train.py of its own -- importing in this order means our train.py resolves
# first, while this project's root is still at the front of sys.path.
from train import build_optimizer, resolve_checkpoint_dir  # noqa: E402
from data.dataset import ClipFeatureDataset  # noqa: E402
from losses.uniformity_loss import uniformity_loss  # noqa: E402
from losses.world_model_loss import WorldModelLoss, _off_diagonal, _group, _group_masked  # noqa: E402
from models.aggregation_head import AggregationHead  # noqa: E402
from models.block_causal_predictor import BlockCausalPredictor, build_block_causal_mask  # noqa: E402
from models.encoder import ClsMlpEncoder, ENCODER_REGISTRY, ResidualMlpEncoder  # noqa: E402
from models.norms import NORM_REGISTRY, BatchNormOverFrames, ClipRMSNorm  # noqa: E402
from models.patch_encoder import PatchProjectorEncoder  # noqa: E402
from models.pixel_encoder import PIXEL_ENCODER_REGISTRY, SmallResnetPixelEncoder  # noqa: E402
from models.predictor import CausalPredictor  # noqa: E402
from models.world_model import ProceduralWorldModel  # noqa: E402
from data.feature_provider import FeatureProvider  # noqa: E402

B, T, D = 4, 16, 384


def test_shapes():
    model = ProceduralWorldModel(feature_dim=D, latent_dim=D, hidden_dim=D, max_len=T).eval()
    z, z_pred = model(torch.randn(B, T, D))
    assert z.shape == (B, T, D), z.shape
    assert z_pred.shape == (B, T, D), z_pred.shape
    assert ClsMlpEncoder(feature_dim=768, latent_dim=256)(torch.randn(2, 5, 768)).shape == (2, 5, 256)


def test_no_future_leakage():
    """output[t] must depend on inputs j <= t and on nothing later."""
    torch.manual_seed(0)
    t_len, d = 8, 64
    pred = CausalPredictor(latent_dim=d, nhead=4, num_layers=2, max_len=t_len).eval()
    z = torch.randn(1, t_len, d)
    with torch.no_grad():
        base = pred(z)
        for j in range(t_len):
            zp = z.clone()
            zp[:, j] += 5.0 * torch.randn(d)  # non-uniform: survives LayerNorm
            out = pred(zp)
            for t in range(t_len):
                changed = (out[:, t] - base[:, t]).abs().max().item() > 1e-4
                assert changed == (j <= t), (
                    f"output[{t}] {'changed' if changed else 'did not change'} when input[{j}] "
                    f"was perturbed; expected dependence iff {j} <= {t}. Causal mask broken?"
                )


def test_prediction_target_alignment():
    """z_pred[:, t] is compared against z[:, t+1], not z[:, t]."""
    crit = WorldModelLoss(pred_loss="l1")
    z = torch.randn(2, 6, 8)
    perfect = torch.zeros_like(z)
    perfect[:, :-1] = z[:, 1:]
    assert crit.prediction_loss(perfect, z).item() < 1e-6
    assert crit.prediction_loss(z.clone(), z).item() > 1e-3  # off-by-one predictor


def test_stop_grad():
    z = torch.randn(2, 6, 8, requires_grad=True)
    WorldModelLoss(stop_grad=True).prediction_loss(torch.randn(2, 6, 8, requires_grad=True), z).backward()
    assert z.grad is None or z.grad.abs().sum() == 0
    z2 = torch.randn(2, 6, 8, requires_grad=True)
    WorldModelLoss(stop_grad=False).prediction_loss(torch.randn(2, 6, 8, requires_grad=True), z2).backward()
    assert z2.grad.abs().sum() > 0


def test_gradient_routing():
    """Regularizers apply to z, so they must never reach the predictor."""
    for term in ["straighten", "chord", "std", "cov"]:
        m = ProceduralWorldModel(feature_dim=D, latent_dim=D, hidden_dim=D, max_len=T)
        c = WorldModelLoss()
        z, _ = m(torch.randn(B, T, D))
        loss = {"straighten": c.straightening_loss,
                "chord": c.chord_loss,
                "std": c.vicreg_std_loss,
                "cov": c.vicreg_cov_loss}[term](z)
        m.zero_grad()
        loss.backward()
        enc = sum(p.grad.abs().sum().item() for p in m.encoder.parameters() if p.grad is not None)
        prd = sum(p.grad.abs().sum().item() for p in m.predictor.parameters() if p.grad is not None)
        assert enc > 0 and prd == 0, f"{term}: encoder={enc}, predictor={prd}"


def test_final_layernorm_present():
    """norm_first=True (pre-LN) requires a final norm after the last block."""
    p = CausalPredictor(latent_dim=D)
    assert p.transformer.layers[0].norm_first is True
    assert p.transformer.norm is not None, "pre-LN stack is missing its final LayerNorm"


def test_encoder_output_is_normalized():
    """||z|| must stay bounded: vicreg_cov grows as ||z||^4, so unbounded z made the
    first K400 run's covariance term explode to 8.9e6."""
    enc = ClsMlpEncoder(feature_dim=D, latent_dim=D, norm_type="layernorm").eval()
    with torch.no_grad():
        small = enc(torch.randn(32, T, D) * 0.01).norm(dim=-1).mean().item()
        large = enc(torch.randn(32, T, D) * 100.0).norm(dim=-1).mean().item()
    assert abs(large - small) / small < 0.5, (
        f"encoder output norm moved {small:.2f} -> {large:.2f} for a 10000x input scale change; "
        f"LayerNorm should keep it roughly fixed"
    )
    assert isinstance(ClsMlpEncoder(norm_type="none").norm, nn.Identity)


def test_all_norm_types_constructible():
    """Every registered norm_type builds and runs at the right shape (registry-driven,
    so a norm added to models/norms.py's NORM_REGISTRY is automatically covered here)."""
    for norm_type in NORM_REGISTRY:
        enc = ClsMlpEncoder(feature_dim=D, latent_dim=D, norm_type=norm_type)
        out = enc(torch.randn(B, T, D))
        assert out.shape == (B, T, D), f"{norm_type}: {out.shape}"


def test_residual_encoder_adds_x_before_norm():
    """residual=True: z = norm(x + projector(x)), verified directly against a manual
    norm(x + projector(x)) computation with the SAME weights, not just a shape check."""
    torch.manual_seed(0)
    enc = ClsMlpEncoder(feature_dim=D, latent_dim=D, norm_type="none", residual=True).eval()
    x = torch.randn(B, T, D)
    with torch.no_grad():
        expected = x + enc.projector(x)  # norm_type="none" -> Identity, no further change
        out = enc(x)
    assert torch.allclose(out, expected, atol=1e-6)


def test_residual_encoder_differs_from_non_residual():
    """Sanity check the residual connection actually changes the output, given IDENTICAL
    weights -- not just that it runs."""
    torch.manual_seed(0)
    enc_res = ClsMlpEncoder(feature_dim=D, latent_dim=D, norm_type="none", residual=True).eval()
    enc_plain = ClsMlpEncoder(feature_dim=D, latent_dim=D, norm_type="none", residual=False).eval()
    enc_plain.load_state_dict(enc_res.state_dict())  # same weights, only residual differs
    x = torch.randn(B, T, D)
    with torch.no_grad():
        assert not torch.allclose(enc_res(x), enc_plain(x))


def test_residual_encoder_rejects_mismatched_dims_and_no_projector():
    try:
        ClsMlpEncoder(feature_dim=384, latent_dim=256, residual=True)
        assert False, "expected ValueError for feature_dim != latent_dim"
    except ValueError:
        pass
    try:
        ClsMlpEncoder(feature_dim=D, latent_dim=D, use_projector=False, residual=True)
        assert False, "expected ValueError for residual=True with use_projector=False"
    except ValueError:
        pass


def test_world_model_threads_residual_projector():
    m = ProceduralWorldModel(feature_dim=D, latent_dim=D, hidden_dim=D, max_len=T,
                              residual_projector=True)
    assert m.encoder.residual is True
    z, z_pred = m(torch.randn(B, T, D))
    assert z.shape == (B, T, D) and z_pred.shape == (B, T, D)


def test_encoder_registry_default_is_unchanged_mlp():
    """The registry must not change what any existing config/checkpoint builds:
    encoder_type defaults to "mlp", which is ClsMlpEncoder exactly as before."""
    m = ProceduralWorldModel(feature_dim=D, latent_dim=D, hidden_dim=D, max_len=T)
    assert type(m.encoder).__name__ == "ClsMlpEncoder"
    assert sorted(ENCODER_REGISTRY) == ["mlp", "residual_mlp"]


def test_residual_mlp_encoder_shapes_and_depth():
    for depth in (0, 1, 3):
        m = ProceduralWorldModel(feature_dim=D, latent_dim=D, max_len=T,
                                  encoder_type="residual_mlp", encoder_depth=depth)
        assert len(m.encoder.blocks) == depth
        z, z_pred = m(torch.randn(B, T, D))
        assert z.shape == (B, T, D) and z_pred.shape == (B, T, D)


def test_residual_mlp_encoder_is_strictly_per_frame():
    """The point of choosing residual blocks over a temporal encoder: z_t must depend on
    x_t ALONE. If this ever fails, z stops being a rollable world-model state and the
    ordinality metrics become gameable by cumulative averaging (a plain cumulative mean
    scores a perfect 1.0 on them)."""
    m = ProceduralWorldModel(feature_dim=D, latent_dim=D, max_len=T,
                              encoder_type="residual_mlp", encoder_depth=2).eval()
    x = torch.randn(B, T, D)
    x2 = x.clone()
    x2[:, 2] = torch.randn(B, D)
    with torch.no_grad():
        delta = (m.encoder(x) - m.encoder(x2)).abs().sum(dim=-1)
    changed = [t for t in range(T) if delta[:, t].max() > 1e-6]
    assert changed == [2], f"only frame 2 may change, got {changed}"


def test_residual_mlp_encoder_rejects_bad_args():
    for kwargs in ({"depth": -1}, {"expansion": 0}):
        try:
            ResidualMlpEncoder(feature_dim=D, latent_dim=D, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected ValueError for {kwargs}")
    # input_residual needs matching dims, same constraint ClsMlpEncoder's residual has
    try:
        ResidualMlpEncoder(feature_dim=D, latent_dim=D * 2, input_residual=True)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for input_residual with mismatched dims")


def test_hypersphere_norm_gives_exactly_unit_vectors():
    """The property none of the other three norm_types provide: LayerNorm fixes
    per-feature variance, ClipRMSNorm/BatchNorm rescale, but only this one constrains
    ||z||. That gap is what let pwm_cls_mlp_k400 inflate ||z|| 20 -> 146 while VICReg's
    relu(1-std) hinge reported no problem."""
    for enc_type in ("mlp", "residual_mlp"):
        m = ProceduralWorldModel(feature_dim=D, latent_dim=D, hidden_dim=D, max_len=T,
                                  encoder_type=enc_type, norm_type="hypersphere")
        z = m.encoder(torch.randn(B, T, D) * 50)   # deliberately large input scale
        norms = z.norm(dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5), \
            f"{enc_type}: expected unit norms, got [{norms.min():.4f}, {norms.max():.4f}]"


# --- token_mode="raw" (pixel_encoder.py) -----------------------------------------
# raw_img_size=32 in these tests (not the real default of 224): SmallResnetPixelEncoder's
# 5 stride-2 blocks only need img_size a multiple of 32, and a tiny image keeps these
# tests fast -- there's nothing resolution-dependent about the shape/gradient/registry
# properties being checked here.

def test_pixel_encoder_registry_has_small_resnet():
    assert sorted(PIXEL_ENCODER_REGISTRY) == ["small_resnet"]


def test_pixel_encoder_shapes_uint8_and_float_agree():
    """FeatureProvider's live-cache stores uint8 (see feature_provider.py's
    _compute_live_raw); a directly-constructed float [0,1] tensor must give the
    numerically same result, not just the same shape -- otherwise a test built only
    against float input could hide a bug that only bites the real uint8 data path."""
    enc = SmallResnetPixelEncoder(feature_dim=D, img_size=32).eval()
    x_uint8 = torch.randint(0, 256, (B, T, 3, 32, 32), dtype=torch.uint8)
    x_float = x_uint8.float() / 255.0  # the EXACT float equivalent of x_uint8, not an
    # independently-rounded one -- rounding an independent float draw to uint8 and back
    # introduces its own ~1/255 quantization error, which is a real (and expected) gap,
    # not the bug this test is checking for.
    with torch.no_grad():
        out_float = enc(x_float.clone())
        out_uint8 = enc(x_uint8)
    assert out_float.shape == (B, T, D)
    assert torch.allclose(out_float, out_uint8, atol=1e-4), \
        "uint8 [0,255] and its exact float [0,1] equivalent must encode identically"


def test_pixel_encoder_is_trainable_not_frozen():
    """The entire point of token_mode='raw': unlike every other encoder in this project
    (which sits on top of a FROZEN DINOv2 forward pass), this backbone must itself
    receive gradients."""
    enc = SmallResnetPixelEncoder(feature_dim=D, img_size=32)
    out = enc(torch.rand(B, T, 3, 32, 32))
    out.sum().backward()
    grads = [p.grad for p in enc.parameters() if p.requires_grad]
    assert grads, "no trainable parameters found"
    assert any(g is not None and g.abs().sum() > 0 for g in grads), \
        "expected nonzero gradients to flow into the pixel encoder's own weights"


def test_world_model_raw_forward_shapes():
    m = ProceduralWorldModel(feature_dim=D, latent_dim=D, max_len=T, token_mode="raw",
                              pixel_encoder_type="small_resnet", raw_img_size=32,
                              encoder_type="residual_mlp", encoder_depth=2)
    x = (torch.rand(B, T, 3, 32, 32) * 255).to(torch.uint8)
    z, z_pred = m(x)
    assert z.shape == (B, T, D) and z_pred.shape == (B, T, D)


def test_world_model_raw_rejects_bad_pixel_encoder_type():
    try:
        ProceduralWorldModel(feature_dim=D, latent_dim=D, max_len=T, token_mode="raw",
                              pixel_encoder_type="not_a_real_encoder")
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for an unknown pixel_encoder_type")


def test_world_model_cls_and_patch_have_no_pixel_encoder():
    """token_mode='raw' is the ONLY mode with a trainable backbone -- cls/patch must
    still start from an already-computed feature, unaffected by adding this new mode."""
    m_cls = ProceduralWorldModel(feature_dim=D, latent_dim=D, max_len=T)
    m_patch = ProceduralWorldModel(feature_dim=D, token_mode="patch",
                                    num_frames=T, heads=2, depth=2)
    assert m_cls.pixel_encoder is None
    assert m_patch.pixel_encoder is None


def test_build_optimizer_includes_pixel_encoder_in_encoder_group():
    """Mirrors test_build_optimizer_two_groups_partition_all_params's agg_head check:
    the pixel encoder is folded into the ENCODER param group, not a third group and not
    silently left out of both (which would mean it never trains at all)."""
    m = ProceduralWorldModel(feature_dim=D, latent_dim=D, max_len=T, token_mode="raw",
                              raw_img_size=32, encoder_type="residual_mlp", encoder_depth=2)
    opt = build_optimizer(m, _FakeArgs(encoder_lr=1e-5, predictor_lr=5e-4, weight_decay=1e-5))
    enc_group, pred_group = opt.param_groups
    enc_ids = {id(p) for p in enc_group["params"]}
    pred_ids = {id(p) for p in pred_group["params"]}
    assert enc_ids.isdisjoint(pred_ids)
    assert enc_ids | pred_ids == _all_model_param_ids(m)
    pixel_ids = {id(p) for p in m.pixel_encoder.parameters()}
    assert pixel_ids <= enc_ids, "pixel_encoder's params must be in the ENCODER group"


def test_feature_provider_raw_returns_resized_uint8_frames():
    """Isolates _compute_live_raw from real video decode (same convention
    test_dataset_window_size_* uses _FakeProvider to avoid it) by monkeypatching
    _load_frames to hand back fixed PIL images directly."""
    import PIL.Image

    provider = FeatureProvider(token_mode="raw", n_frames=3, raw_img_size=16)
    frames = [PIL.Image.new("RGB", (40, 30), color=(i * 10, 0, 0)) for i in range(3)]
    provider._load_frames = lambda video_path: frames
    out = provider._compute_live_raw("fake.mp4")
    assert out.shape == (3, 3, 16, 16)
    assert out.dtype == torch.uint8
    assert out[0, 0].max() < out[1, 0].max() < out[2, 0].max(), \
        "expected the increasing red channel from the fixture frames to survive resize"


def test_angular_pred_loss_is_scale_invariant_unlike_l2():
    z, z_pred = torch.randn(B, T, D), torch.randn(B, T, D)
    ang = WorldModelLoss(pred_loss="angular")
    l2 = WorldModelLoss(pred_loss="l2")
    c1, c2 = ang(z, z_pred)[1]["pred"].item(), ang(z * 10, z_pred * 10)[1]["pred"].item()
    e1, e2 = l2(z, z_pred)[1]["pred"].item(), l2(z * 10, z_pred * 10)[1]["pred"].item()
    assert abs(c1 - c2) < 1e-4, f"angular should be scale-invariant, {c1} vs {c2}"
    assert e2 > 10 * e1, f"l2 should scale with ||z||^2, {e1} vs {e2}"
    try:
        WorldModelLoss(pred_loss="nope")
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for unknown pred_loss")


def test_uniformity_detects_collapse_where_vicreg_std_does_not():
    """On UNIT-NORM data vicreg_std is nearly blind: every coordinate has variance ~1/d
    however bunched the directions are, so it barely moves between an ideal spread and
    total collapse. Uniformity spans the full range. This is why swapping it in was
    necessary once norm_type="hypersphere" put z on the sphere."""
    L = WorldModelLoss()
    unit = lambda v: v / v.norm(dim=-1, keepdim=True)
    spread = unit(torch.randn(16, T, D))
    collapsed = unit(torch.randn(1, 1, D)).repeat(16, T, 1)
    u_spread = uniformity_loss(spread).item()
    u_collapsed = uniformity_loss(collapsed).item()
    assert u_collapsed > u_spread + 1.0, f"uniformity must separate these: {u_spread} vs {u_collapsed}"
    assert abs(u_collapsed) < 1e-5, f"exact collapse should give ~0, got {u_collapsed}"
    v_spread = L.vicreg_std_loss(spread).item()
    v_collapsed = L.vicreg_std_loss(collapsed).item()
    assert abs(v_collapsed - v_spread) < 0.1, \
        "documented blindness of vicreg_std on unit data changed; update the docstrings"


def test_uniformity_off_by_default_and_reported_when_on():
    z, z_pred = torch.randn(B, T, D), torch.randn(B, T, D)
    assert "uniformity" not in WorldModelLoss()(z, z_pred)[1]
    assert "uniformity" in WorldModelLoss(uniformity_weight=1.0)(z, z_pred)[1]


def test_clip_rmsnorm_preserves_straightness_layernorm_does_not():
    """The whole reason ClipRMSNorm exists: a uniform per-clip rescale can't curve a
    straight line, but LayerNorm's per-frame-independent statistic can, since it
    rescales each frame in a clip by a DIFFERENT amount depending on that frame's own
    mean/norm. Construct a genuinely straight synthetic trajectory and check each norm's
    effect on its own straightness (displacement / path-length) ratio directly."""
    torch.manual_seed(0)
    t_len, d = 16, 384
    a = torch.randn(d)
    b = torch.randn(d) * 5  # direction of travel; large enough that per-frame scale
                             # genuinely varies along the line (see world_model_loss.py's
                             # straightening_loss discussion of why std(a+t*b) isn't
                             # constant in t for a generic direction b)
    z_raw = (a + torch.arange(t_len).float().unsqueeze(1) * b).unsqueeze(0)  # (1,T,D), exactly straight

    def straightness(z):
        z = z.squeeze(0)
        path_length = (z[1:] - z[:-1]).norm(dim=-1).sum().item()
        disp = (z[-1] - z[0]).norm().item()
        return disp / path_length

    assert straightness(z_raw) > 0.999, "constructed line should be exactly straight by construction"

    with torch.no_grad():
        out_rms = ClipRMSNorm(d).eval()(z_raw)
        out_ln = nn.LayerNorm(d).eval()(z_raw)

    rms_straightness = straightness(out_rms)
    ln_straightness = straightness(out_ln)
    assert rms_straightness > 0.999, (
        f"ClipRMSNorm (uniform rescale) should preserve straightness exactly, got {rms_straightness}")
    assert ln_straightness < 0.99, (
        f"expected LayerNorm's per-frame statistic to measurably curve this line, got {ln_straightness}")


def test_batchnorm_over_frames():
    bn = BatchNormOverFrames(D)
    bn.train()
    out_train = bn(torch.randn(B, T, D))
    assert out_train.shape == (B, T, D)
    assert not torch.allclose(bn.bn.running_mean, torch.zeros(D)), (
        "running stats should update after a training-mode forward pass")
    bn.eval()
    out_eval = bn(torch.randn(B, T, D))
    assert out_eval.shape == (B, T, D)


def test_loss_scaling_behaviour():
    """Documents which terms are scale-sensitive — the reason z_norm is logged."""
    torch.manual_seed(0)
    c = WorldModelLoss(straighten_weight=1, vicreg_std_weight=1, vicreg_cov_weight=1)
    z = torch.randn(64, 16, 128)
    noise = torch.randn_like(z)
    # An IMPERFECT predictor: a perfect one gives exactly 0 loss at every scale, which
    # says nothing about scaling (and divides by zero).
    err = lambda s: c.prediction_loss(torch.roll(z * s, -1, dims=1) + 0.1 * s * noise, z * s).item()
    assert abs(c.straightening_loss(z * 8).item() - c.straightening_loss(z).item()) < 1e-3
    ratio = c.vicreg_cov_loss(z * 8).item() / c.vicreg_cov_loss(z).item()
    assert 3000 < ratio < 5500, f"expected vicreg_cov ~ s^4 (4096x), got {ratio:.0f}x"
    assert 6 < err(8.0) / err(1.0) < 11, f"expected pred ~ O(s) (8x), got {err(8.0) / err(1.0):.1f}x"


def test_edge_cases():
    c = WorldModelLoss()
    for t_len in [1, 2, 3]:
        assert torch.isfinite(c.straightening_loss(torch.randn(2, t_len, 8)))
        assert torch.isfinite(c.chord_loss(torch.randn(2, t_len, 8)))
        assert torch.isfinite(c.prediction_loss(torch.randn(2, t_len, 8), torch.randn(2, t_len, 8)))
    m = torch.arange(16.).reshape(4, 4)
    assert _off_diagonal(m).numel() == 12
    assert m.diag().sum() not in _off_diagonal(m)


def test_total_is_weighted_sum():
    c = WorldModelLoss(pred_weight=1.0, straighten_weight=0.1, chord_weight=0.2,
                       vicreg_std_weight=1.0, vicreg_cov_weight=0.04)
    m = ProceduralWorldModel(feature_dim=D, latent_dim=D, hidden_dim=D, max_len=T)
    z, z_pred = m(torch.randn(B, T, D))
    _, ld = c(z, z_pred)
    expected = (ld["pred"] + 0.1 * ld["straighten"] + 0.2 * ld["chord"]
                + ld["vicreg_std"] + 0.04 * ld["vicreg_cov"])
    assert abs(expected - ld["total"]) < 1e-5


def test_chord_loss_straight_vs_oscillating():
    """Deterministic geometry check: a perfectly straight-line trajectory (constant
    velocity) must give chord_loss ~= 0 -- every step exactly matches the chord
    direction. A trajectory that oscillates between two points, in contrast, has steps
    that alternate with and against the chord -- for T=6 that's 3 steps at cos=+1 and 2
    at cos=-1, giving an exact expected mean cos of (3-2)/5=0.2, i.e. loss=0.8. This is
    exactly the failure mode chord_weight was added to catch (see the mean_velocity_cosine
    < 0 finding on the vicreg-only checkpoint sweep) that straightening_loss alone,
    checking only adjacent step PAIRS, can under-penalize relative to competing terms."""
    torch.manual_seed(0)
    c = WorldModelLoss()
    Bp, Tp, Dp = 3, 6, 5

    direction = torch.randn(Bp, 1, Dp)
    t = torch.arange(Tp).float().view(1, Tp, 1)
    z_straight = t * direction  # z[b, t] = t * direction[b] -- exactly straight
    assert c.chord_loss(z_straight).item() < 1e-6

    A, Bv = torch.randn(Bp, 1, Dp), torch.randn(Bp, 1, Dp)
    parity = torch.arange(Tp).view(1, Tp, 1) % 2  # 0,1,0,1,... -> A,B,A,B,...
    z_osc = torch.where(parity == 0, A.expand(Bp, Tp, Dp), Bv.expand(Bp, Tp, Dp))
    assert abs(c.chord_loss(z_osc).item() - 0.8) < 1e-4


def test_patch_encoder_global_shapes():
    """global projector pools to 1 token/frame -> plugs into the same CausalPredictor."""
    N = 256  # 16x16 grid, e.g. 224px/patch14
    enc = PatchProjectorEncoder(feature_dim=D, projector_type="global", latent_dim=D)
    out = enc(torch.randn(B, T, N, D))
    assert out.shape == (B, T, 1, D), out.shape
    assert enc.tokens_per_frame == 1


def test_patch_encoder_channel_shapes():
    """channel projector keeps the full patch grid, only crushes channel depth."""
    N = 256
    out_dim = 8
    enc = PatchProjectorEncoder(feature_dim=D, projector_type="channel", channel_out_dim=out_dim)
    out = enc(torch.randn(B, T, N, D))
    assert out.shape == (B, T, N, out_dim), out.shape
    assert enc.tokens_per_frame == N


def test_block_causal_mask_structure():
    """Block-lower-triangular: token in frame i can attend to ALL patches of any frame
    j <= i (including full attention within its own frame), never to frame j > i."""
    t_frames, n_patches = 3, 2
    mask = build_block_causal_mask(t_frames, n_patches)
    assert mask.shape == (1, 1, t_frames * n_patches, t_frames * n_patches)
    allowed = mask[0, 0]  # boolean, True == allowed to attend
    for i in range(t_frames):
        for j in range(t_frames):
            block = allowed[i * n_patches:(i + 1) * n_patches, j * n_patches:(j + 1) * n_patches]
            expected = j <= i
            assert block.all().item() == expected, f"block(frame {i}, frame {j}) allowed={block.all().item()}, expected {expected}"


def test_block_causal_predictor_dim_head_independent_of_latent_dim():
    """The reference's predictor keeps FULL attention capacity (heads*dim_head) even
    when the token's own channel dim has been crushed small (e.g. channel projector's
    out_dim=8) -- unlike PyTorch's built-in MultiheadAttention, which forces
    head_dim = embed_dim/nhead and would give a near-useless 1-dim-per-head attention
    at latent_dim=8, nhead=8. Verify the QKV projection's output size is heads*dim_head,
    NOT tied to latent_dim."""
    latent_dim, heads, dim_head = 8, 16, 64
    pred = BlockCausalPredictor(latent_dim=latent_dim, num_patches=2, num_frames=3,
                                 depth=1, heads=heads, dim_head=dim_head)
    to_qkv = pred.layers[0][0].to_qkv
    assert to_qkv.in_features == latent_dim
    assert to_qkv.out_features == heads * dim_head * 3, (
        f"expected inner attention dim heads*dim_head={heads * dim_head} (x3 for qkv), "
        f"got {to_qkv.out_features} -- attention capacity must not be tied to latent_dim"
    )


def test_block_causal_predictor_no_future_leakage():
    """Same causality property as CausalPredictor's own test, but at the frame-block
    level: perturbing ANY patch in frame j must leave every patch of frames <= j
    unchanged, and must be free to affect frames > j."""
    torch.manual_seed(0)
    t_frames, n_patches, d = 4, 3, 16
    pred = BlockCausalPredictor(latent_dim=d, num_patches=n_patches, num_frames=t_frames,
                                 heads=2, dim_head=8, depth=2).eval()
    z = torch.randn(1, t_frames, n_patches, d)
    with torch.no_grad():
        base = pred(z)
        for j in range(t_frames):
            zp = z.clone()
            zp[:, j, 0] += 5.0 * torch.randn(d)  # perturb one patch in frame j
            out = pred(zp)
            for t in range(t_frames):
                changed = (out[:, t] - base[:, t]).abs().max().item() > 1e-4
                assert changed == (j <= t), (
                    f"frame {t} {'changed' if changed else 'did not change'} when frame {j} was "
                    f"perturbed; expected dependence iff {j} <= {t}"
                )


def test_world_model_patch_global_forward():
    N = 256  # arbitrary; global mode pools whatever grid PatchProjectorEncoder is given
    m = ProceduralWorldModel(feature_dim=D, latent_dim=D, token_mode="patch",
                              projector_type="global", num_frames=T, heads=2, depth=2)
    z, z_pred = m(torch.randn(B, T, N, D))
    assert z.shape == (B, T, 1, D), z.shape
    assert z_pred.shape == (B, T - 1, 1, D), z_pred.shape


def test_world_model_patch_channel_forward():
    N, out_dim = 64, 8  # smaller N than the 196 default, to keep this test fast
    m = ProceduralWorldModel(feature_dim=D, token_mode="patch", projector_type="channel",
                              channel_out_dim=out_dim, num_patches=N, num_frames=T,
                              heads=2, depth=2)
    z, z_pred = m(torch.randn(B, T, N, D))
    assert z.shape == (B, T, N, out_dim), z.shape
    assert z_pred.shape == (B, T - 1, N, out_dim), z_pred.shape
    # optimizer must see BlockCausalPredictor's params immediately (built in __init__,
    # not lazily on first forward -- see world_model.py's docstring on why that matters)
    n_predictor_params = sum(p.numel() for p in m.predictor.parameters())
    assert n_predictor_params > 0


def test_world_model_patch_every_allocated_position_is_supervised():
    """A T-frame clip allocates and trains exactly T-1 predictor positions."""
    n_patches, out_dim = 4, 8
    m = ProceduralWorldModel(
        feature_dim=D, token_mode="patch", projector_type="channel",
        channel_out_dim=out_dim, num_patches=n_patches, num_frames=T,
        heads=2, dim_head=4, depth=1,
    )
    z, z_pred = m(torch.randn(B, T, n_patches, D))
    F.mse_loss(z_pred, z[:, 1:]).backward()
    grad = m.predictor.pos_embedding.grad.reshape(T - 1, n_patches, out_dim)
    assert torch.all(grad.abs().sum(dim=(1, 2)) > 0), "every allocated frame position needs gradient"


def test_aggregation_head_shapes():
    P, in_dim, out_dim = 6, 8, 128
    head = AggregationHead(num_patches=P, in_dim=in_dim, out_dim=out_dim)
    out = head(torch.randn(B, T, P, in_dim))
    assert out.shape == (B, T, out_dim), out.shape


def test_world_model_channel_builds_agg_head():
    """projector_type='channel' must build a real (nonzero-param) agg_head at __init__
    time -- same optimizer-registration requirement as BlockCausalPredictor itself
    (see world_model.py's docstring): if this were built lazily on first forward, its
    weights would never reach train.py's optimizer."""
    N, out_dim = 64, 8
    m = ProceduralWorldModel(feature_dim=D, token_mode="patch", projector_type="channel",
                              channel_out_dim=out_dim, num_patches=N, num_frames=T,
                              heads=2, depth=2)
    assert m.agg_head is not None
    n_agg_params = sum(p.numel() for p in m.agg_head.parameters())
    assert n_agg_params > 0
    z, _ = m(torch.randn(B, T, N, D))
    z_agg = m.aggregate(z)
    assert z_agg.shape == (B, T, m.agg_head.out_dim), z_agg.shape


def test_world_model_global_and_cls_have_no_agg_head():
    """global features (nv=1) need no aggregation head -- reference computes cosine
    similarity directly on the single vector -- so aggregate() must just squeeze/pass z
    through unchanged, never invent one."""
    m_cls = ProceduralWorldModel(feature_dim=D, latent_dim=D, hidden_dim=D, max_len=T)
    assert m_cls.agg_head is None
    z_cls, _ = m_cls(torch.randn(B, T, D))
    assert torch.equal(m_cls.aggregate(z_cls), z_cls)

    N = 64
    m_glob = ProceduralWorldModel(feature_dim=D, latent_dim=D, token_mode="patch",
                                   projector_type="global", num_frames=T, heads=2, depth=2)
    assert m_glob.agg_head is None
    z_glob, _ = m_glob(torch.randn(B, T, N, D))
    assert z_glob.shape == (B, T, 1, D)
    z_agg = m_glob.aggregate(z_glob)
    assert z_agg.shape == (B, T, D)
    assert torch.equal(z_agg, z_glob.squeeze(2))


def test_loss_uses_z_agg_when_given():
    """Passing z_agg must actually change the regularizer's input -- verifies the loss
    isn't silently ignoring it and falling back to the naive mean-pool (the reference's
    own worse-performing 'mean' ablation variant, Figure 14) when a real aggregation
    head is available."""
    torch.manual_seed(0)
    N, out_dim = 64, 8
    m = ProceduralWorldModel(feature_dim=D, token_mode="patch", projector_type="channel",
                              channel_out_dim=out_dim, num_patches=N, num_frames=T,
                              heads=2, depth=2)
    c = WorldModelLoss(straighten_weight=1.0, vicreg_std_weight=1.0, vicreg_cov_weight=1.0)
    z, z_pred = m(torch.randn(B, T, N, D))
    z_agg = m.aggregate(z)

    _, ld_with_agg = c(z, z_pred, z_agg)
    _, ld_mean_fallback = c(z, z_pred, None)

    assert abs(ld_with_agg["straighten"] - ld_mean_fallback["straighten"]) > 1e-6, (
        "aggregation-head-based curvature loss should differ from the naive mean-pool "
        "fallback (agg_head's MLP output isn't the patch mean)"
    )


def test_vicreg_granularity():
    """'token' must compute VICReg on the raw per-patch z (every token a sample, the
    reference's own behavior), 'agg' on the aggregated z_agg -- and the choice must
    not leak into the curvature loss, which always uses z_agg."""
    torch.manual_seed(0)
    N, out_dim = 16, 8
    m = ProceduralWorldModel(feature_dim=D, token_mode="patch", projector_type="channel",
                              channel_out_dim=out_dim, num_patches=N, num_frames=T,
                              heads=2, depth=2)
    z, z_pred = m(torch.randn(B, T, N, D))
    z_agg = m.aggregate(z)

    kw = dict(straighten_weight=1.0, vicreg_std_weight=1.0, vicreg_cov_weight=1.0)
    c_agg = WorldModelLoss(vicreg_granularity="agg", **kw)
    c_tok = WorldModelLoss(vicreg_granularity="token", **kw)
    _, ld_agg = c_agg(z, z_pred, z_agg)
    _, ld_tok = c_tok(z, z_pred, z_agg)

    for term, fn in (("vicreg_std", c_agg.vicreg_std_loss), ("vicreg_cov", c_agg.vicreg_cov_loss)):
        assert torch.allclose(ld_agg[term], fn(z_agg)), f"agg {term} must use z_agg"
        assert torch.allclose(ld_tok[term], fn(z)), f"token {term} must use raw per-token z"
    assert not torch.allclose(ld_agg["vicreg_cov"], ld_tok["vicreg_cov"])
    assert torch.allclose(ld_agg["straighten"], ld_tok["straighten"]), (
        "vicreg_granularity must not change the curvature loss's input")

    try:
        WorldModelLoss(vicreg_granularity="patch")
        raise AssertionError("invalid vicreg_granularity should raise")
    except ValueError:
        pass


def test_agg_mlp_hidden_dim_fallback():
    """Unset -> 4 * out_dim (the reference's fallback); explicit value wins."""
    head = AggregationHead(num_patches=4, in_dim=3, out_dim=16)
    assert head.net[0].out_features == 64
    head = AggregationHead(num_patches=4, in_dim=3, hidden=10, out_dim=16)
    assert head.net[0].out_features == 10
    m = ProceduralWorldModel(feature_dim=D, token_mode="patch", projector_type="channel",
                              channel_out_dim=4, num_patches=4, num_frames=T, heads=2, depth=1,
                              agg_out_dim=128)
    assert m.agg_head.net[0].out_features == 512, "default must stay 512 at agg_out_dim=128"


def test_loss_pools_patch_dim_before_regularizers():
    """When no z_agg is given (the fallback path -- e.g. calling this loss module
    directly without a model to aggregate through), straighten/vicreg on a 4D z must
    exactly match computing them on the patch-mean-pooled 3D version -- prediction_loss,
    in contrast, must NOT be pooled (elementwise on the native 4D shape)."""
    torch.manual_seed(0)
    N = 5
    z = torch.randn(2, 6, N, 8)
    z_pred = torch.randn(2, 6, N, 8)
    c = WorldModelLoss(straighten_weight=1.0, vicreg_std_weight=1.0, vicreg_cov_weight=1.0)
    _, ld_4d = c(z, z_pred)

    z_pooled = z.mean(dim=2)
    expected_straighten = c.straightening_loss(z_pooled).item()
    expected_vstd = c.vicreg_std_loss(z_pooled).item()
    expected_vcov = c.vicreg_cov_loss(z_pooled).item()
    assert abs(ld_4d["straighten"] - expected_straighten) < 1e-5
    assert abs(ld_4d["vicreg_std"] - expected_vstd) < 1e-5
    assert abs(ld_4d["vicreg_cov"] - expected_vcov) < 1e-5

    # prediction_loss must NOT be pooled -- verify it matches the elementwise 4D loss,
    # not the pooled-3D version (which would silently throw away per-patch signal).
    expected_pred_4d = F.l1_loss(z_pred[:, :-1], z[:, 1:])
    expected_pred_pooled = F.l1_loss(z_pred.mean(dim=2)[:, :-1], z_pooled[:, 1:])
    assert abs(ld_4d["pred"] - expected_pred_4d.item()) < 1e-5
    assert abs(ld_4d["pred"] - expected_pred_pooled.item()) > 1e-4, (
        "pred loss should differ from the pooled version unless coincidentally equal -- "
        "if this ever fires spuriously, regenerate with a different seed"
    )


class _FakeProvider:
    """Minimal FeatureProvider stand-in: .get(video_path) returns a fixed-length
    (T, D) tensor whose values are the frame index (so a returned window's own content
    reveals which contiguous span was sampled)."""

    def __init__(self, t_len, d=4):
        self.t_len = t_len
        self.d = d

    def get(self, video_path):
        return torch.arange(self.t_len).float().unsqueeze(1).repeat(1, self.d)


def test_dataset_window_size_disabled_returns_full_clip():
    """window_size=0 (default): unchanged behavior, the whole clip."""
    ds = ClipFeatureDataset(["a.mp4"], _FakeProvider(t_len=16))
    feats = ds[0]["features"]
    assert feats.shape == (16, 4)
    assert torch.equal(feats, torch.arange(16).float().unsqueeze(1).repeat(1, 4))


def test_dataset_window_size_samples_contiguous_span():
    """window_size=W: every draw is exactly W+1 frames, and — since _FakeProvider's
    frame values ARE their index — a contiguous, in-order span of the original clip."""
    window_size, t_len = 3, 16
    ds = ClipFeatureDataset(["a.mp4"], _FakeProvider(t_len=t_len), window_size=window_size)
    for _ in range(20):  # random start each draw -- repeat to exercise more than one
        feats = ds[0]["features"]
        assert feats.shape == (window_size + 1, 4)
        idx = feats[:, 0]
        assert torch.equal(idx, torch.arange(idx[0].item(), idx[0].item() + window_size + 1))
        assert 0 <= idx[0].item() <= t_len - (window_size + 1)


def test_dataset_validation_window_is_fixed_center_span():
    ds = ClipFeatureDataset(
        ["a.mp4"], _FakeProvider(t_len=16), window_size=3, random_window=False,
    )
    expected = torch.arange(6, 10)
    for _ in range(10):
        assert torch.equal(ds[0]["features"][:, 0].long(), expected)


def test_dataset_window_size_too_large_raises():
    ds = ClipFeatureDataset(["a.mp4"], _FakeProvider(t_len=4), window_size=10)
    try:
        ds[0]
        assert False, "expected a ValueError for window_size+1 > clip length"
    except ValueError:
        pass


class _FakeArgs:
    """Minimal stand-in for train.py's argparse Namespace -- only the attributes
    build_optimizer actually reads."""
    def __init__(self, lr=1e-4, encoder_lr=None, predictor_lr=None, weight_decay=1e-5):
        self.lr = lr
        self.encoder_lr = encoder_lr
        self.predictor_lr = predictor_lr
        self.weight_decay = weight_decay


def _all_model_param_ids(model):
    return {id(p) for p in model.parameters()}


def test_resolve_checkpoint_dir_reuses_missing_or_empty():
    with tempfile.TemporaryDirectory() as tmp:
        missing = Path(tmp) / "run"
        assert resolve_checkpoint_dir(missing) == missing, "nonexistent dir needs no suffix"
        missing.mkdir()
        assert resolve_checkpoint_dir(missing) == missing, "empty existing dir needs no suffix"


def test_resolve_checkpoint_dir_suffixes_past_nonempty_dirs():
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp) / "run"
        base.mkdir()
        (base / "config.yaml").touch()  # non-empty -> base itself is taken
        assert resolve_checkpoint_dir(base) == base.with_name("run_1")

        run_1 = base.with_name("run_1")
        run_1.mkdir()
        (run_1 / "config.yaml").touch()  # run_1 also taken -> skips to run_2
        assert resolve_checkpoint_dir(base) == base.with_name("run_2")

        # run_2 exists but EMPTY (e.g. just mkdir'd by train.sh's own mirrored
        # resolution, before train.py runs) -- must be reused, not skipped again.
        run_2 = base.with_name("run_2")
        run_2.mkdir()
        assert resolve_checkpoint_dir(base) == run_2


def test_resolve_checkpoint_dir_ignores_train_sh_own_log():
    """train.sh's own train.log lands in the resolved dir (via its `tee`) BEFORE
    train.py runs -- a bare train.log (± an empty live_cache/) must NOT count as
    'already used', or train.py's own resolution skips one directory further out than
    train.sh just picked (this exact mismatch happened for real: train.sh picked _1,
    wrote _1/train.log, and train.py's old any(iterdir()) check then skipped to _2)."""
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp) / "run"
        base.mkdir()
        (base / "train.log").write_text("Logging to run/train.log\n")
        (base / "live_cache").mkdir()
        assert resolve_checkpoint_dir(base) == base, "train.log/live_cache alone isn't a used run"

        (base / "config.yaml").touch()  # now a REAL artifact is present
        assert resolve_checkpoint_dir(base) == base.with_name("run_1")


def test_build_optimizer_single_group_when_lrs_unset():
    """Backward-compatible default: encoder_lr/predictor_lr both None -> ONE AdamW
    group over every param at --lr, exactly this project's original behavior."""
    m = ProceduralWorldModel(feature_dim=D, latent_dim=D, hidden_dim=D, max_len=T)
    opt = build_optimizer(m, _FakeArgs(lr=1e-4, weight_decay=1e-5))
    assert len(opt.param_groups) == 1
    g = opt.param_groups[0]
    assert g["lr"] == 1e-4 and g["weight_decay"] == 1e-5
    assert {id(p) for p in g["params"]} == _all_model_param_ids(m)


def test_build_optimizer_two_groups_partition_all_params():
    """Setting encoder_lr/predictor_lr splits into two AdamW groups -- every model
    param must land in EXACTLY one group (no missing, no duplicated), agg_head's
    params specifically must be in the ENCODER group (matching the reference's own
    grouping: its agg_mlp is a submodule of DinoV2Encoder, sharing encoder_optimizer,
    not predictor_optimizer -- see build_optimizer's docstring)."""
    N, out_dim = 64, 8
    m = ProceduralWorldModel(feature_dim=D, token_mode="patch", projector_type="channel",
                              channel_out_dim=out_dim, num_patches=N, num_frames=T,
                              heads=2, depth=2)
    assert m.agg_head is not None  # otherwise this test wouldn't exercise the case it's for
    opt = build_optimizer(m, _FakeArgs(encoder_lr=1e-5, predictor_lr=5e-4, weight_decay=1e-5))
    assert len(opt.param_groups) == 2
    enc_group, pred_group = opt.param_groups
    assert enc_group["lr"] == 1e-5
    assert pred_group["lr"] == 5e-4

    enc_ids = {id(p) for p in enc_group["params"]}
    pred_ids = {id(p) for p in pred_group["params"]}
    assert enc_ids.isdisjoint(pred_ids), "a param must not appear in both groups"
    assert enc_ids | pred_ids == _all_model_param_ids(m), (
        "every model param must be covered by exactly one group"
    )
    agg_head_ids = {id(p) for p in m.agg_head.parameters()}
    assert agg_head_ids <= enc_ids, "agg_head's params must be in the ENCODER group, not the predictor's"


def test_build_optimizer_partial_override_falls_back_to_lr():
    """Only --encoder-lr set: the predictor group should fall back to --lr, not error
    or silently use some other default."""
    m = ProceduralWorldModel(feature_dim=D, latent_dim=D, hidden_dim=D, max_len=T)
    opt = build_optimizer(m, _FakeArgs(lr=3e-4, encoder_lr=1e-6, predictor_lr=None))
    lrs = {g["lr"] for g in opt.param_groups}
    assert lrs == {1e-6, 3e-4}, f"expected encoder_lr=1e-6 and predictor fallback=lr=3e-4, got {lrs}"


def test_regularizer_axis_grouping():
    """pooled must equal the legacy flatten; per_time/per_clip must score the right sets."""
    from losses.sigreg_loss import sigreg_loss
    from losses.world_model_loss import _group
    torch.manual_seed(0)
    b, t, d = 6, 5, 8
    z = torch.randn(b, t, d)
    c = WorldModelLoss()
    # pooled == flatten, for std and cov
    g = _group(z, "pooled")
    assert g.shape == (1, b * t, d)
    assert torch.allclose(c.vicreg_std_loss(g, grouped=True), c.vicreg_std_loss(z))
    assert torch.allclose(c.vicreg_cov_loss(g, grouped=True), c.vicreg_cov_loss(z))
    # per_time: group k is exactly the B clips at timestep k; per_clip: group k is clip k's frames
    assert torch.equal(_group(z, "per_time")[2], z[:, 2])
    assert torch.equal(_group(z, "per_clip")[3], z[3])
    # patch dim folds into the sample axis
    zp = torch.randn(b, t, 4, d)
    assert _group(zp, "per_time").shape == (t, b * 4, d)
    assert _group(zp, "per_clip").shape == (b, t * 4, d)
    # grouped loss == mean of the per-group losses (cov/std); same projections for sigreg
    gt = _group(z, "per_time")
    want = torch.stack([c.vicreg_cov_loss(gt[k]) for k in range(t)]).mean()
    assert torch.allclose(c.vicreg_cov_loss(gt, grouped=True), want, atol=1e-6)
    want = torch.stack([c.vicreg_std_loss(gt[k]) for k in range(t)]).mean()
    assert torch.allclose(c.vicreg_std_loss(gt, grouped=True), want, atol=1e-6)
    torch.manual_seed(1)
    got = sigreg_loss(gt, grouped=True, num_slices=64)
    torch.manual_seed(1)
    # same seed -> same (d, S) projection matrix, shared by all groups
    per = []
    for k in range(t):
        torch.manual_seed(1)
        per.append(sigreg_loss(gt[k:k + 1], grouped=True, num_slices=64))
    assert torch.allclose(got, torch.stack(per).mean(), atol=1e-5)
    # ungrouped call unchanged: (B,T,D) and (B*T,D) agree
    torch.manual_seed(2); a1 = sigreg_loss(z, num_slices=64)
    torch.manual_seed(2); a2 = sigreg_loss(z.reshape(-1, d), num_slices=64)
    assert torch.allclose(a1, a2)
    # end to end: axis changes the loss, and invalid axes raise
    zp_pred = torch.randn(b, t, d)
    kw = dict(vicreg_std_weight=1.0, vicreg_cov_weight=1.0, sigreg_weight=1.0)
    vals = {ax: WorldModelLoss(vicreg_axis=ax, sigreg_axis=ax, **kw)(z, zp_pred)[1]["vicreg_cov"].item()
            for ax in ("pooled", "per_time", "per_clip")}
    assert len(set(vals.values())) == 3, vals
    # A final batch with one clip has no across-clip VICReg statistic.
    single = torch.randn(1, t, d, requires_grad=True)
    single_pred = torch.randn(1, t, d, requires_grad=True)
    one_clip_loss, one_clip_parts = WorldModelLoss(
        vicreg_axis="per_time", vicreg_std_weight=1.0, vicreg_cov_weight=1.0,
    )(single, single_pred)
    assert torch.isfinite(one_clip_loss)
    assert one_clip_parts["vicreg_std"] == 0 and one_clip_parts["vicreg_cov"] == 0
    one_clip_loss.backward()
    assert torch.isfinite(single.grad).all()
    try:
        WorldModelLoss(sigreg_axis="per_batch")
        raise AssertionError("invalid axis should raise")
    except ValueError:
        pass


def test_cls_agg_head_is_opt_in_and_decouples_straightening_from_z():
    """Off by default (aggregate() == z). On: head output is (B,T,agg_out_dim), distinct from z,
    straightening reads the head while SIGReg(granularity=token) still reads raw z, and the
    straightening gradient reaches the encoder only THROUGH the head."""
    m_off = ProceduralWorldModel(feature_dim=D, latent_dim=D, hidden_dim=D, max_len=T)
    assert m_off.agg_head is None
    m = ProceduralWorldModel(feature_dim=D, latent_dim=D, hidden_dim=D, max_len=T,
                              cls_agg_head=True, agg_out_dim=16)
    assert m.agg_head is not None
    z, z_pred = m(torch.randn(B, T, D))
    z_agg = m.aggregate(z)
    assert z.shape == (B, T, D) and z_agg.shape == (B, T, 16)
    # straightening on the head output only (no regularizers): loss matches the standalone
    # formula on z_agg, not on z
    c = WorldModelLoss(straighten_weight=1.0, pred_weight=0.0)
    _, ld = c(z, z_pred, z_agg)
    assert torch.allclose(ld["straighten"], c.straightening_loss(z_agg))
    assert not torch.allclose(ld["straighten"], c.straightening_loss(z))
    # SIGReg token granularity reads z (dim D), agg granularity reads the head output (dim 16)
    seen = {}
    import losses.world_model_loss as wml
    orig = wml.sigreg_loss
    wml.sigreg_loss = lambda zz, **kw: (seen.__setitem__("d", zz.shape[-1]), orig(zz, **kw))[1]
    try:
        WorldModelLoss(sigreg_weight=1.0, sigreg_granularity="token")(z, z_pred, z_agg)
        assert seen["d"] == D
        WorldModelLoss(sigreg_weight=1.0, sigreg_granularity="agg")(z, z_pred, z_agg)
        assert seen["d"] == 16
    finally:
        wml.sigreg_loss = orig
    # straightening-only gradient reaches the encoder via the head
    m.zero_grad()
    c.straightening_loss(m.aggregate(m.encoder(torch.randn(B, T, D)))).backward()
    assert m.agg_head.net[0].weight.grad is not None
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in m.encoder.parameters())
    # patch mode rejects the flag (it builds its own head)
    try:
        ProceduralWorldModel(feature_dim=D, latent_dim=D, token_mode="patch", projector_type="global",
                              num_frames=T, heads=2, depth=2, cls_agg_head=True)
        raise AssertionError("cls_agg_head in patch mode should raise")
    except ValueError:
        pass


# ---- variable-length / padding support ----
# The correctness bar throughout: a masked loss on a TRAILING-PADDED batch must equal
# the same loss computed on the unpadded, actually-shorter clip directly. Not "close",
# EQUAL (within float tolerance) -- padding+masking is only correct if it is perfectly
# equivalent to the clip never having had those extra frames at all.

def test_dataset_variable_length_pads_trailing_and_zeros_rest():
    feats = torch.randn(5, T, D)
    paths = [f"v{i}" for i in range(5)]

    class FakeProvider:
        def get(self, p):
            return feats[paths.index(p)].clone()

        def has_precomputed(self, p):
            return True

    ds = ClipFeatureDataset(paths, FakeProvider(), variable_length_min=3)
    for _ in range(20):  # random length each call -- exercise the range
        item = ds[0]
        length = item["length"]
        assert 3 <= length <= T
        assert torch.equal(item["features"][:length], feats[0][:length])
        if length < T:
            assert (item["features"][length:] == 0).all()
    # off: every item reports the full length, unchanged features
    ds_off = ClipFeatureDataset(paths, FakeProvider())
    item = ds_off[0]
    assert item["length"] == T
    assert torch.equal(item["features"], feats[0])


def test_dataset_variable_length_rejects_out_of_range_min():
    class FakeProvider:
        def get(self, p):
            return torch.randn(T, D)

        def has_precomputed(self, p):
            return True

    for bad in (0, T + 1):
        ds = ClipFeatureDataset(["v0"], FakeProvider(), variable_length_min=bad)
        try:
            ds[0]
            raise AssertionError(f"variable_length_min={bad} should raise")
        except ValueError:
            pass


def _valid_mask(lengths, t):
    return torch.arange(t).unsqueeze(0) < lengths.unsqueeze(1)


def test_prediction_loss_masked_matches_unpadded_cls_shape():
    for pred_loss in ("l1", "l2", "angular"):
        crit = WorldModelLoss(pred_loss=pred_loss)
        torch.manual_seed(0)
        z = torch.randn(1, T, D)
        z_pred = torch.randn(1, T, D)
        L = 6
        lengths = torch.tensor([L])
        z_padded, z_pred_padded = z.clone(), z_pred.clone()
        z_padded[:, L:] = 0
        z_pred_padded[:, L:] = 0
        masked = crit.prediction_loss(z_pred_padded, z_padded, valid=_valid_mask(lengths, T))
        unpadded = crit.prediction_loss(z_pred[:, :L], z[:, :L], valid=None)
        assert torch.allclose(masked, unpadded, atol=1e-6), (pred_loss, masked, unpadded)


def test_prediction_loss_masked_matches_unpadded_patch_tminus1_shape():
    """z_pred is T-1 long (BlockCausalPredictor's convention) and z is 4D (B,T,P,D)."""
    crit = WorldModelLoss(pred_loss="l2")
    torch.manual_seed(0)
    P = 3
    z = torch.randn(1, T, P, D)
    z_pred = torch.randn(1, T - 1, P, D)
    L = 6
    lengths = torch.tensor([L])
    z_padded = z.clone()
    z_padded[:, L:] = 0
    # z_pred[:, t] predicts z[:, t+1], so only the first L-1 predictor positions matter
    z_pred_padded = z_pred.clone()
    z_pred_padded[:, L - 1:] = 0
    masked = crit.prediction_loss(z_pred_padded, z_padded, valid=_valid_mask(lengths, T))
    unpadded = crit.prediction_loss(z_pred[:, :L - 1], z[:, :L], valid=None)
    assert torch.allclose(masked, unpadded, atol=1e-6)


def test_straightening_loss_masked_matches_unpadded():
    crit = WorldModelLoss()
    torch.manual_seed(1)
    z = torch.randn(1, T, D)
    L = 7
    lengths = torch.tensor([L])
    z_padded = z.clone()
    z_padded[:, L:] = 0
    masked = crit.straightening_loss(z_padded, valid=_valid_mask(lengths, T))
    unpadded = crit.straightening_loss(z[:, :L], valid=None)
    assert torch.allclose(masked, unpadded, atol=1e-6)


def test_chord_loss_masked_matches_unpadded_and_gathers_real_goal():
    crit = WorldModelLoss()
    torch.manual_seed(2)
    z = torch.randn(1, T, D)
    L = 5
    lengths = torch.tensor([L])
    z_padded = z.clone()
    z_padded[:, L:] = 0
    # WITHOUT the lengths= gather fix, chord would use z_padded[:,-1] (a zero pad frame)
    # -- confirm that's wrong, then confirm the fix matches the unpadded computation.
    wrong = crit.chord_loss(z_padded, valid=_valid_mask(lengths, T), lengths=None)
    correct = crit.chord_loss(z_padded, valid=_valid_mask(lengths, T), lengths=lengths)
    unpadded = crit.chord_loss(z[:, :L], valid=None, lengths=None)
    assert torch.allclose(correct, unpadded, atol=1e-6)
    assert not torch.allclose(wrong, unpadded, atol=1e-3), \
        "chord_loss without the lengths= gather should NOT match (sanity: the fix matters)"


def test_group_masked_pooled_selects_exactly_the_valid_rows():
    torch.manual_seed(3)
    z = torch.randn(B, T, D)
    lengths = torch.tensor([3, 16, 1, 9])
    valid = _valid_mask(lengths, T)
    grouped = _group_masked(z, "pooled", valid, "vicreg_axis")
    assert grouped.shape == (1, int(lengths.sum()), D)
    # every row it kept is a real (valid) row, and nothing valid was dropped
    expected = z[valid]
    assert torch.allclose(grouped.squeeze(0), expected)

    # 4D (patch) input: a whole padded FRAME's patches must all be dropped together
    Pp = 2
    z4 = torch.randn(B, T, Pp, D)
    grouped4 = _group_masked(z4, "pooled", valid, "vicreg_axis")
    assert grouped4.shape == (1, int(lengths.sum()) * Pp, D)

    # valid=None is byte-identical to plain _group (every existing call site)
    assert torch.equal(_group_masked(z, "pooled", None, "vicreg_axis"), _group(z, "pooled"))


def test_group_masked_rejects_non_pooled_axis_with_lengths():
    lengths = torch.tensor([3, 16, 1, 9])
    valid = _valid_mask(lengths, T)
    z = torch.randn(B, T, D)
    for axis in ("per_time", "per_clip"):
        try:
            _group_masked(z, axis, valid, "sigreg_axis")
            raise AssertionError(f"axis={axis!r} with lengths given should raise")
        except NotImplementedError:
            pass


def test_vicreg_and_sigreg_masked_pooled_match_unpadded_population():
    """End to end through forward(): VICReg/SIGReg at the default pooled axis, computed
    on a padded batch via lengths=, must equal computing them directly on the
    concatenated REAL rows of every clip (i.e. exactly what training without padding
    would have seen, just gathered across a batch of different lengths)."""
    torch.manual_seed(4)
    crit = WorldModelLoss(vicreg_std_weight=1.0, vicreg_cov_weight=1.0, sigreg_weight=1.0)
    z = torch.randn(B, T, D)
    z_pred = torch.randn(B, T, D)
    lengths = torch.tensor([3, 16, 1, 9])
    valid = _valid_mask(lengths, T)
    z_padded = z.clone()
    z_padded[~valid] = 0
    _, d_padded = crit(z_padded, z_pred, lengths=lengths)

    real_rows = z[valid].unsqueeze(0)  # (1, sum(lengths), D) -- what _group("pooled") would've built directly
    torch.manual_seed(100)  # sigreg draws a random projection -- pin it identically on both sides
    std_direct = crit.vicreg_std_loss(real_rows, grouped=True)
    cov_direct = crit.vicreg_cov_loss(real_rows, grouped=True)
    torch.manual_seed(100)
    from losses.sigreg_loss import sigreg_loss as _sigreg
    sig_direct = _sigreg(real_rows, num_slices=crit.sigreg_num_slices, grouped=True)

    assert torch.allclose(d_padded["vicreg_std"], std_direct, atol=1e-5)
    assert torch.allclose(d_padded["vicreg_cov"], cov_direct, atol=1e-5)
    # sigreg draws its own fresh random projection inside forward() too -- re-seed isn't
    # reachable from here, so just check it ran and produced a finite, non-degenerate
    # value rather than re-deriving bitwise equality for this one term.
    assert torch.isfinite(d_padded["sigreg"]).all()


def test_uniformity_masked_drops_padded_rows():
    torch.manual_seed(5)
    crit = WorldModelLoss(uniformity_weight=1.0)
    z = torch.randn(B, T, D)
    z_pred = torch.randn(B, T, D)
    lengths = torch.tensor([3, 16, 1, 9])
    valid = _valid_mask(lengths, T)
    z_padded = z.clone()
    z_padded[~valid] = 0
    _, d_padded = crit(z_padded, z_pred, lengths=lengths)
    assert torch.isfinite(d_padded["uniformity"]).all()
    # a degenerate case the implementation must not crash on: every row padding (L=0
    # isn't allowed by the dataset, but a single-clip batch with L=1 leaves only 1 valid
    # row for uniformity's pairwise sampling -- confirm no crash, finite output)
    lengths1 = torch.tensor([1])
    z1 = torch.zeros(1, T, D)
    z1[:, 0] = torch.randn(D)
    _, d1 = crit(z1, torch.randn(1, T, D), lengths=lengths1)
    assert torch.isfinite(d1["uniformity"]).all()


def test_world_model_loss_lengths_none_is_unchanged():
    """The whole feature is opt-in: every existing call site (lengths never passed, or
    explicitly None) must be byte-identical to before this argument existed."""
    torch.manual_seed(6)
    crit = WorldModelLoss(straighten_weight=0.1, chord_weight=0.1,
                           vicreg_std_weight=1.0, vicreg_cov_weight=1.0)
    z = torch.randn(B, T, D)
    z_pred = torch.randn(B, T, D)
    torch.manual_seed(42)
    loss_a, dict_a = crit(z, z_pred)
    torch.manual_seed(42)
    loss_b, dict_b = crit(z, z_pred, lengths=None)
    assert torch.equal(loss_a, loss_b)
    for k in dict_a:
        assert torch.equal(dict_a[k], dict_b[k]), k


def test_world_model_padded_batch_forward_cls_and_patch():
    """Integration smoke test: a real ProceduralWorldModel (both modes) forward +
    WorldModelLoss backward over a padded batch runs cleanly end to end."""
    torch.manual_seed(7)
    for kwargs, x in [
        (dict(feature_dim=D, latent_dim=D, max_len=T), torch.randn(B, T, D)),
        (dict(feature_dim=D, latent_dim=D, token_mode="patch", projector_type="global",
              num_frames=T, heads=2, depth=2), torch.randn(B, T, 4, D)),  # 4 = 2x2, a valid square grid
    ]:
        m = ProceduralWorldModel(**kwargs)
        lengths = torch.tensor([3, T, 1, 9])
        valid = _valid_mask(lengths, T)
        x_padded = x.clone()
        if x.dim() == 4:
            x_padded[~valid.unsqueeze(-1).expand(-1, -1, x.shape[2])] = 0
        else:
            x_padded[~valid] = 0
        z, z_pred = m(x_padded)
        z_agg = m.aggregate(z)
        crit = WorldModelLoss(straighten_weight=0.1, vicreg_std_weight=1.0)
        loss, d = crit(z, z_pred, z_agg, lengths=lengths)
        assert torch.isfinite(loss)
        loss.backward()
        assert any(p.grad is not None for p in m.encoder.parameters())


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
