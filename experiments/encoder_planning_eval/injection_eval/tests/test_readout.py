"""ModelReadout wiring: pooled latents, prediction-error alignment (pred[t] vs z[t+1])."""
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from run_injection_eval import ModelReadout, RawReadout, build_condition, pick_donors, sample_clips  # noqa: E402


class FakeModel(nn.Module):
    """Frame-wise linear encoder; each test sets `predictor` to the callable it needs."""
    def __init__(self):
        super().__init__()
        self.enc = nn.Linear(6, 4)
        self.predictor = None

    def encoder(self, x):
        return self.enc(x)



def test_patch_mode_alignment_with_an_oracle_predictor():
    torch.manual_seed(0)
    model = FakeModel()
    x = torch.randn(2, 8, 5, 6)
    with torch.no_grad():
        z = model.encoder(x)
    model.predictor = lambda zin: z[:, 1:]        # exact next-latent predictor
    ro = ModelReadout("fake", model, {"token_mode": "patch", "pred_loss": "l2"})
    pooled, pe, tok, copy = ro(x)
    assert pooled.shape == (2, 8, 4) and pe.shape == (2, 7) and tok.shape == (2, 7)
    assert np.allclose(pe, 0, atol=1e-10) and copy.shape == (2, 7) and (copy > 0).all()
    model.predictor = lambda zin: zin             # "nothing changes" predictor
    pooled, pe_copy, _, copy2 = ModelReadout("fake", model, {"token_mode": "patch", "pred_loss": "l2"})(x)
    assert (pe_copy > 0).all() and np.allclose(pe_copy, copy2)   # a copy predictor equals the copy baseline


def test_concat_geometry_keeps_patch_order_and_predictor_scores():
    torch.manual_seed(0)
    model = FakeModel()
    x = torch.randn(2, 8, 5, 6)
    model.predictor = lambda zin: zin
    cfg = {"token_mode": "patch", "pred_loss": "l2"}
    mean, mean_pe, _, _ = ModelReadout("fake", model, cfg)(x)
    concat, concat_pe, _, _ = ModelReadout("fake", model, cfg, geometry_pooling="concat")(x)
    with torch.no_grad():
        z = model.encoder(x).numpy()
    assert concat.shape == (2, 8, 20)
    assert np.allclose(concat, z.reshape(2, 8, 20))
    assert np.allclose(mean, z.mean(axis=2))
    assert np.array_equal(mean_pe, concat_pe)


def test_geometry_only_readout_skips_predictor():
    model = FakeModel()
    model.predictor = lambda zin: (_ for _ in ()).throw(AssertionError("predictor called"))
    x = torch.randn(2, 8, 5, 6)
    readout = ModelReadout("fake", model, {"token_mode": "patch"}, compute_predictor=False)
    pooled, pe, tok, copy = readout(x)
    assert pooled.shape == (2, 8, 4) and tok.shape == (2, 7)
    assert pe is None and copy is None and not readout.has_predictor


def test_cls_mode_alignment():
    torch.manual_seed(0)
    model = FakeModel()
    x = torch.randn(2, 8, 6)
    with torch.no_grad():
        z = model.encoder(x)
    model.predictor = lambda zin: torch.cat([z[:, 1:], z[:, -1:]], dim=1)   # pred[t] = z[t+1]
    _, pe, _, _ = ModelReadout("fake", model, {"token_mode": "cls", "pred_loss": "l2"})(x)
    assert pe.shape == (2, 7) and np.allclose(pe, 0, atol=1e-10)


def test_raw_readout_pools_patches_and_has_no_predictor():
    x = torch.randn(2, 8, 5, 6)
    pooled, pe, tok, copy = RawReadout()(x)
    assert pe is None and copy is None and np.allclose(pooled, x.mean(2).numpy()) and tok.shape == (2, 7)


def test_build_condition_shapes_and_invalid_donor_flag():
    X = torch.randn(6, 16, 3, 4)
    labels = ["a", "a", "b", "c", "c", "d"]
    other, same = pick_donors(labels, 0)
    assert same[5] is None and same[2] is None and same[0] == 1 and all(o is not None for o in other)
    donors = {"other": other, "same": same}
    y, seams, valid = build_condition("splice_same", X, list(range(6)), labels, donors, 0)
    assert y.shape == (6, 16, 3, 4) and valid == [True, True, False, True, True, False]
    y, seams, valid = build_condition("drop", X, [0, 1], labels, donors, 0)
    assert y.shape == (2, 12, 3, 4) and all(valid)
    y, _, _ = build_condition("clean12", X, [0, 1], labels, donors, 0)
    assert y.shape[1] == 12


def test_small_stratified_sample_retains_same_class_donors():
    labels = [f"class_{i // 3}" for i in range(30)]
    indices = sample_clips(list(range(30)), labels, 10, 10, 0)
    sampled = [labels[i] for i in indices]
    assert len(indices) == 10
    assert all(sampled.count(label) >= 2 for label in sampled)


if __name__ == "__main__":
    fns = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    failed = 0
    for n, f in fns:
        try:
            f()
            print(f"PASS  {n}")
        except Exception as e:
            failed += 1
            print(f"FAIL  {n}: {type(e).__name__}: {e}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
