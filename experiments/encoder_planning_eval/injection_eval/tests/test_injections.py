"""Injections change exactly what they claim to, and report the right seams."""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from injections import (BLOCK, INJECTIONS, PIXEL_ONLY, REFERENCE, appearance, blur, clean_window,  # noqa: E402
                        drop, freeze, is_pixels, reverse, rng_for, slow_down, speed_up, splice, swap, shuffle)

T = 16


def clip(p=3, d=5):
    """Frame t is constant t (plus a feature offset), so every output frame's source is readable."""
    x = torch.arange(T, dtype=torch.float32).reshape(T, 1, 1).expand(T, p, d).clone()
    return x


def src_of(y):
    return [round(float(v)) for v in y[:, 0, 0]]


def rng(name="t", i=0):
    return rng_for(0, name, i)


def test_reverse_flips_and_has_no_seam():
    y, s = reverse(clip(), rng())
    assert src_of(y) == list(range(T))[::-1] and s == []


def test_swap_is_a_permutation_with_seams_at_every_jump():
    for i in range(50):
        y, s = swap(clip(), rng("swap", i))
        src = src_of(y)
        assert sorted(src) == list(range(T)) and src != list(range(T))
        jumps = [t for t in range(T - 1) if src[t + 1] != src[t] + 1]
        assert s == jumps and 2 <= len(s) <= 4


def test_shuffle_permutes_blocks_and_marks_block_boundaries():
    for i in range(50):
        y, s = shuffle(clip(), rng("shuffle", i))
        src = src_of(y)
        assert sorted(src) == list(range(T)) and src != list(range(T))
        for b in range(0, T, BLOCK):
            assert src[b:b + BLOCK] == list(range(src[b], src[b] + BLOCK))
        assert all((t + 1) % BLOCK == 0 for t in s) and len(s) >= 1


def test_drop_removes_four_frames_and_marks_the_gap():
    for i in range(50):
        y, s = drop(clip(), rng("drop", i))
        src = src_of(y)
        assert len(src) == 12 and len(s) == 1
        t = s[0]
        assert src[t + 1] - src[t] == BLOCK + 1
        assert src == sorted(src) and len(set(src)) == 12


def test_splice_replaces_exactly_one_block_with_donor_frames():
    x, donor = clip(), clip() + 100
    assert splice(x, rng(), None) is None
    for i in range(50):
        y, s = splice(x, rng("splice", i), donor)
        src = src_of(y)
        changed = [t for t in range(T) if src[t] >= 100]
        assert len(changed) == BLOCK and changed == list(range(changed[0], changed[0] + BLOCK))
        assert s == [changed[0] - 1, changed[-1]]
        assert all(src[t] == t for t in range(T) if t not in changed)


def test_freeze_holds_a_frame_then_jumps():
    for i in range(50):
        y, s = freeze(clip(), rng("freeze", i))
        src = src_of(y)
        t = s[0]
        assert len(set(src[t - BLOCK + 1:t + 1])) == 1 and src[t + 1] - src[t] == BLOCK


def test_speed_up_skips_frames_and_has_twelve_frames():
    y, s = speed_up(clip(), rng())
    assert src_of(y) == list(range(8)) + [9, 11, 13, 15] and s == []


def test_slow_down_interpolates_second_half_at_half_pace():
    y, s = slow_down(clip(), rng())
    assert y.shape[0] == T
    vals = [float(v) for v in y[:, 0, 0]]
    assert vals[:8] == [float(k) for k in range(8)]
    np.testing.assert_allclose(vals[8:], [7.5 + 0.5 * k for k in range(8)], atol=1e-6)


def test_appearance_adds_one_constant_to_every_frame():
    x, donor = clip(), clip() * 0 + 50
    y, s = appearance(x, rng(), donor)
    delta = y - x
    assert torch.allclose(delta, delta[0:1].expand_as(delta)) and s == []
    assert appearance(x, rng(), None) is None


def test_clean_window_is_contiguous():
    for i in range(30):
        y, _ = clean_window(clip(), rng("w", i))
        src = src_of(y)
        assert len(src) == 12 and src == list(range(src[0], src[0] + 12))


def test_every_injection_has_a_reference_and_is_deterministic():
    assert set(INJECTIONS) == set(REFERENCE)
    x, donor = clip(), clip() + 100
    for name, f in INJECTIONS.items():
        a = f(x, rng(name, 3), donor)
        b = f(x, rng(name, 3), donor)
        if name in PIXEL_ONLY:
            assert a is None and b is None          # feature inputs cannot be blurred
            continue
        assert torch.equal(a[0], b[0]) and a[1] == b[1]
        want = 12 if REFERENCE[name] == "clean12" else T
        assert a[0].shape[0] == want, name


def test_works_on_cls_style_two_dimensional_clips():
    x = torch.arange(T * 4, dtype=torch.float32).reshape(T, 4)
    donor = x + 1000
    for name, f in INJECTIONS.items():
        out = f(x, rng(name), donor)
        if name in PIXEL_ONLY:
            continue
        assert out[0].shape[1] == 4


def test_pixel_injections_work_on_rgb_frames():
    g = torch.Generator().manual_seed(0)
    x = torch.rand(T, 3, 32, 32, generator=g)
    donor = torch.rand(T, 3, 32, 32, generator=g)
    y, s = blur(x, rng())
    assert y.shape == x.shape and s == [] and y.var() < x.var() * 0.5   # blur removes high frequencies
    y, s = appearance(x, rng(), donor)
    assert y.shape == x.shape and 0 <= y.min() and y.max() <= 1
    shift = y - x                                  # unclamped pixels move by one constant colour
    inside = (x + 0.0).clamp(0.2, 0.8) == x
    assert shift[inside].numel() == 0 or shift.std(dim=0).mean() < 1e-6 + shift.abs().mean()
    for name, f in INJECTIONS.items():
        out = f(x, rng(name), donor)
        assert out is not None and out[0].shape[1:] == x.shape[1:], name
    assert is_pixels(x) and not is_pixels(torch.zeros(T, 196, 384))


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
