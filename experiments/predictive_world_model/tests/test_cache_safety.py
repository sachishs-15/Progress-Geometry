"""Regression checks for live-cache extraction identity and stored provenance."""
import tempfile
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.feature_provider import FeatureProvider


def test_image_sizes_change_live_cache_identity():
    with tempfile.TemporaryDirectory() as tmp:
        patch_196 = FeatureProvider(live_cache_dir=tmp, token_mode="patch", patch_img_size=196)
        patch_224 = FeatureProvider(live_cache_dir=tmp, token_mode="patch", patch_img_size=224)
        raw_196 = FeatureProvider(live_cache_dir=tmp, token_mode="raw", raw_img_size=196)
        raw_224 = FeatureProvider(live_cache_dir=tmp, token_mode="raw", raw_img_size=224)
        assert patch_196._live_cache_path("clip.mp4") != patch_224._live_cache_path("clip.mp4")
        assert raw_196._live_cache_path("clip.mp4") != raw_224._live_cache_path("clip.mp4")


def test_live_cache_stores_and_validates_metadata():
    with tempfile.TemporaryDirectory() as tmp:
        provider = FeatureProvider(live_cache_dir=tmp, token_mode="raw", n_frames=2,
                                   raw_img_size=8)
        expected = torch.arange(2 * 3 * 8 * 8, dtype=torch.uint8).reshape(2, 3, 8, 8)
        provider._compute_live = lambda _: expected
        assert torch.equal(provider.get("clip.mp4"), expected)

        cache_path = provider._live_cache_path("clip.mp4")
        saved = torch.load(cache_path, map_location="cpu")
        assert saved["metadata"] == provider._live_cache_metadata("clip.mp4")
        assert torch.equal(saved["features"], expected)

        saved["metadata"]["raw_img_size"] = 999
        torch.save(saved, cache_path)
        try:
            provider.get("clip.mp4")
        except RuntimeError as exc:
            assert "metadata mismatch" in str(exc)
        else:
            raise AssertionError("a cache entry with mismatched metadata must be rejected")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in tests:
        fn()
        print(f"PASS  {fn.__name__}")
