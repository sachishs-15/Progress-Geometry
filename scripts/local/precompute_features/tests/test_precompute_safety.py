"""Regression checks for extraction provenance and decode-failure handling."""
import argparse
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from precompute_features import (  # noqa: E402
    VideoDecodeError,
    build_extraction_metadata,
    load_frames_linspace,
    metadata_fingerprint,
    parse_encoder_kwargs,
    validate_metadata,
    write_failures,
)


def _args(csv_path, **overrides):
    values = dict(model="dinov2", n_frames=16, pooling="cls", patch_tokens=False,
                  hf_id=None, encoder_kwargs=None, csv=str(csv_path))
    values.update(overrides)
    return argparse.Namespace(**values)


def test_metadata_fingerprint_covers_extraction_settings():
    with tempfile.TemporaryDirectory() as tmp:
        csv_path = Path(tmp) / "clips.csv"
        csv_path.write_text("video_path\na.mp4\n")
        base_args = _args(csv_path)
        base = build_extraction_metadata(base_args, csv_path, parse_encoder_kwargs(base_args))
        changed_args = _args(csv_path, n_frames=8)
        changed = build_extraction_metadata(changed_args, csv_path,
                                            parse_encoder_kwargs(changed_args))
        assert metadata_fingerprint(base) != metadata_fingerprint(changed)

        payload = {"extraction_metadata": base,
                   "extraction_fingerprint": metadata_fingerprint(base)}
        validate_metadata(payload, base, Path("features.pt"))
        try:
            validate_metadata(payload, changed, Path("features.pt"))
        except SystemExit as exc:
            assert "different extraction settings" in str(exc)
        else:
            raise AssertionError("changed extraction settings must reject an existing output")


def test_decode_failure_raises_instead_of_returning_black_frames():
    try:
        load_frames_linspace("/definitely/missing/video.mp4", 4)
    except VideoDecodeError:
        pass
    else:
        raise AssertionError("decode failure must not produce replacement images")


def test_failure_manifest_is_explicit():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "failures.csv"
        write_failures(path, [{"video_path": "bad.mp4", "stage": "decode",
                               "error": "cannot decode"}])
        text = path.read_text()
        assert "bad.mp4" in text and "decode" in text


def test_encoder_kwargs_require_documented_semicolon_separator():
    args = _args(Path("clips.csv"),
                 encoder_kwargs="capture_attention=False;per_frame_batch_size=16")
    assert parse_encoder_kwargs(args) == {
        "pooling": "cls", "capture_attention": False, "per_frame_batch_size": 16,
    }
    args.encoder_kwargs = "capture_attention=False,per_frame_batch_size=16"
    try:
        parse_encoder_kwargs(args)
    except ValueError:
        pass
    else:
        raise AssertionError("comma-separated kwargs must fail instead of becoming one truthy string")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in tests:
        fn()
        print(f"PASS  {fn.__name__}")
