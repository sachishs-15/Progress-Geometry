"""Feature loading with a precomputed-first, compute-on-the-fly-fallback policy.

Three tiers, checked in order for a given video_path:
  1. A bulk precomputed file (LiFT's precompute_features.py .pt format:
     {"features": (N, T, D) float32, "video_paths": [str] * N}) — a plain in-memory
     dict lookup, no encoder ever touched.
  2. A per-video on-disk "live cache" (--live-cache-dir) of anything this
     FeatureProvider has itself computed in a *previous* run. Its key and stored metadata
     cover every extraction setting, including patch/raw image sizes.
  3. An actual DINOv2 forward pass over the raw video — lazy-loaded (the encoder is only
     ever instantiated if tier 1 and 2 both miss), then its result is written to tier 2
     before being returned, so a given video only ever hits this path once across the
     life of --live-cache-dir.

This means the same training run works unmodified whether a dataset has been bulk
-precomputed ahead of time, not at all, or only partially (the realistic case while
iterating on which token_mode/projector to use) — every video not found in tier 1 is
just computed lazily and cached for next time.

token_mode='cls' (default): LiFT's own DINOv2ForVideo (timm, vit_small_patch14_reg4_
dinov2.lvd142m) — same as always, one CLS vector/frame, (T, D).

token_mode='patch': the native facebookresearch/dinov2 repo (via torch.hub, NOT timm) —
LiFT's DINOv2ForVideo can't cleanly expose patch tokens (its forward_dense() is broken/
unused, per its own docstring). Returns raw patch tokens, (T, N, D), N a perfect square
(256 for the 224px/patch14 resize used here).

token_mode='raw': no backbone at all — decodes the same n_frames-uniformly-sampled
frames the other two modes use (_load_frames), resizes to raw_img_size, and returns them
as (T, 3, H, W) uint8 pixels, untouched otherwise (no normalization, no backbone forward
pass). Consumed by models/pixel_encoder.py's trainable CNN, which is the one encoder in
this project that has gradients all the way back to pixels — see world_model.py's
token_mode='raw' docstring. Tier 2 (live-cache) still applies here, and matters MORE than
for cls/patch: caching the decoded uint8 frames (not a backbone's output) is what turns
"decode this video" from a per-epoch cost into a one-time cost, since token_mode='raw'
has no frozen feature to cache in its place.
"""
import fcntl
import hashlib
import os
import json
import sys
import warnings
from pathlib import Path
from typing import Optional

import torch

_LIFT_ROOT = Path(__file__).resolve().parents[3] / "external" / "chirality_in_action" / "LiFT"
if str(_LIFT_ROOT) not in sys.path:
    sys.path.insert(0, str(_LIFT_ROOT))


class FeatureProvider:
    LIVE_CACHE_SCHEMA_VERSION = 2

    def __init__(self, features_path: Optional[str] = None, live_cache_dir: Optional[str] = None,
                 n_frames: int = 16, model_id: str = "vit_small_patch14_reg4_dinov2.lvd142m",
                 token_mode: str = "cls", patch_hub_id: str = "dinov2_vits14",
                 patch_img_size: int = 196, raw_img_size: int = 224, device: Optional[str] = None):
        if token_mode not in ("cls", "patch", "raw"):
            raise ValueError(f"token_mode must be 'cls', 'patch' or 'raw', got {token_mode!r}")
        self.n_frames = n_frames
        self.model_id = model_id
        self.raw_img_size = raw_img_size
        self.token_mode = token_mode
        self.patch_hub_id = patch_hub_id
        self.patch_img_size = patch_img_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.live_cache_dir = Path(live_cache_dir) if live_cache_dir else None
        if self.live_cache_dir is not None:
            self.live_cache_dir.mkdir(parents=True, exist_ok=True)

        self._path_to_idx = {}
        self._bulk_features = None
        if features_path is not None:
            # mmap=True: page in only the videos actually indexed instead of
            # materializing the whole bulk file in RAM up front -- patch-mode files in
            # particular (~50x cls mode/video, see token_mode='patch' docstring above)
            # can be 100GB+, which OOMs a from-scratch torch.load well before training
            # even starts.
            data = torch.load(features_path, map_location="cpu", mmap=True)
            if "extraction_metadata" not in data or "extraction_fingerprint" not in data:
                warnings.warn(
                    f"{features_path} is a legacy bulk feature file without extraction "
                    "provenance; its encoder/checkpoint/pooling settings cannot be verified. "
                    "Regenerate it with scripts/local/precompute_features/precompute_features.py.",
                    RuntimeWarning,
                )
            self._bulk_features = data["features"]  # (N, T, D) or (N, T, Npatch, D)
            self._path_to_idx = {p: i for i, p in enumerate(data["video_paths"])}

        self._backbone = None  # lazy — only loaded on the first tier-3 miss
        self._preprocess = None

    def has_precomputed(self, video_path: str) -> bool:
        return video_path in self._path_to_idx

    @property
    def video_paths(self):
        """Every video_path present in the bulk precomputed file, in its stored order.
        Lets a fully-precomputed dataset (e.g. K400) be trained on directly, with no
        separate manifest restating what the .pt already lists."""
        return list(self._path_to_idx)

    @property
    def feature_shape(self):
        """Trailing shape of the bulk precomputed features (T,D) or (T,N,D), or None if
        none were loaded."""
        if self._bulk_features is None:
            return None
        return tuple(self._bulk_features.shape[1:])

    def _live_cache_path(self, video_path: str) -> Path:
        key = hashlib.sha256(json.dumps(
            self._live_cache_metadata(video_path), sort_keys=True, separators=(",", ":")
        ).encode()).hexdigest()
        return self.live_cache_dir / f"{key}.pt"

    def _live_cache_metadata(self, video_path: str) -> dict:
        return {
            "schema_version": self.LIVE_CACHE_SCHEMA_VERSION,
            "video_path": video_path,
            "n_frames": self.n_frames,
            "model_id": self.model_id,
            "token_mode": self.token_mode,
            "patch_hub_id": self.patch_hub_id,
            "patch_img_size": self.patch_img_size,
            "raw_img_size": self.raw_img_size,
            "frame_sampling": "linspace_index_v1",
        }

    def _load_backbone_cls(self):
        if self._backbone is None:
            from lift.dinov2 import DINOv2ForVideo, make_classification_eval_transform
            print(f"[FeatureProvider] lazily loading {self.model_id} (cls) for on-the-fly "
                  f"features (no precomputed/live-cache hit)")
            self._backbone = DINOv2ForVideo(model_id=self.model_id).to(self.device).eval()
            self._preprocess = make_classification_eval_transform()

    def _load_backbone_patch(self):
        if self._backbone is None:
            import torchvision.transforms as T
            print(f"[FeatureProvider] lazily loading {self.patch_hub_id} (patch, via torch.hub) "
                  f"for on-the-fly features (no precomputed/live-cache hit)")
            self._backbone = torch.hub.load("facebookresearch/dinov2", self.patch_hub_id)
            self._backbone = self._backbone.to(self.device).eval()
            self._preprocess = T.Compose([
                T.Resize((self.patch_img_size, self.patch_img_size)),
                T.ToTensor(),
                T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ])

    DENSE_SCHEMA_VERSION = 1

    def _dense_metadata(self, video_path: str, fps: float) -> dict:
        if self.token_mode == "cls":
            # New dict (cls dense extraction didn't exist before this), distinct enough in
            # content from the patch dict below that it can never hash-collide with it even
            # when both share one live_cache_dir -- no need to touch schema_version for this.
            return {
                "schema_version": self.DENSE_SCHEMA_VERSION,
                "video_path": video_path,
                "fps": fps,
                "token_mode": "cls",
                "model_id": self.model_id,
                "dtype": "float16",
                "frame_sampling": "time_floor_v1",
            }
        # patch -- UNCHANGED from this function's original (pre-cls-support) dict/schema_
        # version, so the already-extracted ego4dgoal dense cache (17k+ clips here alone)
        # keeps hashing to the same filenames and stays valid; do not add keys here.
        return {
            "schema_version": self.DENSE_SCHEMA_VERSION,
            "video_path": video_path,
            "fps": fps,
            "patch_hub_id": self.patch_hub_id,
            "patch_img_size": self.patch_img_size,
            "dtype": "float16",
            "frame_sampling": "time_floor_v1",
        }

    def _dense_cache_path(self, video_path: str, fps: float) -> Path:
        key = hashlib.sha256(json.dumps(
            self._dense_metadata(video_path, fps), sort_keys=True, separators=(",", ":")
        ).encode()).hexdigest()
        return self.live_cache_dir / f"dense_{key}.npy"

    def _compute_dense_patch(self, video_path: str, fps: float, out_path: Path, batch: int = 16):
        """Every frame of the clip at `fps` (frame k = the video frame at time k / fps from
        the clip start), patch tokens streamed batch by batch into a preallocated .npy so
        memory does not grow with clip length."""
        import numpy as np
        from decord import VideoReader, cpu
        from PIL import Image
        self._load_backbone_patch()
        vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
        video_fps, n_total = vr.get_avg_fps(), len(vr)
        n_out = int((n_total - 1) / video_fps * fps) + 1
        idx = np.minimum((np.arange(n_out) / fps * video_fps).astype(int), n_total - 1)
        n_patch = (self.patch_img_size // 14) ** 2
        out = np.lib.format.open_memmap(out_path, mode="w+", dtype=np.float16,
                                        shape=(n_out, n_patch, 384))
        for i in range(0, n_out, batch):
            frames = vr.get_batch(idx[i:i + batch]).asnumpy()
            x = torch.stack([self._preprocess(Image.fromarray(f)) for f in frames]).to(self.device)
            with torch.no_grad():
                tokens = self._backbone.forward_features(x)["x_norm_patchtokens"]
            out[i:i + len(frames)] = tokens.half().cpu().numpy()
        out.flush()

    def _compute_dense_cls(self, video_path: str, fps: float, out_path: Path, batch: int = 32):
        """Every frame of the clip at `fps`, CLS token streamed batch by batch into a
        preallocated .npy, mirroring _compute_dense_patch but for token_mode='cls'. Reuses
        DINOv2ForVideo's own forward (same call shape as compute_dino_features_for_single_
        video: (1, C, T, H, W) in, (T, D) CLS features out) so this is numerically identical
        to the existing n_frames-linspace cls extraction, just over every fps-spaced frame
        instead of a linspaced subset."""
        import numpy as np
        import einops
        from decord import VideoReader, cpu
        from PIL import Image
        self._load_backbone_cls()
        vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
        video_fps, n_total = vr.get_avg_fps(), len(vr)
        n_out = int((n_total - 1) / video_fps * fps) + 1
        idx = np.minimum((np.arange(n_out) / fps * video_fps).astype(int), n_total - 1)
        out = np.lib.format.open_memmap(out_path, mode="w+", dtype=np.float16,
                                        shape=(n_out, self._backbone.embed_dim))
        for i in range(0, n_out, batch):
            frames = vr.get_batch(idx[i:i + batch]).asnumpy()
            x = torch.stack([self._preprocess(Image.fromarray(f)) for f in frames]).to(self.device)  # (T,C,H,W)
            xin = einops.rearrange(x, "t c h w -> 1 c t h w")
            with torch.no_grad():
                z = self._backbone(xin)[0]  # (T, D)
            out[i:i + len(frames)] = z.half().cpu().numpy()
        out.flush()

    def get_dense(self, video_path: str, fps: float):
        """(L, D) [cls] or (L, N, D) [patch] float16 memmap: features for every frame of
        the clip at `fps`, live-cached per clip (computed once on first touch, then a plain
        file read), so a training window is just a slice."""
        import numpy as np
        if self.token_mode not in ("cls", "patch"):
            raise ValueError("get_dense is implemented for token_mode in ('cls', 'patch') only")
        if self.live_cache_dir is None:
            raise ValueError("get_dense needs a live cache dir: dense clips are too long to recompute per epoch")
        path = self._dense_cache_path(video_path, fps)
        meta_path = path.with_suffix(".json")
        with open(path.with_suffix(".lock"), "w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if path.exists():
                if json.load(open(meta_path)) != self._dense_metadata(video_path, fps):
                    raise RuntimeError(f"dense live cache metadata mismatch in {path}; remove it and recompute")
            else:
                tmp = path.with_suffix(".tmp.npy")
                if self.token_mode == "patch":
                    self._compute_dense_patch(video_path, fps, tmp)
                else:
                    self._compute_dense_cls(video_path, fps, tmp)
                json.dump(self._dense_metadata(video_path, fps), open(meta_path, "w"))
                tmp.rename(path)
        return np.load(path, mmap_mode="r")

    def _load_frames(self, video_path):
        from shared.utils.video import load_frames_linspace
        return load_frames_linspace(video_path, n=self.n_frames)

    def _compute_live_cls(self, video_path: str) -> torch.Tensor:
        from lift.dinov2 import compute_dino_features_for_single_video
        self._load_backbone_cls()
        _, feats = compute_dino_features_for_single_video(
            video_path, self._preprocess, self._backbone, n_frames=self.n_frames, device=self.device,
        )
        # feats is a view/slice into a much larger internal buffer (the model's full
        # patch+CLS+register token output) — torch.save on a view serializes the WHOLE
        # underlying storage, not just this slice, so .clone() here is load-bearing
        # (verified: 6144-element (16,384) tensor without it wrote a 6.4M-element file).
        return feats.cpu().clone()

    def _compute_live_patch(self, video_path: str) -> torch.Tensor:
        self._load_backbone_patch()
        frames = self._load_frames(video_path)
        x = torch.stack([self._preprocess(f) for f in frames]).to(self.device)  # (T,3,H,W)
        with torch.no_grad():
            out = self._backbone.forward_features(x)
        tokens = out["x_norm_patchtokens"]  # (T, N, D)
        return tokens.cpu().clone()

    def _compute_live_raw(self, video_path: str) -> torch.Tensor:
        """No backbone, no float conversion -- unlike _compute_live_cls/patch, which both
        end in a frozen DINOv2 forward pass, this just decodes and resizes. Stays uint8
        end-to-end (resize on the PIL image, then a raw HWC->CHW byte tensor) rather than
        going through T.ToTensor()'s float32 conversion, so the tier-2 live-cache this
        gets written to (see get()) is ~58GB rather than ~230GB for K400 10% -- 4 bytes/
        channel/pixel vs 1. float conversion + normalization is the pixel encoder's job
        (models/pixel_encoder.py), not this provider's."""
        import numpy as np
        import torchvision.transforms as T
        frames = self._load_frames(video_path)
        resize = T.Resize((self.raw_img_size, self.raw_img_size))
        out = [torch.from_numpy(np.array(resize(f.convert("RGB")))) for f in frames]  # (H,W,3) uint8
        return torch.stack(out).permute(0, 3, 1, 2).contiguous()  # (T, 3, H, W) uint8

    def _compute_live(self, video_path: str) -> torch.Tensor:
        if self.token_mode == "cls":
            return self._compute_live_cls(video_path)
        if self.token_mode == "raw":
            return self._compute_live_raw(video_path)
        return self._compute_live_patch(video_path)

    def get(self, video_path: str) -> torch.Tensor:
        """Returns (T, D) [cls] or (T, N, D) [patch] float32 features for one video,
        precomputed-first."""
        idx = self._path_to_idx.get(video_path)
        if idx is not None:
            return self._bulk_features[idx]

        if self.live_cache_dir is not None:
            cache_file = self._live_cache_path(video_path)
            if cache_file.exists():
                cached = torch.load(cache_file, map_location="cpu")
                expected = self._live_cache_metadata(video_path)
                if not isinstance(cached, dict) or cached.get("metadata") != expected:
                    raise RuntimeError(
                        f"live cache metadata mismatch in {cache_file}; remove that entry "
                        "and recompute it"
                    )
                return cached["features"]

        feats = self._compute_live(video_path)
        if self.live_cache_dir is not None:
            # Write-then-rename: a reader (another worker, rank or job) must never see a
            # half-written file, which is what corrupted shared caches before.
            final = self._live_cache_path(video_path)
            tmp = final.with_name(f"{final.name}.{os.getpid()}.tmp")
            torch.save({"features": feats, "metadata": self._live_cache_metadata(video_path)}, tmp)
            os.replace(tmp, final)
        return feats
