"""Dense per-video DINOv2 CLS + patch tokens behind a live cache.

Follows predictive_world_model/data/feature_provider.py's tier-2 convention: one file per
video in --cache-dir, named by a hash of every extraction setting, validated against its
stored metadata on a hit, computed on a miss and written before returning. Same backbone
and preprocessing as that provider's token_mode='patch' (torch.hub dinov2_vits14, no
registers, squash-resize to img_size, ImageNet normalization), but over a whole video at
one feature per `stride` frames instead of 16 evenly spaced frames per clip.

get(uid) -> np.memmap (L, 1 + N, 384) float16, token 0 = CLS, tokens 1: = the N = (img_size
/ 14) ** 2 patch tokens (196 at 196px), so a training window reads only its own rows. This
is the token set ProVideLLM feeds its connector (CLS + patch grid).
"""
import fcntl
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from goalstep_common import STRIDE, VIDEOS


class DensePatchProvider:
    SCHEMA_VERSION = 2

    def __init__(self, cache_dir, patch_hub_id="dinov2_vits14", img_size=196, stride=STRIDE,
                 batch=16, device="cuda"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.patch_hub_id, self.img_size, self.stride = patch_hub_id, img_size, stride
        self.batch, self.device = batch, device
        self._backbone = None

    def _metadata(self, uid):
        return {"schema_version": self.SCHEMA_VERSION, "video_uid": uid, "stride": self.stride,
                "patch_hub_id": self.patch_hub_id, "patch_img_size": self.img_size,
                "tokens": "cls+patch", "dtype": "float16", "frame_sampling": "arange_stride_v1"}

    def _path(self, uid):
        key = hashlib.sha256(json.dumps(self._metadata(uid), sort_keys=True,
                                        separators=(",", ":")).encode()).hexdigest()
        return self.cache_dir / f"{key}.npy"

    def cached_path(self, uid):
        """The cache file for uid if it exists, else None. Never computes."""
        path = self._path(uid)
        return path if path.exists() else None

    def _load_backbone(self):
        if self._backbone is None:
            import torchvision.transforms as T
            self._backbone = torch.hub.load("facebookresearch/dinov2", self.patch_hub_id)
            self._backbone = self._backbone.to(self.device).eval()
            self._preprocess = T.Compose([
                T.Resize((self.img_size, self.img_size)),
                T.ToTensor(),
                T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ])

    def _compute(self, uid, out_path):
        """Stream tokens batch by batch into a preallocated .npy so memory stays flat
        however long the video is (accumulating then concatenating held 2x a whole
        video's tokens, several GB for the longest ones)."""
        from decord import VideoReader, cpu
        from PIL import Image
        self._load_backbone()
        vr = VideoReader(str(VIDEOS / f"{uid}.mp4"), ctx=cpu(0), num_threads=1)
        idx = np.arange(0, len(vr), self.stride)
        n_tok = 1 + (self.img_size // 14) ** 2
        out = np.lib.format.open_memmap(out_path, mode="w+", dtype=np.float16,
                                        shape=(len(idx), n_tok, 384))
        for i in range(0, len(idx), self.batch):
            frames = vr.get_batch(idx[i:i + self.batch]).asnumpy()
            x = torch.stack([self._preprocess(Image.fromarray(f)) for f in frames]).to(self.device)
            with torch.no_grad():
                f = self._backbone.forward_features(x)
                tokens = torch.cat([f["x_norm_clstoken"][:, None], f["x_norm_patchtokens"]], dim=1)
            out[i:i + len(frames)] = tokens.half().cpu().numpy()
        out.flush()
        del out

    def get(self, uid):
        path = self._path(uid)
        meta_path = path.with_suffix(".json")
        with open(path.with_suffix(".lock"), "w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if path.exists():
                if json.load(open(meta_path)) != self._metadata(uid):
                    raise RuntimeError(f"live cache metadata mismatch in {path}; remove it and recompute")
            else:
                self._compute(uid, path.with_suffix(".tmp.npy"))
                json.dump(self._metadata(uid), open(meta_path, "w"))
                path.with_suffix(".tmp.npy").rename(path)
        return np.load(path, mmap_mode="r")
