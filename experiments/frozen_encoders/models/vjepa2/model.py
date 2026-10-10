"""V-JEPA2 (video transformer, spacetime patches, no CLS token) plugin.

Genuinely different architecture from the plain-ViT image encoders: it consumes a whole
clip (`config.frames_per_clip` frames, e.g. 64) at once and produces one token per
spacetime "tubelet" (`tubelet_size` consecutive frames x patch x patch), not one token
per frame — and that frame count is architecturally fixed (baked into the model's
positional embeddings), not something a caller can shrink by passing fewer frames in.

Two ways to turn this into per-frame embeddings, both implemented here:

per_frame=True (default): TRUE per-frame isolation. Each frame is independently
replicated into its own fake fpc-length "clip" (every frame in it identical), run
through the model, mean-pooled over the now-redundant temporal/tubelet axis, then pooled
over spatial patches via `pooling`. Since every frame the model's self-attention can see
is a copy of the SAME image, no information from any other real frame can reach this
frame's output — genuine per-frame independence, at the cost of one full forward pass
per frame instead of one per clip. Batched (per_frame_batch_size frames/forward pass, not
one Python-loop iteration each) to actually use the GPU's parallelism for this — verified
this matters: unbatched was ~2s/frame serially, which is 107+ GPU-hours for SSv2 train
alone.

per_frame=False: the original clip-based mode — your whole frame list is resampled onto
the model's real fpc-length clip (nearest-neighbor upsampling, e.g. 16->64), producing
n_temporal = fpc // tubelet_size embeddings (e.g. 32) that each genuinely mix information
from several nearby original frames via real self-attention over the resampled clip.
Cheap (one forward pass total), but not one embedding per input frame and not
per-frame-independent.

Attention: there's no CLS token, so there's no single "what is the embedding looking at"
query the way there is for DINO. Instead, for the last block we take the intra-tubelet
attention block (each tubelet's spatial tokens attending to each other) and average how
much attention each patch *receives* from the rest of its own tubelet — the video
analogue of the DINO CLS->patch map. We hook just the last layer directly (instead of
`output_attentions=True`, which would accumulate all layers' seq x seq matrices at once)
and skip the predictor submodule (`skip_predictor=True`), unused for embeddings and
otherwise the actual OOM risk, not the encoder itself.
"""
from typing import List

import numpy as np
import torch
from PIL import Image
from transformers import AutoVideoProcessor, VJEPA2Model

from common.base import EncodingResult, FrozenEncoder, register
from common.pooling import pool_patch_tokens


@register
class VJEPA2Encoder(FrozenEncoder):
    name = "vjepa2"
    supports_attention = True
    attention_label = "intra-tubelet attention received (avg. over heads/queries)"
    supports_patch_tokens = True
    pooling_choices = ["mean_patch", "max_patch"]

    def __init__(self, hf_id: str = "facebook/vjepa2-vitl-fpc64-256", block_idx: int = -1,
                 per_frame: bool = True, per_frame_batch_size: int = 8,
                 capture_attention: bool = True, **kwargs):
        kwargs.setdefault("pooling", "mean_patch")
        super().__init__(**kwargs)
        self.hf_id = hf_id
        self.block_idx = block_idx
        self.per_frame = per_frame
        self.per_frame_batch_size = per_frame_batch_size
        # attn_implementation="eager" is ONLY needed to make attention weights readable
        # via the forward hook below -- but it also forces the model to materialize the
        # full (B, heads, seq, seq) attention matrix explicitly as an intermediate
        # tensor, REGARDLESS of whether anything actually reads it out. That's what was
        # OOMing at batch_size>1 (a single fpc=64 clip's own attention matrix already
        # used ~18GB; batching multiplied it linearly). precompute_features.py never
        # touches EncodingResult.attention_maps at all -- only run.py's interactive
        # visualization does -- so capture_attention=False (skip attention entirely,
        # load with the efficient default/sdpa implementation instead) is what actually
        # makes per_frame batching usable at scale.
        self.capture_attention = capture_attention

    def load(self) -> None:
        self.processor = AutoVideoProcessor.from_pretrained(self.hf_id)
        attn_impl = "eager" if self.capture_attention else None
        self.model = VJEPA2Model.from_pretrained(self.hf_id, attn_implementation=attn_impl)
        self.model.to(self.device).eval()

    @torch.no_grad()
    def encode(self, frames: List[Image.Image]) -> EncodingResult:
        if self.per_frame:
            return self._encode_per_frame(frames)
        return self._encode_clip(frames)

    def _tubelet_forward_batch(self, clips: List[List[Image.Image]], capture_attn: bool):
        """One model forward pass over a BATCH of fpc-length clips. Returns
        (tokens, grid, attn_maps): tokens is (B, n_temporal, P, D), attn_maps is
        (B, n_temporal, grid, grid) or None if capture_attn=False or
        self.capture_attention=False (the model wasn't even loaded with eager attention
        in that case, so there's nothing to hook regardless of what's requested here)."""
        capture_attn = capture_attn and self.capture_attention
        inputs = self.processor(clips, return_tensors="pt").to(self.device)
        cfg = self.model.config
        n_temporal = cfg.frames_per_clip // cfg.tubelet_size
        b = len(clips)

        captured = {}
        handle = None
        if capture_attn:
            def _hook(module, inp, out):
                captured["attn"] = out[1].detach()  # (B, num_heads, seq, seq)
            attn_module = self.model.encoder.layer[self.block_idx].attention
            handle = attn_module.register_forward_hook(_hook)
        try:
            out = self.model(**inputs, skip_predictor=True)
        finally:
            if handle is not None:
                handle.remove()

        total_tokens = out.last_hidden_state.shape[1]
        grid = int(round((total_tokens / n_temporal) ** 0.5))
        tokens = out.last_hidden_state.reshape(b, n_temporal, grid * grid, -1)  # (B, n_temporal, P, D)

        attn_maps = None
        if capture_attn:
            attn = captured["attn"].mean(dim=1)  # (B, seq, seq)
            p = grid * grid
            attn_maps = np.stack([
                np.stack([
                    attn[bi, t_ * p:(t_ + 1) * p, t_ * p:(t_ + 1) * p].mean(dim=0).reshape(grid, grid).cpu().numpy()
                    for t_ in range(n_temporal)
                ])
                for bi in range(b)
            ])
        return tokens, grid, attn_maps

    @torch.no_grad()
    def _encode_per_frame(self, frames: List[Image.Image]) -> EncodingResult:
        """Each frame replicated into its own fake fpc-length clip and encoded in
        isolation — see module docstring. Batched per_frame_batch_size frames/forward
        pass rather than one Python-loop iteration each."""
        cfg = self.model.config
        fpc = cfg.frames_per_clip
        embeddings, attn_maps = [], []
        for start in range(0, len(frames), self.per_frame_batch_size):
            chunk = frames[start:start + self.per_frame_batch_size]
            clips = [[f] * fpc for f in chunk]
            tokens, grid, chunk_attn = self._tubelet_forward_batch(clips, capture_attn=True)  # (b,n_temporal,P,D)
            tokens = tokens.mean(dim=1)  # (b, P, D): collapse the (redundant, since every
                                          # clip frame is identical) temporal/tubelet axis
            pooled = pool_patch_tokens(tokens, self.pooling)  # (b, D)
            embeddings.append(pooled.cpu().numpy())
            if chunk_attn is not None:
                attn_maps.append(chunk_attn.mean(axis=1))  # (b, grid, grid), same redundant-axis average
        return EncodingResult(
            embeddings=np.concatenate(embeddings, axis=0),
            frame_indices=list(range(len(frames))),
            attention_maps=np.concatenate(attn_maps, axis=0) if attn_maps else None,
        )

    @torch.no_grad()
    def _encode_clip(self, frames: List[Image.Image]) -> EncodingResult:
        """Original clip-based mode: one forward pass over the WHOLE resampled clip,
        producing n_temporal embeddings that each mix real information from several
        nearby original frames. See module docstring for the tradeoff vs per_frame=True."""
        cfg = self.model.config
        fpc, tubelet = cfg.frames_per_clip, cfg.tubelet_size

        t = len(frames)
        clip_idx = np.linspace(0, t - 1, fpc).round().astype(int)
        clip = [frames[i] for i in clip_idx]

        tokens, grid, attn_maps = self._tubelet_forward_batch([clip], capture_attn=True)
        tokens = tokens[0]  # drop the batch-of-1 dim
        attn_maps = attn_maps[0] if attn_maps is not None else None
        pooled = pool_patch_tokens(tokens, self.pooling)  # (n_temporal, D)

        n_temporal = fpc // tubelet
        # Each temporal tubelet covers clip frames [t*tubelet, t*tubelet+tubelet-1] ->
        # map the tubelet's center back to a position in the *input* frames list.
        frame_indices = [int(clip_idx[min(t_ * tubelet + tubelet // 2, fpc - 1)]) for t_ in range(n_temporal)]

        return EncodingResult(
            embeddings=pooled.cpu().numpy(),
            frame_indices=frame_indices,
            attention_maps=attn_maps,
        )

    @torch.no_grad()
    def get_patch_tokens(self, frames: List[Image.Image]):
        """One patch-token grid per *input* frame. Respects self.per_frame the same way
        encode() does.

        per_frame=True: each frame's own grid, from its own isolated fake-clip forward
        pass (temporal axis mean-pooled first, same as _encode_per_frame), batched.
        per_frame=False: V-JEPA2's patch embedding (Conv3d, kernel/stride (tubelet, 16,
        16)) merges every tubelet_size consecutive frames into a single spacetime token
        before the transformer ever runs -- there's no way to recover separate tokens for
        the frames inside one tubelet after the fact, so each frame gets its *tubelet's*
        grid, duplicated across every frame in that tubelet.
        """
        cfg = self.model.config
        fpc, tubelet = cfg.frames_per_clip, cfg.tubelet_size

        if self.per_frame:
            per_frame_tokens = []
            grid = None
            for start in range(0, len(frames), self.per_frame_batch_size):
                chunk = frames[start:start + self.per_frame_batch_size]
                clips = [[f] * fpc for f in chunk]
                tokens, grid, _ = self._tubelet_forward_batch(clips, capture_attn=False)  # (b,n_temporal,P,D)
                per_frame_tokens.append(tokens.mean(dim=1))  # (b, P, D)
            return torch.cat(per_frame_tokens, dim=0), (grid, grid)

        t = len(frames)
        clip_idx = np.linspace(0, t - 1, fpc).round().astype(int)
        clip = [frames[i] for i in clip_idx]
        tokens, grid, _ = self._tubelet_forward_batch([clip], capture_attn=False)
        tokens = tokens[0]  # (n_temporal, P, D)

        n_temporal = fpc // tubelet
        nearest_clip_pos = np.array([int(np.argmin(np.abs(clip_idx - i))) for i in range(t)])
        tubelet_idx = np.minimum(nearest_clip_pos // tubelet, n_temporal - 1)
        per_frame_tokens = tokens[tubelet_idx]  # (T, P, D), tubelets duplicated across their frame pairs
        return per_frame_tokens, (grid, grid)
