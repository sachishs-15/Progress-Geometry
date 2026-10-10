"""Token-level frame input for LSTR: (B, T, 1 + N, 384) CLS + patch tokens -> (B, T, d_model).

The pooling happens here, inside the model, so it runs online during training on whatever
the live cache holds. MODEL.TOKEN_HEAD.MODE:
  cls    CLS token only (sanity check against the CLS-only pipeline)
  avg    CLS + the patch grid average-pooled to GRID x GRID (GRID must divide the grid side)
  query  CLS + NUM_QUERIES learned queries cross-attending over the patch tokens
Each frame's tokens are concatenated and mapped by a linear + ReLU to d_model, as
LSTR's own BaseFeatureHead does for a single vector.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from yacs.config import CfgNode as CN

DIM = 384


def add_defaults(defaults):
    defaults._C.MODEL.TOKEN_HEAD = CN()
    defaults._C.MODEL.TOKEN_HEAD.MODE = "avg"
    defaults._C.MODEL.TOKEN_HEAD.GRID = 2
    defaults._C.MODEL.TOKEN_HEAD.NUM_PATCHES = 196
    defaults._C.MODEL.TOKEN_HEAD.NUM_QUERIES = 4
    defaults._C.MODEL.TOKEN_HEAD.NUM_HEADS = 8


class TokenFeatureHead(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        h = cfg.MODEL.TOKEN_HEAD
        self.mode, self.grid = h.MODE, h.GRID
        self.side = int(round(h.NUM_PATCHES ** 0.5))
        assert self.side ** 2 == h.NUM_PATCHES
        if self.mode == "cls":
            n_tok = 1
        elif self.mode == "avg":
            assert self.side % self.grid == 0, f"grid {self.grid} does not divide {self.side}"
            n_tok = 1 + self.grid ** 2
        elif self.mode == "query":
            self.queries = nn.Parameter(torch.randn(h.NUM_QUERIES, DIM) * 0.02)
            self.attn = nn.MultiheadAttention(DIM, h.NUM_HEADS, batch_first=True)
            n_tok = 1 + h.NUM_QUERIES
        else:
            raise ValueError(f"unknown TOKEN_HEAD.MODE {self.mode!r}")
        self.d_model = cfg.MODEL.FEATURE_HEAD.LINEAR_OUT_FEATURES
        self.input_linear = nn.Sequential(nn.Linear(n_tok * DIM, self.d_model), nn.ReLU(inplace=True))

    def forward(self, visual_input, motion_input=None):
        B, T = visual_input.shape[:2]
        cls = visual_input[:, :, 0]
        tokens = [cls[:, :, None]]
        if self.mode == "avg":
            p = visual_input[:, :, 1:].reshape(B * T, self.side, self.side, DIM).permute(0, 3, 1, 2)
            p = F.adaptive_avg_pool2d(p, self.grid).flatten(2).transpose(1, 2)
            tokens.append(p.reshape(B, T, -1, DIM))
        elif self.mode == "query":
            kv = visual_input[:, :, 1:].reshape(B * T, -1, DIM)
            q = self.queries.unsqueeze(0).expand(B * T, -1, -1)
            out, _ = self.attn(q, kv, kv)
            tokens.append(out.reshape(B, T, -1, DIM))
        return self.input_linear(torch.cat(tokens, dim=2).flatten(2))
