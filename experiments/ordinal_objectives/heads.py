"""The small trainable heads objectives put on top of the encoder's per-frame latents.

Nothing here is the thing under test. The encoder (pwm_bridge.ClsMlpEncoder) is held
fixed across every objective and IS what gets evaluated; these heads exist only because
most objectives cannot express their loss on raw latents alone -- a progress regressor
needs a scalar readout, a sorting objective needs something that can look at several
frames at once. They are kept deliberately small and are NOT saved into the part of the
checkpoint that progress-metric evaluation reads (see train.py's save_checkpoint), since
a bigger head would let an objective "succeed" while leaving the latent space untouched,
which is the exact failure mode this comparison is meant to detect.

One consequence worth stating plainly: because the encoder is per-frame, an
order-sensitive objective (shuffle/sort, arrow of time, pace) can only shape the latent
space through what its SEQUENCE head can read off a bag of independently-computed frame
latents. That is the correct setup for this question -- it asks whether the ordering
signal forces per-frame latents into an ordered arrangement -- but it does mean those
objectives have a structurally harder job than the per-frame ones, and their head
capacity (d_model/num_layers) is a real confound. The configs hold it fixed WITHIN that
family (every sequence head is d_model=256, num_layers=2), but head size cannot be held
fixed ACROSS families, and the spread is large:

    rank_n_contrast / scl / tcn / tcc / vip        0 params (loss applied to z directly)
    progress_regression / ordinal_* / pairwise_rank ~150K
    shuffle_* / arrow_of_time / pace               ~1.68M
    next_step (the world-model control's predictor) ~3.55M

against an encoder of only ~296K. This is inherent -- a sorting objective needs a
sequence model and a contrastive one needs nothing -- but it is not neutral: an objective
with a 1.7M-parameter head can drive its loss down with far less help from the encoder
than one with no head at all, which is the "the head solved it, the latent space did not"
failure this project exists to detect, distributed unevenly across the table. Two things
keep it honest, and neither should be removed: evaluation never sees the head (train.py's
save_checkpoint puts it outside the scored state dict), and train.py prints and stores the
head parameter count so it can be read alongside every row. --head-lr is the knob if a
large head needs slowing down.
"""
import torch
import torch.nn as nn


def mlp(in_dim: int, out_dim: int, hidden_dim=None, depth: int = 2, dropout: float = 0.0):
    """depth=1 is a bare nn.Linear (no hidden layer); depth>=2 stacks (depth-1) GELU
    hidden layers of width hidden_dim (default: in_dim)."""
    if depth < 1:
        raise ValueError(f"depth must be >= 1, got {depth}")
    if depth == 1 and dropout > 0:
        # depth=1 is a bare Linear with nowhere to put a dropout layer; accepting the
        # argument and ignoring it would make a config look like it had regularization
        # that was never applied.
        raise ValueError("depth=1 is a bare nn.Linear and cannot apply dropout; "
                         "use depth >= 2 or dropout=0")
    hidden_dim = hidden_dim or in_dim
    if depth == 1:
        return nn.Linear(in_dim, out_dim)
    layers = [nn.Linear(in_dim, hidden_dim), nn.GELU()]
    for _ in range(depth - 2):
        layers += [nn.Linear(hidden_dim, hidden_dim), nn.GELU()]
    if dropout > 0:
        layers.append(nn.Dropout(dropout))
    layers.append(nn.Linear(hidden_dim, out_dim))
    return nn.Sequential(*layers)


class FrameHead(nn.Module):
    """Per-frame readout: (B, T, d) -> (B, T, out_dim), shared weights across frames and
    with no interaction between them. Used by every objective whose supervision is a
    per-frame quantity (progress regression, ordinal classification, pairwise ranking)."""

    def __init__(self, latent_dim: int, out_dim: int = 1, hidden_dim=None, depth: int = 2,
                 dropout: float = 0.0):
        super().__init__()
        self.net = mlp(latent_dim, out_dim, hidden_dim=hidden_dim, depth=depth, dropout=dropout)

    def forward(self, z):
        return self.net(z)


class CoralHead(nn.Module):
    """CORAL (Cao et al., 'Rank consistent ordinal regression'): num_bins-1 binary
    "is progress past threshold k?" logits computed from ONE shared scalar projection
    plus a per-threshold bias.

    The shared weight vector is the whole point. Following the paper and its official
    coral-pytorch layer, threshold biases start in descending order; together with the
    shared scalar and CORAL loss, this is the construction used for rank-monotonic
    cumulative probabilities. K independent binary classifiers provide no such guarantee.
    For this project that matters twice
    over -- it is also the most direct way an objective can ask for a single 1-D ordinal
    AXIS in latent space rather than K separate directions.
    """

    def __init__(self, latent_dim: int, num_bins: int, hidden_dim=None, depth: int = 2,
                 dropout: float = 0.0):
        super().__init__()
        self.num_bins = num_bins
        self.trunk = mlp(latent_dim, 1, hidden_dim=hidden_dim, depth=depth, dropout=dropout)
        # Official CORAL initialization: descending values in (0, 1]. The paper's
        # ordered-bias result assumes ordered initialization; all-zero biases made every
        # threshold identical initially and discarded that part of the construction.
        self.bias = nn.Parameter(
            torch.arange(num_bins - 1, 0, -1, dtype=torch.float32) / (num_bins - 1)
        )

    def forward(self, z):
        """(B, T, d) -> (B, T, num_bins-1) cumulative logits."""
        return self.trunk(z) + self.bias


class SequenceHead(nn.Module):
    """A small bidirectional Transformer over a SET of frame latents, with a learned
    positional embedding over the position each frame occupies in the (possibly shuffled)
    input sequence.

    output="token":  (B, K, num_classes) -- one prediction per input position. Used by
                     the sorting objective, which predicts each shuffled position's
                     original rank.
    output="pooled": (B, num_classes) -- one prediction per sequence, read off a prepended
                     learned CLS token. Used by order verification, arrow of time, and
                     pace prediction.

    The positional embedding is over INPUT position, never over the frame's true time
    index -- feeding the true index would hand the objective its own answer.
    """

    def __init__(self, latent_dim: int, num_classes: int, d_model: int = 256, nhead: int = 4,
                 num_layers: int = 2, dropout: float = 0.1, max_len: int = 16,
                 output: str = "pooled"):
        super().__init__()
        if output not in ("token", "pooled"):
            raise ValueError(f"output must be 'token' or 'pooled', got {output!r}")
        self.output = output
        self.input_proj = nn.Linear(latent_dim, d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_len + 1, d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4, dropout=dropout,
            batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers,
                                                 norm=nn.LayerNorm(d_model))
        self.out = nn.Linear(d_model, num_classes)

    def forward(self, z):
        """z: (B, K, latent_dim) in whatever order the objective arranged them."""
        b, k, _ = z.shape
        limit = self.pos_embed.size(1) - (1 if self.output == "pooled" else 0)
        if k > limit:
            # Otherwise this surfaces as a bare broadcast error from the positional add
            # ("size of tensor a (21) must match tensor b (17)"), which says nothing about
            # which knob is wrong.
            raise ValueError(f"SequenceHead got {k} frames but was built for at most {limit} "
                             f"(max_len={self.pos_embed.size(1) - 1}); raise max_len or lower "
                             f"the objective's num_frames / the clip length")
        x = self.input_proj(z)
        if self.output == "pooled":
            x = torch.cat([self.cls_token.expand(b, -1, -1), x], dim=1)
        x = x + self.pos_embed[:, :x.size(1)]
        x = self.transformer(x)
        if self.output == "pooled":
            return self.out(x[:, 0])
        return self.out(x)


class ProjectionHead(nn.Module):
    """Optional MLP the contrastive objectives (RnC, TCN, SCL, TCC) can compute their
    loss in instead of directly on z.

    Defaults to OFF everywhere here, which is the opposite of the SimCLR-lineage
    convention these losses come from, and deliberately so: a projection head exists to
    let the backbone keep information the contrastive loss would otherwise discard, i.e.
    to DECOUPLE the loss's geometry from the backbone's. This project measures the
    backbone's geometry, so decoupling it is precisely what we do not want by default.
    Exposed anyway because "does this objective only order the space when it is allowed a
    projection head?" is a real question about each objective, and the papers' own
    reported settings use one.
    """

    def __init__(self, latent_dim: int, out_dim: int = 128, hidden_dim=None):
        super().__init__()
        self.net = mlp(latent_dim, out_dim, hidden_dim=hidden_dim or latent_dim, depth=2)

    def forward(self, z):
        return self.net(z)


def maybe_projection(latent_dim: int, project: bool, proj_dim: int = 128, proj_hidden=None):
    """ProjectionHead or nn.Identity, so callers never branch on `project` at forward
    time."""
    return ProjectionHead(latent_dim, out_dim=proj_dim, hidden_dim=proj_hidden) if project else nn.Identity()
