"""Representation-learning heads that sit on top of the shared `FeatureEncoder`.

`OrderingTransformer` is the shuffle/temporal-ordering objective (section 6-8): it
deliberately receives NO positional encoding on its input tokens — the K encoded features
are fed to it as an unordered set, so the only way it can recover their original order is
from what the shared encoder put into their content. Adding positions would make the task
trivial and defeat the point of the objective.

`TimestampHead` is the control (section 9): a plain per-timestep linear regression head,
with no cross-timestep interaction at all, so it can supervise the same encoder with
"some" temporal signal without being able to use relational/ordering information the way
the transformer can.
"""
import torch
import torch.nn as nn


class OrderingTransformer(nn.Module):
    def __init__(self, latent_dim: int = 512, k: int = 8, num_layers: int = 2, num_heads: int = 8,
                 ff_dim: int = 1024, dropout: float = 0.1):
        super().__init__()
        self.k = k
        layer = nn.TransformerEncoderLayer(
            d_model=latent_dim, nhead=num_heads, dim_feedforward=ff_dim,
            dropout=dropout, batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.rank_head = nn.Linear(latent_dim, k)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """z: [B, K, latent_dim] (already shuffled) -> logits [B, K, K], logits[b, i, :]
        is the predicted distribution over original positions for shuffled slot i."""
        h = self.transformer(z)
        return self.rank_head(h)


class TimestampHead(nn.Module):
    def __init__(self, latent_dim: int = 512):
        super().__init__()
        self.linear = nn.Linear(latent_dim, 1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """z: [..., latent_dim] -> [...] predicted normalized timestamp, independently
        per timestep (no cross-timestep interaction)."""
        return self.linear(z).squeeze(-1)
