"""Pooling strategies to turn patch tokens into one embedding per frame/clip-step."""
import torch


def pool_patch_tokens(patch_tokens: torch.Tensor, method: str, cls_token: torch.Tensor = None) -> torch.Tensor:
    """
    Args:
        patch_tokens: (N, P, D) patch token embeddings (no CLS/register tokens).
        method: one of "mean_patch", "max_patch", "cls".
        cls_token: (N, D), required if method == "cls".

    Returns:
        (N, D) pooled embeddings.
    """
    if method == "mean_patch":
        return patch_tokens.mean(dim=1)
    if method == "max_patch":
        return patch_tokens.max(dim=1).values
    if method == "cls":
        if cls_token is None:
            raise ValueError("cls pooling requires cls_token")
        return cls_token
    raise ValueError(f"Unknown pooling method: {method}")
