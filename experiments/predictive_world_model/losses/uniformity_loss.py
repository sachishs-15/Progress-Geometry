"""Hyperspherical uniformity (Wang & Isola, "Understanding Contrastive Representation
Learning through Alignment and Uniformity on the Hypersphere", ICML 2020) --
a sphere-native anti-collapse regularizer:

    L_uniform = log E_{x,y} [ exp(-t * ||x - y||^2) ]

exp(-t*d^2) is a Gaussian kernel: near 1 for close pairs, ~0 for distant ones. Averaging
it over pairs scores how crowded the point cloud is, and minimizing it makes every point
repel every other -- a potential energy whose unique minimizer on the hypersphere is the
uniform distribution. That is the paper's result and the reason this is a principled
"use the whole space" objective rather than a heuristic.

Why this exists alongside vicreg_std/vicreg_cov in world_model_loss.py, all three
measured on this project's own data:

  1. VICReg's std term is SATISFIABLE BY SCALE. It is relu(1 - std) on unnormalized z, so
     it hits exactly 0 (zero gradient) once per-dimension std >= 1, and nothing then
     resists further inflation. checkpoints/pwm_cls_mlp_k400 rode that loophole: ||z||
     20 -> 146 while the penalty fell to 0.003. Uniformity is scale-free for unit vectors
     (see the identity below), so there is no equivalent move.
  2. VICReg's std term is NEARLY BLIND ON UNIT-NORM DATA. Sweeping an ideal spread to
     total collapse it moves only 0.948 -> 0.990, because after L2 projection every
     coordinate has variance ~1/d however bunched the directions are. Uniformity spans
     -3.99 -> 0.00 over the same sweep.
  3. VICReg's cov term has a d/N NOISE FLOOR. It builds a dxd covariance from N samples,
     so off-diagonals carry ~1/N of sampling noise each and the loss bottoms out at ~d/N
     (predicted 384/512 = 0.750, measured 0.7504). At d=384/batch=32 it stalled at 4.0
     against that floor while consuming 21% of the objective and pushing prediction loss
     UP. Uniformity never forms a dxd matrix -- it is a Monte-Carlo estimate over pairs --
     so it has no such floor: on iid data it sits at -3.7..-4.0 across d in {32, 384} and
     N in {512, 8192}.

UNIT-NORM ASSUMPTION. The theory holds for points ON the sphere. For unit vectors
||x-y||^2 = 2 - 2cos(x,y), so the kernel is exp(-2t)*exp(2t*cos) -- purely a function of
angle, hence scale-free. Off the sphere that identity fails and the loss becomes
minimizable by simply inflating ||z|| to push points apart in Euclidean space, which is
exactly the failure mode item 1 describes. This function therefore L2-normalizes its
input itself rather than trusting the caller to have set norm_type="hypersphere": the
alternative is a regularizer that silently measures the wrong thing and rewards the scale blowup it
was added to prevent. Normalizing is a no-op when the encoder already emits unit vectors.
"""
import numpy as np
import torch


def uniformity_loss(z, t=2.0, n_pairs=2048, eps=1e-8, generator=None):
    """z: (..., D) -- flattened to (N, D) internally, one row per frame/token.

    Returns log(mean(exp(-t * ||xi - xj||^2))) over `n_pairs` uniformly sampled pairs,
    computed via logsumexp for stability. Range: 0 at exact collapse (every pair at
    distance 0), decreasing as the cloud spreads.

    Sampling `n_pairs` rather than all N^2 pairs keeps this O(n_pairs) instead of O(N^2);
    the estimator is unbiased in the kernel mean and its noise averages out over steps.
    One consequence: repeated calls on identical input differ slightly.
    """
    x = z.reshape(-1, z.shape[-1])
    x = x / (x.norm(dim=-1, keepdim=True) + eps)  # see UNIT-NORM ASSUMPTION above
    n = x.shape[0]
    if n < 2:
        return x.new_zeros(())
    i = torch.randint(0, n, (n_pairs,), device=x.device, generator=generator)
    # j = i + offset in [1, n-1] (mod n), NOT an independent randint: independent draws
    # hit i == j with probability 1/n, and a self-pair has distance 0 so it contributes
    # exp(0) = 1, the maximum possible, against a typical pair's ~exp(-4) = 0.018. That
    # single outlier per ~n samples biases the kernel mean UP by roughly
    # log(1 + (1/n)/mean_kernel) -- measured +0.086 at n=512 and +0.039 at n=2048 against
    # the reference. The reference (SsnL/align_uniform) never sees this because
    # torch.pdist enumerates i<j only. This offset guarantees j != i while staying
    # uniform over the other n-1 elements.
    offset = torch.randint(1, n, (n_pairs,), device=x.device, generator=generator)
    j = (i + offset) % n
    sq_dist = (x[i] - x[j]).pow(2).sum(dim=-1)
    return torch.logsumexp(-t * sq_dist, dim=0) - np.log(n_pairs)
