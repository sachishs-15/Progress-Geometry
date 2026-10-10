"""Recover the temporal order of a shuffled set of frames -- the oldest and most-cited
family of temporal self-supervision, and the one already tried in
experiments/temporal_ordering/ (on a different data path and a different probe, so its
numbers are not comparable to this project's; re-implemented here rather than imported,
so it runs under the identical encoder/optimiser/evaluation as everything else).

Three variants spanning the family's history:

  "verify"       Misra et al., "Shuffle and Learn" (ECCV 2016): binary -- is this
                 three-frame tuple in correct temporal order or not? From five ordered
                 frames a<b<c<d<e, their positive is (b,c,d) and their negatives are
                 (b,a,d) and (b,e,d); every tuple may also be reversed. Thus the objective
                 asks for monotonicity, not direction. The 25% positive class ratio used
                 for their final model is the default here.
  "permutation"  Lee et al.'s Order Prediction Network / Xu et al.'s VCOP: classify WHICH
                 of the k! permutations was applied. Much stronger supervision per
                 example, but only tractable for small k (k=4 -> 24 classes; k=8 ->
                 40320, which is why nobody uses it there).
  "sort"         Yang, Xie & Zisserman, "Made to Order" (ECCV 2024, arXiv:2404.16828):
                 a transformer over the shuffled sequence predicts each position's
                 original rank, so it scales to any k. Their central claim is exactly this
                 project's question -- only changes that are MONOTONIC with time can
                 produce a correct ordering, so a model trained to order is a model forced
                 to find monotonic structure. Default here for that reason.

All three read the frames through the same heads.SequenceHead, differing only in what it
predicts, so head capacity is held fixed across the variants.

Note what the encoder's per-frame structure implies: the shuffling cannot be undone by
the encoder (it sees one frame at a time, in no order), so the entire ordering signal has
to arrive through the latent geometry the head can read. That is the point, but it also
means this family is doing something structurally different from the per-frame objectives
-- it constrains the SET of latents, not each one individually.
"""
import itertools

import torch
import torch.nn.functional as F

from heads import SequenceHead

from .base import Objective, gather_frames, random_permutations, sorted_subset_indices


class ShuffleSort(Objective):
    def __init__(self, latent_dim, num_frames=8, variant="sort", d_model=256, nhead=4,
                 num_layers=2, dropout=0.1, max_permutation_frames=5,
                 positive_fraction=0.25):
        super().__init__()
        if variant not in ("sort", "permutation", "verify"):
            raise ValueError(f"variant must be sort/permutation/verify, got {variant!r}")
        if num_frames < 3:
            raise ValueError(f"num_frames must be >= 3 to have a non-trivial order, got {num_frames}")
        if variant == "verify" and num_frames != 3:
            raise ValueError(
                "Shuffle and Learn verification uses three-frame tuples; "
                f"set num_frames=3, got {num_frames}"
            )
        if not 0.0 < positive_fraction < 1.0:
            raise ValueError(
                f"positive_fraction must be strictly between 0 and 1, got {positive_fraction}"
            )
        self.variant = variant
        self.num_frames = num_frames
        self.positive_fraction = positive_fraction
        if variant == "sort":
            num_classes, output = num_frames, "token"
        elif variant == "verify":
            num_classes, output = 2, "pooled"
        else:
            if num_frames > max_permutation_frames:
                raise ValueError(
                    f"variant='permutation' enumerates all {num_frames}! permutations as classes "
                    f"({num_frames}! = {len(list(itertools.permutations(range(num_frames))))}); "
                    f"num_frames > {max_permutation_frames} is not tractable -- use variant='sort', "
                    f"which predicts per-position ranks and scales to any length.")
            perms = torch.tensor(list(itertools.permutations(range(num_frames))))
            self.register_buffer("perms", perms)
            num_classes, output = perms.size(0), "pooled"
        self.head = SequenceHead(latent_dim, num_classes, d_model=d_model, nhead=nhead,
                                 num_layers=num_layers, dropout=dropout,
                                 max_len=num_frames, output=output)

    def _verification_indices(self, batch_size, clip_length, device):
        """Sample the three-frame verification tuples from Misra et al. (ECCV 2016).

        Five source frames are sampled with a<b<c<d<e. Correct tuples use (b,c,d),
        incorrect tuples replace only the middle frame with a or e, and either tuple can
        be reversed. Keeping the endpoints identical prevents the classifier from using
        a different endpoint distribution as a shortcut.
        """
        five = sorted_subset_indices(batch_size, clip_length, 5, device)
        a, b, c, d, e = five.unbind(1)

        # The paper's best downstream result used 75% negatives / 25% positives. Use an
        # exact per-batch quota (when divisible, as the configured batch size of 128 is)
        # instead of allowing accidental monotone negatives to change the class prior.
        n_positive = round(batch_size * self.positive_fraction)
        if batch_size > 1:
            n_positive = min(max(n_positive, 1), batch_size - 1)
        labels = torch.zeros(batch_size, dtype=torch.long, device=device)
        labels[torch.randperm(batch_size, device=device)[:n_positive]] = 1

        outside = torch.where(torch.rand(batch_size, device=device) < 0.5, a, e)
        middle = torch.where(labels.bool(), c, outside)
        order = torch.stack((b, middle, d), dim=1)
        reverse = torch.rand(batch_size, device=device) < 0.5
        order[reverse] = order[reverse].flip(1)
        return order, labels

    def forward(self, z, progress, frame_index=None, z_pair=None, progress_pair=None,
                frame_index_pair=None):
        b, t, _ = z.shape
        k = self.num_frames

        if self.variant == "verify":
            order, labels = self._verification_indices(b, t, z.device)
            logits = self.head(gather_frames(z, order))
            loss = F.cross_entropy(logits, labels)
            acc = (logits.argmax(-1) == labels).float().mean()
            return loss, {"acc": acc.detach()}

        idx = sorted_subset_indices(b, t, k, z.device)   # (B, K) ascending in time
        ordered = gather_frames(z, idx)                   # (B, K, d), true temporal order

        if self.variant == "permutation":
            cls = torch.randint(0, self.perms.size(0), (b,), device=z.device)
            order = self.perms[cls]                       # (B, K)
            logits = self.head(gather_frames(ordered, order))
            loss = F.cross_entropy(logits, cls)
            return loss, {"acc": (logits.argmax(-1) == cls).float().mean().detach()}

        # "sort": position p of the shuffled sequence holds the frame whose true rank is
        # order[p], and the head predicts that rank for every position.
        order = random_permutations(b, k, z.device)
        logits = self.head(gather_frames(ordered, order))  # (B, K, K)
        loss = F.cross_entropy(logits.reshape(b * k, k), order.reshape(b * k))
        acc = (logits.argmax(-1) == order).float().mean()
        exact = (logits.argmax(-1) == order).all(1).float().mean()
        return loss, {"rank_acc": acc.detach(), "exact_match": exact.detach()}
