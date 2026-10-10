"""Prediction loss (configurable L1/L2) + optional curvature (straightening) and VICReg
regularizers on the encoder's own latents — same formulas as LiFT's
PredictiveStraighteningLoss/MaskedJEPALoss (external/temporal-straightening's own), kept
here as a self-contained copy since this project isn't a LiFT dependent.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .sigreg_loss import sigreg_loss
from .uniformity_loss import uniformity_loss


def _off_diagonal(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


AXES = ("pooled", "per_time", "per_clip")


def _group(z, axis):
    """Regroup z -- (B, T, D) or (B, T, P, D) -- into (G, N, D): G independent sample sets
    of N rows, which a regularizer scores separately and averages.
      pooled   G=1,  N=B*T*P  every frame/token one sample (the original behavior)
      per_time G=T,  N=B*P    at each timestep, the distribution ACROSS clips (LeWM)
      per_clip G=B,  N=T*P    inside each clip, the distribution ACROSS its frames
    """
    d = z.shape[-1]
    if axis == "pooled":
        return z.reshape(1, -1, d)
    if z.dim() not in (3, 4):
        raise ValueError(f"axis={axis!r} needs z of shape (B, T, D) or (B, T, P, D), got {tuple(z.shape)}")
    if z.dim() == 3:
        z = z.unsqueeze(2)
    b, t, p, _ = z.shape
    if axis == "per_time":
        return z.permute(1, 0, 2, 3).reshape(t, b * p, d)
    return z.reshape(b, t * p, d)


def _group_masked(z, axis, valid, axis_param_name):
    """_group, but padding-aware: valid=None is byte-identical to _group(z, axis) (every
    existing call site/config, where lengths was never passed to WorldModelLoss.forward).

    valid is not None: only axis="pooled" is supported. There G=1 already (every frame/
    token is one independent sample, no per-group alignment), so invalid rows can just be
    dropped before the single group is built -- vicreg_std_loss/vicreg_cov_loss/
    sigreg_loss need no change, they just see a smaller N. "per_time"/"per_clip" are NOT
    supported with padding: sigreg_loss's grouped=True (and vicreg's) pack G groups into
    ONE dense (G, N, D) tensor with a single shared N across every group (sigreg_loss's
    own .mean(dim=1) assumes this) -- a per_time group at a late timestep can have a
    different valid-row count per clip-length than an early one, which would need ragged/
    per-group-N support inside those functions, not just here. Raising rather than
    silently computing per-group N as if it were uniform (wrong) or falling back to
    unmasked (silently ignores padding)."""
    if valid is None:
        return _group(z, axis)
    if axis != "pooled":
        raise NotImplementedError(
            f"{axis_param_name}={axis!r} with padded/variable-length input (lengths given to "
            f"WorldModelLoss.forward) is not supported -- only 'pooled' composes with masking "
            f"today (see _group_masked's docstring for why 'per_time'/'per_clip' would need "
            f"ragged per-group N support inside vicreg_*_loss/sigreg_loss, not just here)."
        )
    if z.dim() == 4:  # (B, T, P, D): every patch of a padded frame is itself padding
        valid = valid.unsqueeze(-1).expand(-1, -1, z.shape[2])  # (B, T, P)
    return z[valid].unsqueeze(0)  # (1, N_valid, D)


class WorldModelLoss(nn.Module):
    def __init__(self, pred_loss="l1", pred_weight=1.0, straighten_weight=0.0,
                 vicreg_std_weight=0.0, vicreg_cov_weight=0.0, vicreg_granularity="agg",
                 vicreg_axis="pooled",
                 sigreg_weight=0.0, sigreg_num_slices=1024, sigreg_granularity="agg",
                 sigreg_axis="pooled",
                 chord_weight=0.0, stop_grad=True,
                 step_thresh=1e-6, uniformity_weight=0.0, uniformity_t=2.0,
                 uniformity_pairs=2048):
        super().__init__()
        if pred_loss not in ("l1", "l2", "angular"):
            raise ValueError(f"pred_loss must be 'l1', 'l2' or 'angular', got {pred_loss!r}")
        if vicreg_granularity not in ("agg", "token"):
            raise ValueError(f"vicreg_granularity must be 'agg' or 'token', got {vicreg_granularity!r}")
        if sigreg_granularity not in ("agg", "token"):
            raise ValueError(f"sigreg_granularity must be 'agg' or 'token', got {sigreg_granularity!r}")
        for name, axis in (("vicreg_axis", vicreg_axis), ("sigreg_axis", sigreg_axis)):
            if axis not in AXES:
                raise ValueError(f"{name} must be one of {AXES}, got {axis!r}")
        self.vicreg_axis = vicreg_axis
        self.sigreg_axis = sigreg_axis
        self.vicreg_granularity = vicreg_granularity
        self.sigreg_granularity = sigreg_granularity
        # angular: 1 - cos(pred, target). l1/l2 measure Euclidean
        # distance, which is the wrong geometry once norm_type="hypersphere" puts z on the unit
        # sphere -- there the only thing that can differ between two latents is their
        # angle, and an L2 penalty additionally charges for a radius difference that the
        # projection has already removed. Wrapped to F.l1_loss's (input, target)
        # signature so prediction_loss() below needs no branch. Works for 3D (B,T,D) and
        # 4D (B,T,P,D) alike since dim=-1, though in patch mode that means cosine over
        # 8-dim per-patch channel vectors, which is a different notion than cls mode's.
        self.pred_loss_fn = {
            "l1": F.l1_loss,
            "l2": F.mse_loss,
            "angular": lambda a, b: (1 - F.cosine_similarity(a, b, dim=-1)).mean(),
        }[pred_loss]
        self.pred_weight = pred_weight
        self.straighten_weight = straighten_weight
        self.vicreg_std_weight = vicreg_std_weight
        self.vicreg_cov_weight = vicreg_cov_weight
        self.sigreg_weight = sigreg_weight
        self.sigreg_num_slices = sigreg_num_slices
        self.chord_weight = chord_weight
        self.uniformity_weight = uniformity_weight
        self.uniformity_t = uniformity_t
        self.uniformity_pairs = uniformity_pairs
        self.stop_grad = stop_grad
        self.step_thresh = step_thresh

    def prediction_loss(self, z_pred, z, valid=None):
        """valid: optional (B, T) bool, True=real frame (see forward()'s lengths= docstring
        for where this comes from). A (t, t+1) pair is excluded if EITHER end is padding --
        with trailing-only padding that's exactly the pairs touching or inside the padded
        region, since a real frame's own next-step target is never padding until the pair
        that straddles the boundary."""
        if z.size(1) < 2:
            # No (t, t+1) pair exists; the slices below would be empty and reduction
            # 'mean' over an empty tensor is NaN.
            return z.new_zeros(())
        if z_pred.size(1) == z.size(1):
            # cls/raw CausalPredictor retains its historical T-output interface.
            z_src = z_pred[:, :-1]
        elif z_pred.size(1) == z.size(1) - 1:
            # Patch BlockCausalPredictor is called on the T-1 supervised source frames.
            z_src = z_pred
        else:
            raise ValueError(
                f"prediction length must be T or T-1 for a T-frame target, got "
                f"z_pred.shape={tuple(z_pred.shape)}, z.shape={tuple(z.shape)}"
            )
        z_tgt = z[:, 1:]
        if self.stop_grad:
            z_tgt = z_tgt.detach()
        if valid is None:
            return self.pred_loss_fn(z_src, z_tgt)
        pair_valid = valid[:, :-1] & valid[:, 1:]  # (B, T-1)
        if not pair_valid.any():
            return z.new_zeros(())
        # Reduce the feature dim(s) per-position FIRST (matching what pred_loss_fn's own
        # 'mean' reduction would average over per element), then mask-average over
        # (B, T-1) -- same per-position-then-masked-mean shape every pred_loss_fn needs.
        if self.pred_loss_fn is F.l1_loss:
            per_pos = (z_src - z_tgt).abs().mean(dim=-1)
        elif self.pred_loss_fn is F.mse_loss:
            per_pos = (z_src - z_tgt).pow(2).mean(dim=-1)
        else:  # angular
            per_pos = 1 - F.cosine_similarity(z_src, z_tgt, dim=-1)
        while pair_valid.dim() < per_pos.dim():  # patch mode: per_pos keeps a P dim, pair_valid doesn't
            pair_valid = pair_valid.unsqueeze(-1)
        return per_pos[pair_valid.expand_as(per_pos)].mean()

    def straightening_loss(self, z, valid=None):
        """1 - cos(v1, v2) between consecutive step-vectors, masking near-zero-norm
        steps (step_thresh) to avoid numerically meaningless cosine similarities.

        valid: optional (B, T) bool, True=real frame. A STEP z[t+1]-z[t] additionally
        needs valid[t] and valid[t+1] (step_valid below); this is belt-and-suspenders
        with step_thresh for steps deep inside the padded region (literal zero frames ->
        a zero-norm step, already caught by step_thresh), but it is NOT redundant for the
        one step that matters: real-frame -> first-pad-frame, which has a large, very
        much nonzero norm and would otherwise look like a real (fake) motion."""
        steps = z[:, 1:] - z[:, :-1]  # (B, T-1, D)
        v1, v2 = steps[:, :-1], steps[:, 1:]  # (B, T-2, D)
        norm1, norm2 = v1.norm(dim=-1), v2.norm(dim=-1)
        ok = (norm1 > self.step_thresh) & (norm2 > self.step_thresh)
        if valid is not None:
            step_valid = valid[:, 1:] & valid[:, :-1]  # (B, T-1)
            ok = ok & step_valid[:, :-1] & step_valid[:, 1:]
        if not ok.any():
            return z.new_zeros(())
        cos_sim = F.cosine_similarity(v1, v2, dim=-1)
        return (1 - cos_sim[ok]).mean()

    def chord_loss(self, z, valid=None, lengths=None):
        """1 - cos(step, chord) between EVERY step vector (not just adjacent pairs, unlike
        straightening_loss) and the clip's global chord z[-1] - z[0] -- ties every step to
        a single whole-clip reference direction instead of only checking local step-to-step
        agreement, so an oscillation that straightening_loss might only weakly penalize
        (each pair individually near -1 cos, but easy to trade off against other loss terms)
        gets caught directly: half its steps would point backward relative to the chord.
        Purely directional (scale-invariant), so unlike straightening_weight this doesn't
        pull start/end back together -- complements straightening_weight, doesn't replace it.
        Same masking convention as straightening_loss/mean_chord_alignment (see
        experiments/encoder_planning_eval/progress_metrics/compute_progress_metrics.py's
        eval-only counterpart this mirrors as a trainable loss).

        lengths: optional (B,), each clip's valid frame count. z[:, -1] is the clip's
        LAST frame only when every clip is full-length; with trailing padding it is a
        padded (zero) frame for any clip shorter than T, which would make the chord a
        meaningless "goal minus start" against a zero vector. When given, gathers each
        clip's own last VALID frame instead. valid (B, T) bool) masks steps the same way
        straightening_loss does, for the same reason (the real->pad boundary step)."""
        if lengths is None:
            goal = z[:, -1]  # (B, D), original behavior, byte-identical when unused
        else:
            last_idx = (lengths - 1).clamp(min=0)
            goal = z[torch.arange(z.size(0), device=z.device), last_idx]  # (B, D)
        chord = goal - z[:, 0]  # (B, D)
        chord_norm = chord.norm(dim=-1)  # (B,)
        steps = z[:, 1:] - z[:, :-1]  # (B, T-1, D)
        step_norms = steps.norm(dim=-1)  # (B, T-1)
        ok = (chord_norm > self.step_thresh).unsqueeze(1) & (step_norms > self.step_thresh)
        if valid is not None:
            step_valid = valid[:, 1:] & valid[:, :-1]  # (B, T-1)
            ok = ok & step_valid
        if not ok.any():
            return z.new_zeros(())
        cos_sim = F.cosine_similarity(steps, chord.unsqueeze(1), dim=-1)
        return (1 - cos_sim[ok]).mean()

    # grouped=True: z is already (G, N, D) from _group -- each group scored on its own
    # N rows, then averaged. Default (False) flattens everything into one group, as before.
    def vicreg_std_loss(self, z, grouped=False):
        x = z if grouped else z.reshape(1, -1, z.shape[-1])
        if x.shape[1] < 2:
            # Sample standard deviation is undefined for a one-clip final batch.
            return x.sum() * 0.0
        std_x = torch.sqrt(x.var(dim=1) + 1e-4)  # (G, D)
        return torch.mean(F.relu(1 - std_x))

    def vicreg_cov_loss(self, z, grouped=False):
        x = z if grouped else z.reshape(1, -1, z.shape[-1])
        _, n, d = x.shape
        if n < 2:
            return x.sum() * 0.0
        x = x - x.mean(dim=1, keepdim=True)
        cov_x = (x.transpose(1, 2) @ x) / (n - 1)  # (G, D, D)
        off_diag = cov_x.pow(2).sum(dim=(1, 2)) - cov_x.diagonal(dim1=1, dim2=2).pow(2).sum(dim=1)
        return (off_diag / d).mean()

    def forward(self, z, z_pred, z_agg=None, lengths=None):
        # lengths: optional (B,) int, each clip's valid (non-padded) frame count --
        # see train.py's --variable-length-min and data/dataset.py's ClipFeatureDataset.
        # None (default, every existing call site/config): no masking at all, identical
        # to before this argument existed. Padding is assumed TRAILING ONLY (a clip's
        # frames 0..lengths[b]-1 are real, lengths[b]..T-1 are zeroed) -- that is what
        # lets every mask below reduce to simple slicing/AND rather than needing a
        # general interior-gap mask, and it's what the causal predictor already relies
        # on structurally: position t can only attend to positions <= t, so a real frame
        # can never attend to a LATER padded one regardless of any mask here. Only the
        # LOSS needs this argument; models/world_model.py's encoder/predictor need no
        # change for padding to be safe.
        valid = None
        if lengths is not None:
            t = z.size(1)
            valid = torch.arange(t, device=z.device).unsqueeze(0) < lengths.unsqueeze(1)  # (B, T)

        # prediction_loss operates at the model's native token granularity (3D or 4D --
        # F.l1_loss/F.mse_loss are elementwise, so no change needed there). curvature and
        # VICReg need one representative vector per frame instead -- z_agg, if given, is
        # ProceduralWorldModel.aggregate(z)'s output: the reference's own
        # best-performing "agg head" cosine-similarity variant for spatial features
        # (Section B.6/Figure 14), a LEARNABLE MLP, not a naive mean-pool. When z_agg is
        # omitted (e.g. calling this loss module directly, outside train.py's wiring),
        # fall back to mean-pooling z's patch dimension -- worse per the reference's own
        # ablation, but a reasonable default when no aggregation head is available.
        #
        # loss_dict values are DETACHED TENSORS, not .item() floats: calling .item()
        # here would force a host<->device sync on every one of these (up to 5+ terms)
        # EVERY training step, serializing the GPU behind the CPU and killing any
        # overlap/pipelining. Staying on-device lets a caller (train.py's run_epoch)
        # accumulate a whole epoch's worth of these and pull them to Python floats with
        # ONE sync at the end, instead of one sync per term per step.
        pred_loss = self.prediction_loss(z_pred, z, valid=valid)
        total_loss = self.pred_weight * pred_loss
        loss_dict = {"pred": pred_loss.detach()}

        z_reg = z_agg if z_agg is not None else (z.mean(dim=2) if z.dim() == 4 else z)

        if self.straighten_weight > 0:
            straighten_loss = self.straightening_loss(z_reg, valid=valid)
            total_loss = total_loss + self.straighten_weight * straighten_loss
            loss_dict["straighten"] = straighten_loss.detach()

        if self.chord_weight > 0:
            chord_loss = self.chord_loss(z_reg, valid=valid, lengths=lengths)
            total_loss = total_loss + self.chord_weight * chord_loss
            loss_dict["chord"] = chord_loss.detach()

        if self.vicreg_std_weight > 0 or self.vicreg_cov_weight > 0:
            # "token": z at its native granularity -- vicreg_*_loss flatten every leading
            # dim, so each (batch, frame, patch) token is its own sample, as in the
            # reference. "agg": the same per-frame z_reg the curvature loss uses.
            z_vic = z if self.vicreg_granularity == "token" else z_reg
            z_vic = _group_masked(z_vic, self.vicreg_axis, valid, "vicreg_axis")
            if self.vicreg_std_weight > 0:
                std_loss = self.vicreg_std_loss(z_vic, grouped=True)
                total_loss = total_loss + self.vicreg_std_weight * std_loss
                loss_dict["vicreg_std"] = std_loss.detach()
            if self.vicreg_cov_weight > 0:
                cov_loss = self.vicreg_cov_loss(z_vic, grouped=True)
                total_loss = total_loss + self.vicreg_cov_weight * cov_loss
                loss_dict["vicreg_cov"] = cov_loss.detach()

        if self.sigreg_weight > 0:
            # Principled alternative to the two VICReg terms above (see
            # losses/sigreg_loss.py) -- independent of them, so either or both can be
            # active at once; set vicreg_*_weight to 0 to use SIGReg alone.
            # sigreg_granularity mirrors vicreg_granularity above: "token" points it at
            # the raw per-patch z (every (batch, frame, patch) token its own sample)
            # instead of z_reg. sigreg_loss's magnitude is not simply proportional to n
            # (the row count after flattening, see sigreg_loss.py) -- it's an
            # Epps-Pulley-style test statistic whose expectation stays roughly constant
            # under near-Gaussian data regardless of n, but grows with n's statistical
            # power to detect a REAL deviation from Gaussianity the larger n gets. Since
            # "token" granularity has ~num_patches more rows than "agg", expect the loss
            # to run higher there specifically because it's more sensitive to whatever
            # non-Gaussianity is actually present, not from a fixed multiplicative
            # scale factor -- sigreg_weight may still need retuning, but not by a
            # precomputable ratio.
            z_sig = z if self.sigreg_granularity == "token" else z_reg
            z_sig = _group_masked(z_sig, self.sigreg_axis, valid, "sigreg_axis")
            sig_loss = sigreg_loss(z_sig, num_slices=self.sigreg_num_slices, grouped=True)
            total_loss = total_loss + self.sigreg_weight * sig_loss
            loss_dict["sigreg"] = sig_loss.detach()

        if self.uniformity_weight > 0:
            # Third anti-collapse option, independent of the two above (any combination
            # can be active). Unlike VICReg it is scale-free and has no d/N noise floor,
            # and unlike both it is defined on the unit sphere -- see
            # losses/uniformity_loss.py for the measurements motivating each claim. It
            # L2-normalizes z_reg internally, so it is safe (though least meaningful)
            # under norm_type values that leave ||z|| free. Always effectively "pooled"
            # (no axis concept), so masking is just dropping invalid rows before it ever
            # sees them.
            z_unif = z_reg[valid] if valid is not None else z_reg
            unif_loss = uniformity_loss(z_unif, t=self.uniformity_t,
                                         n_pairs=self.uniformity_pairs)
            total_loss = total_loss + self.uniformity_weight * unif_loss
            loss_dict["uniformity"] = unif_loss.detach()

        loss_dict["total"] = total_loss.detach()
        return total_loss, loss_dict
