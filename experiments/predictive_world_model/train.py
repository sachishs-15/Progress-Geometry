"""Train ProceduralWorldModel (cls-token mode, first slice of this project).

Precomputed-or-live features: --features points at a bulk .pt (LiFT's
precompute_features.py format) if you have one; any clip whose video_path isn't in it is
transparently computed live and cached under --live-cache-dir. Either source can be used
alone or together:

  --features only (no manifest)   train on every video in the .pt, nothing computed live.
                                  The fully-precomputed case, e.g. K400.
  --train-manifest only           train on the manifest's clips, all computed live and
                                  cached (nothing precomputed yet).
  both                            train on the manifest's clips, using the .pt where it
                                  covers them and computing the rest live.

Usage:
    # fully-precomputed dataset, no manifest needed
    python train.py --config configs/cls_mlp_k400.yaml --checkpoint-dir checkpoints/k400_cls

    # manifest-driven, mixed precomputed/live
    python train.py --config configs/cls_mlp.yaml \\
        --train-manifest /path/to/ssv2_train.csv --val-manifest /path/to/ssv2_val.csv \\
        --features /path/to/ssv2_train_raw.pt --checkpoint-dir checkpoints/run1
"""
import argparse
import csv
import os
import random
import sys
from pathlib import Path

import torch
import torch.distributed as dist
import yaml
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler, random_split

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data.dataset import ClipFeatureDataset, RandomWindowDataset
from data.feature_provider import FeatureProvider
from losses.world_model_loss import WorldModelLoss
from models.encoder import ENCODER_REGISTRY
from models.pixel_encoder import PIXEL_ENCODER_REGISTRY
from models.world_model import ProceduralWorldModel


class _Tee:
    """Duplicates writes to every given stream -- lets train.py write its own
    <checkpoint-dir>/train.log directly (opened line-buffered, so it stays live rather
    than sitting in a block-buffer until the process exits) instead of relying on
    train.sh's external `tee`. That external tee had to resolve the checkpoint-dir
    ITSELF, before train.py even started, duplicating (and, once, disagreeing with)
    train.py's own resolve_checkpoint_dir call -- owning the log here removes the need
    for train.sh to know the checkpoint-dir at all, so there's only ever one place that
    decides it."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)

    def flush(self):
        for s in self.streams:
            s.flush()


def _has_training_artifacts(ckpt_dir: Path) -> bool:
    """True if ckpt_dir holds actual training output (config.yaml, train_log.csv, or any
    *.ckpt) -- NOT just any file at all. A directory can hold a bare train.log (and
    nothing else) without ever having actually trained anything in it -- e.g. one left
    over from before train.py wrote its own log directly (see _Tee), when an external
    `tee` could land a train.log in a directory train.py itself hadn't touched yet.
    Treating "any file present" as "already used" would make such a directory
    permanently unreusable even though no real run ever happened there."""
    markers = ("config.yaml", "train_log.csv")
    return any((ckpt_dir / m).exists() for m in markers) or any(ckpt_dir.glob("*.ckpt"))


def resolve_checkpoint_dir(base: Path) -> Path:
    """base if empty/unused/nonexistent, else base_1/base_2/... (first one with no
    training artifacts yet). There is no --resume: every invocation trains fresh from
    scratch, so a checkpoint-dir with real training artifacts in it can only be a
    PREVIOUS run's leftovers, never a deliberate continuation. Silently reusing it means
    config.yaml gets overwritten immediately but train_log.csv/checkpoints don't until
    epoch 0 finishes -- stale data sitting next to a freshly-rewritten config.yaml,
    indistinguishable without checking mtimes by hand. Auto-suffixing instead means a
    new run always gets its own directory.

    The ONLY place this runs -- train.sh no longer resolves/knows the checkpoint-dir at
    all (see its own comment); it just execs train.py, which resolves this itself and
    then writes its own train.log into whatever directory it lands on."""
    ckpt_dir = base
    i = 1
    while _has_training_artifacts(ckpt_dir):
        ckpt_dir = base.with_name(f"{base.name}_{i}")
        i += 1
    return ckpt_dir


def predictor_num_frames(args):
    """Clip length used to size the patch predictor. ProceduralWorldModel allocates
    one fewer source position because a T-frame clip contains T-1 next-step pairs."""
    return args.window_size + 1 if args.window_size > 0 else args.n_frames


def build_model(args):
    return ProceduralWorldModel(
        feature_dim=args.feature_dim, latent_dim=args.latent_dim, hidden_dim=args.hidden_dim,
        use_projector=args.use_projector, norm_type=args.norm_type,
        token_mode=args.token_mode, projector_type=args.projector_type,
        channel_out_dim=args.channel_out_dim, channel_hidden=args.channel_hidden,
        # cls mode only -- unchanged, see ProceduralWorldModel's docstring
        nhead=args.nhead, pred_num_layers=args.pred_num_layers, max_len=args.max_len,
        residual_projector=args.residual_projector,
        encoder_type=args.encoder_type, encoder_depth=args.encoder_depth,
        encoder_expansion=args.encoder_expansion, encoder_dropout=args.encoder_dropout,
        # raw mode only
        pixel_encoder_type=args.pixel_encoder_type, raw_img_size=args.raw_img_size,
        # patch mode only
        num_patches=args.num_patches, num_frames=predictor_num_frames(args),
        depth=args.depth, heads=args.heads, dim_head=args.dim_head, mlp_dim=args.mlp_dim,
        emb_dropout=args.emb_dropout, agg_mlp_hidden_dim=args.agg_mlp_hidden_dim,
        agg_out_dim=args.agg_out_dim, cls_agg_head=args.cls_agg_head,
        # shared
        dropout=args.dropout,
    )


def build_optimizer(model, args):
    """Single AdamW over all params (backward-compatible default, this project's
    original behavior) unless --encoder-lr/--predictor-lr are set, in which case two
    param groups are built instead -- matching the reference's own split exactly
    (train.py's init_optimizers(): encoder_optimizer = Adam(self.encoder.parameters(),
    lr=encoder_lr), predictor_optimizer = AdamW(self.predictor.parameters(),
    lr=predictor_lr); Table 3: encoder_lr=1e-5 for spatial/channel projectors, 1e-6 for
    the global projector, predictor_lr=5e-4 for both). The reference lumps its
    aggregation head into the SAME encoder optimizer as the projector (agg_mlp is a
    submodule of their DinoV2Encoder, sharing its param group) -- our agg_head is a
    sibling of encoder/predictor on ProceduralWorldModel instead, so it's folded into
    the encoder param group explicitly here to match that grouping. Both groups use
    AdamW with the same --weight-decay here (the reference's encoder_optimizer is plain
    Adam, no weight decay; simplified to one optimizer class/weight-decay value since
    only the LR split -- the actual point of Table 3 -- was asked for)."""
    if args.encoder_lr is None and args.predictor_lr is None:
        return torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    encoder_params = list(model.encoder.parameters())
    if model.agg_head is not None:
        encoder_params += list(model.agg_head.parameters())
    if model.pixel_encoder is not None:
        encoder_params += list(model.pixel_encoder.parameters())
    param_groups = [
        {"params": encoder_params, "lr": args.encoder_lr if args.encoder_lr is not None else args.lr},
        {"params": model.predictor.parameters(),
         "lr": args.predictor_lr if args.predictor_lr is not None else args.lr},
    ]
    return torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)


def build_criterion(args):
    return WorldModelLoss(
        pred_loss=args.pred_loss, pred_weight=args.pred_weight,
        straighten_weight=args.straighten_weight, vicreg_std_weight=args.vicreg_std_weight,
        vicreg_cov_weight=args.vicreg_cov_weight, vicreg_granularity=args.vicreg_granularity,
        vicreg_axis=args.vicreg_axis, sigreg_axis=args.sigreg_axis,
        sigreg_weight=args.sigreg_weight,
        sigreg_num_slices=args.sigreg_num_slices, sigreg_granularity=args.sigreg_granularity,
        chord_weight=args.chord_weight,
        uniformity_weight=args.uniformity_weight, uniformity_t=args.uniformity_t,
        uniformity_pairs=args.uniformity_pairs,
        stop_grad=args.stop_grad,
    )


def init_distributed(device_arg):
    """(rank, world_size, device). Multi-GPU runs are launched by torchrun (train.sh with
    NUM_GPUS>1), which sets RANK/WORLD_SIZE/LOCAL_RANK; a plain `python train.py` has none
    of those and gets (0, 1, device_arg) with no process group, i.e. the original
    single-process behaviour."""
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if world == 1:
        return 0, 1, torch.device(device_arg)
    rank, local_rank = int(os.environ["RANK"]), int(os.environ["LOCAL_RANK"])
    if torch.device(device_arg).type == "cuda":
        torch.cuda.set_device(local_rank)
        dist.init_process_group("nccl")
        return rank, world, torch.device("cuda", local_rank)
    dist.init_process_group("gloo")
    return rank, world, torch.device("cpu")


class TrainStep(torch.nn.Module):
    """model + loss as ONE module returning (loss, loss_dict). DistributedDataParallel only
    syncs gradients for what runs inside its own forward(); model.aggregate() and the
    criterion are called separately in the single-process loop, which under DDP would leave
    the aggregation head's gradients unsynced. Wrapping the whole step avoids that, and
    single-process training calls this unwrapped, so the math is unchanged."""

    def __init__(self, model, criterion):
        super().__init__()
        self.model = model
        self.criterion = criterion

    def forward(self, x, lengths=None):
        z, z_pred = self.model(x)
        z_agg = self.model.aggregate(z)
        loss, loss_dict = self.criterion(z, z_pred, z_agg, lengths=lengths)
        # Diagnostic, not part of the objective: the prediction loss scales with
        # ||z|| while the curvature term is scale-invariant, so without this the
        # loss curves alone can't distinguish "prediction got worse" from "z grew".
        loss_dict["z_norm"] = z.detach().norm(dim=-1).mean()
        return loss, loss_dict


def run_epoch(model, criterion, loader, device, optimizer=None, grad_clip=0.0,
              desc="", progress_every=0, amp=False, step=None, world_size=1):
    """progress_every=0 (default): silent, as before. >0: print a running-average
    progress line every that many batches (real newlines, not tqdm's carriage-return
    in-place updates -- those get mangled into a wall of separate lines rather than
    overwriting in place when piped through train.sh's `tee` into a log FILE instead of
    a live terminal). This is the only visibility into epoch 0's live feature-extraction
    pass at K400 scale: every batch there pays a real DINOv2 forward pass per video, not
    just a training step, so per-batch time is far higher than usual and worth seeing
    progress on directly rather than waiting silently for the epoch summary.

    totals accumulates loss_dict's tensors ON-DEVICE all epoch (WorldModelLoss.forward
    returns them detached, not .item()'d, for exactly this reason -- see its own
    docstring) and only converts to Python floats ONCE at the very end (or at a
    progress_every print point, an explicit opt-in) -- calling .item() every batch
    forces a host<->device sync that serializes the GPU behind the CPU every step.

    step: the module called on each batch, returning (loss, loss_dict) -- pass the DDP-
    wrapped TrainStep for multi-GPU; default is a plain TrainStep(model, criterion).
    world_size>1: the epoch averages are all-reduced so every rank returns the same
    numbers (per-batch progress lines stay local to the rank that prints them).
    """
    train = optimizer is not None
    if step is None:
        step = TrainStep(model, criterion)
    model.train(train)
    totals = {}
    n_batches = 0
    total_batches = len(loader)
    amp_enabled = amp and device.type == "cuda"
    with torch.set_grad_enabled(train):
        for batch in loader:
            x = batch["features"].to(device, non_blocking=True)
            # "length" is only present for ClipFeatureDataset (and only varies per-clip
            # when --variable-length-min is set) -- RandomWindowDataset has no such key.
            # None here means "every position is valid", i.e. identical to today's
            # behavior; see WorldModelLoss.forward's lengths= docstring.
            lengths = batch["length"].to(device, non_blocking=True) if "length" in batch else None
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=amp_enabled):
                loss, loss_dict = step(x, lengths)
            if train:
                optimizer.zero_grad()
                loss.backward()
                if grad_clip > 0:
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    loss_dict["grad_norm"] = grad_norm.detach()
                optimizer.step()
            loss_dict["z_norm"] = loss_dict.pop("z_norm")  # keep it last, after grad_norm
            for k, v in loss_dict.items():
                # .float() is load-bearing, not defensive. bf16 has 8 mantissa bits, so a
                # running sum STOPS ADVANCING once its exponent makes the ULP exceed 2x the
                # per-batch value: 754 batches of ~10.9 pin the total at 4096 partway
                # through the epoch, and every later batch adds exactly nothing. That made
                # train_sigreg log as a constant 5.4375 (= 4096/754) for 50 straight epochs
                # while its true value was ~10.9 and falling ~17%/epoch -- indistinguishable
                # from "the regularizer is doing nothing".
                #
                # Only sigreg was hit, and only under --amp: autocast promotes mse_loss /
                # cosine_similarity / var to fp32, so pred, straighten, vicreg_std and
                # vicreg_cov already come back fp32, but sigreg_loss's matmul -> cos/sin ->
                # logsumexp chain stays bf16 all the way out. Casting here covers whichever
                # term is bf16 today without depending on autocast's op-promotion table
                # staying put. Logs only, never the objective: `loss` is a separate tensor
                # that never passes through this accumulator.
                totals[k] = totals.get(k, 0.0) + v.float()
            n_batches += 1
            if progress_every > 0 and (n_batches % progress_every == 0 or n_batches == total_batches):
                avg = {k: (v / n_batches).item() for k, v in totals.items()}
                stats = ", ".join(f"{k}={v:.4f}" for k, v in avg.items())
                print(f"  [{desc}] batch {n_batches}/{total_batches} "
                      f"({100 * n_batches / total_batches:.1f}%): {stats}", flush=True)
    if world_size > 1:
        keys = list(totals)  # insertion order is identical on every rank
        vec = torch.stack([totals[k] for k in keys]).to(device)
        dist.all_reduce(vec)
        totals = {k: v / world_size for k, v in zip(keys, vec)}
    return {k: (v / n_batches).item() for k, v in totals.items()}


if __name__ == "__main__":
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", default=None)
    pre_args, _ = pre_parser.parse_known_args()

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None, help="yaml overriding arch/loss defaults below")

    parser.add_argument("--train-manifest", default=None,
                         help="CSV with a video_path column; omit to train on every video in --features")
    parser.add_argument("--val-manifest", default=None)
    parser.add_argument("--val-fraction", type=float, default=0.0,
                         help="random val split off the training set, when no --val-manifest is given")
    parser.add_argument("--path-col", default="video_path")
    parser.add_argument("--features", default=None, help="precomputed bulk .pt (optional)")
    parser.add_argument("--live-cache-dir", default=None,
                         help="on-disk cache for on-the-fly-computed features (default: <checkpoint-dir>/live_cache)")

    parser.add_argument("--feature-dim", type=int, default=384)
    parser.add_argument("--latent-dim", type=int, default=384)
    parser.add_argument("--hidden-dim", type=int, default=None)
    parser.add_argument("--use-projector", dest="use_projector", action="store_true", default=True)
    parser.add_argument("--no-projector", dest="use_projector", action="store_false")
    parser.add_argument("--residual-projector", action="store_true", default=False,
                         help="cls mode + use_projector=True only: add the raw feature back after "
                              "the projector (z = norm(x + projector(x))) -- see ClsMlpEncoder")
    parser.add_argument("--encoder-type", choices=sorted(ENCODER_REGISTRY), default="mlp",
                         help="cls mode only. mlp: the original 2-layer Linear-GELU-Linear "
                              "projector (default, what every existing config/checkpoint uses). "
                              "residual_mlp: `encoder_depth` pre-norm residual blocks, each "
                              "expanding to encoder_expansion*latent_dim -- deepenable, unlike mlp. "
                              "Both are strictly per-frame; neither mixes across time.")
    parser.add_argument("--encoder-depth", type=int, default=4,
                         help="encoder_type=residual_mlp only: number of residual blocks (0 = "
                              "in_proj + norm, a linear-probe floor).")
    parser.add_argument("--encoder-expansion", type=int, default=4,
                         help="encoder_type=residual_mlp only: FFN expansion factor per block.")
    parser.add_argument("--encoder-dropout", type=float, default=0.0,
                         help="encoder_type=residual_mlp only: dropout inside each block.")
    parser.add_argument("--norm-type", choices=["layernorm", "clip_rmsnorm", "batchnorm", "hypersphere", "none"],
                         default="layernorm",
                         help="normalization on the encoder output — bounds ||z||, which the "
                              "reference gets for free and we otherwise don't (see models/norms.py "
                              "for what each option does and the tradeoffs between them)")
    # cls mode only (predictor.py's CausalPredictor, unchanged) -- NOT used by
    # token_mode=patch, which has its own predictor hyperparameters below.
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--pred-num-layers", type=int, default=2)
    parser.add_argument("--max-len", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.1,
                         help="shared by both predictors (cls mode's CausalPredictor and patch "
                              "mode's BlockCausalPredictor)")
    parser.add_argument("--n-frames", type=int, default=16)
    parser.add_argument("--sample-fps", type=float, default=None,
                         help="cls or patch mode + --train-manifest: instead of n_frames spread evenly over "
                              "each clip, draw n_frames CONSECUTIVE frames at this fps from a random "
                              "position in the clip (--n-frames 32 --sample-fps 4 = a random 8s). "
                              "Every clip is extracted once at this fps into the live cache, so each "
                              "epoch draws fresh windows from the cache.")
    parser.add_argument("--windows-per-clip", type=int, default=1,
                         help="--sample-fps only: independent random windows per clip per epoch")
    parser.add_argument("--short-clip-policy", choices=["drop", "pad"], default="drop",
                         help="--sample-fps only: clips shorter than n_frames/fps are dropped, or padded "
                              "by repeating their last frame")
    parser.add_argument("--duration-col", default="duration",
                         help="--sample-fps only: manifest column with clip duration in seconds")
    parser.add_argument("--model-id", default="vit_small_patch14_reg4_dinov2.lvd142m")
    parser.add_argument("--token-mode", choices=["cls", "patch", "raw"], default="cls",
                         help="cls: DINOv2 CLS features (LiFT's own timm pipeline), 1 token/frame. "
                              "patch: raw DINOv2 patch tokens (native torch.hub repo) through a "
                              "CNN projector (see --projector-type, models/projectors.py). "
                              "raw: raw RGB frames (no DINOv2 at all) through a trainable CNN "
                              "(see --pixel-encoder-type), then identical to cls downstream.")
    parser.add_argument("--pixel-encoder-type", choices=sorted(PIXEL_ENCODER_REGISTRY),
                         default="small_resnet",
                         help="token_mode=raw only: the trainable backbone over raw pixels feeding "
                              "into --encoder-type (mlp/residual_mlp), which is otherwise unchanged.")
    parser.add_argument("--raw-img-size", type=int, default=224,
                         help="token_mode=raw only: frames are decoded then resized to this "
                              "(square) size before the pixel encoder.")
    parser.add_argument("--projector-type", choices=["global", "channel"], default="global",
                         help="patch mode only. global: pools to 1 token/frame. channel: keeps "
                              "the full patch grid. Both use BlockCausalPredictor (see below), "
                              "NOT cls mode's CausalPredictor.")
    parser.add_argument("--channel-out-dim", type=int, default=8,
                         help="channel projector's output channel depth (reference default: 8)")
    parser.add_argument("--channel-hidden", type=int, default=192,
                         help="channel projector's hidden channel depth (reference default: 192)")
    # patch mode only (block_causal_predictor.py's BlockCausalPredictor -- reference-exact,
    # matches external/temporal-straightening/conf/predictor/vit.yaml: depth=6, heads=16,
    # dim_head=64 (independent of the token's own channel dim -- see
    # models/block_causal_predictor.py for why that matters), mlp_dim=2048, emb_dropout=0).
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--dim-head", type=int, default=64)
    parser.add_argument("--mlp-dim", type=int, default=2048)
    parser.add_argument("--emb-dropout", type=float, default=0.0)
    parser.add_argument("--agg-mlp-hidden-dim", type=int, default=None,
                         help="aggregation head's hidden dim (projector_type=channel, or --cls-agg-head). "
                              "Unset -> 4 * agg_out_dim, the reference's own fallback "
                              "(conf/encoder/dino_channel.yaml sets it explicitly to 512)")
    parser.add_argument("--cls-agg-head", action="store_true",
                         help="token_mode cls/raw: add the same learnable aggregation head patch mode "
                              "uses (3-layer MLP + LayerNorm, 1 token/frame -> --agg-out-dim) and "
                              "apply the curvature/chord terms -- plus VICReg/SIGReg unless their "
                              "--*-granularity is token -- to its output instead of the latent z the "
                              "predictor consumes. Off by default (no head, losses read z).")
    parser.add_argument("--agg-out-dim", type=int, default=128,
                         help="aggregation head's output dim (reference: agg_out_dim=128)")
    parser.add_argument("--window-size", type=int, default=0,
                         help="patch mode only. 0 (default): train causally over the whole "
                              "--n-frames clip in one pass (every next-step supervised per clip). "
                              ">0: match the reference's actual training regime instead -- sample "
                              "a random (window_size+1)-frame span per clip each step (num_hist="
                              "window_size, num_pred=1; reference default window_size=3), and "
                              "size the predictor's positional embedding to window_size+1 instead "
                              "of the full clip. See data/dataset.py's window_size.")
    parser.add_argument("--variable-length-min", type=int, default=None,
                         help="TRAIN split only (never validation, which stays fixed-length for "
                              "stable cross-epoch metrics). None (default): off, every clip is the "
                              "full --n-frames (or window_size+1) length, unchanged from before. "
                              "Set to an int N: each clip/window draws a random valid length in "
                              "[N, length] per __getitem__ and zero-pads frames past it -- "
                              "simulates clips shorter than what the architecture's max_len "
                              "supports, without a new data-loading path (see "
                              "data/dataset.py's ClipFeatureDataset). Padding is always TRAILING, "
                              "so causal attention already can't let a real frame see a padded "
                              "one; only the loss needs the resulting --length batch key (see "
                              "WorldModelLoss.forward's lengths= argument) to exclude padded "
                              "positions from every term's reduction.")
    parser.add_argument("--num-patches", type=int, default=196,
                         help="patch grid size for projector_type=channel (fixed by image "
                              "resolution/patch size -- 196 (14x14) is the reference's own grid "
                              "size, Table 1's '14x14x...' dims; must match --patch-img-size/14 "
                              "squared, and must be known up front, see ProceduralWorldModel's "
                              "docstring for why)")
    parser.add_argument("--patch-hub-id", default="dinov2_vits14",
                         help="torch.hub facebookresearch/dinov2 checkpoint name for token_mode=patch "
                              "-- reference: conf/encoder/dino_channel.yaml/dino_global.yaml's "
                              "name='dinov2_vits14' (NO registers -- a different pretrained checkpoint "
                              "from dinov2_vits14_reg, not just extra output tokens)")
    parser.add_argument("--patch-img-size", type=int, default=196,
                         help="resize applied before the patch-token DINOv2 forward pass -- 196 "
                              "gives a 14x14=196 patch grid (patch_size=14), matching the "
                              "reference's Table 1 dims exactly (it resizes DINOv2's input to keep "
                              "the grid at 14x14 regardless of source resolution)")

    parser.add_argument("--pred-loss", choices=["l1", "l2", "angular"], default="l1",
                         help="angular = 1 - cos(pred, target), rather than Euclidean "
                              "error. Pair it with --norm-type hypersphere: on the unit sphere the only "
                              "thing two latents can differ by is angle, so an L2 penalty also "
                              "charges for a radius gap the projection already removed.")
    parser.add_argument("--pred-weight", type=float, default=1.0)
    parser.add_argument("--straighten-weight", type=float, default=0.0)
    parser.add_argument("--vicreg-std-weight", type=float, default=0.0)
    parser.add_argument("--vicreg-cov-weight", type=float, default=0.0)
    parser.add_argument("--vicreg-granularity", choices=["agg", "token"], default="agg",
                         help="agg: VICReg on the aggregated per-frame vector (same input as the "
                              "curvature loss). token: VICReg on the raw per-token latents, every "
                              "(batch, frame, patch) token a separate sample -- the reference's "
                              "own behavior (VWorldModel applies it to visual_prop(z), never the "
                              "agg head's output). Only differs when an aggregation head exists "
                              "(projector_type=channel, or --cls-agg-head).")
    axis_help = ("which samples the regularizer scores together. pooled: all B*T frames in one "
                 "set (original). per_time: at each timestep t the B clips form the set, "
                 "averaged over t (LeWM's SIGReg -- statistics ACROSS clips, no pooling over "
                 "time). per_clip: inside each clip its T frames form the set, averaged over "
                 "clips (needs T >> D for cov/std to be meaningful)")
    parser.add_argument("--vicreg-axis", choices=["pooled", "per_time", "per_clip"],
                         default="pooled", help=axis_help)
    parser.add_argument("--sigreg-axis", choices=["pooled", "per_time", "per_clip"],
                         default="pooled", help=axis_help)
    parser.add_argument("--uniformity-weight", type=float, default=0.0,
                         help="Wang & Isola (ICML 2020) hyperspherical uniformity, a third "
                              "anti-collapse option next to VICReg and SIGReg (see "
                              "losses/uniformity_loss.py). Scale-free and with no d/N noise "
                              "floor, unlike vicreg_cov_weight. Intended with --norm-type hypersphere; "
                              "it L2-normalizes internally either way.")
    parser.add_argument("--uniformity-t", type=float, default=2.0,
                         help="uniformity kernel width t in exp(-t*||xi-xj||^2). Paper default 2.0.")
    parser.add_argument("--uniformity-pairs", type=int, default=2048,
                         help="random pairs sampled per call for the uniformity Monte-Carlo "
                              "estimate (all-pairs would be O(N^2)).")
    parser.add_argument("--sigreg-weight", type=float, default=0.0,
                         help="LeJEPA's Sketched Isotropic Gaussian Regularization -- "
                              "independent of vicreg_*_weight, so set those to 0 to use "
                              "SIGReg alone (see losses/sigreg_loss.py)")
    parser.add_argument("--sigreg-num-slices", type=int, default=1024,
                         help="random projection directions SIGReg averages its "
                              "per-slice normality statistic over")
    parser.add_argument("--sigreg-granularity", choices=["agg", "token"], default="agg",
                         help="agg: SIGReg on the aggregated per-frame vector (same input as "
                              "the curvature loss, and SIGReg's only behavior before this flag "
                              "existed). token: SIGReg on the raw per-token latents, every "
                              "(batch, frame, patch) token a separate sample -- mirrors "
                              "--vicreg-granularity=token. Only differs when an aggregation head exists (projector_type=channel, or --cls-agg-head). "
                              "NOTE: expect the loss magnitude to differ from agg (it's an "
                              "Epps-Pulley-style test statistic, more samples means more power "
                              "to detect real non-Gaussianity, not a fixed multiplicative scale "
                              "factor) -- --sigreg-weight may need retuning when switching.")
    parser.add_argument("--chord-weight", type=float, default=0.0,
                         help="ties every step in a clip to the clip's own global chord "
                              "(z[-1]-z[0]) direction, not just adjacent steps to each other "
                              "like straighten_weight -- complements it, doesn't replace it "
                              "(see losses/world_model_loss.py's chord_loss)")
    parser.add_argument("--stop-grad", dest="stop_grad", action="store_true", default=True)
    parser.add_argument("--no-stop-grad", dest="stop_grad", action="store_false")

    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-4,
                         help="used for the WHOLE model when --encoder-lr/--predictor-lr are "
                              "both unset (this project's original single-optimizer behavior); "
                              "otherwise the fallback for whichever of the two isn't set")
    parser.add_argument("--encoder-lr", type=float, default=None,
                         help="separate LR for encoder+projector+agg_head (reference Table 3: "
                              "1e-5 for spatial/channel projectors, 1e-6 for the global "
                              "projector). Leave unset to use --lr for the whole model.")
    parser.add_argument("--predictor-lr", type=float, default=None,
                         help="separate LR for the predictor (reference Table 3: 5e-4). "
                              "Leave unset to use --lr for the whole model.")
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--grad-clip", type=float, default=1.0,
                         help="max grad norm before optimizer.step(); 0 disables")
    parser.add_argument("--batch-size", type=int, default=32,
                         help="per GPU: a multi-GPU run (NUM_GPUS>1 in train.sh) sees this times the "
                              "world size per optimizer step, with no automatic LR scaling")
    parser.add_argument("--amp", action="store_true", default=False,
                         help="bf16 autocast on the forward+loss pass (CUDA only, no-op "
                              "elsewhere) -- off by default so existing configs/checkpoints "
                              "stay numerically reproducible; opt in for real speed on new runs "
                              "(tensor cores, and a better shot at scaled_dot_product_attention's "
                              "fused/flash kernel, which prefers fp16/bf16 over fp32)")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--checkpoint-dir", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--progress-every", type=int, default=0,
                         help="print a running-average progress line every N batches "
                              "within an epoch; 0 (default) disables, only the "
                              "end-of-epoch summary prints. Most useful for a slow "
                              "first epoch where every video's features are computed "
                              "live (nothing precomputed yet) -- e.g. 200 for K400-scale.")
    parser.add_argument("--save-every", type=int, default=50,
                         help="also save a standalone epoch{N}.ckpt every N epochs, "
                              "alongside the always-on last.ckpt/best.ckpt; 0 disables")

    if pre_args.config:
        with open(pre_args.config) as f:
            cfg = yaml.safe_load(f) or {}
        # Any argparse dest is settable from the yaml — including data/optim keys like
        # features and checkpoint_dir, so a config fully describes a run. Unknown keys are
        # an error rather than silently ignored: a typo (or a key copied from LiFT's
        # configs, e.g. ckpt_dir) would otherwise quietly train something else entirely.
        valid_dests = {a.dest for a in parser._actions}
        unknown = set(cfg) - valid_dests
        if unknown:
            raise SystemExit(f"{pre_args.config}: unknown config key(s) {sorted(unknown)}. "
                              f"Valid keys: {sorted(valid_dests - {'help'})}")
        parser.set_defaults(**cfg)
    args = parser.parse_args()

    if not args.checkpoint_dir:
        raise SystemExit("--checkpoint-dir is required (pass it directly, or set "
                          "'checkpoint_dir:' in --config)")
    if not args.train_manifest and not args.features:
        raise SystemExit("Pass --train-manifest, --features, or both: with neither there's "
                          "nothing to say which clips to train on.")
    if args.val_manifest and args.val_fraction > 0:
        raise SystemExit("Pass only one of --val-manifest / --val-fraction")
    if args.sample_fps is not None:
        if args.token_mode not in ("cls", "patch") or not args.train_manifest or args.features:
            raise SystemExit("--sample-fps needs --token-mode cls or patch with --train-manifest "
                              "and no --features file (frames are extracted live at that fps)")
        if args.window_size != 0 or args.val_fraction > 0:
            raise SystemExit("--sample-fps does not support --window-size > 0 or --val-fraction; "
                              "use --val-manifest for validation")
    if args.token_mode == "raw" and args.features:
        # Bulk-precomputed files are a DINOv2-feature format (N,T,D)/(N,T,N,D) -- there's
        # no raw-pixel equivalent, and no reason to want one: at K400 10% scale, raw
        # frames are ~58GB even as uint8 (vs <1GB for CLS features), so "precompute a
        # bulk file up front" loses to "decode once, live-cache does the rest" (see
        # feature_provider.py's token_mode='raw' docstring) rather than just being an
        # unsupported combination worth adding support for.
        raise SystemExit("--features is not supported with --token-mode raw (no raw-pixel bulk "
                          "format exists) -- use --train-manifest, which live-caches decoded "
                          "frames on first touch (see --live-cache-dir).")
    if args.token_mode in ("cls", "raw") and args.n_frames != args.max_len:
        # cls/raw mode's CausalPredictor sizes its positional table to max_len, and
        # expects every clip to be exactly that long (live-computed clips get n_frames
        # frames, precomputed ones have max_len; if a batch ever mixed the two sources
        # they wouldn't stack). Patch mode has no such constraint -- its predictor gets
        # T-1 source positions from the T-frame clip sized by predictor_num_frames().
        raise SystemExit(f"--n-frames ({args.n_frames}) must equal --max-len ({args.max_len}): "
                          f"on-the-fly features would otherwise have a different T than "
                          f"precomputed ones and fail to collate.")
    if args.token_mode == "patch" and args.window_size > 0 and args.window_size + 1 > args.n_frames:
        raise SystemExit(f"--window-size {args.window_size} needs {args.window_size + 1} "
                          f"frames/clip, but --n-frames is only {args.n_frames}.")

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    rank, world_size, device = init_distributed(args.device)
    is_main = rank == 0
    # Only rank 0 resolves the checkpoint-dir (each rank doing it would race on the
    # _1/_2 auto-suffixing) and owns every file in it; the others learn the result by
    # broadcast, and keep only stderr so a crash on any rank is still visible.
    ckpt_str = [None]
    if is_main:
        ckpt_str[0] = str(resolve_checkpoint_dir(Path(args.checkpoint_dir)))
    if world_size > 1:
        dist.broadcast_object_list(ckpt_str, src=0)
    ckpt_dir = Path(ckpt_str[0])
    if is_main:
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        # Own train.log directly (line-buffered, so it stays live) instead of leaving it to
        # train.sh's external `tee` -- see _Tee's own docstring for why.
        log_handle = open(ckpt_dir / "train.log", "a", buffering=1)
        sys.stdout = _Tee(sys.__stdout__, log_handle)
        sys.stderr = _Tee(sys.__stderr__, log_handle)
        print(f"Logging to {ckpt_dir / 'train.log'}")
        if str(ckpt_dir) != args.checkpoint_dir:
            print(f"--checkpoint-dir {args.checkpoint_dir} already has content -- using {ckpt_dir} instead")
        if world_size > 1:
            print(f"Distributed: {world_size} GPUs, batch_size {args.batch_size}/GPU "
                  f"-> {args.batch_size * world_size} per optimizer step")
    else:
        sys.stdout = open(os.devnull, "w")
    live_cache_dir = args.live_cache_dir or str(ckpt_dir / "live_cache")

    provider = FeatureProvider(
        features_path=args.features, live_cache_dir=live_cache_dir,
        n_frames=args.n_frames, model_id=args.model_id, device=str(device),
        token_mode=args.token_mode, patch_hub_id=args.patch_hub_id, patch_img_size=args.patch_img_size,
        raw_img_size=args.raw_img_size,
    )
    if args.features:
        feat_shape = provider.feature_shape  # (T, D) for cls, (T, N, D) for patch
        print(f"Loaded {len(provider.video_paths)} precomputed videos, features shape (N, {feat_shape})")
        # cls mode's clip length must match max_len (the CausalPredictor's own
        # positional-table size); patch mode's BlockCausalPredictor sizing is decoupled
        # from clip length (see predictor_num_frames()), so only n_frames matters here.
        expected_t = args.max_len if args.token_mode == "cls" else args.n_frames
        if feat_shape[0] != expected_t:
            raise SystemExit(f"expected T={expected_t} frames/video but {args.features} has "
                              f"T={feat_shape[0]}.")
        if args.token_mode == "cls":
            if feat_shape[1] != args.feature_dim:
                raise SystemExit(f"--feature-dim={args.feature_dim} but {args.features} has "
                                  f"D={feat_shape[1]}.")
        else:
            if len(feat_shape) != 3:
                raise SystemExit(f"--token-mode=patch expects (T,N,D) precomputed features, "
                                  f"got shape (T, {', '.join(map(str, feat_shape[1:]))}) from {args.features}.")
            if feat_shape[1] != args.num_patches:
                raise SystemExit(f"--num-patches={args.num_patches} but {args.features} has "
                                  f"N={feat_shape[1]} patches/frame.")
            if feat_shape[2] != args.feature_dim:
                raise SystemExit(f"--feature-dim={args.feature_dim} but {args.features} has "
                                  f"D={feat_shape[2]}.")

    if args.sample_fps is not None:
        train_ds = RandomWindowDataset.from_manifest(
            args.train_manifest, provider, args.sample_fps, args.n_frames, path_col=args.path_col,
            duration_col=args.duration_col, short_clip_policy=args.short_clip_policy,
            windows_per_clip=args.windows_per_clip)
    elif args.train_manifest:
        train_ds = ClipFeatureDataset.from_manifest(
            args.train_manifest, provider, path_col=args.path_col, window_size=args.window_size,
            variable_length_min=args.variable_length_min)
    else:
        train_ds = ClipFeatureDataset(provider.video_paths, provider, window_size=args.window_size,
                                       variable_length_min=args.variable_length_min)

    n_precomputed = sum(provider.has_precomputed(p) for p in train_ds.video_paths)
    print(f"Train set: {len(train_ds)} clips, {n_precomputed} precomputed / "
          f"{len(train_ds) - n_precomputed} on-the-fly")

    val_ds = None
    if args.val_manifest and args.sample_fps is not None:
        val_ds = RandomWindowDataset.from_manifest(
            args.val_manifest, provider, args.sample_fps, args.n_frames, path_col=args.path_col,
            duration_col=args.duration_col, short_clip_policy=args.short_clip_policy,
            random_window=False,
        )
    elif args.val_manifest:
        val_ds = ClipFeatureDataset.from_manifest(
            args.val_manifest, provider, path_col=args.path_col,
            window_size=args.window_size, random_window=False,
        )
    elif args.val_fraction > 0:
        base_train_ds = train_ds
        n_val = max(1, int(len(train_ds) * args.val_fraction))
        train_subset, val_subset = random_split(
            range(len(base_train_ds)), [len(base_train_ds) - n_val, n_val],
            generator=torch.Generator().manual_seed(args.seed),
        )
        train_ds = torch.utils.data.Subset(base_train_ds, list(train_subset))
        val_base = ClipFeatureDataset(
            base_train_ds.video_paths, provider, window_size=args.window_size,
            random_window=False,
        )
        val_ds = torch.utils.data.Subset(val_base, list(val_subset))
        print(f"Split: {len(train_ds)} train / {len(val_ds)} val")

    # spawn, not the platform default fork: FeatureProvider lazily loads its DINOv2
    # backbone onto CUDA on first .get() call (see feature_provider.py), which happens
    # for the first time INSIDE whichever worker draws the first not-yet-cached video.
    # A forked worker can't touch CUDA at all -- "Cannot re-initialize CUDA in forked
    # subprocess" -- since it inherits the parent's already-initialized CUDA context.
    # spawn gives each worker a fresh process that initializes its own CUDA context
    # cleanly, keeping real multi-worker parallelism (vs. num_workers=0, which would
    # serialize video decode + DINOv2 forward + training step with no prefetch overlap
    # -- a real cost at K400-scale, where nothing is precomputed and every video's
    # features get computed live on first touch).
    loader_kwargs = {"multiprocessing_context": "spawn"} if args.num_workers > 0 else {}
    # pin_memory: lets the h2d copy in run_epoch (non_blocking=True) actually overlap
    # with compute instead of blocking -- only meaningful for a CUDA device.
    pin_memory = device.type == "cuda"
    # Multi-GPU: each rank draws its own disjoint 1/world_size of the clips per epoch.
    # drop_last for train so no clip is repeated to even out the shards (a repeated clip
    # would be touched by two ranks at once, including in the live cache); val pads
    # instead, so every val clip is scored (at most world_size-1 of them twice).
    train_sampler = val_sampler = None
    if world_size > 1:
        train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank,
                                           shuffle=True, seed=args.seed, drop_last=True)
        if val_ds is not None:
            val_sampler = DistributedSampler(val_ds, num_replicas=world_size, rank=rank, shuffle=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=train_sampler is None,
                               sampler=train_sampler, num_workers=args.num_workers,
                               pin_memory=pin_memory, **loader_kwargs)
    val_loader = None
    if val_ds is not None:
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, sampler=val_sampler,
                                 num_workers=args.num_workers, pin_memory=pin_memory, **loader_kwargs)

    model = build_model(args).to(device)
    criterion = build_criterion(args)
    optimizer = build_optimizer(model, args)
    step = None
    if world_size > 1:
        if args.norm_type == "batchnorm":
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        # DDP broadcasts rank 0's weights, so identical init regardless of seed; reseed per
        # rank afterwards so dropout masks differ across GPUs.
        # find_unused_parameters: with e.g. --straighten-weight 0 the aggregation head gets
        # no gradient, which plain DDP treats as an error.
        step = DistributedDataParallel(
            TrainStep(model, criterion), device_ids=[device.index] if device.type == "cuda" else None,
            find_unused_parameters=True)
        torch.manual_seed(args.seed + rank)
        random.seed(args.seed + rank)

    if is_main:
        with open(ckpt_dir / "config.yaml", "w") as f:
            yaml.safe_dump(vars(args), f)

    log_path = ckpt_dir / "train_log.csv"
    best_val = float("inf")
    # Written per epoch rather than once at the end: a job killed by walltime or a crash
    # would otherwise lose its entire loss history despite having saved checkpoints.
    log_file = None
    writer = None

    # Everything above this (feature loading, model/optimizer construction) can take a
    # while on its own (e.g. torch.load of a multi-GB precomputed features file) -- this
    # line is what actually distinguishes "still starting up" from "stuck/hung" when
    # checking train.log, since nothing else prints between here and the first epoch's
    # own summary line (which itself can be a long time out at batch=16/window_size=0).
    print(f"Training started: {args.epochs} epochs, {len(train_loader)} batches/epoch, "
          f"device={device}", flush=True)

    for epoch in range(args.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)  # new shuffle each epoch, same on every rank
        train_metrics = run_epoch(model, criterion, train_loader, device, optimizer,
                                   grad_clip=args.grad_clip, desc=f"epoch {epoch} train",
                                   progress_every=args.progress_every, amp=args.amp,
                                   step=step, world_size=world_size)
        row = {"epoch": epoch, **{f"train_{k}": v for k, v in train_metrics.items()}}

        if val_loader is not None:
            val_metrics = run_epoch(model, criterion, val_loader, device, optimizer=None,
                                     desc=f"epoch {epoch} val", progress_every=args.progress_every,
                                     amp=args.amp, step=step, world_size=world_size)
            row.update({f"val_{k}": v for k, v in val_metrics.items()})
            monitor = val_metrics["total"]
        else:
            monitor = train_metrics["total"]

        if is_main:
            if writer is None:
                log_file = open(log_path, "w", newline="")
                writer = csv.DictWriter(log_file, fieldnames=list(row))
                writer.writeheader()
            writer.writerow(row)
            log_file.flush()

            if epoch % args.log_every == 0 or epoch == args.epochs - 1:
                print(f"epoch {epoch}: " + ", ".join(f"{k}={v:.4f}" for k, v in row.items() if k != "epoch"))

            torch.save({"model": model.state_dict(), "epoch": epoch}, ckpt_dir / "last.ckpt")
            if monitor < best_val:
                best_val = monitor
                torch.save({"model": model.state_dict(), "epoch": epoch}, ckpt_dir / "best.ckpt")
            if args.save_every > 0 and (epoch + 1) % args.save_every == 0:
                torch.save({"model": model.state_dict(), "epoch": epoch}, ckpt_dir / f"epoch{epoch}.ckpt")

    if log_file is not None:
        log_file.close()
    print(f"Done. Checkpoints + config + log -> {ckpt_dir}")
    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()
