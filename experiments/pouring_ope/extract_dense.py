"""Every-frame DINOv2 patch tokens for Pouring, preprocessed exactly like FeatureProvider's
token_mode='patch' training path: torch.hub dinov2_vits14, Resize((196, 196)) (no crop),
ImageNet normalize, x_norm_patchtokens. Writes the same cache layout as extract.py under
the tag `hubdinov2_vits14_i196_s<stride>`.

    python extract_dense.py [--stride 1] [--limit N]
Then: python run_ope.py --encoder hubdinov2_vits14 --image-size 196 --pool mean
"""
import argparse
import time

import torch
import torchvision.transforms as T
from decord import VideoReader, cpu
from PIL import Image

from pouring_data import cache_dir, load_split

HUB_ID, SIZE, TAG = "dinov2_vits14", 196, "hubdinov2_vits14"

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--splits", nargs="+", default=["train", "val"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    out_dir = cache_dir(TAG, None, SIZE, args.stride)
    out_dir.mkdir(parents=True, exist_ok=True)
    model = torch.hub.load("facebookresearch/dinov2", HUB_ID).to(args.device).eval()
    prep = T.Compose([T.Resize((SIZE, SIZE)), T.ToTensor(),
                      T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])
    for split in args.splits:
        for v in load_split(split)[:args.limit]:
            path = out_dir / f"{v.name}.patch.pt"
            if path.exists():
                continue
            t0 = time.time()
            vr = VideoReader(v.path, ctx=cpu(0), num_threads=1)
            idx = list(range(0, len(vr), args.stride))
            rows = []
            for i in range(0, len(idx), args.batch):
                frames = vr.get_batch(idx[i:i + args.batch]).asnumpy()
                x = torch.stack([prep(Image.fromarray(f)) for f in frames]).to(args.device)
                with torch.no_grad():
                    rows.append(model.forward_features(x)["x_norm_patchtokens"].half().cpu())
            torch.save({"tokens": torch.cat(rows), "frame_idx": torch.tensor(idx), "grid_hw": (SIZE // 14, SIZE // 14)}, path)
            print(f"{v.name}: {len(idx)} frames, {time.time() - t0:.1f}s", flush=True)
