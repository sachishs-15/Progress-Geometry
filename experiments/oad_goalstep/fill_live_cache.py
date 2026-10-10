"""Fill the dense patch-token live cache ahead of training, sharded over processes.

  python fill_live_cache.py --cache-dir <dir> --data-info <data_info.json> --shard-id I --num-shards N [--batch 32]

Same DensePatchProvider.get() the training data layer calls on a miss; running it ahead
just keeps decoding (about 9 features/s per thread) out of the training loop.
"""
import argparse
import json

from dense_provider import DensePatchProvider


def main(args):
    info = json.load(open(args.data_info))["GoalStep"]
    uids = (info["train_session_set"] + info["test_session_set"])[args.shard_id::args.num_shards]
    provider = DensePatchProvider(args.cache_dir, batch=args.batch, device="cuda:0")
    for uid in uids:
        try:
            print(uid, provider.get(uid).shape, flush=True)
        except Exception as e:
            print(f"FAILED {uid}: {e}", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--cache-dir", required=True)
    p.add_argument("--data-info", required=True)
    p.add_argument("--shard-id", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--batch", type=int, default=32)
    main(p.parse_args())
