import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import synthetic_latent as sl
from metrics_batch import per_clip_metrics

def test_closed_forms():
    for r in (0.0, 0.05, 0.3, 1.0, 5.0, 50.0):
        pc, keep = per_clip_metrics(sl.simulate(400, 16, 384, r, seed=1))
        vc, st = pc["mean_velocity_cosine_raw"][keep].mean(), pc["trajectory_straightness_raw"][keep].mean()
        assert abs(vc - sl.vel_cos_theory(r)) < 0.02, (r, vc, sl.vel_cos_theory(r))
        assert abs(st - sl.straight_theory(r)) < 0.02, (r, st, sl.straight_theory(r))
        if r > 0.05:
            assert abs(sl.infer_r_velcos(vc) - r) / r < 0.15
            assert abs(sl.infer_r_straight(st) - r) / r < 0.2
    print("ok")

if __name__ == "__main__":
    test_closed_forms()
