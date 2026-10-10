"""Render selected offline examples from the 200-calibration/800-test SSv2 run."""
import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from injections import INJECTIONS, rng_for
from make_offline_explanation_videos import (FINAL_HOLD, FPS, HEIGHT, REPEAT, WIDTH,
                                             contribution, display_traces, render)
from run_injection_eval import DATASETS, load_world_model, pick_donors

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
BASE = RESULTS / "egoper_style_ssv2_1000_cal200_eval800_vicreg_concat"
CLS = RESULTS / "egoper_style_ssv2_1000_cal200_eval800_dinov2_cls"
OUT = RESULTS / "threshold_examples_1000"
CASES = (("drop", 5), ("swap", 283), ("shuffle", 793),
         ("splice_other", 417), ("speed_up", 587))
READOUTS = (("dinov2_cls", CLS), ("raw", BASE), ("vicreg_clean", BASE))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    protocol = json.loads((BASE / "protocol.json").read_text())
    paths = protocol["test_paths"]
    manifest = pd.read_csv(DATASETS["ssv2"]["manifest"], dtype=str)
    label_of = dict(zip(manifest.video_path, manifest.verb))
    donors = pick_donors([label_of[p] for p in paths], protocol["args"]["seed"])[0]
    seams = json.loads((BASE / "test_seams.json").read_text())
    metric = {}
    for readout, folder in READOUTS:
        metric.update({(readout, row["injection"]): row
                       for row in csv.DictReader((folder / "segment_metrics.csv").open())
                       if row["readout"] == readout})
    cached = torch.load(OUT / "candidate_frames.pt", mmap=True, weights_only=False)
    frame_cache = dict(zip(cached["paths"], cached["frames"]))
    sys.path.insert(0, str(HERE.parents[1] / "predictive_world_model"))
    from data.feature_provider import FeatureProvider
    provider = FeatureProvider(None, live_cache_dir=None, n_frames=16,
                               token_mode="raw", raw_img_size=224, device="cpu")

    def frames_for(path):
        if path not in frame_cache:
            frame_cache[path] = provider._compute_live_raw(path)
        return frame_cache[path].permute(0, 2, 3, 1).numpy()

    with np.load(BASE / "per_clip_scores.npz") as base_scores, \
         np.load(CLS / "per_clip_scores.npz") as cls_scores:
        saved = {"dinov2_cls": cls_scores, "raw": base_scores,
                 "vicreg_clean": base_scores}
        rows = []
        for injection, index in CASES:
            score = metric["raw", injection]["score"]
            donor_index = donors[index] if injection == "splice_other" else None
            donor_path = paths[donor_index] if donor_index is not None else None
            source_path = paths[index]
            source_frames = frames_for(source_path)
            donor_frames = frames_for(donor_path) if donor_path else None
            seed = protocol["args"]["seed"]
            source_indices, _ = INJECTIONS[injection](
                torch.arange(16), rng_for(seed, injection, index),
                torch.arange(16) + 100 if donor_index is not None else None)
            source_indices = [int(v) for v in source_indices.tolist()]
            paths_by_readout = {}
            if injection == "splice_other":
                patch_data = torch.load(DATASETS["ssv2"]["features"], mmap=True, weights_only=False)
                patch_lookup = {p: j for j, p in enumerate(patch_data["video_paths"])}
                source = patch_data["features"][patch_lookup[source_path]].float()
                donor = patch_data["features"][patch_lookup[donor_path]].float()
                edited, _ = INJECTIONS[injection](source, rng_for(seed, injection, index), donor)
                paths_by_readout["raw"] = edited.mean(1).numpy()
                checkpoint = Path(protocol["models"]["vicreg_clean"]["checkpoint"])
                model, _, _ = load_world_model(checkpoint, checkpoint.parent / "config.yaml",
                                                torch.device("cpu"), with_predictor=False)
                torch.set_num_threads(4)
                with torch.no_grad():
                    paths_by_readout["vicreg_clean"] = model.encoder(edited[None]).flatten(2)[0].numpy()
                cls_data = torch.load(json.loads((CLS / "protocol.json").read_text())["features"],
                                      mmap=True, weights_only=False)
                cls_lookup = {p: j for j, p in enumerate(cls_data["video_paths"])}
                cls_source = cls_data["features"][cls_lookup[source_path]].float()
                cls_donor = cls_data["features"][cls_lookup[donor_path]].float()
                cls_edited, _ = INJECTIONS[injection](cls_source,
                                                    rng_for(seed, injection, index), cls_donor)
                paths_by_readout["dinov2_cls"] = cls_edited.numpy()
            profiles, values, thresholds = {}, {}, {}
            for readout, _ in READOUTS:
                archive = saved[readout]
                step = archive[f"test|{readout}|{injection}|step"][index]
                profile = contribution(score, step, paths_by_readout.get(readout))
                profiles[readout] = profile if score == "chord_dev" else np.r_[0, profile]
                values[readout] = float(archive[f"test|{readout}|{injection}|{score}"][index])
                thresholds[readout] = float(metric[readout, injection]["threshold"])
                computed = profile.max() if score != "step_cv" else np.sqrt(np.mean(profile ** 2))
                if not np.isclose(computed, values[readout], rtol=1e-4, atol=1e-5):
                    raise RuntimeError(f"{injection}/{index}/{readout} score mismatch")
            if not (values["raw"] < thresholds["raw"] and
                    values["vicreg_clean"] > thresholds["vicreg_clean"]):
                raise RuntimeError(f"{injection}/{index} is not a DINOv2 miss / VICREG hit")
            seam_frames = tuple(j + 2 for j in seams[injection][index])
            peak = int(np.argmax(profiles["vicreg_clean"])) + 1
            if score == "jump_spike" and peak not in seam_frames:
                raise RuntimeError(f"{injection}/{index}: strongest jump misses edit seam")
            if score == "chord_dev" and not (seam_frames[0] <= peak < seam_frames[-1]):
                raise RuntimeError(f"{injection}/{index}: strongest deviation outside donor span")
            traces = display_traces(profiles, score, thresholds)
            for readout in traces:
                shown = traces[readout][-1] if score == "step_cv" else traces[readout].max()
                if not np.isclose(shown, values[readout] / thresholds[readout], rtol=1e-4):
                    raise RuntimeError(f"{injection}/{index}/{readout} plotted score mismatch")
            out = OUT / f"offline_{injection}.mp4"
            ffmpeg = subprocess.Popen(["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo",
                                       "-pix_fmt", "bgr24", "-s", f"{WIDTH}x{HEIGHT}",
                                       "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-crf", "20",
                                       "-pix_fmt", "yuv420p", str(out)], stdin=subprocess.PIPE)
            try:
                for canvas in render(injection, index, source_frames, source_indices, donor_frames,
                                     profiles, score, values, thresholds, seam_frames):
                    ffmpeg.stdin.write(canvas.tobytes())
            finally:
                ffmpeg.stdin.close()
                if ffmpeg.wait() != 0:
                    raise RuntimeError(f"ffmpeg failed for {injection}")
            rows.append({"injection": injection, "test_index": index, "source_path": source_path,
                         "donor_index": donor_index, "score": score,
                         "source_frame_indices": source_indices, "seam_frames": seam_frames,
                         "scores": values, "thresholds": thresholds, "vicreg_peak_frame": peak})
            print(f"saved {out}", flush=True)
    (OUT / "cases.json").write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
