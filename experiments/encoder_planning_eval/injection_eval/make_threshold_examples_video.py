"""Render aligned synthetic edits with calibrated prefix-score threshold crossings."""
import argparse
import csv
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

from injections import INJECTIONS, rng_for
from run_injection_eval import DATASETS, load_world_model, pick_donors


HERE = Path(__file__).resolve().parent
RUN = HERE / "results" / "egoper_style_ssv2_vicreg_clean_concat_epoch49"
FRAME_CACHE = HERE / "results" / "frame_cache" / "frames_ssv2_n397_seed0_224.pt"
OUT = HERE / "results" / "threshold_examples"
CASES = (("drop", 4), ("splice_other", 308), ("speed_up", 123))
EXTRA_CASES = {"shuffle": ("shuffle", 4), "swap": ("swap", 293)}
FPS = 12
REPEAT = 6
FINAL_HOLD = 24
WIDTH, HEIGHT = 1280, 720
BLUE = (220, 150, 50)       # DINOv2, BGR
GREEN = (90, 220, 80)       # VICREG concat, BGR
WHITE = (245, 245, 245)
RED = (75, 90, 230)


def prefix_score(steps, score):
    if score == "jump_spike":
        return steps.max(axis=-1) / (np.median(steps, axis=-1) + 1e-8)
    if score == "step_cv":
        return steps.std(axis=-1) / (steps.mean(axis=-1) + 1e-8)
    raise ValueError(score)


def curves(saved, injection, index, score, reference):
    output = {}
    for readout in ("raw", "vicreg_clean"):
        step = saved[f"test|{readout}|{injection}|step"][index]
        clean = saved[f"calibration|{readout}|{reference}|step"]
        ratio = np.full(len(step) + 1, np.nan)
        values = np.full(len(step) + 1, np.nan)
        thresholds = np.full(len(step) + 1, np.nan)
        if injection in ("shuffle", "swap"):
            # A local signal: transition t reaches edited frame t+2 (one-based).
            values[1:] = step
            thresholds[1:] = np.quantile(clean, .95, axis=0)
            ratio[1:] = values[1:] / thresholds[1:]
        else:
            for length in range(5, len(step) + 2):
                values[length - 1] = prefix_score(step[:length - 1], score)
                thresholds[length - 1] = np.quantile(
                    prefix_score(clean[:, :length - 1], score), .95)
                ratio[length - 1] = values[length - 1] / thresholds[length - 1]
        output[readout] = (ratio, values, thresholds)
    return output


def chord_score(z, length):
    """Prefix version of scores.geometry_scores(...)[chord_dev]."""
    z = np.asarray(z[:, :length], dtype=np.float64)
    steps = np.linalg.norm(np.diff(z, axis=1), axis=-1)
    rel = z - z[:, :1]
    chord = z[:, -1] - z[:, 0]
    progress = (rel * chord[:, None]).sum(-1) / ((chord * chord).sum(-1)[:, None] + 1e-8)
    dev = np.linalg.norm(rel - progress[..., None] * chord[:, None], axis=-1)
    return dev.max(1) / (steps.mean(1) + 1e-8)


def splice_curves(saved, index, paths, calibration_paths, return_paths=False):
    """Reconstruct a splice path and clean prefix thresholds in both feature spaces."""
    feature_file = torch.load(DATASETS["ssv2"]["features"], mmap=True, weights_only=False)
    lookup = {path: i for i, path in enumerate(feature_file["video_paths"])}
    features = feature_file["features"]
    manifest = pd.read_csv(DATASETS["ssv2"]["manifest"], dtype=str)
    label_of = dict(zip(manifest["video_path"], manifest["verb"]))
    donor_index = pick_donors([label_of[p] for p in paths], 0)[0][index]
    source = features[lookup[paths[index]]].float()
    donor = features[lookup[paths[donor_index]]].float()
    edited, _ = INJECTIONS["splice_other"](source, rng_for(0, "splice_other", index), donor)
    model_dir = RUN.parent / "egoper_style_ssv2_vicreg_clean_epoch49_snapshot"
    model, _, _ = load_world_model(model_dir / "best.ckpt", model_dir / "config.yaml",
                                    torch.device("cpu"), with_predictor=False)
    torch.set_num_threads(4)
    clean = features[torch.tensor([lookup[p] for p in calibration_paths])].float()
    with torch.no_grad():
        clean_latent = np.concatenate([model.encoder(x).flatten(2).numpy()
                                       for x in clean.split(8)])
        edited_latent = model.encoder(edited[None]).flatten(2).numpy()
    paths_by_readout = {"raw": (clean.mean(2).numpy(), edited.mean(1).numpy()[None]),
                        "vicreg_clean": (clean_latent, edited_latent)}
    output = {}
    for readout, (normal, changed) in paths_by_readout.items():
        ratio = np.full(16, np.nan)
        values = np.full(16, np.nan)
        thresholds = np.full(16, np.nan)
        for length in range(5, 17):
            values[length - 1] = chord_score(changed, length)[0]
            thresholds[length - 1] = np.quantile(chord_score(normal, length), .95)
            ratio[length - 1] = values[length - 1] / thresholds[length - 1]
        saved_full = saved[f"test|{readout}|splice_other|chord_dev"][index]
        if not np.isclose(values[-1], saved_full):
            raise RuntimeError(f"reconstructed splice score differs for {readout}")
        output[readout] = (ratio, values, thresholds)
    source_index, _ = INJECTIONS["splice_other"](torch.arange(16),
                                                   rng_for(0, "splice_other", index),
                                                   torch.arange(16) + 100)
    result = output, donor_index, [int(v) for v in source_index.tolist()]
    return (*result, paths_by_readout) if return_paths else result


def put(canvas, value, point, scale=.7, color=WHITE, thick=2):
    cv2.putText(canvas, value, point, cv2.FONT_HERSHEY_SIMPLEX, scale,
                (0, 0, 0), thick + 3, cv2.LINE_AA)
    cv2.putText(canvas, value, point, cv2.FONT_HERSHEY_SIMPLEX, scale,
                color, thick, cv2.LINE_AA)


def panel(canvas, frame, x, y, title, source_index, highlight=False):
    picture = cv2.resize(frame[..., ::-1], (390, 390), interpolation=cv2.INTER_LINEAR)
    canvas[y:y + 390, x:x + 390] = picture
    cv2.rectangle(canvas, (x, y), (x + 389, y + 389), GREEN if highlight else (140, 140, 140), 3)
    put(canvas, title, (x, y - 12), .75)
    label = (f"donor frame {source_index - 99}/16" if source_index >= 100
             else f"source frame {source_index + 1}/16")
    put(canvas, label, (x + 8, y + 365), .58)


def plot(canvas, curves_by_readout, frame_index, count, first_cross,
         signal_kind="prefix", seam_frames=()):
    x0, x1, y0, y1 = 890, 1240, 178, 510
    cv2.rectangle(canvas, (x0, y0), (x1, y1), (62, 62, 62), -1)
    ymax = max(2.8, *(np.nanmax(x[0]) + .15 for x in curves_by_readout.values()))
    first_valid = 2 if signal_kind == "step" else 5
    def xy(k, ratio):
        x = x0 + 28 + int((k - first_valid) / max(count - first_valid, 1) * (x1 - x0 - 48))
        y = y1 - 28 - int(ratio / ymax * (y1 - y0 - 52))
        return x, y
    threshold_y = xy(first_valid, 1.0)[1]
    cv2.line(canvas, (x0 + 28, threshold_y), (x1 - 20, threshold_y), WHITE, 2)
    put(canvas, "threshold = 1.0", (x0 + 35, threshold_y - 8), .50)
    for seam_frame in seam_frames:
        seam_x = xy(seam_frame, 0)[0]
        for top in range(y0 + 48, y1 - 25, 14):
            cv2.line(canvas, (seam_x, top), (seam_x, min(top + 7, y1 - 25)),
                     (50, 190, 240), 1)
    for readout, color in (("raw", BLUE), ("vicreg_clean", GREEN)):
        ratio = curves_by_readout[readout][0]
        for k in range(first_valid + 1, min(frame_index + 2, count + 1)):
            if np.isfinite(ratio[k - 2]) and np.isfinite(ratio[k - 1]):
                cv2.line(canvas, xy(k - 1, ratio[k - 2]), xy(k, ratio[k - 1]), color, 3)
        current = ratio[frame_index] if frame_index >= first_valid - 1 else np.nan
        if np.isfinite(current):
            cv2.circle(canvas, xy(frame_index + 1, current), 7, color, -1)
    concat_ratio = curves_by_readout["vicreg_clean"][0]
    marked = ([n for n in seam_frames if concat_ratio[n - 1] > 1.0]
              if signal_kind == "step" else [first_cross])
    for crossing_frame in marked:
        if frame_index + 1 >= crossing_frame:
            px, py = xy(crossing_frame, concat_ratio[crossing_frame - 1])
            cv2.circle(canvas, (px, py), 13, RED, 3)
    if frame_index + 1 >= first_cross:
        label = ("crosses: frames " + ", ".join(map(str, marked))
                 if signal_kind == "step" else f"first cross: frame {first_cross}")
        put(canvas, label, (x0 + 20, y0 + 35), .52, RED)
    put(canvas, "Frames observed", (x0 + 75, y1 + 28), .56)
    put(canvas, "Step / clean p95" if signal_kind == "step" else "Score / clean p95",
        (x0 + 5, y0 - 12), .62)


def render_case(title, injection, index, source_frames, source_indices, curves_by_readout,
                first_cross, donor_frames=None, seam_frames=()):
    count = len(source_indices)
    for frame_index in range(count):
        canvas = np.full((HEIGHT, WIDTH, 3), 27, dtype=np.uint8)
        put(canvas, title, (35, 48), .95)
        put(canvas, "Synthetic edit on an SSv2 held-out clip", (35, 80), .60)
        edited_source = source_indices[frame_index]
        panel(canvas, source_frames[min(frame_index, 15)], 35, 160,
              "Original sampled video", min(frame_index, 15))
        edited_frame = (donor_frames[edited_source - 100] if edited_source >= 100
                        else source_frames[edited_source])
        panel(canvas, edited_frame, 455, 160,
              f"Edited: {injection.replace('_', ' ')}", edited_source,
              frame_index + 1 >= first_cross)
        signal_kind = "step" if injection in ("shuffle", "swap") else "prefix"
        plot(canvas, curves_by_readout, frame_index, count, first_cross,
             signal_kind, seam_frames)
        ratio_raw = curves_by_readout["raw"][0][frame_index]
        ratio_concat = curves_by_readout["vicreg_clean"][0][frame_index]
        if frame_index >= (1 if signal_kind == "step" else 4):
            put(canvas, f"DINOv2 patch mean: {ratio_raw:.2f}x", (895, 565), .62,
                RED if ratio_raw > 1 else BLUE)
            put(canvas, f"VICREG patch concat: {ratio_concat:.2f}x", (895, 602), .62,
                GREEN if ratio_concat > 1 else WHITE)
        else:
            put(canvas, "Score starts after 2 frames" if signal_kind == "step"
                else "Scores start after 5 frames", (895, 565), .57)
        if frame_index == count - 1:
            conclusion = ("AT EDIT SEAMS: DINOv2 BELOW  |  VICREG ABOVE"
                          if signal_kind == "step" else
                          "FULL CLIP: DINOv2 MISS  |  VICREG CONCAT HIT")
            put(canvas, conclusion, (35, 630), .68, GREEN)
        else:
            put(canvas, f"Observed frame {frame_index + 1}/{count}", (35, 630), .65)
        note = ("Dashed lines = edit seams. Threshold is clean p95 at each transition."
                if signal_kind == "step" else
                "Each prefix uses its own clean-calibration p95 threshold.")
        put(canvas, note, (35, 668), .59)
        for _ in range(FINAL_HOLD if frame_index == count - 1 else REPEAT):
            yield canvas


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", choices=EXTRA_CASES,
                        help="render one additional example as its own video")
    args = parser.parse_args()
    selected_cases = (EXTRA_CASES[args.only],) if args.only else CASES
    video_name = f"{args.only}.mp4" if args.only else "dino_miss_concat_hit.mp4"
    details_name = f"{args.only}_case.json" if args.only else "cases.json"
    OUT.mkdir(parents=True, exist_ok=True)
    protocol = json.loads((RUN / "protocol.json").read_text())
    paths = protocol["test_paths"]
    cached = torch.load(FRAME_CACHE, mmap=True, weights_only=False)
    lookup = {path: i for i, path in enumerate(cached["paths"])}
    if any(paths[i] not in lookup for _, i in selected_cases):
        raise RuntimeError("selected video is absent from frame cache")
    metrics = {(r["readout"], r["injection"]): r
               for r in csv.DictReader((RUN / "segment_metrics.csv").open())
               if r["channel"] == "geometry"}
    ffmpeg = subprocess.Popen(["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo",
                               "-pix_fmt", "bgr24", "-s", f"{WIDTH}x{HEIGHT}",
                               "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-crf", "20",
                               "-pix_fmt", "yuv420p", str(OUT / video_name)],
                              stdin=subprocess.PIPE)
    details = []
    try:
        with np.load(RUN / "per_clip_scores.npz") as saved:
            for injection, index in selected_cases:
                reference = "clean12" if injection in ("drop", "speed_up") else "clean16"
                score = metrics["raw", injection]["score"]
                if injection == "splice_other":
                    curve, donor_index, source_indices = splice_curves(
                        saved, index, paths, protocol["calibration_paths"])
                    donor_frames = cached["frames"][lookup[paths[donor_index]]].permute(
                        0, 2, 3, 1).numpy()
                else:
                    curve = curves(saved, injection, index, score, reference)
                    source_index, _ = INJECTIONS[injection](torch.arange(16),
                                                             rng_for(0, injection, index))
                    source_indices = [int(v) for v in source_index.tolist()]
                    donor_frames = None
                first = np.flatnonzero(curve["vicreg_clean"][0] > 1.0)
                if len(first) == 0 or np.any(curve["raw"][0] > 1.0):
                    raise RuntimeError(f"case {injection}/{index} lacks clean threshold crossing")
                first_cross = int(first[0]) + 1
                frames = cached["frames"][lookup[paths[index]]].permute(0, 2, 3, 1).numpy()
                for readout in ("raw", "vicreg_clean"):
                    value = float(saved[f"test|{readout}|{injection}|{score}"][index])
                    threshold = float(metrics[readout, injection]["threshold"])
                    if injection not in ("shuffle", "swap") and (not np.isclose(value, curve[readout][1][-1])
                            or not np.isclose(threshold, curve[readout][2][-1])):
                        raise RuntimeError("prefix final score differs from saved clip score")
                label = f"Case {len(details) + 1}: {injection.replace('_', ' ')}"
                seam_frames = ()
                if injection in ("shuffle", "swap"):
                    seams = json.loads((RUN / "test_seams.json").read_text())
                    seam_frames = tuple(t + 2 for t in seams[injection][index])
                for canvas in render_case(label, injection, index, frames, source_indices,
                                          curve, first_cross, donor_frames, seam_frames):
                    ffmpeg.stdin.write(canvas.tobytes())
                details.append({"injection": injection, "test_index": index,
                                "video_path": paths[index], "score": score,
                                "donor_test_index": donor_index if injection == "splice_other" else None,
                                "source_frame_indices_0_based": source_indices,
                                "concat_first_cross_frame_1_based": first_cross,
                                "edit_seam_target_frames_1_based": seam_frames,
                                "signal": "per_transition_step" if injection in ("shuffle", "swap")
                                          else f"prefix_{score}",
                                "dino_last_displayed_score": float(curve["raw"][1][-1]),
                                "dino_last_displayed_threshold": float(curve["raw"][2][-1]),
                                "concat_last_displayed_score": float(curve["vicreg_clean"][1][-1]),
                                "concat_last_displayed_threshold": float(curve["vicreg_clean"][2][-1]),
                                "dino_full_clip_score": float(saved[
                                    f"test|raw|{injection}|{score}"][index]),
                                "dino_full_clip_threshold": float(metrics[
                                    "raw", injection]["threshold"]),
                                "concat_full_clip_score": float(saved[
                                    f"test|vicreg_clean|{injection}|{score}"][index]),
                                "concat_full_clip_threshold": float(metrics[
                                    "vicreg_clean", injection]["threshold"])})
    finally:
        ffmpeg.stdin.close()
        if ffmpeg.wait() != 0:
            raise RuntimeError("ffmpeg failed")
    (OUT / details_name).write_text(json.dumps(details, indent=2))
    print(f"saved -> {OUT / video_name}")


if __name__ == "__main__":
    main()
