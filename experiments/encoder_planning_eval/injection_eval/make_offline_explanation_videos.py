"""Render whole-clip score decisions with post-hoc per-frame contributions."""
import argparse
import csv
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np
import torch

from injections import INJECTIONS, rng_for
from make_threshold_examples_video import (BLUE, FRAME_CACHE, GREEN, HEIGHT, OUT,
                                           RED, RUN, WHITE, WIDTH, panel, put,
                                           splice_curves)

CLS_COLOR = (190, 110, 215)

# Chosen from whole-clip scores: DINOv2 mean below its threshold, VICREG
# concat above its threshold, with large margins and visually distinct clips.
CASES = (("drop", 222), ("splice_other", 232), ("speed_up", 4),
         ("swap", 293), ("shuffle", 231))
FPS = 12
REPEAT = 6
FINAL_HOLD = 24


def contribution(score, step=None, path=None):
    if score == "jump_spike":
        median = np.median(step)
        return step / (median + 1e-8)
    if score == "step_cv":
        mean = step.mean()
        return np.abs(step - mean) / (mean + 1e-8)
    if score == "chord_dev":
        z = np.asarray(path, dtype=np.float64)
        chord = z[-1] - z[0]
        rel = z - z[:1]
        projection = (rel * chord).sum(-1) / ((chord * chord).sum() + 1e-8)
        distance = np.linalg.norm(rel - projection[:, None] * chord, axis=-1)
        average_step = np.linalg.norm(np.diff(z, axis=0), axis=-1).mean()
        return distance / (average_step + 1e-8)
    raise ValueError(score)


def display_traces(profiles, score, thresholds):
    """Offline contributions revealed in playback order; final value matches decision."""
    traces = {}
    for readout, p in profiles.items():
        if score == "step_cv":
            # p[1:] contains |step - full-clip mean| / full-clip mean. Its
            # cumulative RMS reaches the exact full-clip CV at the final frame.
            cumulative = np.r_[0, np.sqrt(np.cumsum(p[1:] ** 2) / (len(p) - 1))]
            traces[readout] = cumulative / thresholds[readout]
        else:
            traces[readout] = p / thresholds[readout]
    return traces


def plot(canvas, traces, score, frame_number, seam_frames):
    x0, x1, y0, y1 = 885, 1250, 260 if "dinov2_cls" in traces else 230, 510
    cv2.rectangle(canvas, (x0, y0), (x1, y1), (55, 55, 55), -1)
    n = len(next(iter(traces.values())))
    ymax = max(1.4, *(float(np.max(p)) * 1.2 for p in traces.values()))

    def xy(frame, value):
        x = x0 + 24 + int((frame - 1) / max(n - 1, 1) * (x1 - x0 - 48))
        y = y1 - 25 - int(value / ymax * (y1 - y0 - 50))
        return x, y

    y = xy(1, 1)[1]
    cv2.line(canvas, (x0 + 24, y), (x1 - 24, y), WHITE, 2)
    put(canvas, "full-clip threshold", (x0 + 33, y - 8), .42)
    for frame in seam_frames:
        x = xy(frame, 0)[0]
        for y in range(y0 + 20, y1 - 20, 14):
            cv2.line(canvas, (x, y), (x, y + 6), (30, 200, 230), 1)
    for readout, color in (("dinov2_cls", CLS_COLOR), ("raw", BLUE),
                           ("vicreg_clean", GREEN)):
        if readout not in traces:
            continue
        values = traces[readout]
        start = 1 if score == "chord_dev" else 2
        for frame in range(start + 1, frame_number + 1):
            a, b = float(values[frame - 2]), float(values[frame - 1])
            left, right = xy(frame - 1, a), xy(frame, b)
            if readout == "raw" or (a <= 1 and b <= 1):
                cv2.line(canvas, left, right, color, 3)
            elif a > 1 and b > 1:
                cv2.line(canvas, left, right, RED, 3)
            else:
                fraction = (1 - a) / (b - a)
                cross = xy((frame - 1) + fraction, 1)
                cv2.line(canvas, left, cross, GREEN if a <= 1 else RED, 3)
                cv2.line(canvas, cross, right, RED if b > 1 else GREEN, 3)
        if frame_number >= start:
            cv2.circle(canvas, xy(frame_number, values[frame_number - 1]), 5,
                       RED if readout == "vicreg_clean" and values[frame_number - 1] > 1 else color, -1)
    x = xy(frame_number, 0)[0]
    cv2.line(canvas, (x, y0 + 8), (x, y1 - 8), WHITE, 1)
    title = ("Offline chord contribution / threshold" if score == "chord_dev" else
             "Offline jump contribution / threshold" if score == "jump_spike" else
             "Offline cumulative CV / threshold")
    put(canvas, title, (x0 + 3, y0 - 15), .50)
    put(canvas, "Red = VICREG trace above threshold", (x0 + 18, y1 + 28), .45)


def render(injection, index, frames, source_indices, donor_frames, profiles,
           score, values, thresholds, seam_frames):
    count = len(source_indices)
    verdict = {r: values[r] > thresholds[r] for r in values}
    traces = display_traces(profiles, score, thresholds)
    for current in range(1, count + 1):
        canvas = np.full((HEIGHT, WIDTH, 3), 27, dtype=np.uint8)
        put(canvas, f"{injection.replace('_', ' ').title()} | offline whole-clip decision", (30, 45), .85)
        put(canvas, "Offline scores revealed frame by frame for visualization", (30, 75), .52)
        original_index = min(current - 1, 15)
        edited_index = source_indices[current - 1]
        panel(canvas, frames[original_index], 30, 165, "Original sampled video", original_index)
        edited_frame = (donor_frames[edited_index - 100] if edited_index >= 100
                        else frames[edited_index])
        panel(canvas, edited_frame, 445, 165,
              f"Edited: {injection.replace('_', ' ')}", edited_index,
              edited_index >= 100 or edited_index != current - 1)
        if traces["vicreg_clean"][current - 1] > 1:
            cv2.rectangle(canvas, (445, 165), (834, 554), RED, 4)
        put(canvas, f"Score: {score}", (890, 105), .59)
        score_lines = ((135, "dinov2_cls", CLS_COLOR, "DINOv2 CLS"),
                       (168, "raw", BLUE, "DINOv2 mean"),
                       (201, "vicreg_clean", GREEN, "VICREG concat")) if "dinov2_cls" in values else (
                       (148, "raw", BLUE, "DINOv2 mean"),
                       (185, "vicreg_clean", GREEN, "VICREG concat"))
        for line, readout, color, title in score_lines:
            decision = "FLAG" if verdict[readout] else "NO FLAG"
            put(canvas, f"{title}: {values[readout]:.2f} / {thresholds[readout]:.2f}  {decision}",
                (890, line), .51, color)
        plot(canvas, traces, score, current, seam_frames)
        put(canvas, f"Playback frame {current}/{count}  |  decision uses all {count} frames",
            (30, 630), .62)
        put(canvas, "Dashed lines = edit seams; red = VICREG above full-clip threshold",
            (30, 665), .53)
        for _ in range(FINAL_HOLD if current == count else REPEAT):
            yield canvas


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", choices=("drop", "splice_other", "speed_up", "swap", "shuffle"))
    parser.add_argument("--index", type=int, help="test clip index for --only")
    parser.add_argument("--suffix", default="offline_reveal", help="output filename prefix")
    args = parser.parse_args()
    if (args.only is None) != (args.index is None):
        parser.error("--only and --index must be provided together")
    cases = ((args.only, args.index),) if args.only else CASES
    OUT.mkdir(parents=True, exist_ok=True)
    protocol = json.loads((RUN / "protocol.json").read_text())
    paths = protocol["test_paths"]
    cached = torch.load(FRAME_CACHE, mmap=True, weights_only=False)
    lookup = {path: i for i, path in enumerate(cached["paths"])}
    metrics = {(r["readout"], r["injection"]): r
               for r in csv.DictReader((RUN / "segment_metrics.csv").open())
               if r["channel"] == "geometry"}
    seams = json.loads((RUN / "test_seams.json").read_text())
    selected = []
    with np.load(RUN / "per_clip_scores.npz") as saved:
        for injection, index in cases:
            score = metrics["raw", injection]["score"]
            frames = cached["frames"][lookup[paths[index]]].permute(0, 2, 3, 1).numpy()
            donor_frames = None
            paths_by_readout = None
            if injection == "splice_other":
                _, donor_index, source_indices, paths_by_readout = splice_curves(
                    saved, index, paths, protocol["calibration_paths"], return_paths=True)
                donor_frames = cached["frames"][lookup[paths[donor_index]]].permute(
                    0, 2, 3, 1).numpy()
            else:
                source_index, _ = INJECTIONS[injection](torch.arange(16),
                                                        rng_for(0, injection, index))
                source_indices = [int(v) for v in source_index.tolist()]
            profiles, values, thresholds = {}, {}, {}
            for readout in ("raw", "vicreg_clean"):
                step = saved[f"test|{readout}|{injection}|step"][index]
                z = paths_by_readout[readout][1][0] if paths_by_readout else None
                profile = contribution(score, step, z)
                profiles[readout] = profile if score == "chord_dev" else np.r_[0, profile]
                values[readout] = float(saved[f"test|{readout}|{injection}|{score}"][index])
                thresholds[readout] = float(metrics[readout, injection]["threshold"])
                computed = (profile.max() if score != "step_cv" else
                            float(np.sqrt(np.mean(profile ** 2))))
                if not np.isclose(values[readout], computed, rtol=1e-4, atol=1e-5):
                    raise RuntimeError(f"{injection}/{readout}: contribution does not match score")
                if readout == "vicreg_clean" and not values[readout] > thresholds[readout]:
                    raise RuntimeError(f"{injection}: VICREG does not flag selected clip")
                if readout == "raw" and not values[readout] < thresholds[readout]:
                    raise RuntimeError(f"{injection}: DINOv2 flags selected clip")
            seam_frames = tuple(t + 2 for t in seams[injection][index]) if injection in seams else ()
            if score == "jump_spike" and int(np.argmax(profiles["vicreg_clean"])) + 1 not in seam_frames:
                raise RuntimeError(f"{injection}: strongest VICREG jump is not at an edit seam")
            if score == "chord_dev":
                peak = int(np.argmax(profiles["vicreg_clean"])) + 1
                if not (seam_frames[0] <= peak < seam_frames[1]):
                    raise RuntimeError(f"{injection}: strongest VICREG deviation is outside donor span")
            traces = display_traces(profiles, score, thresholds)
            for readout in traces:
                displayed = traces[readout][-1] if score == "step_cv" else traces[readout].max()
                if not np.isclose(displayed, values[readout] / thresholds[readout], rtol=1e-4):
                    raise RuntimeError(f"{injection}/{readout}: displayed trace differs from final score")
            out = OUT / f"{args.suffix}_{injection}.mp4"
            ffmpeg = subprocess.Popen(["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo",
                                       "-pix_fmt", "bgr24", "-s", f"{WIDTH}x{HEIGHT}",
                                       "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-crf", "20",
                                       "-pix_fmt", "yuv420p", str(out)], stdin=subprocess.PIPE)
            try:
                for canvas in render(injection, index, frames, source_indices, donor_frames,
                                     profiles, score, values, thresholds, seam_frames):
                    ffmpeg.stdin.write(canvas.tobytes())
            finally:
                ffmpeg.stdin.close()
                if ffmpeg.wait() != 0:
                    raise RuntimeError(f"ffmpeg failed for {injection}")
            selected.append({"edit": injection, "test_index": index, "video_path": paths[index],
                             "score": score, "source_frame_indices_0_based": source_indices,
                             "edit_seam_target_frames_1_based": seam_frames,
                             "dino_score": values["raw"], "dino_threshold": thresholds["raw"],
                             "vicreg_score": values["vicreg_clean"],
                             "vicreg_threshold": thresholds["vicreg_clean"],
                             "dino_peak_contributor_frame": int(np.argmax(profiles["raw"])) + 1,
                             "vicreg_peak_contributor_frame": int(np.argmax(profiles["vicreg_clean"])) + 1})
            print(f"saved {out}")
    (OUT / f"{args.suffix}_cases.json").write_text(json.dumps(selected, indent=2))


if __name__ == "__main__":
    main()
