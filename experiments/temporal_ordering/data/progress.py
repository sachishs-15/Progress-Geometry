"""Progress-label definitions. `timestamp` is the primary/default definition used by the
main experiment (section 3 of the spec); `step` is the annotated-procedural-step definition
(section 16), added on top once the timestamp-based experiment works — both share the same
downstream training/eval code, selected via `dataset.progress_type` in a config.
"""
import numpy as np


def timestamp_progress(T: int) -> np.ndarray:
    """progress[t] = t / (T - 1); a single-observation video gets progress 0.0."""
    if T <= 1:
        return np.zeros(T, dtype=np.float32)
    return (np.arange(T, dtype=np.float32) / (T - 1)).astype(np.float32)


def step_progress(step_labels: np.ndarray) -> np.ndarray:
    """Normalized procedural-step progress: step k (0-indexed, N total steps) covers the
    range [k/N, (k+1)/N); a timestep's progress is linearly interpolated across its own
    step's span using its position within that step's run of timesteps, so progress is
    still monotonically increasing within a step rather than flat.
    """
    step_labels = np.asarray(step_labels)
    n_steps = int(step_labels.max()) + 1
    progress = np.zeros(len(step_labels), dtype=np.float32)
    for k in range(n_steps):
        idx = np.where(step_labels == k)[0]
        if len(idx) == 0:
            continue
        span_lo, span_hi = k / n_steps, (k + 1) / n_steps
        within = timestamp_progress(len(idx)) if len(idx) > 1 else np.zeros(1, dtype=np.float32)
        progress[idx] = span_lo + within * (span_hi - span_lo)
    return progress


def compute_progress(record, progress_type: str = "timestamp") -> np.ndarray:
    T = record.features.shape[0]
    if progress_type == "timestamp":
        return timestamp_progress(T)
    if progress_type == "step":
        if record.step_labels is None:
            raise ValueError(f"video {record.video_id} has no step_labels; can't use progress_type='step'")
        return step_progress(record.step_labels)
    raise ValueError(f"Unknown progress_type: {progress_type!r}")
