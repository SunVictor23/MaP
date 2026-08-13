"""Raw-video reading for motion-guided adaptive sampling (route 2).

Route 2 must run CoTracker on the *full-framerate* video, because fps=1
frames have already discarded the inter-frame motion we want to recover.
This module reads a raw mp4 and (optionally) resamples it to a tracking fps.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def fps1_budget(src_num_frames: int, src_fps: float, target_fps: float = 1.0) -> int:
    """Number of frames an fps=1 extraction would produce for this video.

    Mirrors the reference frame-index utility.compute_frame_indices exactly, so the
    adaptive budget equals the appearance-path frame count frame-for-frame:
        n = max(1, round(total / orig_fps * target_fps)), capped at total.
    """
    if src_num_frames <= 0:
        return 0
    fps = src_fps if src_fps and src_fps > 0 else 1.0
    n = max(1, round(src_num_frames / fps * target_fps))
    return int(min(n, src_num_frames))


def qwen3vl_budget(
    src_num_frames: int,
    src_fps: float,
    target_fps: float = 2.0,
    min_frames: int = 2,
    max_frames: int = 768,
) -> int:
    """Frame count Qwen3-VL's native video processor would sample.

    Verbatim reproduction of transformers Qwen3VLVideoProcessor.sample_frames:
        num_frames = int(total / meta_fps * fps)                    # truncate
        num_frames = min(min(max(num_frames, min_frames), max_frames), total)
    The processor's own class default is min_frames=4; we lower it to 2 so a
    frame-budget sweep can actually reach the sparse regime on short clips
    (e.g. CLEVRER ~5s), where min_frames=4 would floor fps=0.5 and fps=1 to
    nearly the same count. max_frames=768 unchanged. Note int() truncates.
    """
    if src_num_frames <= 0:
        return 0
    fps = src_fps if src_fps and src_fps > 0 else 24.0
    n = int(src_num_frames / fps * target_fps)
    n = min(min(max(n, min_frames), max_frames), src_num_frames)
    return int(n)


def qwen3vl_frame_indices(total_num_frames: int, num_frames: int) -> np.ndarray:
    """Frame indices Qwen3-VL samples for a given count (endpoint-inclusive).

    Verbatim: np.linspace(0, total-1, num_frames).round().astype(int).
    Uniformly spans the WHOLE clip (both endpoints included) — NOT one frame
    per integer second. This is what the native processor feeds the model.
    """
    if total_num_frames <= 0 or num_frames <= 0:
        return np.array([], dtype=np.int64)
    return np.linspace(0, total_num_frames - 1, num_frames).round().astype(np.int64)


@dataclass
class RawVideo:
    frames: np.ndarray          # (T, H, W, 3) uint8, RGB, at tracking fps
    timestamps: np.ndarray      # (T,) float seconds, original position of each kept frame
    fps: float                  # tracking fps (frames-per-second of `frames`)
    src_fps: float              # native fps of the source file
    src_num_frames: int         # native frame count of the source file

    @property
    def num_frames(self) -> int:
        return int(self.frames.shape[0])

    @property
    def hw(self) -> tuple[int, int]:
        return int(self.frames.shape[1]), int(self.frames.shape[2])

    def fps1_budget(self, target_fps: float = 1.0) -> int:
        """fps=1 frame count for the SOURCE video (== appearance-path budget)."""
        return fps1_budget(self.src_num_frames, self.src_fps, target_fps)

    def qwen3vl_budget(self, target_fps: float = 2.0, min_frames: int = 2,
                       max_frames: int = 768) -> int:
        """Frame count Qwen3-VL's native video processor would sample."""
        return qwen3vl_budget(self.src_num_frames, self.src_fps, target_fps,
                              min_frames, max_frames)


def read_video(
    path: str,
    tracking_fps: float | None = None,
    max_frames: int | None = None,
    bound: tuple[float, float] | None = None,
) -> RawVideo:
    """Read a video and resample to `tracking_fps`.

    Args:
        path: path to a video file (mp4, ...).
        tracking_fps: target fps for tracking. None keeps the native fps.
            Lower fps -> cheaper tracking but coarser motion; CoTracker3 was
            trained on real-time-ish video, ~8-12 fps is a good cost/signal
            tradeoff for motion-peak detection.
        max_frames: hard cap on number of frames fed to the tracker (uniform
            subsample after fps resampling). Guards against OOM on long clips.
        bound: optional (start_s, end_s) window in seconds. Only frames inside
            [start, end] on the original timeline are read (MVBench-style
            temporally-bounded clips). Returned timestamps stay on the original
            timeline.

    Returns:
        RawVideo. `timestamps` records where each kept frame sits on the
        original time axis (seconds), so the selector can emit true,
        non-uniform timestamps back to the VLM.
    """
    import imageio.v3 as iio

    meta = iio.immeta(path, plugin="pyav")
    src_fps = float(meta.get("fps") or 0.0)

    frames = iio.imread(path, plugin="pyav")  # (T, H, W, 3) uint8
    if frames.ndim == 3:  # grayscale -> add channel
        frames = np.repeat(frames[..., None], 3, axis=-1)
    if frames.shape[-1] == 4:  # RGBA -> RGB
        frames = frames[..., :3]
    total_frames = int(frames.shape[0])

    if src_fps <= 0:
        # Fallback: some containers don't report fps. Assume 25 so timestamps
        # are at least monotonic; absolute seconds may be off but relative
        # spacing (what the VLM prompt needs) stays correct after resampling.
        src_fps = 25.0

    # Restrict to a temporal window (kept on the ORIGINAL timeline).
    lo = 0
    if bound is not None:
        start_s, end_s = bound
        lo = max(0, int(round(start_s * src_fps)))
        hi = min(total_frames - 1, int(round(end_s * src_fps)))
        if hi <= lo:  # degenerate bound -> keep at least 2 frames
            hi = min(total_frames - 1, lo + 1)
        frames = frames[lo : hi + 1]
    src_num_frames = int(frames.shape[0])

    keep_idx = _resample_indices(src_num_frames, src_fps, tracking_fps, max_frames)
    kept = frames[keep_idx]
    # timestamps are absolute seconds on the original file timeline.
    timestamps = (keep_idx + lo).astype(np.float64) / src_fps
    out_fps = tracking_fps if tracking_fps and tracking_fps > 0 else src_fps

    return RawVideo(
        frames=np.ascontiguousarray(kept),
        timestamps=timestamps,
        fps=float(out_fps),
        src_fps=float(src_fps),
        src_num_frames=src_num_frames,
    )


def read_frames_dir(
    frames_dir: str,
    src_fps: float = 3.0,
    tracking_fps: float | None = None,
    max_frames: int | None = None,
    bound: tuple[float, float] | None = None,
) -> RawVideo:
    """Read a directory of pre-extracted JPG frames as a RawVideo.

    Used by MVBench's episodic_reasoning task (tvqa/frames_fps3_hq): frames are
    dumped at `src_fps` (default 3) and named 1-based `00001.jpg`, `00002.jpg`.

    Args mirror `read_video`; `bound` is a (start_s, end_s) window in seconds.
    """
    import os

    from PIL import Image

    names = sorted(
        f for f in os.listdir(frames_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    )
    if not names:
        raise FileNotFoundError(f"no image frames in {frames_dir}")
    total_frames = len(names)
    if src_fps <= 0:
        src_fps = 3.0

    lo = 0
    if bound is not None:
        start_s, end_s = bound
        lo = max(0, int(round(start_s * src_fps)))
        hi = min(total_frames - 1, int(round(end_s * src_fps)))
        if hi <= lo:
            hi = min(total_frames - 1, lo + 1)
        names = names[lo : hi + 1]
    src_num_frames = len(names)

    keep_idx = _resample_indices(src_num_frames, src_fps, tracking_fps, max_frames)
    imgs = [np.asarray(Image.open(os.path.join(frames_dir, names[i])).convert("RGB"))
            for i in keep_idx]
    kept = np.stack(imgs, axis=0)
    timestamps = (keep_idx + lo).astype(np.float64) / src_fps
    out_fps = tracking_fps if tracking_fps and tracking_fps > 0 else src_fps

    return RawVideo(
        frames=np.ascontiguousarray(kept),
        timestamps=timestamps,
        fps=float(out_fps),
        src_fps=float(src_fps),
        src_num_frames=src_num_frames,
    )


def _resample_indices(
    n: int,
    src_fps: float,
    tracking_fps: float | None,
    max_frames: int | None,
) -> np.ndarray:
    """Pick source-frame indices for the given tracking fps and frame cap."""
    if tracking_fps and tracking_fps > 0 and tracking_fps < src_fps:
        duration = n / src_fps
        n_target = max(2, int(round(duration * tracking_fps)))
        idx = np.linspace(0, n - 1, num=min(n_target, n)).round().astype(np.int64)
        idx = np.unique(idx)
    else:
        idx = np.arange(n, dtype=np.int64)

    if max_frames is not None and len(idx) > max_frames:
        sub = np.linspace(0, len(idx) - 1, num=max_frames).round().astype(np.int64)
        idx = idx[np.unique(sub)]

    return idx
