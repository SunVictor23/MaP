"""End-to-end motion-guided adaptive sampling pipeline (route 2).

mp4 --> [read @ tracking fps] --> [frozen CoTracker3 grid tracks]
     --> [motion energy M(t)] --> [budget selection] --> selected frames
     + non-uniform timestamps (seconds on the original timeline) for the VLM.

Usage (library):
    sampler = MotionAdaptiveSampler(cotracker_ckpt="/.../scaled_offline.pth")
    res = sampler.sample("/path/video.mp4", budget=16)
    res.frames        # (B, H, W, 3) uint8  -> feed to VLM
    res.timestamps    # (B,) seconds        -> non-uniform timestamp prompt
    res.prompt_hint   # ready-made string describing the non-uniform sampling

Usage (CLI):
    python -m map_kit.pipeline --video X.mp4 --budget 16 \
        --cotracker_ckpt /.../scaled_offline.pth --out_dir /tmp/out
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass

import numpy as np

# Make the bundled co-tracker importable without requiring PYTHONPATH export.
_PKG_DIR = os.path.dirname(os.path.abspath(__file__))
_COTRACKER_DIR = os.path.normpath(os.path.join(_PKG_DIR, "..", "co-tracker"))
if os.path.isdir(_COTRACKER_DIR) and _COTRACKER_DIR not in sys.path:
    sys.path.insert(0, _COTRACKER_DIR)

from map_kit.data.video_io import read_video, read_frames_dir
from map_kit.core.motion_energy import compute_motion_energy
from map_kit.core.frame_selector import select_frames

DEFAULT_COTRACKER_CKPT = "/path/to/basemodels/scaled_offline.pth"


@dataclass
class SampleResult:
    frames: np.ndarray          # (B, H, W, 3) uint8, selected key frames
    timestamps: np.ndarray      # (B,) seconds on the ORIGINAL timeline
    indices: np.ndarray         # (B,) indices into the tracking-fps timeline
    M: np.ndarray               # (T,) motion energy curve (tracking timeline)
    track_timestamps: np.ndarray  # (T,) seconds for each tracking-fps frame
    anchors: np.ndarray         # anchor indices (subset of `indices`)
    peaks: np.ndarray           # motion-peak indices (subset of `indices`)
    components: dict            # normalized M(t) sub-components
    tracks: np.ndarray | None = None       # (T, N, 2) full-clip point tracks (route 1)
    visibility: np.ndarray | None = None   # (T, N) bool
    obj_vel: np.ndarray | None = None      # (T, N, 2) camera-compensated velocity

    @property
    def prompt_hint(self) -> str:
        ts = ", ".join(f"{t:.2f}s" for t in self.timestamps)
        return (
            "The following frames were sampled non-uniformly in time, "
            "concentrated on moments of strong motion. Their timestamps are: "
            f"[{ts}]."
        )


class MotionAdaptiveSampler:
    def __init__(
        self,
        cotracker_ckpt: str = DEFAULT_COTRACKER_CKPT,
        device: str | None = None,
        tracking_fps: float | None = 8.0,
        max_track_frames: int | None = 200,
        grid_size: int = 30,
        compensate_camera: bool = True,
        motion_weights: dict | None = None,
        reseed_every_s: float | None = 1.0,
        segment_track: bool = False,
        online: bool = False,
        online_checkpoint: str | None = None,
    ):
        from map_kit.data.cotracker_runner import CoTrackerRunner

        self.tracking_fps = tracking_fps
        self.max_track_frames = max_track_frames
        self.compensate_camera = compensate_camera
        self.motion_weights = motion_weights
        self.runner = CoTrackerRunner(
            checkpoint_path=cotracker_ckpt, device=device, grid_size=grid_size,
            reseed_every_s=reseed_every_s, segment_track=segment_track,
            online=online, online_checkpoint=online_checkpoint,
        )

    def sample(
        self,
        video_path: str,
        budget: int | None = None,
        n_anchor: int | None = None,
        target_fps: float | None = None,
        mode: str = "adaptive",
        bound: tuple[float, float] | None = None,
        data_type: str = "video",
        frame_fps: float = 3.0,
        with_tracks: bool = False,
        sampling_rule: str = "qwen3vl",
    ) -> SampleResult:
        """Adaptively sample key frames from a video.

        Args:
            with_tracks: in "uniform" mode, also run the tracker and attach
                tracks/visibility/obj_vel to the result (so route-1 marking can
                draw trajectories on uniformly-sampled frames). No effect on
                which frames are chosen; adaptive already computes tracks.
            budget: number of frames to select. None (default) uses the frame
                count implied by `sampling_rule` (see below).
            target_fps: the fps VALUE fed into the sampling rule when `budget`
                is None. None (default) resolves to the rule's own default fps
                (1.0 for both qwen3vl and fps1). Pass a number to override — fps
                is a tunable knob, NOT part of Qwen3-VL's frame-selection
                mechanism (its processor default of 2 is just a config value).
            sampling_rule: how the uniform baseline picks its budget AND frame
                positions when `budget` is None (the MECHANISM; fps is separate):
                  "fps1" (default): budget = round(total/fps*target_fps) frames,
                    positions at integer-second marks round(k*tracking_fps).
                    Preserves the project's existing fps=1 equal-budget setup.
                  "qwen3vl": VERBATIM Qwen3-VL native rule —
                    budget = int(total/fps*2) clamped to [4,768], positions via
                    linspace(0, N-1, budget).round() (endpoint-inclusive, spans
                    the whole clip). Timestamps stay real frame-seconds, matching
                    the native <t.t seconds> text-timestamp prompt.
            mode: "adaptive" (motion-guided, route 2) or "uniform" (evenly
                spaced frames, the equal-budget baseline). Uniform skips
                tracking entirely so the comparison isolates *which* frames.
            bound: optional (start_s, end_s) temporal window (MVBench clips).
            data_type: "video" reads a video file; "frame" reads a directory of
                pre-extracted JPGs at `frame_fps` (MVBench episodic_reasoning).
            frame_fps: fps of the pre-extracted frames when data_type=="frame".
        """
        if data_type == "frame":
            raw = read_frames_dir(
                video_path,
                src_fps=frame_fps,
                tracking_fps=self.tracking_fps,
                max_frames=self.max_track_frames,
                bound=bound,
            )
        else:
            raw = read_video(
                video_path,
                tracking_fps=self.tracking_fps,
                max_frames=self.max_track_frames,
                bound=bound,
            )
        if budget is None:
            # sampling_rule fixes the MECHANISM; target_fps is the (tunable) fps
            # VALUE fed into it. fps is NOT part of Qwen3-VL's rule — the
            # processor's fps=2 is just a configurable default. We default to
            # 1.0 for both rules; pass target_fps to override.
            if sampling_rule == "qwen3vl":
                q_fps = target_fps if target_fps is not None else 1.0
                budget = raw.qwen3vl_budget(target_fps=q_fps)
            else:
                f_fps = target_fps if target_fps is not None else 1.0
                budget = raw.fps1_budget(target_fps=f_fps)
        if raw.num_frames <= budget:
            # Nothing to select from; return everything.
            idx = np.arange(raw.num_frames)
            M = np.zeros(raw.num_frames, dtype=np.float64)
            return SampleResult(
                frames=raw.frames, timestamps=raw.timestamps, indices=idx,
                M=M, track_timestamps=raw.timestamps,
                anchors=idx, peaks=np.array([], dtype=int), components={},
            )

        if mode == "uniform":
            if sampling_rule == "qwen3vl":
                # VERBATIM Qwen3-VL native positions: linspace over the WHOLE
                # clip, endpoint-inclusive (not integer-second marks).
                idx = np.unique(
                    np.linspace(0, raw.num_frames - 1, num=budget)
                    .round().astype(np.int64)
                )
            else:
                # fps=1 rule: one frame per second at integer-second positions
                # t=0,1,2,... (indices round(k*tracking_fps)), so frame content
                # matches the <k.0 seconds> labels. NOT linspace(0,N-1,budget),
                # which lands on fractional seconds that desync the labels.
                fps = raw.fps if raw.fps and raw.fps > 0 else 1.0
                idx = np.array([int(round(k * fps)) for k in range(budget)],
                               dtype=np.int64)
                idx = np.unique(np.clip(idx, 0, raw.num_frames - 1))
            tr = obj_vel = None
            if with_tracks:
                # run the tracker on the full-fps frames so route-1 marking can
                # draw trajectories; frame selection stays uniform (unchanged).
                # Pass the sampled indices as segment bounds so (with
                # segment_track) each CoTracker segment == the motion between two
                # sampled frames — one independent trajectory per frame interval.
                tr = self.runner.track(raw.frames, fps=raw.fps, seg_bounds=idx)
                energy = compute_motion_energy(
                    tr["tracks"], tr["visibility"], tr["hw"],
                    compensate_camera=self.compensate_camera,
                    weights=self.motion_weights,
                )
                obj_vel = energy["obj_vel"]
            return SampleResult(
                frames=raw.frames[idx], timestamps=raw.timestamps[idx], indices=idx,
                M=np.zeros(raw.num_frames), track_timestamps=raw.timestamps,
                anchors=idx, peaks=np.array([], dtype=int), components={},
                tracks=(tr["tracks"] if tr else None),
                visibility=(tr["visibility"] if tr else None),
                obj_vel=obj_vel,
            )
        if mode != "adaptive":
            raise ValueError(f"unknown mode {mode!r}; use 'adaptive' or 'uniform'")

        tr = self.runner.track(raw.frames, fps=raw.fps)
        energy = compute_motion_energy(
            tr["tracks"], tr["visibility"], tr["hw"],
            compensate_camera=self.compensate_camera,
            weights=self.motion_weights,
        )
        M = energy["M"]
        sel = select_frames(M, budget=budget, n_anchor=n_anchor)
        idx = sel["indices"]

        # Second tracking pass, frame-aligned: adaptive selection is a chicken-
        # and-egg — the frames aren't known until AFTER M(t), so the first pass
        # can't use them as segment bounds. With segment_track on, re-track using
        # the selected frames as boundaries so each returned trajectory is the
        # motion BETWEEN two selected frames (matching uniform's semantics, and
        # what route-1's track_span="frame" draws). M(t)/selection stay from the
        # first pass; only the tracks handed to route-1 are refreshed.
        out_tr, out_vis, out_objvel = (
            tr["tracks"], tr["visibility"], energy["obj_vel"])
        if self.runner.segment_track and len(idx) >= 2:
            tr2 = self.runner.track(raw.frames, fps=raw.fps, seg_bounds=idx)
            energy2 = compute_motion_energy(
                tr2["tracks"], tr2["visibility"], tr2["hw"],
                compensate_camera=self.compensate_camera,
                weights=self.motion_weights,
            )
            out_tr, out_vis, out_objvel = (
                tr2["tracks"], tr2["visibility"], energy2["obj_vel"])

        return SampleResult(
            frames=raw.frames[idx],
            timestamps=raw.timestamps[idx],
            indices=idx,
            M=M,
            track_timestamps=raw.timestamps,
            anchors=sel["anchors"],
            peaks=sel["peaks"],
            components=energy["components"],
            tracks=out_tr,
            visibility=out_vis,
            obj_vel=out_objvel,
        )


def _save_debug(res: SampleResult, out_dir: str, video_path: str) -> None:
    import imageio.v3 as iio

    os.makedirs(out_dir, exist_ok=True)
    for i, (f, t) in enumerate(zip(res.frames, res.timestamps)):
        iio.imwrite(os.path.join(out_dir, f"sel_{i:03d}_t{t:.2f}.jpg"), f)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        tt = res.track_timestamps
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(tt, res.M, label="M(t) motion energy", color="black")
        for a in res.anchors:
            ax.axvline(tt[a], color="tab:blue", alpha=0.3, lw=1)
        for p in res.peaks:
            ax.axvline(tt[p], color="tab:red", alpha=0.6, lw=1.5)
        ax.scatter(res.timestamps, res.M[res.indices], color="tab:red",
                   zorder=5, label="selected")
        ax.set_xlabel("time (s)")
        ax.set_ylabel("motion energy")
        ax.set_title(f"Adaptive sampling: {os.path.basename(video_path)}  "
                     f"(blue=anchor, red=peak)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "motion_energy.png"), dpi=110)
        plt.close(fig)
    except Exception as e:  # matplotlib optional
        print(f"[debug] skipped plot: {e}")


def main() -> None:
    import argparse
    import json

    ap = argparse.ArgumentParser(description="Motion-guided adaptive frame sampling")
    ap.add_argument("--video", required=True)
    ap.add_argument("--budget", type=int, default=None,
                    help="frames to select; default = fps=1 frame count of the video")
    ap.add_argument("--cotracker_ckpt", default=DEFAULT_COTRACKER_CKPT)
    ap.add_argument("--tracking_fps", type=float, default=8.0)
    ap.add_argument("--max_track_frames", type=int, default=200)
    ap.add_argument("--grid_size", type=int, default=30)
    ap.add_argument("--no_camera_comp", action="store_true")
    ap.add_argument("--device", default=None)
    ap.add_argument("--out_dir", default=None, help="save selected frames + M(t) plot")
    args = ap.parse_args()

    sampler = MotionAdaptiveSampler(
        cotracker_ckpt=args.cotracker_ckpt,
        device=args.device,
        tracking_fps=args.tracking_fps,
        max_track_frames=args.max_track_frames,
        grid_size=args.grid_size,
        compensate_camera=not args.no_camera_comp,
    )
    res = sampler.sample(args.video, budget=args.budget)

    print(json.dumps({
        "video": args.video,
        "tracking_window_len": sampler.runner.window_len,
        "n_tracking_frames": int(res.track_timestamps.shape[0]),
        "budget": int(res.indices.shape[0]),
        "budget_source": "fps=1" if args.budget is None else "explicit",
        "selected_indices": res.indices.tolist(),
        "selected_timestamps_s": [round(float(t), 3) for t in res.timestamps],
        "n_anchors": int(len(res.anchors)),
        "n_peaks": int(len(res.peaks)),
    }, indent=2))
    print("\nprompt hint:\n" + res.prompt_hint)

    if args.out_dir:
        _save_debug(res, args.out_dir, args.video)
        print(f"\nsaved debug artifacts to {args.out_dir}")


if __name__ == "__main__":
    main()
