"""Frozen CoTracker3 runner for motion-guided adaptive sampling (route 2).

Loads the offline CoTracker3 checkpoint with `window_len` auto-detected from
the checkpoint's `time_emb` buffer (per project convention), runs a regular
grid of tracks over a video, and returns tracks + visibility on the CPU.

The tracker is used purely as a frozen motion sensor; no gradients, no VLM.
"""
from __future__ import annotations

import numpy as np
import torch


def detect_window_len(checkpoint_path: str) -> int:
    """Read window_len from the checkpoint's time_emb buffer shape.

    CoTrackerThreeBase registers `time_emb` of shape (1, window_len, input_dim);
    reading dim 1 recovers the training window without hardcoding it.
    """
    sd = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "model" in sd:
        sd = sd["model"]
    if "time_emb" not in sd:
        raise KeyError(
            f"'time_emb' not found in {checkpoint_path}; cannot detect window_len"
        )
    return int(sd["time_emb"].shape[1])


class CoTrackerRunner:
    def __init__(
        self,
        checkpoint_path: str,
        device: str | None = None,
        grid_size: int = 30,
        reseed_every_s: float | None = 1.0,
        backward_track: bool = False,
        segment_track: bool = False,
        online: bool = False,
        online_checkpoint: str | None = None,
    ):
        """
        Args:
            checkpoint_path: path to scaled_offline.pth (offline mode).
            device: 'cuda' / 'cpu'. Defaults to cuda if available.
            grid_size: NxN regular grid of query points (route 2 needs a dense
                enough grid to capture object motion but not so dense it OOMs).
            reseed_every_s: re-drop a fresh grid of query points every this many
                seconds (using the video's tracking fps), so objects that enter
                the scene AFTER frame 0 also get tracked. Seeds are placed at
                frame 0 and every reseed step after. None / <=0 keeps a single
                grid at frame 0.
            segment_track: if True, do NOT track any point across the whole clip.
                Instead split the video into consecutive ~reseed_every_s chunks
                (overlapping by 1 frame) and track a fresh grid INDEPENDENTLY
                within each chunk. Rationale: because the VLM re-samples frames
                anyway, cross-second trajectory continuity is unnecessary — we
                only need in-segment motion of whatever moves in that second.
                This removes long-track drift (a frame-0 point wandering for 5s)
                and re-covers newly-entered objects every segment. Each chunk
                puts only ~step+1 frames on the GPU, so it is inherently
                low-memory (online mode is redundant and rejected). The overlap
                means every consecutive frame-pair lies fully inside one segment,
                so M(t) has no boundary garbage. Offline predictor only.
            backward_track: if True, points seeded at a later frame are also
                tracked BACKWARDS so they have positions before they were seeded.
                Default False: a point only has a trajectory from its seed frame
                onward, so a mid-clip entrant shows no line before it appears
                (physically truthful; visibility marks earlier frames invalid).
                Offline mode only (online is inherently causal / forward-only).
            online: if True, use the streaming CoTrackerOnlinePredictor, which
                processes the video in overlapping windows so only ~window_len
                frames sit on the GPU at once (much lower peak memory on long
                clips). Default False keeps the exact offline behavior below.
            online_checkpoint: path to scaled_online.pth; required when
                online=True. Defaults to sibling scaled_online.pth of
                checkpoint_path if not given.
        """
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.grid_size = grid_size
        self.reseed_every_s = reseed_every_s
        self.backward_track = backward_track
        self.segment_track = segment_track
        self.online = online

        if segment_track and online:
            # segment tracking is already low-memory (only ~step+1 frames per
            # chunk on the GPU), so the online predictor adds nothing but the
            # complexity of two windowing schemes; keep them mutually exclusive.
            raise ValueError(
                "segment_track and online are mutually exclusive; segment "
                "tracking is already low-memory (use offline)."
            )

        if online:
            import os

            from cotracker.predictor import CoTrackerOnlinePredictor

            ckpt = online_checkpoint or os.path.join(
                os.path.dirname(checkpoint_path), "scaled_online.pth"
            )
            self.window_len = detect_window_len(ckpt)
            self.predictor = CoTrackerOnlinePredictor(
                checkpoint=ckpt, v2=False, window_len=self.window_len,
            )
        else:
            from cotracker.predictor import CoTrackerPredictor

            self.window_len = detect_window_len(checkpoint_path)
            self.predictor = CoTrackerPredictor(
                checkpoint=checkpoint_path,
                offline=True,
                v2=False,
                window_len=self.window_len,
            )
        self.predictor = self.predictor.to(self.device).eval()
        for p in self.predictor.parameters():
            p.requires_grad_(False)

    def _grid_queries(self, T: int, H: int, W: int, fps: float) -> torch.Tensor:
        """Build (1, N, 3) grid queries (t, x, y) re-seeded every reseed_every_s.

        A grid of grid_size x grid_size points is placed at frame 0 and again at
        each subsequent re-seed frame, so objects entering mid-clip are queried
        once they appear.
        """
        g = self.grid_size
        margin_x = W / (g + 1)
        margin_y = H / (g + 1)
        xs = torch.linspace(margin_x, W - margin_x, g)
        ys = torch.linspace(margin_y, H - margin_y, g)
        gy, gx = torch.meshgrid(ys, xs, indexing="ij")
        grid_xy = torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=1)  # (g*g, 2)

        # frames at which to (re)seed the grid
        if self.reseed_every_s and self.reseed_every_s > 0 and fps > 0:
            step = max(1, int(round(self.reseed_every_s * fps)))
            seed_frames = list(range(0, T, step))
        else:
            seed_frames = [0]

        q = []
        for t in seed_frames:
            tcol = torch.full((grid_xy.shape[0], 1), float(t))
            q.append(torch.cat([tcol, grid_xy], dim=1))  # (g*g, 3) = (t,x,y)
        queries = torch.cat(q, dim=0)[None]  # (1, N, 3)
        return queries.to(self.device)

    @torch.no_grad()
    def track(self, frames: np.ndarray, fps: float | None = None,
              seg_bounds: np.ndarray | None = None) -> dict:
        """Run grid tracking on a (T, H, W, 3) uint8 RGB video.

        Args:
            frames: (T, H, W, 3) uint8 RGB.
            fps: tracking-timeline fps, used to space periodic re-seeds. If None
                or reseed disabled, falls back to a single grid at frame 0.
            seg_bounds: (only for segment_track) sorted tracking-timeline indices
                to use as segment boundaries — each pair (b[k], b[k+1]) becomes
                one independently-tracked segment. Pass the sampled-frame indices
                so each CoTracker segment is exactly the motion BETWEEN two
                sampled frames. If None, segments fall back to fixed
                reseed_every_s spacing.

        Returns dict with:
            tracks:     (T, N, 2) float32 pixel coords in ORIGINAL frame space
            visibility: (T, N)   bool
            hw:         (H, W)
        """
        assert frames.ndim == 4 and frames.shape[-1] == 3, frames.shape
        T, H, W = int(frames.shape[0]), int(frames.shape[1]), int(frames.shape[2])

        if self.segment_track:
            return self._track_segments(frames, fps, H, W, seg_bounds=seg_bounds)

        if self.online:
            return self._track_online(frames, fps, H, W)

        video = (
            torch.from_numpy(frames)
            .permute(0, 3, 1, 2)[None]        # (1, T, 3, H, W)
            .float()
            .to(self.device)
        )

        reseed = self.reseed_every_s and self.reseed_every_s > 0 and fps and fps > 0
        if reseed and T > 1:
            # Custom multi-frame queries so mid-clip entrants get tracked.
            queries = self._grid_queries(T, H, W, float(fps))
            tracks, vis = self.predictor(
                video, queries=queries, backward_tracking=self.backward_track,
            )
        else:
            # CoTracker3 uses float32 internally (bf16 grid_sample is unstable).
            tracks, vis = self.predictor(
                video,
                grid_size=self.grid_size,
                grid_query_frame=0,
                backward_tracking=False,
            )
        # tracks: (1, T, N, 2), vis: (1, T, N)
        return {
            "tracks": tracks[0].float().cpu().numpy(),
            "visibility": vis[0].bool().cpu().numpy(),
            "hw": (H, W),
        }

    @torch.no_grad()
    def _track_segments(self, frames: np.ndarray, fps, H: int, W: int,
                        seg_bounds: np.ndarray | None = None) -> dict:
        """Independent per-segment grid tracking (no cross-segment continuity).

        Split the clip into consecutive segments, overlapping neighbours by 1
        frame, and track a fresh grid_size x grid_size grid INSIDE each segment
        only. Each segment's points become their own columns in the global
        (T, N_total, *) output, visible only over that segment's frame span;
        everywhere else they are invisible (vis=False) and their coords are
        never read by M(t) because compute_motion_energy masks frame-pairs with
        `vis[t] & vis[t+1]`.

        Segment boundaries:
          * seg_bounds given -> each pair (b[k], b[k+1]) is one segment, so a
            segment == the motion BETWEEN two sampled frames (the intended
            "one trajectory per sampled-frame interval").
          * else -> fixed step = round(reseed_every_s * fps) frames.

        The 1-frame overlap guarantees every consecutive pair (t, t+1) lies
        fully inside exactly one segment, so there is no boundary in which a
        point is visible at t but gone at t+1 — M(t) stays clean at seams.
        """
        T = int(frames.shape[0])
        if T <= 1:
            # nothing to track across; single grid, single frame
            return self._track_offline_single_grid(frames, H, W)

        # Build (start, end_inclusive) segment spans overlapping by 1 frame.
        if seg_bounds is not None and len(seg_bounds) >= 2:
            b = np.unique(np.asarray(seg_bounds, dtype=np.int64))
            b = b[(b >= 0) & (b <= T - 1)]
            spans = [(int(b[k]), int(b[k + 1])) for k in range(len(b) - 1)
                     if b[k + 1] > b[k]]
            if not spans:
                spans = [(0, T - 1)]
        else:
            if self.reseed_every_s and self.reseed_every_s > 0 and fps and fps > 0:
                step = max(1, int(round(self.reseed_every_s * fps)))
            else:
                step = T - 1  # one segment = whole clip
            spans = [(s, min(s + step, T - 1)) for s in range(0, T - 1, step)]

        video = (
            torch.from_numpy(frames)
            .permute(0, 3, 1, 2)[None]        # (1, T, 3, H, W)
            .float()
            .to(self.device)
        )

        seg_tracks: list[np.ndarray] = []
        seg_vis: list[np.ndarray] = []
        for s, e in spans:                    # e is inclusive; overlaps next by 1
            sub = video[:, s : e + 1]         # (1, L, 3, H, W)
            tr, vis = self.predictor(
                sub, grid_size=self.grid_size, grid_query_frame=0,
                backward_tracking=False,
            )
            L = e + 1 - s
            g2 = tr.shape[2]
            # scatter this segment's points into a full-timeline block
            blk_t = np.zeros((T, g2, 2), dtype=np.float32)
            blk_v = np.zeros((T, g2), dtype=bool)
            blk_t[s : e + 1] = tr[0, :L].float().cpu().numpy()
            blk_v[s : e + 1] = vis[0, :L].bool().cpu().numpy()
            seg_tracks.append(blk_t)
            seg_vis.append(blk_v)

        tracks = np.concatenate(seg_tracks, axis=1)   # (T, N_total, 2)
        visibility = np.concatenate(seg_vis, axis=1)  # (T, N_total)
        return {"tracks": tracks, "visibility": visibility, "hw": (H, W)}

    @torch.no_grad()
    def _track_offline_single_grid(self, frames: np.ndarray, H: int, W: int) -> dict:
        """Fallback: one grid at frame 0 over the whole (tiny) clip."""
        video = (
            torch.from_numpy(frames).permute(0, 3, 1, 2)[None].float().to(self.device)
        )
        tracks, vis = self.predictor(
            video, grid_size=self.grid_size, grid_query_frame=0,
            backward_tracking=False,
        )
        return {
            "tracks": tracks[0].float().cpu().numpy(),
            "visibility": vis[0].bool().cpu().numpy(),
            "hw": (H, W),
        }

    @torch.no_grad()
    def _track_online(self, frames: np.ndarray, fps, H: int, W: int) -> dict:
        """Streaming tracking with CoTrackerOnlinePredictor (low peak memory).

        Processes the clip in overlapping windows of `step*2` frames, so only
        ~window_len frames occupy the GPU at any time instead of the whole clip.
        The same multi-frame reseed grid is registered up front (queries carry
        their seed-frame index t); a point only starts contributing once the
        stream reaches its seed frame, which matches the offline reseed behavior
        and keeps output shape/semantics identical: tracks (T,N,2), vis (T,N).
        """
        T = int(frames.shape[0])
        step = self.predictor.step  # window_len // 2

        reseed = self.reseed_every_s and self.reseed_every_s > 0 and fps and fps > 0
        if reseed and T > 1:
            queries = self._grid_queries(T, H, W, float(fps))
        else:
            queries = self._grid_queries(T, H, W, 0.0)  # single grid at frame 0

        def _chunk(win_frames):
            # always feed at most step*2 (== window_len) frames to the model
            sub = win_frames[-step * 2:]
            return (
                torch.from_numpy(np.stack(sub))
                .permute(0, 3, 1, 2)[None]        # (1, t<=window_len, 3, H, W)
                .float()
                .to(self.device)
            )

        # Register queries once (is_first_step), then stream one window at a
        # time, mirroring co-tracker's online_demo loop.
        tracks = vis = None
        window = []
        is_first = True
        for i in range(T):
            if i % step == 0 and i != 0:
                out = self.predictor(
                    _chunk(window), is_first_step=is_first,
                    queries=queries, add_support_grid=True,
                )
                if not is_first:
                    tracks, vis = out
                is_first = False
            window.append(frames[i])
        # flush remaining tail frames
        out = self.predictor(
            _chunk(window), is_first_step=is_first,
            queries=queries, add_support_grid=True,
        )
        if not is_first:
            tracks, vis = out
        # The online model pads the final window up to window_len, so its output
        # can be longer than T; trim back to the real clip length.
        return {
            "tracks": tracks[0, :T].float().cpu().numpy(),
            "visibility": vis[0, :T].bool().cpu().numpy(),
            "hw": (H, W),
        }
