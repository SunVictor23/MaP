"""Route 1: Track-as-Mark — render CoTracker3 trajectories as visual marks.

Turns the frozen tracker's output into Set-of-Marks style overlays the VLM can
read directly: cluster the high-motion points into a few object-level groups,
then draw one numbered arrow-polyline per group on each selected key frame, with
a blue->red time gradient and a small legend. Relatively-static points are not
drawn, so the overlay stays uncluttered.

Public API:
    groups = cluster_motion_groups(tracks, visibility, obj_vel, hw, ...)
    marked = draw_marks_on_frames(frames, sel_indices, groups, timestamps, ...)
    legend = legend_text(len(groups))
"""
from __future__ import annotations

import numpy as np

# Circled numbers ①..⑳ for group labels in the legend text.
_CIRCLED = "①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳"


def _circled(i: int) -> str:
    return _CIRCLED[i] if 0 <= i < len(_CIRCLED) else f"({i + 1})"


def cluster_motion_groups(
    tracks: np.ndarray,
    visibility: np.ndarray,
    obj_vel: np.ndarray,
    hw: tuple[int, int],
    max_groups: int = 3,
    motion_frac: float = 0.6,
    min_points: int = 4,
) -> list[dict]:
    """Cluster high-motion points into <=max_groups object-level motion groups.

    Args:
        tracks: (T, N, 2) pixel coords on the tracking timeline.
        visibility: (T, N) bool.
        obj_vel: (T, N, 2) camera-compensated per-point velocity (from
            compute_motion_energy); used to rank and cluster moving points.
        hw: (H, W) for scale-independent thresholds.
        max_groups: cap on number of motion groups (numbered marks).
        motion_frac: fraction of point-motion energy to DROP as static, i.e.
            keep points whose total motion exceeds this quantile. 0.6 keeps the
            top ~40% most-moving points.
        min_points: below this many surviving points, return a single group (or
            none if there is essentially no motion).

    Returns:
        list of groups, each = {
            "point_ids": (k,) int indices into N,
            "centroid_track": (T, 2) float mean track of member points,
        }, sorted by descending total motion.
    """
    T, N, _ = tracks.shape
    H, W = hw
    diag = float(np.hypot(H, W)) + 1e-8
    vis = visibility.astype(bool)

    # Per-point total object motion over the clip (visible frames only).
    speed_pt = np.linalg.norm(obj_vel, axis=2)          # (T, N)
    speed_pt = np.where(vis, speed_pt, 0.0)
    motion = speed_pt.sum(axis=0) / diag                # (N,)

    if not np.any(motion > 0):
        return []

    thr = np.quantile(motion[motion > 0], motion_frac)
    moving = np.where(motion > max(thr, 1e-6))[0]
    if moving.size < min_points:
        # too few moving points -> one group of whatever moves (or nothing)
        moving = np.where(motion > 0)[0]
        if moving.size == 0:
            return []
        return [_make_group(moving, tracks, vis)]

    # Feature per moving point: mean visible position + mean velocity direction,
    # both normalized, so clustering groups points that are co-located AND move
    # the same way (i.e. belong to the same object).
    feats = []
    for n in moving:
        m = vis[:, n]
        pos = tracks[m, n, :].mean(axis=0) / np.array([W, H])
        v = obj_vel[:, n, :][m]
        vmean = v.mean(axis=0)
        vn = vmean / (np.linalg.norm(vmean) + 1e-8)
        feats.append([pos[0], pos[1], vn[0], vn[1]])
    feats = np.asarray(feats, dtype=np.float64)

    k = int(min(max_groups, moving.size))
    try:
        from sklearn.cluster import KMeans

        labels = KMeans(n_clusters=k, n_init=5, random_state=0).fit_predict(feats)
    except Exception:
        labels = np.zeros(moving.size, dtype=int)

    groups = []
    for lab in range(labels.max() + 1 if labels.size else 0):
        ids = moving[labels == lab]
        if ids.size == 0:
            continue
        groups.append(_make_group(ids, tracks, vis))

    # Rank by total motion of the group (most salient first -> becomes ①).
    groups.sort(key=lambda g: -float(motion[g["point_ids"]].sum()))
    return groups


def _make_group(ids: np.ndarray, tracks: np.ndarray, vis: np.ndarray) -> dict:
    """Build a group dict with a per-frame centroid track over member points.

    `valid[t]` is True only where at least one member point is visible at frame
    t. Positions at invalid frames are still filled (for a continuous polyline
    across brief occlusions), but callers should NOT draw segments that touch an
    invalid frame — that is where CoTracker emits hallucinated coordinates for a
    point before it enters / after it leaves the scene.
    """
    T = tracks.shape[0]
    centroid = np.full((T, 2), np.nan, dtype=np.float64)
    valid = np.zeros(T, dtype=bool)
    for t in range(T):
        m = vis[t, ids]
        if m.any():
            centroid[t] = tracks[t, ids[m], :].mean(axis=0)
            valid[t] = True
    # fill gaps so the polyline stays continuous across brief occlusions
    centroid = _fill_nan(centroid)
    return {"point_ids": ids, "centroid_track": centroid, "valid": valid}


def _moving_point_ids(
    tracks: np.ndarray,
    vis: np.ndarray,
    hw: tuple[int, int],
    fps: float = 8.0,
    static_thresh_frac: float = 0.03,
) -> tuple[np.ndarray, np.ndarray]:
    """Point ids that MOVE, by the displacement test (shared by dbscan & points).

    A point is kept if its LARGEST net displacement over any ~1-second window of
    visible frames is >= static_thresh_frac * frame_width. Points that never
    move that far (static background / jitter) are dropped. This is the exact
    filter DBSCAN uses as its step 1, factored out so select_motion_points uses
    the same criterion instead of a quantile of total motion.

    Returns (moving_ids, total_motion) where total_motion (N,) = per-point
    summed object-speed / diagonal, for ranking.
    """
    T, N, _ = tracks.shape
    H, W = hw
    diag = float(np.hypot(H, W)) + 1e-8
    win = max(1, int(round(fps)))
    thr = static_thresh_frac * float(W)

    speed_pt = np.linalg.norm(np.diff(tracks, axis=0, prepend=tracks[:1]), axis=2)
    speed_pt = np.where(vis, speed_pt, 0.0)
    total_motion = speed_pt.sum(axis=0) / diag

    moving = []
    for n in range(N):
        vt = np.where(vis[:, n])[0]
        if vt.size < 2:
            continue
        xy = tracks[vt, n, :]
        max_disp = 0.0
        for a in range(len(vt)):
            b = a
            while b + 1 < len(vt) and (vt[b + 1] - vt[a]) <= win:
                b += 1
            d = float(np.linalg.norm(xy[b] - xy[a]))
            if d > max_disp:
                max_disp = d
        if max_disp >= thr:
            moving.append(n)
    return np.asarray(moving, dtype=int), total_motion


def select_motion_points(
    tracks: np.ndarray,
    visibility: np.ndarray,
    obj_vel: np.ndarray,
    hw: tuple[int, int],
    fps: float = 8.0,
    static_thresh_frac: float = 0.03,
    max_points: int | None = None,
) -> list[dict]:
    """Draw each MOVING point's own trajectory, WITHOUT clustering.

    Each returned "group" is a single point's own trajectory (no numbering /
    object grouping). Moving points are selected with the SAME displacement test
    as DBSCAN step 1 (largest ~1s-window displacement >= static_thresh_frac *
    frame width), not a motion quantile — so static background is dropped by the
    same criterion.

    Args:
        fps: tracking fps, for the ~1s static-filter window.
        static_thresh_frac: 1s-window displacement below this * frame width is
            treated as static and dropped (default 0.03 = 3%).
        max_points: optional hard cap on how many point-paths to draw (keeps the
            most-moving ones), to avoid clutter on busy scenes.

    Returns:
        list of {"point_ids": (1,) int, "centroid_track": (T,2)} — one per
        surviving point, sorted by descending total motion.
    """
    vis = visibility.astype(bool)
    moving, total_motion = _moving_point_ids(
        tracks, vis, hw, fps=fps, static_thresh_frac=static_thresh_frac)
    if moving.size == 0:
        return []

    # sort by motion (desc) so an optional cap keeps the most-moving points
    moving = moving[np.argsort(-total_motion[moving])]
    if max_points is not None and moving.size > max_points:
        moving = moving[:max_points]

    return [_make_group(np.array([n]), tracks, vis) for n in moving]


def select_object_tracks(
    tracks: np.ndarray,
    visibility: np.ndarray,
    obj_vel: np.ndarray,
    hw: tuple[int, int],
    motion_frac: float = 0.8,
    nms_radius_frac: float = 0.12,
    dir_cos_thresh: float = 0.8,
    max_objects: int = 8,
) -> list[dict]:
    """One representative trajectory per moving object, via spatial NMS.

    Instead of clustering with a fixed k (which merges or drops small objects),
    greedily pick the highest-motion point as an object "seed", then suppress
    other points that are BOTH spatially close to it AND move in a similar
    direction — i.e. belong to the same object. Repeat on what remains. The
    number of objects is data-driven, so a small object that is a local motion
    maximum still yields its own track and never gets absorbed by a bigger one.

    Args:
        motion_frac: quantile of per-point total motion to DROP as static.
        nms_radius_frac: suppression radius as a fraction of the frame diagonal.
            Points within this distance of a seed AND moving the same way are
            treated as the same object.
        dir_cos_thresh: cosine-similarity threshold on mean velocity direction;
            a nearby point is only suppressed if its direction matches the seed
            above this (so two objects crossing paths stay separate).
        max_objects: safety cap on number of representative tracks.

    Returns:
        list of {"point_ids": (1,) int seed, "centroid_track": (T,2)}, one per
        object, sorted by descending motion (most salient first).
    """
    T, N, _ = tracks.shape
    H, W = hw
    diag = float(np.hypot(H, W)) + 1e-8
    vis = visibility.astype(bool)

    speed_pt = np.linalg.norm(obj_vel, axis=2)
    speed_pt = np.where(vis, speed_pt, 0.0)
    motion = speed_pt.sum(axis=0) / diag                 # (N,)
    if not np.any(motion > 0):
        return []

    thr = np.quantile(motion[motion > 0], motion_frac)
    moving = np.where(motion > max(thr, 1e-6))[0]
    if moving.size == 0:
        moving = np.where(motion > 0)[0]
    if moving.size == 0:
        return []

    # mean visible position + mean velocity direction per moving point
    mean_pos = np.full((N, 2), np.nan)
    mean_dir = np.zeros((N, 2))
    for n in moving:
        m = vis[:, n]
        if not m.any():
            continue
        mean_pos[n] = tracks[m, n, :].mean(axis=0)
        v = obj_vel[m, n, :].mean(axis=0)
        nv = np.linalg.norm(v)
        if nv > 1e-8:
            mean_dir[n] = v / nv

    radius = nms_radius_frac * diag
    order = moving[np.argsort(-motion[moving])]          # high motion first
    suppressed = np.zeros(N, dtype=bool)
    seeds = []
    for n in order:
        if suppressed[n] or np.isnan(mean_pos[n]).any():
            continue
        seeds.append(n)
        if len(seeds) >= max_objects:
            break
        # suppress nearby same-direction points (same object)
        for m in order:
            if suppressed[m] or m == n or np.isnan(mean_pos[m]).any():
                continue
            dist = np.linalg.norm(mean_pos[m] - mean_pos[n])
            if dist > radius:
                continue
            cos = float(mean_dir[m] @ mean_dir[n])
            if cos >= dir_cos_thresh:
                suppressed[m] = True

    # each seed's OWN trajectory is the representative line for its object
    return [_make_group(np.array([s]), tracks, vis) for s in seeds]


def cluster_object_tracks(
    tracks: np.ndarray,
    visibility: np.ndarray,
    obj_vel: np.ndarray,
    hw: tuple[int, int],
    fps: float = 8.0,
    static_thresh_frac: float = 0.03,
    eps: float = 0.15,
    min_samples: int = 3,
    dir_weight: float = 0.75,
    max_objects: int = 12,
) -> list[dict]:
    """One representative (centroid) trajectory per object, via DBSCAN.

    Three-step pipeline (per user spec):
      1. Drop static points: a point is static if its LARGEST displacement over
         any ~1-second window is below `static_thresh_frac` of the frame width.
         (1s window = round(fps) frames; uses raw track displacement.)
      2. Cluster the survivors with DBSCAN on a [position, motion-direction]
         feature so points that are BOTH spatially near AND move alike group
         into one object. DBSCAN needs no preset cluster count (we don't know
         how many objects there are) and labels outliers as noise (-1), which
         drops stray points for free.
      3. One representative track per cluster: the MEDOID — the member point
         whose mean position is nearest the cluster's spatial centroid — and
         its OWN real trajectory (not a synthesized centroid average, which can
         smear across a long segment's motion).

    Args:
        fps: tracking-timeline fps, for the 1-second static-filter window.
        static_thresh_frac: 1s-window displacement below this * frame width is
            treated as static and dropped (default 0.03 = 3%).
        eps: DBSCAN neighbourhood radius in the normalized feature space
            ([x/W, y/H] in [0,1] plus dir_weight * unit-velocity).
        min_samples: DBSCAN core-point threshold (also the min points/object).
        dir_weight: how strongly motion direction (vs raw position) drives the
            clustering; 0 = position-only, higher = split co-located objects
            moving different ways.
        max_objects: safety cap on returned tracks (keeps the largest clusters).

    Returns:
        list of {"point_ids", "centroid_track", "valid"} (same shape as the
        other selectors), sorted by descending cluster motion.
    """
    from sklearn.cluster import DBSCAN

    T, N, _ = tracks.shape
    H, W = hw
    diag = float(np.hypot(H, W)) + 1e-8
    vis = visibility.astype(bool)

    # --- step 1: drop static points (shared displacement filter) ---
    # A point survives if its largest ~1s-window displacement >= 3% frame width.
    moving, _ = _moving_point_ids(
        tracks, vis, hw, fps=fps, static_thresh_frac=static_thresh_frac)
    if moving.size == 0:
        return []
    # camera-compensated total motion for ranking clusters (most salient first)
    speed_pt = np.where(vis, np.linalg.norm(obj_vel, axis=2), 0.0)
    total_motion = speed_pt.sum(axis=0) / diag           # (N,)

    # --- step 2: DBSCAN on [position, motion-direction] features ---
    feats = np.zeros((moving.size, 4), dtype=np.float64)
    for i, n in enumerate(moving):
        m = vis[:, n]
        pos = tracks[m, n, :].mean(axis=0) / np.array([W, H])   # in [0,1]
        v = obj_vel[m, n, :].mean(axis=0)
        nv = np.linalg.norm(v)
        vdir = v / nv if nv > 1e-8 else np.zeros(2)
        feats[i] = [pos[0], pos[1], dir_weight * vdir[0], dir_weight * vdir[1]]

    if moving.size < min_samples:
        # too few to cluster: treat all survivors as a single object
        labels = np.zeros(moving.size, dtype=int)
    else:
        labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(feats)

    groups = []
    for lab in sorted(set(labels.tolist())):
        if lab == -1:                       # DBSCAN noise -> dropped
            continue
        ids = moving[labels == lab]
        if ids.size == 0:
            continue
        # step 3: pick the ONE member point closest to the cluster centroid and
        # use ITS OWN trajectory (not a synthesized centroid track). The medoid
        # is a real tracked point, so its line is a genuine trajectory rather
        # than an average that can smear across a long segment's motion.
        rep = _cluster_medoid(ids, tracks, vis, W, H)
        groups.append(_make_group(np.array([rep]), tracks, vis))

    if not groups and moving.size:
        # everything was labelled noise: fall back to the medoid of all survivors
        rep = _cluster_medoid(moving, tracks, vis, W, H)
        groups = [_make_group(np.array([rep]), tracks, vis)]

    groups.sort(key=lambda g: -float(total_motion[g["point_ids"]].sum()))
    return groups[:max_objects]


def _cluster_medoid(ids: np.ndarray, tracks: np.ndarray, vis: np.ndarray,
                    W: int, H: int) -> int:
    """Return the member point id whose mean visible position is nearest the
    cluster's spatial centroid (the medoid). Falls back to ids[0] if none of
    the members is ever visible."""
    pts = []
    valid_ids = []
    for n in ids:
        m = vis[:, n]
        if m.any():
            pts.append(tracks[m, n, :].mean(axis=0))
            valid_ids.append(int(n))
    if not valid_ids:
        return int(ids[0])
    pts = np.asarray(pts)                       # (k, 2) mean positions
    centroid = pts.mean(axis=0)
    d = np.linalg.norm(pts - centroid, axis=1)
    return valid_ids[int(np.argmin(d))]


def _fill_nan(arr: np.ndarray) -> np.ndarray:
    """Forward then backward fill NaN rows in a (T,2) array."""
    out = arr.copy()
    last = None
    for t in range(len(out)):
        if np.isnan(out[t]).any():
            if last is not None:
                out[t] = last
        else:
            last = out[t]
    last = None
    for t in range(len(out) - 1, -1, -1):
        if np.isnan(out[t]).any():
            if last is not None:
                out[t] = last
        else:
            last = out[t]
    return out


def _time_color(frac: float) -> tuple[int, int, int]:
    """Blue (early) -> red (late) in RGB for a time fraction in [0,1]."""
    frac = float(np.clip(frac, 0.0, 1.0))
    r = int(255 * frac)
    b = int(255 * (1.0 - frac))
    g = int(60 * (1.0 - abs(frac - 0.5) * 2))  # slight green mid for visibility
    return (r, g, b)


def _row_color(y: float, H: int) -> tuple[int, int, int]:
    """Per-row color (RGB) from a pixel y-position, mirroring the diagnostic
    tracks GIF: hue = y / (H-1) in HSV, so each grid row gets its own hue."""
    import cv2

    hue = int(float(np.clip(y / max(H - 1, 1), 0.0, 1.0)) * 179)
    rgb = cv2.cvtColor(np.uint8([[[hue, 220, 255]]]), cv2.COLOR_HSV2RGB)[0, 0]
    return int(rgb[0]), int(rgb[1]), int(rgb[2])


def draw_marks_on_frames(
    frames: np.ndarray,
    sel_indices: np.ndarray,
    groups: list[dict],
    timestamps: np.ndarray | None = None,
    linewidth: int = 2,
    alpha: float = 0.7,
    step: int = 1,
    span: str = "full",
    number: bool = True,
    color_mode: str = "time",
    endpoint_radius: int = 3,
) -> np.ndarray:
    """Overlay numbered motion trajectories onto selected frames.

    Args:
        frames: (B, H, W, 3) uint8 — the selected key frames to mark.
        sel_indices: (B,) tracking-timeline indices of each selected frame.
            Used to slice the per-frame trajectory segment when span=="frame".
        groups: output of cluster_motion_groups (centroid_track is on the FULL
            tracking timeline, same pixel space as frames).
        timestamps: optional (unused; kept for API symmetry).
        linewidth: polyline thickness.
        alpha: overlay opacity (original frame kept for VLM semantics).
        step: draw every `step`-th trajectory segment (thin dense paths).
        span: "full" draws the whole start->end path on EVERY frame (all frames
            look the same overlay). "frame" draws only the segment between the
            previous selected frame and the current one, so each frame shows the
            incremental motion in that interval (frame-to-frame). Time color is
            always on the global timeline, so blue->red = absolute time in both.
        color_mode: "time" (default) colors each segment blue->red by its time
            within the drawn span. "row" gives each group ONE fixed color from
            its first-visible y-position (per-row hue), matching the diagnostic
            tracks GIF, so a trajectory reads as one continuous colored line.

    Returns:
        (B, H, W, 3) uint8 marked frames (copies; input untouched).
    """
    import cv2

    if not groups:
        return frames.copy()
    if span not in ("full", "frame"):
        raise ValueError(f"span must be 'full' or 'frame', got {span!r}")
    if color_mode not in ("time", "row"):
        raise ValueError(f"color_mode must be 'time' or 'row', got {color_mode!r}")

    B, H, W, _ = frames.shape
    out = np.empty_like(frames)
    sel = np.asarray(sel_indices).astype(int)

    # In "row" mode each group gets one fixed color from its first-visible y.
    group_colors = None
    if color_mode == "row":
        group_colors = []
        for g in groups:
            path = g["centroid_track"]
            valid = g.get("valid")
            if valid is not None and valid.any():
                t0 = int(np.argmax(valid))
            else:
                t0 = 0
            group_colors.append(_row_color(float(path[t0][1]), H))

    for bi in range(B):
        base = frames[bi]
        overlay = base.copy()
        # Segment [lo, hi] of the tracking timeline to draw on this frame.
        if span == "full":
            lo, hi = 0, None  # whole path
        else:  # frame-to-frame: prev selected index -> this selected index
            hi = int(sel[bi])
            lo = int(sel[bi - 1]) if bi > 0 else max(0, hi - 1)

        for gi, g in enumerate(groups):
            path = g["centroid_track"]  # (T,2) full timeline
            T = path.shape[0]
            pts = path.astype(np.int32)
            seg_hi = (T - 1) if hi is None else min(hi, T - 1)
            seg_lo = min(lo, seg_hi)
            if seg_hi - seg_lo < 1:
                # degenerate interval: nudge to show at least one segment
                seg_lo = max(0, seg_hi - 1)
            valid = g.get("valid")
            # gradient polyline over [seg_lo, seg_hi]. In "time" mode color runs
            # blue->red LOCAL to THIS frame's drawn span (seg_lo=blue,
            # seg_hi=red); in "row" mode the whole line is the group's fixed
            # per-row color. Skip any segment touching an invalid (pre-seed /
            # hallucinated) frame so a mid-clip entrant shows no line before it
            # appears.
            seg_span = max(seg_hi - seg_lo, 1)
            last_valid_t = None
            for t in range(seg_lo, seg_hi, step):
                t1 = min(t + step, seg_hi)
                if valid is not None and not (valid[t] and valid[t1]):
                    continue
                p0 = tuple(pts[t]); p1 = tuple(pts[t1])
                if color_mode == "row":
                    col = group_colors[gi]
                else:
                    col = _time_color((t - seg_lo) / seg_span)  # local blue->red
                cv2.line(overlay, p0, p1, col, linewidth, lineType=cv2.LINE_AA)
                last_valid_t = t1
            # endpoint dot: mark where this frame's trajectory ends (the
            # object's current position) with a larger filled circle.
            if endpoint_radius > 0 and last_valid_t is not None:
                end_col = group_colors[gi] if color_mode == "row" else _time_color(1.0)
                cv2.circle(overlay, tuple(pts[last_valid_t]), endpoint_radius,
                           end_col, -1, cv2.LINE_AA)
            # numbered label near the segment start (skipped when not grouping)
            if number and (valid is None or valid[seg_lo]):
                lbl = str(gi + 1)
                org = (int(pts[seg_lo][0]) + 4, int(pts[seg_lo][1]) - 4)
                cv2.putText(overlay, lbl, org, cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (255, 255, 255), 3, cv2.LINE_AA)
                cv2.putText(overlay, lbl, org, cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (0, 0, 0), 1, cv2.LINE_AA)

        marked = cv2.addWeighted(overlay, alpha, base, 1.0 - alpha, 0.0)
        out[bi] = marked

    return out


def legend_text(n_groups: int, span: str = "full", number: bool = True,
                color_mode: str = "time", version: str = "prompt-v1") -> str:
    """Prompt sentence describing the marks; placed right before the question
    (after the frames) — this position scored best in prompt-position tests.

    version: "prompt-v1" (default, original wording) or "prompt-v2" (row-color
    only) which adds three clarifications shown to help on stronger models:
    plural framing ("some of the frames"), an explicit note that line colors
    only distinguish points (no other meaning), and a caveat that the overlays
    are motion aids — NOT objects — so counting/identity should ignore them.
    Falls back to prompt-v1 for time-color mode (prompt-v2 only revises the
    row-color legend)."""
    if n_groups <= 0:
        return ""
    if version == "prompt-v2" and color_mode == "row":
        return (
            "Some of the frames have motion trajectories drawn on moving points "
            "as an overlay: each colored line shows the path a tracked point "
            "moved along between the previous frame and the current one, ending "
            "in a filled dot at the point's current position. The line colors "
            "are only to tell different points apart — they carry no other "
            "meaning. These overlays are visual aids for motion, not real "
            "objects in the scene: do not count the lines or dots as objects, "
            "and judge object identity, attributes, and counts from the "
            "underlying scene as if the overlays were not there. Use the "
            "trajectories only to reason about how things move."
        )
    frame_clause = (
        "each line shows the path the object point moved along between the "
        "previous frame and this frame"
        if span == "frame"
        else "each line shows the object point's full motion path across the whole clip"
    )
    if color_mode == "row":
        return (
            "The image has motion trajectories drawn on some moving object "
            f"points: {frame_clause}, and each object point's line has its own "
            "single color with a larger filled dot of that color marking where "
            "the point is in this frame. Use these trajectories to reason about "
            "how things move."
        )
    return (
        "The image has motion trajectories drawn as lines on some moving "
        "objects. Each line's color goes from blue (earlier in time) to red "
        f"(later); {frame_clause}. Use these trajectories to reason about how "
        "things move."
    )

