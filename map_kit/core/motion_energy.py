"""Motion energy curve M(t) from CoTracker tracks (route 2).

Given per-frame point tracks, compute a 1-D "motion energy" signal that peaks
at the moments a human would call motion-relevant: fast motion, direction
changes, accelerations (contact / collision), and points spreading apart or
coming together (approach / separation).

Camera self-motion is removed first by estimating a global similarity motion
per frame-pair and subtracting it, so M(t) reflects *object* motion rather
than panning.
"""
from __future__ import annotations

import numpy as np


def _robust_norm(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Scale a non-negative signal to [0,1] using a robust (95th pct) max."""
    if x.size == 0:
        return x
    hi = np.percentile(x, 95)
    if hi < eps:
        return np.zeros_like(x)
    return np.clip(x / hi, 0.0, 1.0)


def _estimate_camera_flow(
    p0: np.ndarray, p1: np.ndarray, mask: np.ndarray
) -> np.ndarray:
    """Estimate global per-point displacement from a similarity transform.

    Fits p1 ~= s R p0 + t (translation + rotation + scale) by least squares on
    visible points, then returns the predicted displacement for ALL points.
    This is the camera/background component to subtract.
    """
    n = p0.shape[0]
    if mask.sum() < 3:
        return np.zeros((n, 2), dtype=np.float64)

    src = p0[mask]
    dst = p1[mask]
    # Solve for [a -b tx; b a ty] (similarity) via linear least squares.
    # [x -y 1 0; y x 0 1] @ [a b tx ty]^T = [x'; y']
    x, y = src[:, 0], src[:, 1]
    A = np.zeros((2 * len(src), 4), dtype=np.float64)
    A[0::2] = np.stack([x, -y, np.ones_like(x), np.zeros_like(x)], axis=1)
    A[1::2] = np.stack([y, x, np.zeros_like(x), np.ones_like(x)], axis=1)
    b = dst.reshape(-1)
    try:
        params, *_ = np.linalg.lstsq(A, b, rcond=None)
    except np.linalg.LinAlgError:
        return np.zeros((n, 2), dtype=np.float64)

    a, bb, tx, ty = params
    pred_x = a * p0[:, 0] - bb * p0[:, 1] + tx
    pred_y = bb * p0[:, 0] + a * p0[:, 1] + ty
    pred = np.stack([pred_x, pred_y], axis=1)
    return pred - p0


def compute_motion_energy(
    tracks: np.ndarray,
    visibility: np.ndarray,
    hw: tuple[int, int],
    compensate_camera: bool = True,
    weights: dict | None = None,
    smooth_window: int = 3,
) -> dict:
    """Compute the per-frame motion energy curve M(t).

    Args:
        tracks: (T, N, 2) pixel coords.
        visibility: (T, N) bool.
        hw: (H, W) of the frames the coords live in (for scale normalization).
        compensate_camera: subtract estimated global similarity motion.
        weights: relative weights for the 4 components (speed/accel/curv/diverg).
        smooth_window: moving-average window (frames) applied to M(t).

    Returns:
        dict with 'M' (T,), and the four raw normalized components, plus the
        residual (object) velocity field 'obj_vel' (T, N, 2).
    """
    T, N, _ = tracks.shape
    H, W = hw
    diag = float(np.hypot(H, W)) + 1e-8
    # divergence defaults to 0: point-cloud spread change was noisy and did not
    # help on TempCompass; speed/accel/curvature carry the motion signal.
    w = {"speed": 1.0, "accel": 1.0, "curvature": 1.0, "divergence": 0.0}
    if weights:
        w.update(weights)

    tracks = tracks.astype(np.float64)
    vis = visibility.astype(bool)

    # Per-frame-pair velocity (frame t -> t+1), object-only if compensating.
    obj_vel = np.zeros((T, N, 2), dtype=np.float64)
    for t in range(T - 1):
        disp = tracks[t + 1] - tracks[t]
        pair_vis = vis[t] & vis[t + 1]
        if compensate_camera:
            cam = _estimate_camera_flow(tracks[t], tracks[t + 1], pair_vis)
            disp = disp - cam
        disp[~pair_vis] = 0.0
        obj_vel[t] = disp
    obj_vel[T - 1] = obj_vel[T - 2] if T >= 2 else 0.0

    speed_pt = np.linalg.norm(obj_vel, axis=2) / diag           # (T, N)
    vis_f = vis.astype(np.float64)
    denom = np.maximum(vis_f.sum(axis=1), 1.0)

    # 1) Speed: mean object speed over visible points.
    speed = (speed_pt * vis_f).sum(axis=1) / denom

    # 2) Acceleration: change in velocity between consecutive frames.
    accel_vec = np.zeros_like(obj_vel)
    accel_vec[1:] = obj_vel[1:] - obj_vel[:-1]
    accel_pt = np.linalg.norm(accel_vec, axis=2) / diag
    accel = (accel_pt * vis_f).sum(axis=1) / denom

    # 3) Curvature: angle change of each point's velocity (direction reversal).
    curv_pt = np.zeros((T, N), dtype=np.float64)
    v_prev = obj_vel[:-1]
    v_next = obj_vel[1:]
    dot = (v_prev * v_next).sum(axis=2)
    nrm = np.linalg.norm(v_prev, axis=2) * np.linalg.norm(v_next, axis=2)
    cos = np.divide(dot, nrm, out=np.ones_like(dot), where=nrm > 1e-8)
    ang = np.arccos(np.clip(cos, -1.0, 1.0))            # (T-1, N)
    # weight by motion magnitude so still points don't emit spurious angles
    mag = np.minimum(np.linalg.norm(v_prev, axis=2), np.linalg.norm(v_next, axis=2))
    curv_pt[1:] = ang * (mag / diag)
    curvature = (curv_pt * vis_f).sum(axis=1) / denom

    # 4) Divergence: rate of change of mean pairwise distance to the centroid
    #    (points approaching/separating => contact, expansion, collapse).
    diverg = np.zeros(T, dtype=np.float64)
    spread = np.zeros(T, dtype=np.float64)
    for t in range(T):
        m = vis[t]
        if m.sum() >= 2:
            pts = tracks[t][m]
            c = pts.mean(axis=0)
            spread[t] = np.linalg.norm(pts - c, axis=1).mean() / diag
    diverg[1:] = np.abs(spread[1:] - spread[:-1])

    comp = {
        "speed": _robust_norm(speed) * w["speed"],
        "accel": _robust_norm(accel) * w["accel"],
        "curvature": _robust_norm(curvature) * w["curvature"],
        "divergence": _robust_norm(diverg) * w["divergence"],
    }
    M = comp["speed"] + comp["accel"] + comp["curvature"] + comp["divergence"]
    M = _robust_norm(M)
    if smooth_window and smooth_window > 1:
        M = _moving_average(M, smooth_window)
        M = _robust_norm(M)

    return {"M": M, "components": comp, "obj_vel": obj_vel}


def _moving_average(x: np.ndarray, k: int) -> np.ndarray:
    if k <= 1 or x.size == 0:
        return x
    pad = k // 2
    xp = np.pad(x, (pad, pad), mode="edge")
    kernel = np.ones(k, dtype=np.float64) / k
    return np.convolve(xp, kernel, mode="valid")[: len(x)]
