"""Frame budget allocation from a motion energy curve M(t) (route 2).

Given M(t) over the tracking-fps timeline and a fixed frame budget B, pick B
frame indices that (a) keep a few uniform "anchor" frames for temporal
coverage, and (b) spend the rest on high-motion moments via peak detection +
non-maximum suppression, falling back to importance sampling.

Returns indices into the tracking-fps timeline; the caller maps them back to
original timestamps for the VLM prompt.
"""
from __future__ import annotations

import numpy as np


def select_frames(
    M: np.ndarray,
    budget: int,
    n_anchor: int | None = None,
    nms_radius: int | None = None,
) -> dict:
    """Allocate a frame budget over the motion energy curve.

    Args:
        M: (T,) motion energy in [0,1] on the tracking-fps timeline.
        budget: total number of frames to select (== VLM frame budget).
        n_anchor: number of uniform anchor frames guaranteeing time coverage.
            Defaults to ceil(budget/4), min 2 (first & last).
        nms_radius: min index gap between two motion peaks. Defaults to
            T // (4*budget), i.e. one quarter of the uniform sampling interval
            T/budget. This lets motion peaks pack up to ~4x denser than uniform
            while still preventing the whole peak budget from collapsing onto a
            single frame. Time coverage is further guaranteed by the anchors.

    Returns:
        dict with 'indices' (sorted int array, len == min(budget, T)),
        'anchors' and 'peaks' (the two sub-selections, for debugging).
    """
    T = int(M.shape[0])
    budget = int(budget)
    if budget >= T:
        return {"indices": np.arange(T), "anchors": np.arange(T), "peaks": np.array([], dtype=int)}
    if budget <= 0:
        return {"indices": np.array([], dtype=int), "anchors": np.array([], dtype=int), "peaks": np.array([], dtype=int)}

    if n_anchor is None:
        n_anchor = max(2, int(np.ceil(budget / 4)))
    n_anchor = min(n_anchor, budget)
    if nms_radius is None:
        nms_radius = max(1, T // (2 * budget))

    # 1) Uniform anchors (always include first & last for coverage).
    anchors = np.unique(np.linspace(0, T - 1, num=n_anchor).round().astype(np.int64))

    # 2) Motion peaks for the remaining budget.
    selected = set(anchors.tolist())
    n_peaks = budget - len(selected)
    peaks: list[int] = []
    if n_peaks > 0:
        peaks = _greedy_peaks(M, n_peaks, nms_radius, exclude=selected)
        selected.update(peaks)

    # 3) If NMS left us short (flat/short curves), top up by importance sampling.
    if len(selected) < budget:
        order = np.argsort(-M)  # highest energy first
        for idx in order:
            if len(selected) >= budget:
                break
            selected.add(int(idx))

    indices = np.array(sorted(selected)[:budget], dtype=np.int64)
    return {
        "indices": indices,
        "anchors": anchors,
        "peaks": np.array(sorted(peaks), dtype=np.int64),
    }


def _greedy_peaks(
    M: np.ndarray, n: int, radius: int, exclude: set[int]
) -> list[int]:
    """Greedily pick the top-n energy indices with a min-gap (NMS) constraint."""
    order = np.argsort(-M)
    chosen: list[int] = []
    blocked = np.zeros(len(M), dtype=bool)
    for idx in exclude:
        lo, hi = max(0, idx - radius), min(len(M), idx + radius + 1)
        blocked[lo:hi] = True
    for idx in order:
        if len(chosen) >= n:
            break
        if blocked[idx]:
            continue
        chosen.append(int(idx))
        lo, hi = max(0, idx - radius), min(len(M), idx + radius + 1)
        blocked[lo:hi] = True
    return chosen
