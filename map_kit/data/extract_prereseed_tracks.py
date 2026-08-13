"""Extract per-segment trajectories (the paths accumulated right BEFORE each
re-seed) for one CLEVRER sample. With reseed_every_s=1s and tracking_fps=8,
CoTracker re-drops a fresh grid every 8 frames; segments overlap neighbours by
1 frame: [0-8],[8-16],[16-24],[24-32],[32-40] for a 41-frame clip.

Each segment's grid points live in their own columns, visible only over that
segment's span. For segment k we draw that segment's full trajectory (blue->red
over its frames) onto the segment's LAST frame (== the frame just before the
next re-seed) and save one PNG per segment.
"""
import argparse
import os
import sys

import cv2
import numpy as np

_PKG_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.normpath(os.path.join(_PKG_DIR, ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from map_kit.data.video_io import read_video
from map_kit.data.cotracker_runner import CoTrackerRunner

COT = "/path/to/basemodels/scaled_offline.pth"


def _draw_seg(frame, tracks, vis, s, e, radius=3, lw=2):
    """Draw columns visible over [s,e] as blue(early)->red(late) polylines on a
    copy of `frame`. tracks (T,N,2), vis (T,N)."""
    img = frame.copy()
    N = tracks.shape[1]
    span = list(range(s, e + 1))
    for n in range(N):
        pts = [(t, tracks[t, n]) for t in span if vis[t, n]]
        if len(pts) < 2:
            continue
        for i in range(len(pts) - 1):
            frac = i / max(1, len(pts) - 2)          # 0->1 over the segment
            # blue (255,0,0 in BGR) -> red (0,0,255)
            color = (int(255 * (1 - frac)), 0, int(255 * frac))
            p0 = tuple(np.round(pts[i][1]).astype(int))
            p1 = tuple(np.round(pts[i + 1][1]).astype(int))
            cv2.line(img, p0, p1, color, lw, cv2.LINE_AA)
        # mark the last (pre-reseed) position
        cv2.circle(img, tuple(np.round(pts[-1][1]).astype(int)), radius,
                   (0, 0, 255), -1, cv2.LINE_AA)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="counterfactual_inference")
    ap.add_argument("--item_id", type=int, default=1)
    ap.add_argument("--tracking_fps", type=float, default=8.0)
    ap.add_argument("--reseed_every_s", type=float, default=1.0)
    ap.add_argument("--grid_size", type=int, default=10)
    ap.add_argument("--out_dir", default=None)
    args = ap.parse_args()

    import json
    mvroot = "/path/to/benchmarks/MVBench"
    items = json.load(open(f"{mvroot}/json/{args.task}.json"))
    vid = items[args.item_id]["video"]
    vpath = f"{mvroot}/video/clevrer/video_validation/{vid}"
    out_dir = args.out_dir or os.path.join(
        _ROOT, "visualizations", "clevrer_ada_win",
        f"{args.task}_item{args.item_id}", "presreed_segments")
    os.makedirs(out_dir, exist_ok=True)

    raw = read_video(vpath, tracking_fps=args.tracking_fps, max_frames=200)
    T = raw.num_frames
    runner = CoTrackerRunner(checkpoint_path=COT, grid_size=args.grid_size,
                             segment_track=True, reseed_every_s=args.reseed_every_s)
    tr = runner.track(raw.frames, fps=raw.fps)      # segment spacing = reseed step
    tracks, vis = tr["tracks"], tr["visibility"]

    step = max(1, int(round(args.reseed_every_s * args.tracking_fps)))
    spans = [(s, min(s + step, T - 1)) for s in range(0, T - 1, step)]
    print(f"video={vid} T={T} step={step} segments={spans}")

    for k, (s, e) in enumerate(spans):
        img = _draw_seg(raw.frames[e], tracks, vis, s, e)
        p = os.path.join(out_dir, f"seg{k}_f{s:02d}-{e:02d}_prereseed.png")
        cv2.imwrite(p, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        print(f"  seg{k} frames[{s}..{e}] -> {p}")
    print(f"saved {len(spans)} segment images -> {out_dir}")


if __name__ == "__main__":
    main()
