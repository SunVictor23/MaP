"""SSv2 inference on GPT-5.5 (remote) with ADAPTIVE sampling + route-1 tracks.

Hybrid pipeline (mirrors run_mvbench_gpt_adaptive.py) for the Something-
Something-v2 4-way MC subset: local GPU does motion-guided frame selection
(route 2) and draws CoTracker3 trajectories (route 1); marked frames go to the
remote GPT-5.5 vision endpoint. GPU sampling is sequential in the main thread;
GPT calls are dispatched to a thread pool.

Reuses ssv2 templating/scoring + the MotionAdaptiveSampler + track_marker.
Single task key "action_recognition"; output/scoring identical to run_ssv2.
"""
import argparse
import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

_PKG_DIR = os.path.dirname(os.path.abspath(__file__))
_MOTIONVLM_ROOT = os.path.normpath(os.path.join(_PKG_DIR, ".."))
if _MOTIONVLM_ROOT not in sys.path:
    sys.path.insert(0, _MOTIONVLM_ROOT)

from map_kit import MotionAdaptiveSampler
from map_kit.models.gpt_client import GPTClient, build_vision_content, DEFAULT_MODEL
from map_kit.core.track_marker import (
    cluster_motion_groups, select_motion_points, select_object_tracks,
    cluster_object_tracks, draw_marks_on_frames, legend_text,
)
from map_kit.benchmarks.run_mvbench import (
    SYSTEM_PROMPT, QUESTION_SUFFIX, check_ans, _acc_from_results, _dump,
)
from map_kit.data.ssv2_data import (
    TASK_KEY, DEFAULT_SUBSET, DEFAULT_VIDEO_ROOT, ssv2_qa_template,
)

DEFAULT_API_KEY = os.environ.get("MAP_API_KEY", "")  # set via env var; never hardcode
DEFAULT_COTRACKER_CKPT = "/path/to/basemodels/scaled_offline.pth"


def _mark_frames(res, args):
    frames = res.frames
    legend = None
    if res.tracks is None:
        return frames, legend
    hw = res.frames.shape[1:3]
    if args.mark_mode == "cluster":
        groups = cluster_motion_groups(res.tracks, res.visibility, res.obj_vel,
                                       hw=hw, max_groups=args.max_groups,
                                       motion_frac=args.motion_frac)
    elif args.mark_mode == "points":
        groups = select_motion_points(res.tracks, res.visibility, res.obj_vel,
                                      hw=hw, fps=args.tracking_fps,
                                      max_points=args.max_points)
    elif args.mark_mode == "dbscan":
        groups = cluster_object_tracks(res.tracks, res.visibility, res.obj_vel,
                                       hw=hw, fps=args.tracking_fps,
                                       max_objects=args.max_objects)
    else:
        groups = select_object_tracks(res.tracks, res.visibility, res.obj_vel,
                                      hw=hw, motion_frac=args.motion_frac,
                                      nms_radius_frac=args.nms_radius_frac,
                                      max_objects=args.max_objects)
    numbered = args.mark_mode == "cluster"
    if groups:
        frames = draw_marks_on_frames(res.frames, res.indices, groups,
                                      res.timestamps, span=args.track_span,
                                      number=numbered, color_mode=args.track_color,
                                      linewidth=args.track_linewidth)
        legend = legend_text(len(groups), span=args.track_span, number=numbered,
                             color_mode=args.track_color,
                             version=args.legend_version)
    return frames, legend


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api_key", default=DEFAULT_API_KEY)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--cotracker_ckpt", default=DEFAULT_COTRACKER_CKPT)
    ap.add_argument("--subset", default=DEFAULT_SUBSET)
    ap.add_argument("--video_root", default=DEFAULT_VIDEO_ROOT)
    ap.add_argument("--output_path", default=None)
    ap.add_argument("--mode", default="adaptive", choices=["adaptive", "uniform"])
    ap.add_argument("--sampling_rule", default="qwen3vl", choices=["qwen3vl", "fps1"])
    ap.add_argument("--video_fps", type=float, default=None)
    ap.add_argument("--tracking_fps", type=float, default=8.0)
    ap.add_argument("--grid_size", type=int, default=10)
    ap.add_argument("--max_track_frames", type=int, default=200)
    ap.add_argument("--segment_track", action="store_true")
    ap.add_argument("--no_camera_comp", action="store_true")
    ap.add_argument("--mark_mode", default="points",
                    choices=["object", "points", "cluster", "dbscan"])
    ap.add_argument("--track_span", default="frame", choices=["full", "frame"])
    ap.add_argument("--track_color", default="row", choices=["row", "time"])
    ap.add_argument("--legend_version", default="prompt-v2",
                    choices=["prompt-v1", "prompt-v2"])
    ap.add_argument("--track_linewidth", type=int, default=1)
    ap.add_argument("--max_points", type=int, default=40)
    ap.add_argument("--max_objects", type=int, default=8)
    ap.add_argument("--max_groups", type=int, default=3)
    ap.add_argument("--nms_radius_frac", type=float, default=0.12)
    ap.add_argument("--motion_frac", type=float, default=0.8)
    ap.add_argument("--no_tracks", action="store_true")
    ap.add_argument("--jpeg_quality", type=int, default=90)
    ap.add_argument("--no_timestamps", action="store_true")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--max_new_tokens", type=int, default=8192)
    ap.add_argument("--refill_empty", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--num_shards", type=int, default=1)
    ap.add_argument("--shard_id", type=int, default=0)
    args = ap.parse_args()

    tag_suffix = "" if args.no_tracks else f"-marks-{args.mark_mode}"
    output_path = args.output_path or os.path.join(
        _MOTIONVLM_ROOT, "predictions", f"gpt55-ssv2-{args.mode}{tag_suffix}")
    output_path = os.path.normpath(output_path)
    os.makedirs(output_path, exist_ok=True)

    print(f"[gpt-adapt] loading CoTracker from {args.cotracker_ckpt}")
    sampler = MotionAdaptiveSampler(
        cotracker_ckpt=args.cotracker_ckpt,
        tracking_fps=args.tracking_fps,
        grid_size=args.grid_size,
        max_track_frames=args.max_track_frames,
        compensate_camera=not args.no_camera_comp,
        segment_track=args.segment_track,
    )
    client = GPTClient(api_key=args.api_key, model=args.model)

    with open(args.subset) as f:
        items = json.load(f)
    my_items = list(enumerate(items))[args.shard_id::args.num_shards]

    pred_file = os.path.join(
        output_path,
        f"{TASK_KEY}.shard{args.shard_id}.json" if args.num_shards > 1
        else f"{TASK_KEY}.json")

    lock = threading.Lock()
    results, done_ids = [], set()
    if os.path.isfile(pred_file):
        with open(pred_file) as f:
            results = json.load(f).get("results", [])
        if args.refill_empty:
            kept = [r for r in results if str(r.get("prediction", "")).strip()]
            print(f"[refill] re-running {len(results)-len(kept)} empty predictions")
            results = kept
        done_ids = {r["item_id"] for r in results if "item_id" in r}

    todo = [(i, d) for i, d in my_items if i not in done_ids]
    if args.limit:
        todo = todo[:args.limit]

    def gpt_call(item_id, question, gt, content, system):
        pred = client.chat(content, system=system, max_tokens=args.max_new_tokens)
        return {"item_id": item_id, "task_type": TASK_KEY, "question": question,
                "answer": gt, "prediction": pred, "correct": bool(check_ans(pred, gt))}

    n_since = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = []
        for item_id, data in tqdm(todo, desc=f"gpt-adapt:ssv2 s{args.shard_id} (sample)"):
            video_path = os.path.join(args.video_root, f"{data['video_id']}.webm")
            if not os.path.isfile(video_path):
                continue
            question, gt = ssv2_qa_template(data)
            try:
                res = sampler.sample(video_path, budget=None, mode=args.mode,
                                     bound=None, data_type="video",
                                     with_tracks=not args.no_tracks,
                                     sampling_rule=args.sampling_rule,
                                     target_fps=args.video_fps)
            except Exception as e:
                print(f"[warn] {data['video_id']}: sampling failed: {e}")
                continue
            if args.no_tracks:
                frames, legend = res.frames, None
            else:
                try:
                    frames, legend = _mark_frames(res, args)
                except Exception as e:
                    print(f"[warn] {data['video_id']}: marking failed: {e}")
                    frames, legend = res.frames, None
            q = (legend + "\n" if legend else "") + question + QUESTION_SUFFIX
            content = build_vision_content(
                list(frames),
                timestamps=(None if args.no_timestamps else res.timestamps),
                question=q, jpeg_quality=args.jpeg_quality)
            futs.append(ex.submit(gpt_call, item_id, question, gt, content, SYSTEM_PROMPT))

        for fut in tqdm(as_completed(futs), total=len(futs),
                        desc=f"gpt-adapt:ssv2 s{args.shard_id} (gpt)"):
            r = fut.result()
            with lock:
                results.append(r); n_since += 1
                if n_since >= 10:
                    _dump(pred_file, results); n_since = 0

    _dump(pred_file, results)
    a = _acc_from_results(results).get(TASK_KEY, {})
    print(f"[gpt-adapt s{args.shard_id}] ssv2: {a.get('correct',0)}/"
          f"{a.get('total',0)} = {a.get('acc',0):.2f}%  -> {pred_file}")


if __name__ == "__main__":
    main()
