"""MVBench inference on GPT-5.5 (remote) with ADAPTIVE sampling + route-1 tracks.

Hybrid pipeline: local GPU does motion-guided frame selection (route 2) AND
draws CoTracker3 point trajectories on the selected frames (route 1, points +
frame-span + row color, matching the CATER/TempCompass track runs); the marked
frames are then sent to the remote GPT-5.5 vision endpoint.

Because CoTracker is GPU-bound (one model on the GPU) and GPT is HTTP-bound, the
GPU sampling+marking runs sequentially in the main thread while the (slow) GPT
calls are dispatched to a thread pool — the GPU keeps sampling the next item
while earlier items' HTTP requests are in flight.

Reuses run_mvbench's task list / MC templating / scoring, the MotionAdaptiveSampler,
and track_marker. Output/scoring identical to run_mvbench.
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
    DATA_LIST, TASK_KEYS, KEY_TO_TASK, SYSTEM_PROMPT, QUESTION_SUFFIX,
    qa_template, check_ans, _acc_from_results, _dump,
)

DEFAULT_API_KEY = os.environ.get("MAP_API_KEY", "")  # set via env var; never hardcode
DEFAULT_COTRACKER_CKPT = "/path/to/basemodels/scaled_offline.pth"


def _mark_frames(res, args):
    """Draw route-1 trajectories on the sampled frames. Returns (frames, legend)."""
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


def _run_task(sampler, client, args, task_name, output_path):
    json_file, prefix, data_type, has_bound = DATA_LIST[task_name]
    key = TASK_KEYS[task_name]

    qfile = os.path.join(args.mvbench_root, "json", json_file)
    if not os.path.isfile(qfile):
        print(f"[skip] {task_name}: missing json {qfile}")
        return
    with open(qfile) as f:
        items = json.load(f)

    my_items = list(enumerate(items))[args.shard_id :: args.num_shards]

    if args.num_shards > 1:
        pred_file = os.path.join(output_path, f"{key}.shard{args.shard_id}.json")
    else:
        pred_file = os.path.join(output_path, f"{key}.json")

    lock = threading.Lock()
    results = []
    done_ids = set()
    if os.path.isfile(pred_file):
        with open(pred_file) as f:
            prev = json.load(f)
        results = prev.get("results", [])
        if args.refill_empty:
            # keep only non-empty predictions; empties get re-run & backfilled.
            kept = [r for r in results if str(r.get("prediction", "")).strip()]
            n_refill = len(results) - len(kept)
            results = kept
            print(f"[refill] {task_name}: re-running {n_refill} empty predictions")
        done_ids = {r["item_id"] for r in results if "item_id" in r}

    video_root = os.path.join(args.mvbench_root, "video", prefix)
    todo = [(i, d) for i, d in my_items if i not in done_ids]
    if args.limit:
        todo = todo[:args.limit]

    def gpt_call(item_id, task_name, question, gt, content, system):
        pred = client.chat(content, system=system, max_tokens=args.max_new_tokens)
        correct = check_ans(pred, gt)
        return {"item_id": item_id, "task_type": task_name, "question": question,
                "answer": gt, "prediction": pred, "correct": bool(correct)}

    n_since_dump = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = []
        # GPU sampling+marking sequential; HTTP dispatched as each item is ready.
        for item_id, data in tqdm(todo, desc=f"gpt-adapt:{key} shard{args.shard_id} (sample)"):
            video_path = os.path.join(video_root, data["video"])
            exists = os.path.isdir(video_path) if data_type == "frame" else os.path.isfile(video_path)
            if not exists:
                continue
            bound = (data["start"], data["end"]) if has_bound else None
            question, gt = qa_template(data)
            try:
                res = sampler.sample(video_path, budget=None, mode=args.mode,
                                     bound=bound, data_type=data_type,
                                     with_tracks=not args.no_tracks,
                                     sampling_rule=args.sampling_rule,
                                     target_fps=args.video_fps)
            except Exception as e:
                print(f"[warn] {task_name}/{data['video']}: sampling failed: {e}")
                continue
            if args.no_tracks:
                frames, legend = res.frames, None  # adaptive selection, no marks
            else:
                try:
                    frames, legend = _mark_frames(res, args)
                except Exception as e:
                    print(f"[warn] {task_name}/{data['video']}: marking failed: {e}")
                    frames, legend = res.frames, None
            prefix_txt = legend  # route-1 legend before the question
            q = (prefix_txt + "\n" if prefix_txt else "") + question + QUESTION_SUFFIX
            content = build_vision_content(list(frames),
                                          timestamps=(None if args.no_timestamps else res.timestamps),
                                          question=q, jpeg_quality=args.jpeg_quality)
            futs.append(ex.submit(gpt_call, item_id, task_name, question, gt,
                                  content, SYSTEM_PROMPT))

        for fut in tqdm(as_completed(futs), total=len(futs),
                        desc=f"gpt-adapt:{key} shard{args.shard_id} (gpt)"):
            r = fut.result()
            with lock:
                results.append(r)
                n_since_dump += 1
                if n_since_dump >= 10:
                    _dump(pred_file, results)
                    n_since_dump = 0

    _dump(pred_file, results)
    acc = _acc_from_results(results)
    a = acc.get(task_name, {})
    print(f"[gpt-adapt shard{args.shard_id}] {task_name}: "
          f"{a.get('correct',0)}/{a.get('total',0)} = {a.get('acc',0):.2f}%  -> {pred_file}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api_key", default=DEFAULT_API_KEY)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--cotracker_ckpt", default=DEFAULT_COTRACKER_CKPT)
    ap.add_argument("--mvbench_root", default="/path/to/benchmarks/MVBench")
    ap.add_argument("--output_path", default=None,
                    help="default: predictions/gpt55-mvbench-adaptive-points")
    ap.add_argument("--task", default="all")
    ap.add_argument("--mode", default="adaptive", choices=["adaptive", "uniform"],
                    help="frame selection: adaptive (M(t) route-2) or uniform "
                         "(evenly spaced, same budget); tracks are drawn in both")
    ap.add_argument("--sampling_rule", default="qwen3vl", choices=["qwen3vl", "fps1"])
    ap.add_argument("--video_fps", type=float, default=None)
    ap.add_argument("--tracking_fps", type=float, default=8.0)
    ap.add_argument("--grid_size", type=int, default=10)
    ap.add_argument("--max_track_frames", type=int, default=200)
    ap.add_argument("--segment_track", action="store_true",
                    help="frame-aligned segment tracking (offline; recommended "
                         "with track_span=frame so each line = motion between two "
                         "selected frames)")
    ap.add_argument("--no_camera_comp", action="store_true")
    # route-1 marking
    ap.add_argument("--mark_mode", default="points",
                    choices=["object", "points", "cluster", "dbscan"])
    ap.add_argument("--track_span", default="frame", choices=["full", "frame"])
    ap.add_argument("--track_color", default="row", choices=["row", "time"])
    ap.add_argument("--legend_version", default="prompt-v1",
                    choices=["prompt-v1", "prompt-v2"],
                    help="route-1 legend wording: prompt-v1 (original) or "
                         "prompt-v2 (adds plural framing + 'colors carry no "
                         "meaning' + 'overlays are not objects, do not count them')")
    ap.add_argument("--track_linewidth", type=int, default=1)
    ap.add_argument("--max_points", type=int, default=40)
    ap.add_argument("--max_objects", type=int, default=8)
    ap.add_argument("--max_groups", type=int, default=3)
    ap.add_argument("--nms_radius_frac", type=float, default=0.12)
    ap.add_argument("--motion_frac", type=float, default=0.8)
    # gpt / io
    ap.add_argument("--no_tracks", action="store_true",
                    help="skip drawing trajectories: adaptive frame SELECTION "
                         "only, no route-1 marks (isolates selection from the "
                         "track overlay). Also skips CoTracker where possible.")
    ap.add_argument("--jpeg_quality", type=int, default=90)
    ap.add_argument("--no_timestamps", action="store_true",
                    help="omit the <t.t seconds> text token before each frame")
    ap.add_argument("--workers", type=int, default=12,
                    help="concurrent HTTP requests (GPU sampling is sequential)")
    ap.add_argument("--max_new_tokens", type=int, default=2048,
                    help="GPT-5.5 is a reasoning model: needs a large budget")
    ap.add_argument("--refill_empty", action="store_true",
                    help="only re-run items whose stored prediction is empty "
                         "(e.g. truncated at a lower token budget) and backfill "
                         "them into the existing JSON; non-empty items are kept.")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--num_shards", type=int, default=1)
    ap.add_argument("--shard_id", type=int, default=0)
    args = ap.parse_args()

    output_path = args.output_path or os.path.join(
        _PKG_DIR, "..", "predictions", "gpt55-mvbench-adaptive-points")
    output_path = os.path.normpath(output_path)
    os.makedirs(output_path, exist_ok=True)

    if args.task == "all":
        task_names = list(DATA_LIST.keys())
    else:
        keys = [k.strip() for k in args.task.split(",") if k.strip()]
        unknown = [k for k in keys if k not in KEY_TO_TASK]
        if unknown:
            raise SystemExit(f"unknown task(s) {unknown}; choices: {sorted(KEY_TO_TASK)} or 'all'")
        task_names = [KEY_TO_TASK[k] for k in keys]

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
    for task_name in task_names:
        _run_task(sampler, client, args, task_name, output_path)


if __name__ == "__main__":
    main()
