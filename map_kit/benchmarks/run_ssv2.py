"""SSv2 inference: Qwen3-VL with motion-guided adaptive sampling (+ route-1 marks).

The project's own method on Something-Something-v2 (4-way MC subset, see
ssv2_data.py), mirroring run_mvbench.py's adaptive path but reading the
SSv2 subset json + .webm videos. Same equal-budget comparison:

    --mode adaptive : motion-guided frame selection (route 2)
    --mode uniform  : evenly spaced frames, SAME frame count (baseline)
    --draw_tracks   : overlay CoTracker3 trajectories (route 1) on the frames

Budget = fps=1 frame count (budget=None), matching the baseline runners so the
comparison isolates which frames / what marks.

Output: predictions/qwen3vl-2b-ssv2-<mode><-marks-...>/action_recognition[.shardN].json
scored inline by option letter (reuse check_ans); merge/score via
merge_mvbench.py / score_mvbench.py.
"""
import argparse
import json
import os
import sys

from tqdm import tqdm

_PKG_DIR = os.path.dirname(os.path.abspath(__file__))
_MOTIONVLM_ROOT = os.path.normpath(os.path.join(_PKG_DIR, ".."))
if _MOTIONVLM_ROOT not in sys.path:
    sys.path.insert(0, _MOTIONVLM_ROOT)

from map_kit.models.qwen_infer import (
    MotionAdaptiveQwen3VL, build_messages,
    DEFAULT_QWEN_CKPT, DEFAULT_COTRACKER_CKPT,
)
from map_kit.core.track_marker import (
    cluster_motion_groups, select_motion_points, select_object_tracks,
    draw_marks_on_frames, legend_text,
)
from map_kit.benchmarks.run_mvbench import (
    SYSTEM_PROMPT, QUESTION_SUFFIX, check_ans, _acc_from_results, _answer,
)
from map_kit.data.ssv2_data import (
    TASK_KEY, DEFAULT_SUBSET, DEFAULT_VIDEO_ROOT, ssv2_qa_template,
)


def _dump(pred_file, results):
    with open(pred_file, "w") as f:
        json.dump({"acc": _acc_from_results(results), "results": results},
                  f, indent=2, ensure_ascii=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="adaptive", choices=["adaptive", "uniform"])
    ap.add_argument("--model_path", default=DEFAULT_QWEN_CKPT)
    ap.add_argument("--cotracker_ckpt", default=DEFAULT_COTRACKER_CKPT)
    ap.add_argument("--subset", default=DEFAULT_SUBSET)
    ap.add_argument("--video_root", default=DEFAULT_VIDEO_ROOT)
    ap.add_argument("--output_path", default=None)
    ap.add_argument("--num_frames", type=int, default=0,
                    help="0 = fps=1 budget (equal to baselines)")
    ap.add_argument("--sampling_rule", default="qwen3vl")
    ap.add_argument("--video_fps", type=float, default=None)
    ap.add_argument("--tracking_fps", type=float, default=8.0)
    ap.add_argument("--grid_size", type=int, default=20)
    ap.add_argument("--max_track_frames", type=int, default=200)
    ap.add_argument("--online", action="store_true")
    ap.add_argument("--online_checkpoint", default=None)
    ap.add_argument("--segment_track", action="store_true")
    ap.add_argument("--no_camera_comp", action="store_true")
    ap.add_argument("--prompt_hint", action="store_true")
    ap.add_argument("--draw_tracks", action="store_true")
    ap.add_argument("--mark_mode", default="object",
                    choices=["object", "points", "cluster"])
    ap.add_argument("--max_groups", type=int, default=3)
    ap.add_argument("--max_points", type=int, default=40)
    ap.add_argument("--max_objects", type=int, default=8)
    ap.add_argument("--nms_radius_frac", type=float, default=0.12)
    ap.add_argument("--motion_frac", type=float, default=0.8)
    ap.add_argument("--track_span", default="full", choices=["full", "frame"])
    ap.add_argument("--no_flash_attn", action="store_true")
    ap.add_argument("--max_new_tokens", type=int, default=8192)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--num_shards", type=int, default=1)
    ap.add_argument("--shard_id", type=int, default=0)
    args = ap.parse_args()

    tag_suffix = ""
    if args.draw_tracks:
        tag_suffix = f"-marks-{args.mark_mode}"
        if args.track_span == "frame":
            tag_suffix += "-f2f"
    output_path = args.output_path or os.path.join(
        _MOTIONVLM_ROOT, "predictions", f"qwen3vl-2b-ssv2-{args.mode}{tag_suffix}")
    output_path = os.path.normpath(output_path)
    os.makedirs(output_path, exist_ok=True)

    print(f"[{args.mode}] loading Qwen3-VL from {args.model_path}")
    engine = MotionAdaptiveQwen3VL(
        qwen_ckpt=args.model_path,
        cotracker_ckpt=args.cotracker_ckpt,
        tracking_fps=args.tracking_fps,
        grid_size=args.grid_size,
        max_track_frames=args.max_track_frames,
        compensate_camera=not args.no_camera_comp,
        attn_implementation=None if args.no_flash_attn else "flash_attention_2",
        segment_track=args.segment_track,
        online=args.online,
        online_checkpoint=args.online_checkpoint,
    )

    with open(args.subset) as f:
        items = json.load(f)
    my_items = list(enumerate(items))[args.shard_id::args.num_shards]

    pred_file = os.path.join(
        output_path,
        f"{TASK_KEY}.shard{args.shard_id}.json" if args.num_shards > 1
        else f"{TASK_KEY}.json")

    done_ids, results = set(), []
    if os.path.isfile(pred_file):
        with open(pred_file) as f:
            results = json.load(f).get("results", [])
        done_ids = {r["item_id"] for r in results if "item_id" in r}

    n_new = 0
    for item_id, data in tqdm(my_items, desc=f"{args.mode}:ssv2 s{args.shard_id}"):
        if args.limit and n_new >= args.limit:
            break
        if item_id in done_ids:
            continue
        video_path = os.path.join(args.video_root, f"{data['video_id']}.webm")
        if not os.path.isfile(video_path):
            print(f"[skip] missing media {video_path}")
            continue

        question, gt = ssv2_qa_template(data)
        budget = args.num_frames if args.num_frames > 0 else None
        try:
            res = engine.sampler.sample(
                video_path, budget=budget, mode=args.mode, bound=None,
                data_type="video", with_tracks=args.draw_tracks,
                sampling_rule=args.sampling_rule, target_fps=args.video_fps)
        except Exception as e:
            print(f"[warn] {data['video_id']}: sampling failed: {e}")
            continue

        hint = res.prompt_hint if (args.prompt_hint and args.mode == "adaptive") else None
        frames = res.frames
        legend = None
        if args.draw_tracks and res.tracks is not None:
            try:
                hw = res.frames.shape[1:3]
                if args.mark_mode == "cluster":
                    groups = cluster_motion_groups(
                        res.tracks, res.visibility, res.obj_vel, hw=hw,
                        max_groups=args.max_groups, motion_frac=args.motion_frac)
                elif args.mark_mode == "points":
                    groups = select_motion_points(
                        res.tracks, res.visibility, res.obj_vel, hw=hw,
                        fps=args.tracking_fps, max_points=args.max_points)
                else:
                    groups = select_object_tracks(
                        res.tracks, res.visibility, res.obj_vel, hw=hw,
                        motion_frac=args.motion_frac,
                        nms_radius_frac=args.nms_radius_frac,
                        max_objects=args.max_objects)
                numbered = args.mark_mode == "cluster"
                if groups:
                    frames = draw_marks_on_frames(
                        res.frames, res.indices, groups, res.timestamps,
                        span=args.track_span, number=numbered,
                        color_mode="row")
                    legend = legend_text(len(groups), span=args.track_span,
                                         number=numbered, color_mode="row")
            except Exception as e:
                print(f"[warn] {data['video_id']}: marking failed: {e}")

        inp = SYSTEM_PROMPT + question + QUESTION_SUFFIX
        prefix_txt = "\n".join(x for x in (legend, hint) if x)
        try:
            pred = _answer(engine, frames, res.timestamps, inp,
                           None, prefix_txt or None, args.max_new_tokens)
        except Exception as e:
            print(f"[warn] {data['video_id']}: infer failed: {e}")
            continue

        results.append({
            "item_id": item_id, "task_type": TASK_KEY,
            "video_id": data["video_id"],
            "question": question, "answer": gt, "prediction": pred,
            "correct": bool(check_ans(pred, gt)),
        })
        n_new += 1
        if n_new % 20 == 0:
            _dump(pred_file, results)

    _dump(pred_file, results)
    a = _acc_from_results(results).get(TASK_KEY, {})
    print(f"[{args.mode} s{args.shard_id}] ssv2: {a.get('correct',0)}/"
          f"{a.get('total',0)} = {a.get('acc',0):.2f}%  -> {pred_file}")


if __name__ == "__main__":
    main()
