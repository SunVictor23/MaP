"""TempCompass inference: plain Qwen3-VL with motion-guided adaptive sampling.

Route-2 evaluation harness. For each video we run the sampler ONCE (frames are
identical across a video's questions) and answer every question with plain
Qwen3-VL. Two modes give the fair equal-budget comparison the proposal needs:

    --mode adaptive : motion-guided frame selection (route 2)
    --mode uniform  : evenly spaced frames, SAME frame count (baseline)

Both use the fps=1 frame budget, so the only difference is *which* frames are
shown. Output format matches run_qwen3vl_motiontoken.py so the existing
eval_*.py scorers work unchanged:

    predictions[vid][dim] = [{"question","answer","prediction"}, ...]
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
    MotionAdaptiveQwen3VL,
    build_messages,
    DEFAULT_QWEN_CKPT,
    DEFAULT_COTRACKER_CKPT,
)
from map_kit.core.track_marker import (
    cluster_motion_groups,
    select_motion_points,
    select_object_tracks,
    cluster_object_tracks,
    draw_marks_on_frames,
    legend_text,
)

# Answer-format suffixes, identical to run_qwen3vl_motiontoken.py.
ANSWER_PROMPT = {
    "multi-choice": "\nPlease directly give the best option:",
    "yes_no": "\nPlease answer yes or no:",
    "caption_matching": "\nPlease directly give the best option:",
    "captioning": "",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="adaptive", choices=["adaptive", "uniform"])
    ap.add_argument("--model_path", default=DEFAULT_QWEN_CKPT)
    ap.add_argument("--cotracker_ckpt", default=DEFAULT_COTRACKER_CKPT)
    ap.add_argument("--data_path", default="/path/to/benchmarks/TempCompass")
    ap.add_argument("--questions_path", default="/path/to/TempCompass/questions")
    ap.add_argument("--output_path", default=None,
                    help="default: predictions/qwen3vl-2b-<mode>")
    ap.add_argument("--task_type", default="multi-choice",
                    choices=["multi-choice", "captioning", "caption_matching", "yes_no"])
    ap.add_argument("--tracking_fps", type=float, default=8.0)
    ap.add_argument("--grid_size", type=int, default=30)
    ap.add_argument("--max_track_frames", type=int, default=200)
    ap.add_argument("--online", action="store_true")
    ap.add_argument("--online_checkpoint", default=None)
    ap.add_argument("--segment_track", action="store_true",
                    help="track a fresh grid independently within each sampled "
                         "segment (offline only; mutually exclusive with --online)")
    ap.add_argument("--sampling_rule", default="qwen3vl",
                    choices=["qwen3vl", "fps1"])
    ap.add_argument("--video_fps", type=float, default=None)
    ap.add_argument("--draw_tracks", action="store_true",
                    help="route 1: draw CoTracker3 motion trajectories on the "
                         "sampled frames + add a legend to the prompt")
    ap.add_argument("--mark_mode", default="points",
                    choices=["object", "points", "cluster", "dbscan"])
    ap.add_argument("--track_span", default="frame", choices=["full", "frame"])
    ap.add_argument("--track_color", default="row", choices=["row", "time"])
    ap.add_argument("--track_linewidth", type=int, default=1)
    ap.add_argument("--max_points", type=int, default=40)
    ap.add_argument("--max_objects", type=int, default=8)
    ap.add_argument("--max_groups", type=int, default=3)
    ap.add_argument("--nms_radius_frac", type=float, default=0.12)
    ap.add_argument("--motion_frac", type=float, default=0.8)
    ap.add_argument("--no_camera_comp", action="store_true")
    ap.add_argument("--prompt_hint", action="store_true",
                    help="prepend the non-uniform-sampling hint sentence to the "
                         "question (adaptive mode only; default off)")
    ap.add_argument("--no_flash_attn", action="store_true")
    ap.add_argument("--max_new_tokens", type=int, default=128)
    ap.add_argument("--limit", type=int, default=0, help="debug: cap #videos (0=all)")
    ap.add_argument("--num_shards", type=int, default=1)
    ap.add_argument("--shard_id", type=int, default=0)
    args = ap.parse_args()

    output_path = args.output_path or os.path.join(
        _PKG_DIR, "..", "predictions", f"qwen3vl-2b-{args.mode}"
    )
    output_path = os.path.normpath(output_path)

    question_file = os.path.join(args.questions_path, f"{args.task_type}.json")
    with open(question_file) as f:
        input_datas = json.load(f)

    all_vids = sorted(input_datas.keys())
    my_vids = all_vids[args.shard_id :: args.num_shards]

    os.makedirs(output_path, exist_ok=True)
    if args.num_shards > 1:
        pred_file = os.path.join(output_path, f"{args.task_type}.shard{args.shard_id}.json")
    else:
        pred_file = os.path.join(output_path, f"{args.task_type}.json")
    predictions = {}
    if os.path.isfile(pred_file):
        with open(pred_file) as f:
            predictions = json.load(f)

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

    n_done = 0
    for vid in tqdm(my_vids, desc=f"{args.mode} shard{args.shard_id}"):
        if args.limit and n_done >= args.limit:
            break
        if vid in predictions and predictions[vid]:
            n_done += 1
            continue
        data = input_datas[vid]
        video_path = os.path.join(args.data_path, "videos", f"{vid}.mp4")

        # Sample ONCE per video (frames shared across all its questions).
        try:
            res = engine.sampler.sample(
                video_path, mode=args.mode,
                with_tracks=args.draw_tracks,
                sampling_rule=args.sampling_rule,
                target_fps=args.video_fps,
            )
        except Exception as e:
            print(f"[warn] {vid}: sampling failed: {e}")
            n_done += 1
            continue

        hint = None
        if args.prompt_hint and args.mode == "adaptive":
            hint = res.prompt_hint

        # Route 1: draw motion trajectories on the sampled frames + legend.
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
                elif args.mark_mode == "dbscan":
                    groups = cluster_object_tracks(
                        res.tracks, res.visibility, res.obj_vel, hw=hw,
                        fps=args.tracking_fps, max_objects=args.max_objects)
                else:  # object
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
                        color_mode=args.track_color,
                        linewidth=args.track_linewidth)
                    legend = legend_text(len(groups), span=args.track_span,
                                         number=numbered,
                                         color_mode=args.track_color)
            except Exception as e:
                print(f"[warn] {vid}: marking failed: {e}")

        # route1 legend + route2 hint both go right before the question.
        prefix_txt = "\n".join(x for x in (legend, hint) if x) or None

        predictions[vid] = {}
        for dim, questions in data.items():
            predictions[vid][dim] = []
            for q in questions:
                inp = q["question"] + ANSWER_PROMPT[args.task_type]
                try:
                    pred = _answer(engine, frames, res.timestamps, inp,
                                   prefix_txt, args.max_new_tokens)
                except Exception as e:
                    print(f"[warn] {vid}/{dim}: {e}")
                    pred = None
                predictions[vid][dim].append(
                    {"question": q["question"], "answer": q["answer"], "prediction": pred}
                )
        with open(pred_file, "w") as f:
            json.dump(predictions, f, indent=4, ensure_ascii=False)
        n_done += 1

    print(f"[{args.mode} shard {args.shard_id}] saved predictions to {pred_file}")


def _answer(engine, frames, timestamps, question, hint, max_new_tokens):
    """Answer one question from already-sampled (possibly marked) frames."""
    import torch

    messages = build_messages(
        frames, timestamps, question, prompt_hint=hint
    )
    inputs = engine.processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors="pt",
    ).to(engine.model.device)
    with torch.no_grad():
        out = engine.model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False
        )
    trimmed = out[:, inputs["input_ids"].shape[1]:]
    return engine.processor.batch_decode(
        trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0].strip()


if __name__ == "__main__":
    main()
