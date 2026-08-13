"""MVBench inference: plain Qwen3-VL with motion-guided adaptive sampling.

Route-2 evaluation harness for MVBench (20 temporal tasks). Mirrors
run_tempcompass.py but adapts to MVBench conventions:

    * frame budget = fps=1 frame count of the (bounded) clip for BOTH adaptive
      and uniform, matching the appearance path and TempCompass, so the
      comparison isolates *which* frames. (--num_frames>0 forces a fixed budget.)
    * per-clip temporal bounds (start/end) for the 8 bounded tasks; the fps=1
      budget is computed on the clipped segment.
    * frame-directory input for episodic_reasoning (tvqa frames @ fps3).
    * multiple-choice A/B/C/D questions; accuracy is scored inline by matching
      the predicted option letter (same rule as the reference mvbench.ipynb).

Two modes give the fair equal-budget comparison the proposal needs:

    --mode adaptive : motion-guided frame selection (route 2)
    --mode uniform  : evenly spaced frames, SAME frame count (baseline)

Output per shard/task:
    predictions/<tag>/<task_key>.json = {
        "acc": {task_type: {"correct":int,"total":int,"acc":float}},
        "results": [{"task_type","question","answer","prediction","correct"}],
    }
Merge with merge_mvbench.py, then aggregate accuracy with score_mvbench.py.
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
    draw_marks_on_frames,
    legend_text,
)

# task_type -> (json_file, video_prefix, data_type, has_bound)
# Mirrors the reference mvbench.ipynb data_list, with prefixes relative to
# <mvbench_root>/video and json files under <mvbench_root>/json.
DATA_LIST = {
    "Action Sequence": ("action_sequence.json", "star/Charades_v1_480/", "video", True),
    "Action Prediction": ("action_prediction.json", "star/Charades_v1_480/", "video", True),
    "Action Antonym": ("action_antonym.json", "ssv2_video/", "video", False),
    "Fine-grained Action": ("fine_grained_action.json", "Moments_in_Time_Raw/videos/", "video", False),
    "Unexpected Action": ("unexpected_action.json", "FunQA_test/test/", "video", False),
    "Object Existence": ("object_existence.json", "clevrer/video_validation/", "video", False),
    "Object Interaction": ("object_interaction.json", "star/Charades_v1_480/", "video", True),
    "Object Shuffle": ("object_shuffle.json", "perception/videos/", "video", False),
    "Moving Direction": ("moving_direction.json", "clevrer/video_validation/", "video", False),
    "Action Localization": ("action_localization.json", "sta/sta_video/", "video", True),
    "Scene Transition": ("scene_transition.json", "scene_qa/video/", "video", False),
    "Action Count": ("action_count.json", "perception/videos/", "video", False),
    "Moving Count": ("moving_count.json", "clevrer/video_validation/", "video", False),
    "Moving Attribute": ("moving_attribute.json", "clevrer/video_validation/", "video", False),
    "State Change": ("state_change.json", "perception/videos/", "video", False),
    "Fine-grained Pose": ("fine_grained_pose.json", "nturgbd/", "video", False),
    "Character Order": ("character_order.json", "perception/videos/", "video", False),
    "Egocentric Navigation": ("egocentric_navigation.json", "vlnqa/", "video", False),
    "Episodic Reasoning": ("episodic_reasoning.json", "tvqa/frames_fps3_hq/", "frame", True),
    "Counterfactual Inference": ("counterfactual_inference.json", "clevrer/video_validation/", "video", False),
}

# task_type -> filename-safe key (for per-task output files / --tasks selection).
TASK_KEYS = {name: DATA_LIST[name][0][:-5] for name in DATA_LIST}  # strip ".json"
KEY_TO_TASK = {v: k for k, v in TASK_KEYS.items()}

SYSTEM_PROMPT = (
    "Carefully watch the video and pay attention to the cause and sequence of "
    "events, the detail and movement of objects, and the action and pose of "
    "persons. Based on your observations, select the best option that "
    "accurately addresses the question.\n"
)
QUESTION_SUFFIX = "\nOnly give the best option."


def qa_template(data):
    """Build MC question text and the ground-truth "(X) answer" string."""
    question = f"Question: {data['question']}\n"
    question += "Options:\n"
    answer = data["answer"]
    answer_idx = -1
    for idx, c in enumerate(data["candidates"]):
        question += f"({chr(ord('A') + idx)}) {c}\n"
        if c == answer:
            answer_idx = idx
    question = question.rstrip()
    answer = f"({chr(ord('A') + answer_idx)}) {answer}"
    return question, answer


def check_ans(pred, gt):
    """Match predicted option letter against the ground-truth letter.

    Same rule as reference mvbench.ipynb: compare the leading "(X)" token.
    """
    pred = pred.strip()
    pred_list = pred.lower().split(" ")
    pred_option = pred_list[0] if pred_list else ""
    gt_list = gt.lower().split(" ")
    gt_option = gt_list[0]
    pred_option = pred_option.replace(".", "").replace("(", "").replace(")", "").strip()
    gt_option_clean = gt_option.replace("(", "").replace(")", "").strip()
    if not pred_option:
        return False
    if pred_option == gt_option_clean:
        return True
    # tolerate "answer: (A)" / "the answer is A" style leading text
    if gt_option_clean and gt_option_clean in pred.lower().replace("(", "").replace(")", "").split(" "):
        return True
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="adaptive", choices=["adaptive", "uniform"])
    ap.add_argument("--model_path", default=DEFAULT_QWEN_CKPT)
    ap.add_argument("--cotracker_ckpt", default=DEFAULT_COTRACKER_CKPT)
    ap.add_argument("--mvbench_root", default="/path/to/benchmarks/MVBench")
    ap.add_argument("--output_path", default=None,
                    help="default: predictions/qwen3vl-2b-mvbench-<mode>")
    ap.add_argument("--task", default="all",
                    help="task key, comma-separated keys "
                         "(e.g. action_sequence,moving_direction), or 'all'")
    ap.add_argument("--num_frames", type=int, default=0,
                    help="frame budget for BOTH modes. 0 (default) = frame "
                         "count from --sampling_rule, matching the appearance "
                         "path. Set >0 only to force a fixed budget.")
    ap.add_argument("--sampling_rule", default="qwen3vl",
                    choices=["qwen3vl", "fps1"],
                    help="uniform budget + frame positions when --num_frames=0: "
                         "'qwen3vl' (default) = Qwen3-VL native MECHANISM "
                         "(int(total/fps*video_fps), clamp[4,768], linspace over "
                         "whole clip); 'fps1' = legacy round(total/fps) "
                         "sampling. Timestamps are real frame-seconds either way.")
    ap.add_argument("--video_fps", type=float, default=None,
                    help="fps VALUE fed into --sampling_rule (a tunable knob, "
                         "NOT part of the mechanism). Default None = 1.0.")
    ap.add_argument("--tracking_fps", type=float, default=8.0)
    ap.add_argument("--grid_size", type=int, default=20)
    ap.add_argument("--max_track_frames", type=int, default=200)
    ap.add_argument("--online", action="store_true",
                    help="use CoTracker3 streaming (online) predictor: processes "
                         "the clip in overlapping windows so only ~window_len "
                         "frames sit on the GPU at once (lower peak memory). "
                         "Loads scaled_online.pth. Default off = offline.")
    ap.add_argument("--online_checkpoint", default=None,
                    help="path to scaled_online.pth (defaults to sibling of "
                         "--cotracker_ckpt)")
    ap.add_argument("--segment_track", action="store_true",
                    help="track a fresh grid INDEPENDENTLY within each "
                         "~reseed second segment (no cross-segment continuity). "
                         "Removes long-track drift and re-covers new objects "
                         "every segment; inherently low-memory. Offline only "
                         "(mutually exclusive with --online).")
    ap.add_argument("--no_camera_comp", action="store_true")
    ap.add_argument("--prompt_hint", action="store_true")
    ap.add_argument("--draw_tracks", action="store_true",
                    help="route 1: draw CoTracker3 motion trajectories on the "
                         "selected frames + add a legend to the prompt "
                         "(adaptive mode only)")
    ap.add_argument("--mark_mode", default="object",
                    choices=["object", "points", "cluster"],
                    help="route 1 point selection: 'object' = one representative "
                         "trajectory per object via spatial NMS (default, small "
                         "objects survive); 'points' = each high-motion point's "
                         "own path; 'cluster' = KMeans numbered groups (legacy)")
    ap.add_argument("--max_groups", type=int, default=3,
                    help="route 1: max numbered motion groups (mark_mode=cluster)")
    ap.add_argument("--max_points", type=int, default=40,
                    help="route 1: cap on paths for mark_mode=points")
    ap.add_argument("--max_objects", type=int, default=8,
                    help="route 1: cap on objects for mark_mode=object (NMS)")
    ap.add_argument("--nms_radius_frac", type=float, default=0.12,
                    help="route 1: object-NMS suppression radius / frame diagonal")
    ap.add_argument("--motion_frac", type=float, default=0.8,
                    help="route 1: quantile of point-motion to drop as static "
                         "(0.8 keeps top ~20%% moving points)")
    ap.add_argument("--track_span", default="full", choices=["full", "frame"],
                    help="route 1 trajectory span: 'full' draws the whole path on "
                         "every frame; 'frame' draws only the segment between the "
                         "previous and current selected frame (frame-to-frame)")
    ap.add_argument("--no_flash_attn", action="store_true")
    ap.add_argument("--max_new_tokens", type=int, default=100)
    ap.add_argument("--limit", type=int, default=0, help="debug: cap #items per task (0=all)")
    ap.add_argument("--num_shards", type=int, default=1)
    ap.add_argument("--shard_id", type=int, default=0)
    args = ap.parse_args()

    # route1 marking works in both modes now: adaptive already has tracks;
    # uniform runs the tracker on demand (with_tracks) while keeping uniform
    # frame selection — this is the "fps=1 uniform + trajectory marks" setup.
    tag_suffix = ""
    if args.draw_tracks:
        tag_suffix = f"-marks-{args.mark_mode}"
        if args.track_span == "frame":
            tag_suffix += "-f2f"
    output_path = args.output_path or os.path.join(
        _PKG_DIR, "..", "predictions", f"qwen3vl-2b-mvbench-{args.mode}{tag_suffix}"
    )
    output_path = os.path.normpath(output_path)
    os.makedirs(output_path, exist_ok=True)

    if args.task == "all":
        task_names = list(DATA_LIST.keys())
    else:
        # comma-separated list of task keys, e.g. "action_sequence,moving_direction"
        keys = [k.strip() for k in args.task.split(",") if k.strip()]
        unknown = [k for k in keys if k not in KEY_TO_TASK]
        if unknown:
            raise SystemExit(
                f"unknown task(s) {unknown}; choices: {sorted(KEY_TO_TASK)} or 'all'"
            )
        task_names = [KEY_TO_TASK[k] for k in keys]

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

    for task_name in task_names:
        _run_task(engine, args, task_name, output_path)


def _run_task(engine, args, task_name, output_path):
    json_file, prefix, data_type, has_bound = DATA_LIST[task_name]
    key = TASK_KEYS[task_name]

    qfile = os.path.join(args.mvbench_root, "json", json_file)
    if not os.path.isfile(qfile):
        print(f"[skip] {task_name}: missing json {qfile}")
        return
    with open(qfile) as f:
        items = json.load(f)

    # Deterministic stride sharding over this task's items.
    my_items = list(enumerate(items))[args.shard_id :: args.num_shards]

    if args.num_shards > 1:
        pred_file = os.path.join(output_path, f"{key}.shard{args.shard_id}.json")
    else:
        pred_file = os.path.join(output_path, f"{key}.json")

    done_ids = set()
    results = []
    if os.path.isfile(pred_file):
        with open(pred_file) as f:
            prev = json.load(f)
        results = prev.get("results", [])
        done_ids = {r["item_id"] for r in results if "item_id" in r}

    video_root = os.path.join(args.mvbench_root, "video", prefix)
    n_new = 0
    for item_id, data in tqdm(my_items, desc=f"{args.mode}:{key} shard{args.shard_id}"):
        if args.limit and n_new >= args.limit:
            break
        if item_id in done_ids:
            continue
        video_path = os.path.join(video_root, data["video"])
        exists = os.path.isdir(video_path) if data_type == "frame" else os.path.isfile(video_path)
        if not exists:
            print(f"[skip] {task_name}: missing media {video_path}")
            continue

        bound = (data["start"], data["end"]) if has_bound else None
        question, gt = qa_template(data)

        # budget None -> fps=1 frame count of the (bounded) clip, so adaptive and
        # uniform share the same equal-budget as the appearance path. Only a
        # positive --num_frames forces a fixed budget.
        budget = args.num_frames if args.num_frames > 0 else None
        try:
            res = engine.sampler.sample(
                video_path,
                budget=budget,
                mode=args.mode,
                bound=bound,
                data_type=data_type,
                with_tracks=args.draw_tracks,  # uniform needs tracks to mark
                sampling_rule=args.sampling_rule,
                target_fps=args.video_fps,
            )
        except Exception as e:
            print(f"[warn] {task_name}/{data['video']}: sampling failed: {e}")
            continue

        hint = res.prompt_hint if (args.prompt_hint and args.mode == "adaptive") else None

        # Route 1: draw motion trajectories on the selected frames + legend.
        frames = res.frames
        legend = None
        if args.draw_tracks and res.tracks is not None:
            try:
                hw = res.frames.shape[1:3]
                if args.mark_mode == "cluster":
                    groups = cluster_motion_groups(
                        res.tracks, res.visibility, res.obj_vel, hw=hw,
                        max_groups=args.max_groups, motion_frac=args.motion_frac,
                    )
                elif args.mark_mode == "points":
                    groups = select_motion_points(
                        res.tracks, res.visibility, res.obj_vel, hw=hw,
                        fps=args.tracking_fps, max_points=args.max_points,
                    )
                else:  # object (default): one representative track per object
                    groups = select_object_tracks(
                        res.tracks, res.visibility, res.obj_vel, hw=hw,
                        motion_frac=args.motion_frac,
                        nms_radius_frac=args.nms_radius_frac,
                        max_objects=args.max_objects,
                    )
                numbered = args.mark_mode == "cluster"
                if groups:
                    frames = draw_marks_on_frames(
                        res.frames, res.indices, groups, res.timestamps,
                        span=args.track_span, number=numbered,
                        color_mode="row",
                    )
                    legend = legend_text(len(groups), span=args.track_span,
                                         number=numbered, color_mode="row")
            except Exception as e:
                print(f"[warn] {task_name}/{data['video']}: marking failed: {e}")

        inp = SYSTEM_PROMPT + question + QUESTION_SUFFIX
        # route1 legend + route2 hint both go right before the question (after
        # the frames). This placement scored best in prompt-position tests.
        prefix_txt = "\n".join(x for x in (legend, hint) if x)
        try:
            pred = _answer(engine, frames, res.timestamps, inp,
                           None, prefix_txt or None, args.max_new_tokens)
        except Exception as e:
            print(f"[warn] {task_name}/{data['video']}: infer failed: {e}")
            continue

        correct = check_ans(pred, gt)
        results.append({
            "item_id": item_id,
            "task_type": task_name,
            "question": question,
            "answer": gt,
            "prediction": pred,
            "correct": bool(correct),
        })
        n_new += 1
        if n_new % 20 == 0:
            _dump(pred_file, results)

    _dump(pred_file, results)
    acc = _acc_from_results(results)
    a = acc.get(task_name, {})
    print(f"[{args.mode} shard{args.shard_id}] {task_name}: "
          f"{a.get('correct',0)}/{a.get('total',0)} = {a.get('acc',0):.2f}%  -> {pred_file}")


def _acc_from_results(results):
    acc = {}
    for r in results:
        t = r["task_type"]
        d = acc.setdefault(t, {"correct": 0, "total": 0})
        d["total"] += 1
        d["correct"] += int(r["correct"])
    for t, d in acc.items():
        d["acc"] = d["correct"] / d["total"] * 100 if d["total"] else 0.0
    return acc


def _dump(pred_file, results):
    with open(pred_file, "w") as f:
        json.dump({"acc": _acc_from_results(results), "results": results},
                  f, indent=2, ensure_ascii=False)


def _answer(engine, frames, timestamps, question, preamble, hint, max_new_tokens):
    import torch

    messages = build_messages(
        frames, timestamps, question,
        system_prompt=None, prompt_hint=hint, preamble=preamble,
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
    text = engine.processor.batch_decode(
        trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0].strip()
    return text.split("\n")[0].strip()


if __name__ == "__main__":
    main()
