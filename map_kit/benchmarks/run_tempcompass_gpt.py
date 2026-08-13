"""TempCompass inference on GPT-5.5 (remote) with uniform frame sampling, no tracks.

Baseline for the stronger closed model on TempCompass. Uniformly samples frames
(Qwen3-VL-native rule, budget = fps=1 frame count) and sends them as base64
images to the GPT vision endpoint. No CoTracker, no GPU — HTTP-bound, so videos
run in a thread pool. Sample ONCE per video (frames shared across its questions).

Output format matches run_tempcompass.py so the stock eval_*.py --disable_llm
scorers work unchanged:
    predictions[vid][dim] = [{"question","answer","prediction"}, ...]
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

from map_kit.models.gpt_client import GPTClient, build_vision_content, DEFAULT_MODEL
from map_kit.data.video_io import (
    read_video, qwen3vl_budget, qwen3vl_frame_indices, fps1_budget,
)

DEFAULT_API_KEY = os.environ.get("MAP_API_KEY", "")  # set via env var; never hardcode
DEFAULT_COTRACKER_CKPT = "/path/to/basemodels/scaled_offline.pth"

ANSWER_PROMPT = {
    "multi-choice": "\nPlease directly give the best option:",
    "yes_no": "\nPlease answer yes or no:",
    "caption_matching": "\nPlease directly give the best option:",
    "captioning": "",
}


def _uniform_frames(video_path, key_fps, rule, max_frames):
    raw = read_video(video_path, tracking_fps=None, max_frames=max_frames)
    T = int(raw.frames.shape[0])
    n = qwen3vl_budget(T, raw.fps, target_fps=key_fps) if rule == "qwen3vl" \
        else fps1_budget(T, raw.fps, target_fps=key_fps)
    idx = qwen3vl_frame_indices(T, n)
    return [raw.frames[i] for i in idx], raw.timestamps[idx]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api_key", default=DEFAULT_API_KEY)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--data_path", default="/path/to/benchmarks/TempCompass")
    ap.add_argument("--questions_path", default="/path/to/TempCompass/questions")
    ap.add_argument("--output_path", default=None,
                    help="default: predictions/gpt55-tc-uniform")
    ap.add_argument("--task_type", default="multi-choice",
                    choices=["multi-choice", "captioning", "caption_matching", "yes_no"])
    ap.add_argument("--mode", default="uniform", choices=["uniform", "adaptive"],
                    help="uniform (HTTP-only) or adaptive M(t) selection (needs "
                         "local GPU CoTracker; no tracks drawn)")
    ap.add_argument("--cotracker_ckpt", default=DEFAULT_COTRACKER_CKPT)
    ap.add_argument("--tracking_fps", type=float, default=8.0)
    ap.add_argument("--grid_size", type=int, default=10)
    ap.add_argument("--segment_track", action="store_true",
                    help="frame-aligned segment tracking (offline, OOM-safe)")
    ap.add_argument("--key_fps", type=float, default=1.0)
    ap.add_argument("--sampling_rule", default="qwen3vl", choices=["qwen3vl", "fps1"])
    ap.add_argument("--max_frames", type=int, default=None)
    ap.add_argument("--no_timestamps", action="store_true")
    ap.add_argument("--jpeg_quality", type=int, default=90)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--max_new_tokens", type=int, default=2048,
                    help="GPT-5.5 is a reasoning model: needs a large budget or "
                         "it truncates before emitting the answer")
    ap.add_argument("--refill_empty", action="store_true",
                    help="only re-run videos with any empty prediction; backfill")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--num_shards", type=int, default=1)
    ap.add_argument("--shard_id", type=int, default=0)
    args = ap.parse_args()

    output_path = args.output_path or os.path.join(
        _PKG_DIR, "..", "predictions", "gpt55-tc-uniform")
    output_path = os.path.normpath(output_path)
    os.makedirs(output_path, exist_ok=True)

    with open(os.path.join(args.questions_path, f"{args.task_type}.json")) as f:
        input_datas = json.load(f)
    all_vids = sorted(input_datas.keys())
    my_vids = all_vids[args.shard_id :: args.num_shards]

    if args.num_shards > 1:
        pred_file = os.path.join(output_path, f"{args.task_type}.shard{args.shard_id}.json")
    else:
        pred_file = os.path.join(output_path, f"{args.task_type}.json")
    predictions = {}
    if os.path.isfile(pred_file):
        with open(pred_file) as f:
            predictions = json.load(f)

    def has_empty(vid):
        for dim in predictions.get(vid, {}).values():
            for q in dim:
                if not str(q.get("prediction", "")).strip():
                    return True
        return False

    def is_done(vid):
        if vid not in predictions or not predictions[vid]:
            return False
        if args.refill_empty and has_empty(vid):
            return False
        return True

    todo = [v for v in my_vids if not is_done(v)]
    if args.limit:
        todo = todo[:args.limit]

    client = GPTClient(api_key=args.api_key, model=args.model)
    lock = threading.Lock()

    sampler = None
    if args.mode == "adaptive":
        from map_kit import MotionAdaptiveSampler
        sampler = MotionAdaptiveSampler(
            cotracker_ckpt=args.cotracker_ckpt, tracking_fps=args.tracking_fps,
            grid_size=args.grid_size, segment_track=args.segment_track)

    def get_frames(vid):
        """Return (frames_list, timestamps) for a video, per --mode."""
        video_path = os.path.join(args.data_path, "videos", f"{vid}.mp4")
        if args.mode == "adaptive":
            res = sampler.sample(video_path, budget=None, mode="adaptive",
                                 data_type="video", with_tracks=False,
                                 sampling_rule=args.sampling_rule,
                                 target_fps=args.key_fps if args.sampling_rule == "qwen3vl" else None)
            return list(res.frames), res.timestamps
        return _uniform_frames(video_path, args.key_fps, args.sampling_rule,
                              args.max_frames)

    def gpt_answer(vid, frames, ts):
        data = input_datas[vid]
        out = {}
        for dim, questions in data.items():
            out[dim] = []
            for q in questions:
                inp = q["question"] + ANSWER_PROMPT[args.task_type]
                content = build_vision_content(
                    frames, timestamps=(None if args.no_timestamps else ts),
                    question=inp, jpeg_quality=args.jpeg_quality)
                pred = client.chat(content, max_tokens=args.max_new_tokens)
                out[dim].append({"question": q["question"],
                                "answer": q["answer"], "prediction": pred})
        return vid, out

    def save():
        with open(pred_file, "w") as f:
            json.dump(predictions, f, indent=4, ensure_ascii=False)

    n_since = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        if args.mode == "adaptive":
            # GPU sampling sequential; GPT calls dispatched to the pool.
            futs = []
            for vid in tqdm(todo, desc=f"gpt-tc:{args.task_type} shard{args.shard_id} (sample)"):
                try:
                    frames, ts = get_frames(vid)
                except Exception as e:
                    print(f"[warn] {vid}: sampling failed: {e}")
                    continue
                futs.append(ex.submit(gpt_answer, vid, frames, ts))
            it = as_completed(futs)
        else:
            futs = {}
            for vid in todo:
                def _mk(v):
                    def run():
                        try:
                            fr, ts = get_frames(v)
                        except Exception as e:
                            print(f"[warn] {v}: sampling failed: {e}")
                            return v, None
                        return gpt_answer(v, fr, ts)
                    return run
                futs[ex.submit(_mk(vid))] = vid
            it = as_completed(futs)

        for fut in tqdm(it, total=len(futs),
                        desc=f"gpt-tc:{args.task_type} shard{args.shard_id} (gpt)"):
            vid, out = fut.result()
            if out is None:
                continue
            with lock:
                predictions[vid] = out
                n_since += 1
                if n_since >= 10:
                    save(); n_since = 0

    save()
    print(f"[gpt-tc {args.task_type} shard{args.shard_id}] saved -> {pred_file}")


if __name__ == "__main__":
    main()
