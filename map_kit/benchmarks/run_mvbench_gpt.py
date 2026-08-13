"""MVBench inference on GPT-5.5 (remote) with uniform frame sampling, no tracks.

Baseline for evaluating our frame-sampling methods on a stronger closed model.
Uniformly samples frames (Qwen3-VL-native rule, budget = fps=1 frame count of
the bounded clip) and sends them as base64 images to the GPT vision endpoint.
No CoTracker, no GPU — the work is HTTP-bound, so items run in a thread pool.

Reuses run_mvbench's task list / MC templating / option-letter scoring, and
video_io for reading + uniform frame selection (matching the local uniform
baseline exactly, so the only variable is the model).

Output/scoring identical to run_mvbench (merge_mvbench + score_mvbench).
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
    read_video, read_frames_dir, qwen3vl_budget, qwen3vl_frame_indices,
    fps1_budget,
)
from map_kit.benchmarks.run_mvbench import (
    DATA_LIST, TASK_KEYS, KEY_TO_TASK, SYSTEM_PROMPT, QUESTION_SUFFIX,
    qa_template, check_ans, _acc_from_results, _dump,
)

DEFAULT_API_KEY = os.environ.get("MAP_API_KEY", "")  # set via env var; never hardcode


def _uniform_frames(video_path, data_type, bound, key_fps, rule, max_frames):
    """Read + uniformly sample frames (Qwen3-VL native rule). Returns
    (frames list, timestamps) at original resolution."""
    if data_type == "frame":
        raw = read_frames_dir(video_path, tracking_fps=None, bound=bound,
                              max_frames=max_frames)
    else:
        raw = read_video(video_path, tracking_fps=None, bound=bound,
                         max_frames=max_frames)
    T = int(raw.frames.shape[0])
    if rule == "qwen3vl":
        n = qwen3vl_budget(T, raw.fps, target_fps=key_fps)
    else:
        n = fps1_budget(T, raw.fps, target_fps=key_fps)
    idx = qwen3vl_frame_indices(T, n)
    return [raw.frames[i] for i in idx], raw.timestamps[idx]


def _run_task(client, args, task_name, output_path):
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

    def work(item_id, data):
        video_path = os.path.join(video_root, data["video"])
        exists = os.path.isdir(video_path) if data_type == "frame" else os.path.isfile(video_path)
        if not exists:
            return None
        bound = (data["start"], data["end"]) if has_bound else None
        question, gt = qa_template(data)
        try:
            frames, ts = _uniform_frames(video_path, data_type, bound,
                                        args.key_fps, args.sampling_rule,
                                        args.max_frames)
        except Exception as e:
            print(f"[warn] {task_name}/{data['video']}: sampling failed: {e}")
            return None
        content = build_vision_content(
            frames, timestamps=(None if args.no_timestamps else ts),
            question=question + QUESTION_SUFFIX,
            jpeg_quality=args.jpeg_quality)
        pred = client.chat(content, system=SYSTEM_PROMPT,
                           max_tokens=args.max_new_tokens)
        correct = check_ans(pred, gt)
        return {"item_id": item_id, "task_type": task_name, "question": question,
                "answer": gt, "prediction": pred, "correct": bool(correct)}

    n_since_dump = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(work, i, d): i for i, d in todo}
        for fut in tqdm(as_completed(futs), total=len(futs),
                        desc=f"gpt:{key} shard{args.shard_id}"):
            r = fut.result()
            if r is None:
                continue
            with lock:
                results.append(r)
                n_since_dump += 1
                if n_since_dump >= 10:
                    _dump(pred_file, results)
                    n_since_dump = 0

    _dump(pred_file, results)
    acc = _acc_from_results(results)
    a = acc.get(task_name, {})
    print(f"[gpt shard{args.shard_id}] {task_name}: "
          f"{a.get('correct',0)}/{a.get('total',0)} = {a.get('acc',0):.2f}%  -> {pred_file}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api_key", default=DEFAULT_API_KEY)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--mvbench_root", default="/path/to/benchmarks/MVBench")
    ap.add_argument("--output_path", default=None,
                    help="default: predictions/gpt55-mvbench-uniform")
    ap.add_argument("--task", default="all",
                    help="task key, comma-separated keys, or 'all'")
    ap.add_argument("--key_fps", type=float, default=1.0)
    ap.add_argument("--sampling_rule", default="qwen3vl", choices=["qwen3vl", "fps1"])
    ap.add_argument("--max_frames", type=int, default=None)
    ap.add_argument("--jpeg_quality", type=int, default=90)
    ap.add_argument("--no_timestamps", action="store_true",
                    help="omit the <t.t seconds> text token before each frame")
    ap.add_argument("--workers", type=int, default=8,
                    help="concurrent HTTP requests (thread pool)")
    ap.add_argument("--max_new_tokens", type=int, default=2048,
                    help="GPT-5.5 is a reasoning model: it spends tokens on "
                         "hidden reasoning before the answer, so this must be "
                         "large (finish_reason=length -> empty visible content)")
    ap.add_argument("--refill_empty", action="store_true",
                    help="only re-run items whose stored prediction is empty "
                         "(e.g. truncated at a lower token budget) and backfill "
                         "them into the existing JSON; non-empty items are kept.")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--num_shards", type=int, default=1)
    ap.add_argument("--shard_id", type=int, default=0)
    args = ap.parse_args()

    output_path = args.output_path or os.path.join(
        _PKG_DIR, "..", "predictions", "gpt55-mvbench-uniform")
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

    client = GPTClient(api_key=args.api_key, model=args.model)
    for task_name in task_names:
        _run_task(client, args, task_name, output_path)


if __name__ == "__main__":
    main()
