"""Aggregate merged MVBench per-task predictions into the leaderboard table.

Reads predictions/<tag>/<key>.json for every task present and prints per-task
accuracy plus the overall average (mean over items, matching mvbench.ipynb).
"""
import argparse
import glob
import json
import os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output_path", required=True)
    args = ap.parse_args()

    files = sorted(
        p for p in glob.glob(os.path.join(args.output_path, "*.json"))
        if ".shard" not in os.path.basename(p)
        and os.path.basename(p) not in ("summary.json",)
    )
    per_task = {}
    total_c = total_n = 0
    for p in files:
        with open(p) as f:
            data = json.load(f)
        for task, d in data.get("acc", {}).items():
            c = per_task.setdefault(task, {"correct": 0, "total": 0})
            c["correct"] += d["correct"]
            c["total"] += d["total"]

    print(f"{'Task':28s} {'Acc':>7s}  (correct/total)")
    print("-" * 52)
    for task in sorted(per_task):
        d = per_task[task]
        acc = d["correct"] / d["total"] * 100 if d["total"] else 0.0
        total_c += d["correct"]
        total_n += d["total"]
        print(f"{task:28s} {acc:6.2f}%  ({d['correct']}/{d['total']})")
    print("-" * 52)
    avg = total_c / total_n * 100 if total_n else 0.0
    print(f"{'Avg':28s} {avg:6.2f}%  ({total_c}/{total_n})")

    summary = {task: (per_task[task]["correct"] / per_task[task]["total"] * 100
                      if per_task[task]["total"] else 0.0)
               for task in per_task}
    summary["Avg"] = avg
    with open(os.path.join(args.output_path, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
