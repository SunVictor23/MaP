"""Merge per-shard MVBench prediction files into one <key>.json per task.

Each shard writes predictions/<tag>/<key>.shard<N>.json with a "results" list
(disjoint item_ids via stride sharding). This stitches them back together and
recomputes per-task accuracy.
"""
import argparse
import glob
import json
import os


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output_path", required=True)
    ap.add_argument("--task", default="all", help="task key or 'all'")
    args = ap.parse_args()

    if args.task == "all":
        keys = sorted({
            os.path.basename(p).split(".shard")[0]
            for p in glob.glob(os.path.join(args.output_path, "*.shard*.json"))
        })
    else:
        keys = [k.strip() for k in args.task.split(",") if k.strip()]

    for key in keys:
        shards = sorted(glob.glob(os.path.join(args.output_path, f"{key}.shard*.json")))
        if not shards:
            print(f"[merge] {key}: no shards found, skip")
            continue
        merged = {}
        for sp in shards:
            with open(sp) as f:
                for r in json.load(f).get("results", []):
                    merged[r["item_id"]] = r  # dedup by item_id
        results = [merged[i] for i in sorted(merged)]
        if not results:
            print(f"[merge] {key}: 0 items (empty shards), skip writing")
            continue
        out = os.path.join(args.output_path, f"{key}.json")
        with open(out, "w") as f:
            json.dump({"acc": _acc_from_results(results), "results": results},
                      f, indent=2, ensure_ascii=False)
        a = _acc_from_results(results)
        tot = sum(d["total"] for d in a.values())
        cor = sum(d["correct"] for d in a.values())
        pct = cor / tot * 100 if tot else 0.0
        print(f"[merge] {key}: {len(shards)} shards -> {len(results)} items "
              f"({cor}/{tot} = {pct:.2f}%)  -> {out}")


if __name__ == "__main__":
    main()
