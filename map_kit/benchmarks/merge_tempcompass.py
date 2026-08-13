"""Merge per-shard TempCompass prediction files into one <task_type>.json.

Each shard writes predictions/<tag>/<task_type>.shard<N>.json holding a
dict predictions[vid][dim] = [{question, answer, prediction, ...}] over a
disjoint set of vids (stride sharding by video). This unions them back into a
single predictions[vid][dim] dict that the official eval_*.py consumes
(link the merged file into the TempCompass repo's predictions/<tag>/).
"""
import argparse
import glob
import json
import os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output_path", required=True)
    ap.add_argument("--task_type", default="all",
                    help="task_type (multi-choice|yes_no|caption_matching|"
                         "captioning) or 'all'")
    args = ap.parse_args()

    if args.task_type == "all":
        keys = sorted({
            os.path.basename(p).split(".shard")[0]
            for p in glob.glob(os.path.join(args.output_path, "*.shard*.json"))
        })
    else:
        keys = [k.strip() for k in args.task_type.split(",") if k.strip()]

    for key in keys:
        shards = sorted(glob.glob(os.path.join(args.output_path, f"{key}.shard*.json")))
        if not shards:
            print(f"[merge] {key}: no shards found, skip")
            continue
        merged = {}
        for sp in shards:
            with open(sp) as f:
                data = json.load(f)
            for vid, dims in data.items():
                # Deep-merge: a vid's questions may be split across shards
                # (entry-level sharding), so combine at (vid->dim->question)
                # level instead of overwriting the whole vid. Dedup by question.
                vslot = merged.setdefault(vid, {})
                for dim, qs in dims.items():
                    dslot = vslot.setdefault(dim, [])
                    seen = {x["question"] for x in dslot}
                    for x in qs:
                        if x["question"] not in seen:
                            dslot.append(x); seen.add(x["question"])
        if not merged:
            print(f"[merge] {key}: 0 videos (empty shards), skip writing")
            continue
        out = os.path.join(args.output_path, f"{key}.json")
        with open(out, "w") as f:
            json.dump(merged, f, indent=4, ensure_ascii=False)
        n_q = sum(len(qs) for dims in merged.values() for qs in dims.values())
        print(f"[merge] {key}: {len(shards)} shards -> {len(merged)} videos, "
              f"{n_q} questions  -> {out}")


if __name__ == "__main__":
    main()
