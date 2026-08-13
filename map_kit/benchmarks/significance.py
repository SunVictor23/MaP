"""Paired significance testing for two MVBench prediction runs.

Both runs are evaluated on the SAME item_ids (identical videos/questions), so
per-item correctness is paired: for each item we get (a_i, u_i) in {0,1} for
system-A (e.g. SOTA) and system-B (e.g. baseline). Comparing two independent
accuracies throws away this pairing; McNemar + paired bootstrap use it.

For each task and for the pooled set we report:
  * the 2x2 discordance counts b (A-right/B-wrong) and c (A-wrong/B-right)
  * McNemar's EXACT test p-value (two-sided binomial on the discordant pairs)
  * the accuracy delta Δ = acc_A - acc_B with a paired-bootstrap 95% CI
McNemar answers "is the difference in this direction significant"; the bootstrap
CI answers "how big is Δ, and does its 95% interval exclude 0".

Usage:
  python -m map_kit.significance \
      --sys_a predictions/<sota_tag> --sys_b predictions/<baseline_tag> \
      --task object_existence,moving_direction,... [--boot 10000] [--seed 0]
"""
import argparse
import json
import os
from math import comb

import numpy as np

CLEVRER = ("object_existence,moving_direction,moving_count,"
           "moving_attribute,counterfactual_inference")


def _load(tag_dir, task):
    """item_id -> bool(correct) for one task's prediction json."""
    f = os.path.join(tag_dir, f"{task}.json")
    if not os.path.isfile(f):
        return None
    res = json.load(open(f)).get("results", [])
    return {r["item_id"]: bool(r["correct"]) for r in res if "item_id" in r}


def _pair(a_map, b_map):
    """Inner-join on item_id -> aligned arrays (a, b) of 0/1, plus dropped count."""
    ids = sorted(set(a_map) & set(b_map))
    a = np.array([a_map[i] for i in ids], dtype=np.int8)
    b = np.array([b_map[i] for i in ids], dtype=np.int8)
    dropped = (len(a_map) - len(ids), len(b_map) - len(ids))
    return a, b, dropped


def mcnemar_exact(b, c):
    """Two-sided exact McNemar p on discordant counts b, c.

    Under H0 each discordant pair is 50/50; #(b-type) ~ Binomial(n=b+c, 0.5).
    Two-sided p = 2 * P(X >= max(b,c)), capped at 1. If b+c==0, p=1 (no evidence).
    """
    n = b + c
    if n == 0:
        return 1.0
    k = max(b, c)
    tail = sum(comb(n, i) for i in range(k, n + 1)) / (2.0 ** n)
    return min(1.0, 2.0 * tail)


def paired_bootstrap(a, b, R, rng):
    """95% CI for Δacc = mean(a) - mean(b) by resampling ITEMS (keeps pairing)."""
    n = len(a)
    if n == 0:
        return 0.0, (0.0, 0.0)
    diff = a.astype(float) - b.astype(float)   # per-item paired difference
    idx = rng.integers(0, n, size=(R, n))
    boots = diff[idx].mean(axis=1)             # Δ per bootstrap replicate
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(diff.mean()), (float(lo), float(hi))


def _row(name, a, b, R, rng):
    n = len(a)
    both = int(np.sum((a == 1) & (b == 1)))
    bb = int(np.sum((a == 1) & (b == 0)))      # A right, B wrong
    cc = int(np.sum((a == 0) & (b == 1)))      # A wrong, B right
    p = mcnemar_exact(bb, cc)
    delta, (lo, hi) = paired_bootstrap(a, b, R, rng)
    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
    ci_excl = "yes" if (lo > 0 or hi < 0) else "no"
    print(f"{name:26s} N={n:4d}  b={bb:3d} c={cc:3d}  "
          f"Δ={delta*100:+5.2f}% [{lo*100:+5.2f},{hi*100:+5.2f}]  "
          f"CI≠0:{ci_excl:3s}  McNemar p={p:.4f} {sig}")
    return a, b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sys_a", required=True, help="dir of system A (e.g. SOTA)")
    ap.add_argument("--sys_b", required=True, help="dir of system B (e.g. baseline)")
    ap.add_argument("--task", default=CLEVRER, help="comma-separated task keys")
    ap.add_argument("--boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    tasks = [t.strip() for t in args.task.split(",") if t.strip()]
    print(f"A (better?) = {args.sys_a}")
    print(f"B (base)    = {args.sys_b}")
    print(f"Δ = acc_A - acc_B ; paired by item_id ; bootstrap R={args.boot}\n")

    pooled_a, pooled_b = [], []
    for t in tasks:
        am, bm = _load(args.sys_a, t), _load(args.sys_b, t)
        if am is None or bm is None:
            print(f"{t:26s} MISSING ({'A' if am is None else 'B'})")
            continue
        a, b, dropped = _pair(am, bm)
        if dropped != (0, 0):
            print(f"  [warn] {t}: dropped unmatched items A={dropped[0]} B={dropped[1]}")
        _row(t, a, b, args.boot, rng)
        pooled_a.append(a); pooled_b.append(b)

    if pooled_a:
        print("-" * 96)
        A = np.concatenate(pooled_a); B = np.concatenate(pooled_b)
        _row("AGGREGATE (pooled items)", A, B, args.boot, rng)


if __name__ == "__main__":
    main()
