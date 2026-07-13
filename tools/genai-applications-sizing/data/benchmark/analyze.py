#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Summarize client-side benchmark KPIs from a run-benchmark.sh results CSV.

Computes count / min / mean / p50 / p90 / p95 / max for end-to-end latency and
time-to-first-output, per prompt and overall, using only the 'measure' phase
rows (warmups are excluded). Detailed span KPIs live in SigNoz; this covers the
client-observed latency numbers.

Usage:
    python3 analyze.py <results.csv>
"""
import csv
import sys
from statistics import mean


def percentile(values, pct):
    if not values:
        return None
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    k = (len(s) - 1) * (pct / 100.0)
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def stats(values):
    values = [v for v in values if v is not None]
    if not values:
        return None
    return {
        "n": len(values),
        "min": min(values),
        "mean": mean(values),
        "p50": percentile(values, 50),
        "p90": percentile(values, 90),
        "p95": percentile(values, 95),
        "max": max(values),
    }


def fmt(row_label, s):
    if s is None:
        return f"{row_label:<40} (no data)"
    return (
        f"{row_label:<40} n={s['n']:<3} "
        f"min={s['min']:>7.0f} mean={s['mean']:>7.0f} "
        f"p50={s['p50']:>7.0f} p90={s['p90']:>7.0f} "
        f"p95={s['p95']:>7.0f} max={s['max']:>7.0f}"
    )


def main():
    if len(sys.argv) != 2:
        print("Usage: python3 analyze.py <results.csv>", file=sys.stderr)
        sys.exit(1)

    path = sys.argv[1]
    by_prompt = {}          # pidx -> {"prompt": str, "e2e": [], "ttft": []}
    all_e2e, all_ttft = [], []
    failures = 0

    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            if r.get("phase") != "measure":
                continue
            try:
                rc = int(r["exit_code"])
            except (ValueError, KeyError):
                rc = -1
            if rc != 0:
                failures += 1

            pidx = r["prompt_index"]
            entry = by_prompt.setdefault(
                pidx, {"prompt": r.get("prompt", ""), "e2e": [], "ttft": []}
            )
            for key, dest, bucket in (
                ("e2e_ms", "e2e", all_e2e),
                ("ttft_ms", "ttft", all_ttft),
            ):
                val = r.get(key, "").strip()
                if val:
                    try:
                        num = float(val)
                    except ValueError:
                        continue
                    entry[dest].append(num)
                    bucket.append(num)

    print("=" * 100)
    print("CLIENT-SIDE KPI SUMMARY (milliseconds; 'measure' phase only)")
    print("=" * 100)

    print("\nEnd-to-end latency by prompt:")
    for pidx in sorted(by_prompt, key=lambda x: int(x)):
        e = by_prompt[pidx]
        label = f"[{pidx}] {e['prompt'][:32]}"
        print("  " + fmt(label, stats(e["e2e"])))

    print("\nTime-to-first-output by prompt:")
    for pidx in sorted(by_prompt, key=lambda x: int(x)):
        e = by_prompt[pidx]
        label = f"[{pidx}] {e['prompt'][:32]}"
        print("  " + fmt(label, stats(e["ttft"])))

    print("\nOverall:")
    print("  " + fmt("end-to-end latency (all prompts)", stats(all_e2e)))
    print("  " + fmt("time-to-first-output (all prompts)", stats(all_ttft)))

    total = len(all_e2e)
    print(f"\nRuns measured: {total}   failures (exit_code != 0): {failures}")
    print("=" * 100)


if __name__ == "__main__":
    main()
