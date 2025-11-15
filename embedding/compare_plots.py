#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compare two analysis JSONs (from analyze_embedding.py), make comparative plots and a LaTeX table.

Outputs to --outdir:
  - hit_dist_compare.png               # A vs B: All/Some/None bars
  - per_tool_miss_rate_compare.png     # side-by-side bars per tool (A vs B)
  - per_tool_delta_sorted.png          # delta (B - A) sorted bars (negative = better for B)
  - per_tool_miss_rate_comparison.tex  # LaTeX tabular with A/B/Delta

Usage:
  python compare_plots.py --json_a analysis/A.json --json_b analysis/B.json --outdir analysis
"""

import argparse, json, os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import Counter

plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

TOOL_MAP = {
    '0': 'f_select_column','1': 'f_group_by','2': 'print_table','3': 'f_calculate_average',
    '4': 'f_filter_rows','5': 'f_get_data_info','6': 'f_select_row','7': 'f_sort_by',
    '8': 'f_aggregate','9': 'f_compute_column','10': 'f_distinct_count','11': 'f_final_answer',
    '12': 'f_process_datetime','13': 'f_string_operation','14': 'f_undo'
}

def load(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def summarize_hit_stats(qa):
    total = all_hit = some_hit = none_hit = 0
    for qid, q in qa.items():
        rel = {t['tool_id'] for t in q.get('relevant_tools_performance', [])}
        top5 = {t['tool_id'] for t in q.get('top_k_tools', [])[:5] if t.get('is_relevant', False)}
        if not rel:
            continue
        total += 1
        if rel.issubset(top5):
            all_hit += 1
        elif rel & top5:
            some_hit += 1
        else:
            none_hit += 1
    def pct(x): return (x/total*100.0) if total else 0.0
    return {"total": total, "all": all_hit, "some": some_hit, "none": none_hit,
            "all_rate": pct(all_hit), "some_rate": pct(some_hit), "none_rate": pct(none_hit)}

def per_tool_miss_rate(qa):
    annotated = Counter()
    missed = Counter()
    for qid, q in qa.items():
        rel = {t['tool_id'] for t in q.get('relevant_tools_performance', [])}
        top5 = {t['tool_id'] for t in q.get('top_k_tools', [])[:5] if t.get('is_relevant', False)}
        for t in rel:
            annotated[t] += 1
            if t not in top5:
                missed[t] += 1
    rates = {}
    for t, cnt in annotated.items():
        mr = 100.0 * missed[t] / cnt if cnt else 0.0
        rates[t] = {"annotated": cnt, "missed": missed[t], "miss_rate": mr}
    return rates

def figure_hit_distribution(sA, sB, outpath):
    labels = ['All Relevant in Top-5', 'Some Relevant', 'No Relevant']
    A_vals = [sA['all_rate'], sA['some_rate'], sA['none_rate']]
    B_vals = [sB['all_rate'], sB['some_rate'], sB['none_rate']]
    x = np.arange(len(labels))
    width = 0.35

    plt.figure(figsize=(10, 6))
    plt.bar(x - width/2, A_vals, width, label='Model A')
    plt.bar(x + width/2, B_vals, width, label='Model B')
    plt.xticks(x, labels, rotation=15, ha='right')
    plt.ylabel('Percentage (%)')
    plt.title(f'Top-5 Hit Distribution  (nA={sA["total"]}, nB={sB["total"]})')
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(outpath, dpi=300, bbox_inches='tight')

def figure_per_tool_compare(rA, rB, outpath):
    keys = sorted(set(rA.keys()) | set(rB.keys()), key=lambda x: int(x))
    labels = [TOOL_MAP.get(k, k) for k in keys]
    A_vals = [rA.get(k, {"miss_rate":0.0})["miss_rate"] for k in keys]
    B_vals = [rB.get(k, {"miss_rate":0.0})["miss_rate"] for k in keys]

    x = np.arange(len(labels))
    width = 0.38

    plt.figure(figsize=(14, 8))
    plt.bar(x - width/2, A_vals, width, label='Model A')
    plt.bar(x + width/2, B_vals, width, label='Model B')
    plt.xticks(x, labels, rotation=45, ha='right')
    plt.ylabel('Miss Rate (%)')
    plt.title('Per-Tool Miss Rate (Top-5) Comparison')
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(outpath, dpi=300, bbox_inches='tight')

def figure_delta_sorted(rA, rB, outpath):
    keys = sorted(set(rA.keys()) | set(rB.keys()), key=lambda x: int(x))
    items = []
    for k in keys:
        a = rA.get(k, {"miss_rate":0.0})["miss_rate"]
        b = rB.get(k, {"miss_rate":0.0})["miss_rate"]
        items.append((k, b - a))
    items_sorted = sorted(items, key=lambda x: x[1])  # ascending, negatives first (better for B)

    labels = [TOOL_MAP.get(k, k) for k, _ in items_sorted]
    deltas = [d for _, d in items_sorted]

    plt.figure(figsize=(14, 8))
    x = np.arange(len(labels))
    plt.bar(x, deltas)
    plt.xticks(x, labels, rotation=45, ha='right')
    plt.ylabel('Δ Miss Rate (B - A) %')
    plt.title('Per-Tool Miss Rate Δ (Negative = B better)')
    plt.axhline(0, linestyle='--', alpha=0.7)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(outpath, dpi=300, bbox_inches='tight')

def export_latex_table(rA, rB, outpath):
    keys = sorted(set(rA.keys()) | set(rB.keys()), key=lambda x: int(x))
    lines = []
    lines.append(r"\begin{tabular}{lrrr}")
    lines.append(r"\toprule")
    lines.append(r"Tool & A (Miss\%) & B (Miss\%) & $\Delta$ (B$-$A) \\")
    lines.append(r"\midrule")
    for k in keys:
        name = TOOL_MAP.get(k, k)
        a = rA.get(k, {"miss_rate":0.0})["miss_rate"]
        b = rB.get(k, {"miss_rate":0.0})["miss_rate"]
        delta = b - a
        lines.append(f"{name} & {a:.1f} & {b:.1f} & {delta:.1f} \\\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    with open(outpath, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json_a", required=True, help="Analysis JSON of model A")
    ap.add_argument("--json_b", required=True, help="Analysis JSON of model B")
    ap.add_argument("--outdir", type=str, default="analysis", help="Output directory")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    A = load(args.json_a); qaA = A.get("query_analysis", {})
    B = load(args.json_b); qaB = B.get("query_analysis", {})

    sA = summarize_hit_stats(qaA)
    sB = summarize_hit_stats(qaB)

    rA = per_tool_miss_rate(qaA)
    rB = per_tool_miss_rate(qaB)

    # Plots
    figure_hit_distribution(sA, sB, os.path.join(args.outdir, "hit_dist_compare.png"))
    figure_per_tool_compare(rA, rB, os.path.join(args.outdir, "per_tool_miss_rate_compare.png"))
    figure_delta_sorted(rA, rB, os.path.join(args.outdir, "per_tool_delta_sorted.png"))

    # LaTeX table
    export_latex_table(rA, rB, os.path.join(args.outdir, "per_tool_miss_rate_comparison.tex"))

    print("✅ Outputs saved to:", args.outdir)

if __name__ == "__main__":
    main()
