#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analyze embeddings of queries and tools with a SentenceTransformer model and a BEIR-style dataset.
Outputs a JSON file consumable by plotting scripts and (optionally) a detailed text log.

Usage examples:
  # Analyze test split with a fine-tuned model
  python analyze_embedding.py -m runs/my_model_dir -d WikiTable300 --split test --top_k 10

  # Analyze train split (monitor training behavior)
  python analyze_embedding.py -m runs/my_model_dir -d WikiTable300 --split train --top_k 10

Notes:
- Expects BEIR-like dataset under ./datasets/{dataset}/
- corpus.jsonl entries are dicts with fields like {"title": "...", "text": "..."}
- qrels supply the ground-truth relevant tool ids per query
"""

import argparse
import json
import numpy as np
import os
import sys
from datetime import datetime
from typing import Dict, List, Tuple, Any

class DualOutput:
    """Mirror stdout to a file (used for --save_detailed)."""
    def __init__(self, filename: str):
        self.terminal = sys.stdout
        self.file = open(filename, 'w', encoding='utf-8')
    def write(self, message: str):
        self.terminal.write(message)
        self.file.write(message)
    def flush(self):
        self.terminal.flush()
        self.file.flush()
    def close(self):
        try:
            self.file.close()
        except Exception:
            pass

def doc_to_text(doc: Dict[str, Any]) -> str:
    """Join a BEIR corpus entry to a single string 'title + text'."""
    if isinstance(doc, str):
        # Already a string
        return doc.strip()
    if not isinstance(doc, dict):
        # Fallback to string representation
        return str(doc)
    title = (doc.get("title") or "").strip()
    text = (doc.get("text") or "").strip()
    joined = f"{title} {text}".strip()
    return joined if joined else title or text

def get_tool_name_map():
    return {
        '0': 'f_select_column',
        '1': 'f_group_by',
        '2': 'print_table',
        '3': 'f_calculate_average',
        '4': 'f_filter_rows',
        '5': 'f_get_data_info',
        '6': 'f_select_row',
        '7': 'f_sort_by',
        '8': 'f_aggregate',
        '9': 'f_compute_column',
        '10': 'f_distinct_count',
        '11': 'f_final_answer',
        '12': 'f_process_datetime',
        '13': 'f_string_operation',
        '14': 'f_undo'
    }

def build_argparser():
    p = argparse.ArgumentParser(description="Analyze embeddings and produce Top-K tool stats.")
    p.add_argument("-d", "--dataset", default="WikiTable300", type=str, help="Dataset name (under ./datasets/{dataset})")
    p.add_argument("-m", "--model_path", required=True, type=str, help="SentenceTransformer model directory or model name")
    p.add_argument("--top_k", default=10, type=int, help="Top-k tools to compute per query")
    p.add_argument("--split", default="test", choices=["train", "test", "dev"], help="Data split to analyze")
    p.add_argument("--output_dir", default="analysis", type=str, help="Output directory for JSON and (optional) text log")
    p.add_argument("--save_detailed", action="store_true", help="Also save a human-readable text log")
    p.add_argument("--limit_queries", type=int, default=0, help="Optional: limit number of queries for a quick run (0 = all)")
    return p

def main():
    args = build_argparser().parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = os.path.basename(os.path.normpath(args.model_path))
    detailed_output_file = os.path.join(args.output_dir, f"{model_name}_{timestamp}.txt") if args.save_detailed else None

    # Optional dual output
    if detailed_output_file:
        sys.stdout = DualOutput(detailed_output_file)

    print("🚀 Starting Embedding Analysis")
    print(f"  Dataset: {args.dataset}")
    print(f"  Model:   {args.model_path}")
    print(f"  Split:   {args.split}")
    print(f"  Top-K:   {args.top_k}")
    if args.limit_queries:
        print(f"  Limit queries: {args.limit_queries}")
    if detailed_output_file:
        print(f"  Log file: {detailed_output_file}")

    # Lazy imports to keep startup quick
    try:
        from sentence_transformers import SentenceTransformer
        from beir.datasets.data_loader import GenericDataLoader
        from sklearn.metrics.pairwise import cosine_similarity
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Please install required packages:\n  pip install sentence-transformers beir scikit-learn")
        if isinstance(sys.stdout, DualOutput):
            sys.stdout.close()
            sys.stdout = sys.__stdout__
        sys.exit(1)

    data_path = f"./datasets/{args.dataset}"
    if not os.path.exists(data_path):
        print(f"❌ Dataset path does not exist: {data_path}")
        if isinstance(sys.stdout, DualOutput):
            sys.stdout.close()
            sys.stdout = sys.__stdout__
        sys.exit(2)

    if not (os.path.exists(args.model_path) or "://" in args.model_path or "/" not in args.model_path):
        # If it's not a local path, it could still be a model hub name — allow that.
        pass

    print("📚 Loading dataset...")
    corpus, queries, qrels = GenericDataLoader(data_path).load(split=args.split)
    print(f"✅ Loaded {len(queries)} queries, {len(corpus)} tools")

    # Optionally limit queries for quick iteration
    query_ids = list(queries.keys())
    if args.limit_queries and args.limit_queries > 0:
        query_ids = query_ids[:args.limit_queries]

    query_texts = [queries[qid] for qid in query_ids]
    tool_ids = list(corpus.keys())
    tool_texts = [doc_to_text(corpus[tid]) for tid in tool_ids]

    print("🤖 Loading model...")
    model = SentenceTransformer(args.model_path)
    print(f"✅ Model device: {model.device}")

    print("🔄 Encoding queries...")
    query_embeddings = model.encode(query_texts, convert_to_tensor=False, show_progress_bar=True)
    print("🔄 Encoding tools...")
    tool_embeddings = model.encode(tool_texts, convert_to_tensor=False, show_progress_bar=True)

    print(f"✅ Shapes: queries={query_embeddings.shape}, tools={tool_embeddings.shape}")
    print("🧮 Computing cosine similarity...")
    similarity_matrix = cosine_similarity(query_embeddings, tool_embeddings)

    # Assemble analysis payload
    analysis = {
        "model_info": {
            "model_path": args.model_path,
            "dataset": args.dataset,
            "split": args.split,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "embedding_dimension": int(query_embeddings.shape[1]),
            "num_queries": len(query_ids),
            "num_tools": len(tool_ids)
        },
        "query_analysis": {},
        "statistics": {}
    }

    # Detailed per-query block
    print("\n" + "="*100)
    print("📋 DETAILED QUERY-TOOL ANALYSIS")
    print("="*100)

    all_relevant_sims = []
    tool_name_map = get_tool_name_map()

    for i, qid in enumerate(query_ids):
        qtext = queries[qid]
        sims = similarity_matrix[i]
        # top-k indices
        top_idx = np.argsort(sims)[::-1][:args.top_k]
        top_sim = sims[top_idx]
        top_tool_ids = [tool_ids[j] for j in top_idx]

        relevant = qrels.get(qid, {})  # dict of {tool_id: score}
        rel_set = set(relevant.keys())

        print(f"\n🔍 Query {i+1}/{len(query_ids)}: {qid}")
        print(f"📝 Text: {qtext}")
        print(f"🎯 Ground truth relevant tools: {len(rel_set)}")

        print(f"\n🏆 Top {args.top_k} Tools:")
        print(f"{'Rank':<4} {'Tool ID':<6} {'Name':<22} {'Sim':<10} {'Relevant':<9} {'Text'}")
        print("-"*110)
        top_tools_details = []
        for rank, (tid, s) in enumerate(zip(top_tool_ids, top_sim), start=1):
            is_rel = tid in rel_set
            name = tool_name_map.get(str(tid), str(tid))
            text_full = doc_to_text(corpus[tid])
            text_short = (text_full[:48] + "...") if len(text_full) > 48 else text_full
            print(f"{rank:<4} {tid:<6} {name:<22} {s:<10.4f} {str(is_rel):<9} {text_short}")
            top_tools_details.append({
                "rank": rank,
                "tool_id": tid,
                "similarity": float(s),
                "is_relevant": is_rel,
                "tool_text": text_full
            })

        # Relevant tools performance block
        rel_details = []
        if rel_set:
            print("\n✅ Ground Truth Relevant Tools:")
            print(f"{'Tool ID':<6} {'Name':<22} {'Sim':<10} {'Rank':<6} {'Score':<6} {'Text'}")
            print("-"*110)
            ranking = np.argsort(sims)[::-1]
            for tid, rscore in relevant.items():
                if tid in tool_ids:
                    t_idx = tool_ids.index(tid)
                    sim = sims[t_idx]
                    rank = int(np.where(ranking == t_idx)[0][0]) + 1
                    text_full = doc_to_text(corpus[tid])
                    text_short = (text_full[:48] + "...") if len(text_full) > 48 else text_full
                    name = tool_name_map.get(str(tid), str(tid))
                    print(f"{tid:<6} {name:<22} {sim:<10.4f} {rank:<6} {rscore:<6} {text_short}")
                    all_relevant_sims.append(sim)
                    rel_details.append({
                        "tool_id": tid,
                        "similarity": float(sim),
                        "rank": rank,
                        "relevance_score": rscore,
                        "tool_text": text_full
                    })

        analysis["query_analysis"][qid] = {
            "text": qtext,
            "top_k_tools": top_tools_details,
            "relevant_tools_performance": rel_details
        }

    # Overall stats
    print("\n" + "="*80)
    print("📊 OVERALL STATISTICS")
    print("="*80)

    if all_relevant_sims:
        arr = np.array(all_relevant_sims, dtype=float)
        stats_rel = {
            "count": int(arr.size),
            "avg_similarity": float(arr.mean()),
            "std_similarity": float(arr.std()),
            "min_similarity": float(arr.min()),
            "max_similarity": float(arr.max())
        }
        print("🎯 Relevant pairs:",
              f"count={stats_rel['count']}, avg={stats_rel['avg_similarity']:.4f},",
              f"std={stats_rel['std_similarity']:.4f}, min={stats_rel['min_similarity']:.4f},",
              f"max={stats_rel['max_similarity']:.4f}")
        analysis["statistics"]["relevant_pairs"] = stats_rel

    sim_flat = similarity_matrix.flatten().astype(float)
    analysis["statistics"]["all_pairs"] = {
        "count": int(sim_flat.size),
        "avg_similarity": float(sim_flat.mean()),
        "std_similarity": float(sim_flat.std())
    }
    print(f"🌐 All pairs: count={sim_flat.size}, avg={sim_flat.mean():.4f}, std={sim_flat.std():.4f}")

    # Save JSON
    json_path = os.path.join(args.output_dir, f"{model_name}_{timestamp}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(analysis, f, ensure_ascii=False, indent=2)
    print("\n💾 Saved analysis JSON:", json_path)

    # Cleanup dual output
    if isinstance(sys.stdout, DualOutput):
        sys.stdout.close()
        sys.stdout = sys.__stdout__

if __name__ == "__main__":
    main()
