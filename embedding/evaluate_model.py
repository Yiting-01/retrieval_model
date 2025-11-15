"""
python evaluate_model.py --model_path runs/msmarco-roberta-base-ance-firstpWikiTable300-TQA_20250914_233815 --dataset WikiTable300 --output evaluation_results.txt
python evaluate_model.py --model_path runs/msmarco-roberta-base-ance-firstp_WikiTable300_StateAware-state_aware_20250915_000338 --dataset WikiTable300_StateAware --output evaluation_results.txt

"""

import argparse
import json
import os
import glob
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import torch
from datetime import datetime
import sys

class TeeOutput:
    """同时输出到文件和控制台"""
    def __init__(self, file_path):
        self.terminal = sys.stdout
        self.log = open(file_path, 'w', encoding='utf-8')
        
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()
        
    def flush(self):
        self.terminal.flush()
        self.log.flush()
        
    def close(self):
        sys.stdout = self.terminal
        self.log.close()

def find_latest_model(runs_dir="runs"):
    """自动找到最新训练的模型"""
    model_dirs = glob.glob(os.path.join(runs_dir, "*ance*"))
    if not model_dirs:
        return None
    return max(model_dirs, key=os.path.getmtime)

def load_jsonl(filepath):
    """加载JSONL文件"""
    data = {}
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                data[item['_id']] = item
    return data

def load_qrels(filepath):
    """加载qrels文件"""
    qrels = {}
    df = pd.read_csv(filepath, sep='\t')
    
    for _, row in df.iterrows():
        query_id = str(row['query-id'])
        corpus_id = str(row['corpus-id'])
        score = row['score']
        
        if query_id not in qrels:
            qrels[query_id] = {}
        qrels[query_id][corpus_id] = score
    
    return qrels

def evaluate_retrieval(qrels, results):
    """计算检索指标，输出格式类似训练时的日志"""
    
    num_queries = len(qrels)
    num_corpus = 15  # 固定15个工具
    
    # 计算各种指标
    accuracy_at_k = {}
    precision_at_k = {}
    recall_at_k = {}
    
    k_values = [1, 3, 5, 10]
    
    for k in k_values:
        accuracy_at_k[k] = []
        precision_at_k[k] = []
        recall_at_k[k] = []
    
    # 计算NDCG和MRR
    ndcg_scores = []
    mrr_scores = []
    map_scores = []
    
    for query_id in qrels:
        if query_id not in results:
            continue
            
        relevant_docs = set(qrels[query_id].keys())
        retrieved_docs = list(results[query_id].keys())
        
        # 计算MRR
        mrr = 0
        for i, doc_id in enumerate(retrieved_docs[:10]):
            if doc_id in relevant_docs:
                mrr = 1.0 / (i + 1)
                break
        mrr_scores.append(mrr)
        
        # 计算各k值的指标
        for k in k_values:
            top_k = retrieved_docs[:k]
            
            # Accuracy@k (至少命中一个相关文档)
            hit = len(set(top_k) & relevant_docs) > 0
            accuracy_at_k[k].append(1.0 if hit else 0.0)
            
            # Precision@k
            relevant_in_k = len(set(top_k) & relevant_docs)
            precision_at_k[k].append(relevant_in_k / k)
            
            # Recall@k
            recall_at_k[k].append(relevant_in_k / len(relevant_docs))
        
        # 计算NDCG@10
        dcg = 0
        idcg = sum([1/np.log2(i+2) for i in range(min(len(relevant_docs), 10))])
        
        for i, doc_id in enumerate(retrieved_docs[:10]):
            if doc_id in relevant_docs:
                dcg += 1 / np.log2(i + 2)
        
        ndcg = dcg / idcg if idcg > 0 else 0
        ndcg_scores.append(ndcg)
        
        # 计算AP (Average Precision)
        ap = 0
        relevant_count = 0
        for i, doc_id in enumerate(retrieved_docs):
            if doc_id in relevant_docs:
                relevant_count += 1
                precision_at_i = relevant_count / (i + 1)
                ap += precision_at_i
        ap = ap / len(relevant_docs) if relevant_docs else 0
        map_scores.append(ap)
    
    # 输出结果，格式类似训练日志
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    print(f"{current_time} - Queries: {num_queries}")
    print(f"{current_time} - Corpus: {num_corpus}")
    print()
    
    # 使用余弦相似度的结果
    print(f"{current_time} - Score-Function: cos_sim")
    for k in k_values:
        acc = np.mean(accuracy_at_k[k]) * 100
        print(f"{current_time} - Accuracy@{k}: {acc:.2f}%")
    
    for k in k_values:
        prec = np.mean(precision_at_k[k]) * 100
        print(f"{current_time} - Precision@{k}: {prec:.2f}%")
    
    for k in k_values:
        rec = np.mean(recall_at_k[k]) * 100
        print(f"{current_time} - Recall@{k}: {rec:.2f}%")
    
    mrr = np.mean(mrr_scores)
    ndcg = np.mean(ndcg_scores)
    map_score = np.mean(map_scores)
    
    print(f"{current_time} - MRR@10: {mrr:.4f}")
    print(f"{current_time} - NDCG@10: {ndcg:.4f}")
    print(f"{current_time} - MAP@100: {map_score:.4f}")

def main():
    parser = argparse.ArgumentParser(description="在测试集上评估模型")
    parser.add_argument("--model_path", type=str, help="模型路径")
    parser.add_argument("--dataset", default="WikiTable300", help="数据集名称")
    parser.add_argument("--output", type=str, help="输出文件路径 (例如: evaluation_results.txt)")
    
    args = parser.parse_args()
    
    # 自动生成输出文件名（如果没有指定）
    if args.output is None:
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        # 确保log目录存在
        os.makedirs("log/evaluation", exist_ok=True)
        args.output = f"log/evaluation/evaluation_{args.dataset}_{current_time}.txt"
    else:
        # 如果指定了输出文件，自动添加数据集名称
        file_path, file_ext = os.path.splitext(args.output)
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = f"{file_path}_{args.dataset}_{current_time}{file_ext}"
    
    # 设置输出文件
    if args.output:
        # 确保输出目录存在
        output_dir = os.path.dirname(args.output)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # 重定向输出到文件和控制台
        tee = TeeOutput(args.output)
        sys.stdout = tee
    
    try:
        # 自动找到模型路径
        if args.model_path is None:
            args.model_path = find_latest_model()
            if args.model_path is None:
                print("Error: No model found in runs/ directory")
                return
        
        dataset_path = f"./datasets/{args.dataset}"
        
        print(f"Loading model: {args.model_path}")
        
        # 加载模型
        device = "cpu"
        model = SentenceTransformer(args.model_path, device=device)
        print("Model loaded successfully")
        
        # 加载数据
        print("Loading test data...")
        
        # 加载corpus
        corpus_path = os.path.join(dataset_path, 'corpus.jsonl')
        corpus_data = load_jsonl(corpus_path)
        
        # 加载queries
        queries_path = os.path.join(dataset_path, 'queries.jsonl')
        queries_data = load_jsonl(queries_path)
        
        # 加载测试集qrels
        qrels_path = os.path.join(dataset_path, 'qrels', 'test.tsv')
        qrels = load_qrels(qrels_path)
        
        # 筛选测试集查询
        test_queries = {qid: queries_data[qid]['text'] for qid in qrels.keys() if qid in queries_data}
        
        print(f"Test queries: {len(test_queries)}")
        print(f"Corpus documents: {len(corpus_data)}")
        
        # 准备corpus
        corpus_ids = list(corpus_data.keys())
        corpus_texts = [corpus_data[cid]['text'] for cid in corpus_ids]
        
        # 编码
        print("Encoding corpus...")
        corpus_embeddings = model.encode(corpus_texts, show_progress_bar=True)
        
        print("Encoding queries...")
        query_texts = list(test_queries.values())
        query_ids = list(test_queries.keys())
        query_embeddings = model.encode(query_texts, show_progress_bar=True)
        
        # 检索
        print("Performing retrieval...")
        results = {}
        
        for i, query_id in enumerate(query_ids):
            query_embedding = query_embeddings[i].reshape(1, -1)
            
            # 计算相似度
            similarities = cosine_similarity(query_embedding, corpus_embeddings)[0]
            
            # 排序
            sorted_indices = np.argsort(similarities)[::-1]
            
            results[query_id] = {}
            for idx in sorted_indices:
                corpus_id = corpus_ids[idx]
                score = float(similarities[idx])
                results[query_id][corpus_id] = score
        
        # 评估并输出结果
        print("\n" + "="*50)
        print("Information Retrieval Evaluation on TEST dataset:")
        print("="*50)
        
        evaluate_retrieval(qrels, results)
        
        if args.output:
            print(f"\nResults saved to: {args.output}")
            
    finally:
        # 关闭文件输出
        if args.output:
            tee.close()

if __name__ == "__main__":
    main()