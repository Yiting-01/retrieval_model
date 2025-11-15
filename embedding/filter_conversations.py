#!/usr/bin/env python3
"""
过滤conversation文件夹，移除correctness为"wrong"的conversation文件
python filter_conversations.py ./conversations_2_gpt2 evaluate_csv/wikitable_results_2_gpt2.csv --queries datasets/queries_2.jsonl
python filter_conversations.py ./conversations_3_ds evaluate_csv/wikitable_results_3_ds.csv --queries datasets/queries_3.jsonl
python filter_conversations.py ./conversations_4 evaluate_csv/wikitable_results_4.csv --queries datasets/queries_4.jsonl
python filter_conversations.py ./conversations evaluate_csv/wikitable_results_1_openai.csv --queries datasets/queries_1.jsonl

"""

import json
import csv
import os
import shutil
from pathlib import Path
from typing import Set, List, Dict
import argparse


def load_queries_json(queries_file: str) -> Dict[str, str]:
    """
    加载queries.json或queries.jsonl文件,返回conversation_id到query的映射
    """
    id_to_query = {}
    
    with open(queries_file, 'r', encoding='utf-8') as f:
        # 检查文件扩展名来确定格式
        if queries_file.endswith('.jsonl'):
            # JSONL格式：每行一个JSON对象
            for line in f:
                line = line.strip()
                if line:
                    try:
                        item = json.loads(line)
                        if 'conversation_id' in item and 'question' in item:
                            id_to_query[item['conversation_id']] = item['question']
                        elif '_id' in item and 'text' in item:
                            id_to_query[item['_id']] = item['text']
                    except json.JSONDecodeError as e:
                        print(f"警告: 跳过无效的JSON行: {line[:50]}...")
        else:
            # 标准JSON格式
            try:
                queries_data = json.load(f)
                # 如果是列表
                if isinstance(queries_data, list):
                    for item in queries_data:
                        if 'conversation_id' in item and 'question' in item:
                            id_to_query[item['conversation_id']] = item['question']
                        elif '_id' in item and 'text' in item:
                            id_to_query[item['_id']] = item['text']
                # 如果是字典
                elif isinstance(queries_data, dict):
                    if 'conversation_id' in queries_data and 'question' in queries_data:
                        id_to_query[queries_data['conversation_id']] = queries_data['question']
                    elif '_id' in queries_data and 'text' in queries_data:
                        id_to_query[queries_data['_id']] = queries_data['text']
            except json.JSONDecodeError as e:
                print(f"错误: 无法解析JSON文件 {queries_file}: {e}")
    
    return id_to_query


def load_evaluation_csv(csv_file: str) -> Set[str]:
    """
    加载evaluation CSV文件，返回correctness为"wrong"的query集合
    """
    wrong_queries = set()
    
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get('correctness', '').lower() == 'wrong':
                question = row.get('question', '').strip()
                if question:
                    wrong_queries.add(question)
    
    return wrong_queries


def get_conversation_query(conversation_file: str) -> str:
    """
    从conversation文件中提取query
    """
    try:
        with open(conversation_file, 'r', encoding='utf-8') as f:
            content = f.read().strip()
        
        # 尝试按行分割处理JSONL格式
        if conversation_file.endswith('.jsonl'):
            lines = content.split('\n')
            for line in lines:
                line = line.strip()
                if line:
                    try:
                        conversation_data = json.loads(line)
                        # 尝试提取query
                        query = extract_query_from_data(conversation_data)
                        if query:
                            return query
                    except json.JSONDecodeError:
                        continue
        else:
            # 尝试作为单个JSON对象解析
            conversation_data = json.loads(content)
            return extract_query_from_data(conversation_data)
        
        return ""
    except Exception as e:
        print(f"Error reading {conversation_file}: {e}")
        return ""


def extract_query_from_data(conversation_data: dict) -> str:
    """
    从conversation数据中提取query
    """
    # 从conversation中找到用户的问题
    if 'question' in conversation_data:
        return conversation_data['question']
    elif 'messages' in conversation_data:
        for message in conversation_data['messages']:
            if message.get('role') == 'user':
                return message.get('content', '').strip()
    
    return ""


def should_keep_conversation(conversation_file: str, wrong_queries: Set[str], 
                           id_to_query: Dict[str, str] = None) -> bool:
    """
    判断是否应该保留这个conversation文件
    """
    # 方法1: 从文件名中提取conversation_id，然后查找对应的query
    if id_to_query:
        # 提取conversation_id，处理不同的文件名格式
        conversation_id = Path(conversation_file).stem
        # 移除可能的前缀（如 "conversation_"）
        if conversation_id.startswith('conversation_'):
            conversation_id = conversation_id[len('conversation_'):]
        
        if conversation_id in id_to_query:
            query = id_to_query[conversation_id]
            is_wrong = query in wrong_queries
            if is_wrong:
                print(f"根据ID匹配跳过文件: {Path(conversation_file).name} (query: {query[:50]}...)")
            return not is_wrong
    
    # 方法2: 直接从conversation文件中读取query
    query = get_conversation_query(conversation_file)
    if query:
        is_wrong = query in wrong_queries
        if is_wrong:
            print(f"根据内容匹配跳过文件: {Path(conversation_file).name} (query: {query[:50]}...)")
        return not is_wrong
    
    # 如果无法提取query，保守地保留文件
    print(f"警告: 无法提取query，保留文件: {Path(conversation_file).name}")
    return True


def filter_conversations(input_dir: str, output_dir: str, csv_file: str, 
                        queries_file: str = None):
    """
    过滤conversation文件夹
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # 创建输出目录
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 加载wrong queries
    wrong_queries = load_evaluation_csv(csv_file)
    print(f"找到 {len(wrong_queries)} 个标记为wrong的queries")
    
    # 加载queries映射（可选）
    id_to_query = {}
    if queries_file and os.path.exists(queries_file):
        id_to_query = load_queries_json(queries_file)
        print(f"加载了 {len(id_to_query)} 个query映射")
    
    # 遍历输入目录中的所有conversation文件
    conversation_files = list(input_path.glob("*.jsonl")) + list(input_path.glob("*.json"))
    
    kept_count = 0
    removed_count = 0
    
    for conversation_file in conversation_files:
        try:
            if should_keep_conversation(str(conversation_file), wrong_queries, id_to_query):
                # 保留这个文件
                output_file = output_path / conversation_file.name
                shutil.copy2(conversation_file, output_file)
                kept_count += 1
            else:
                # 跳过这个文件
                removed_count += 1
                print(f"跳过文件: {conversation_file.name}")
                
        except Exception as e:
            print(f"处理文件 {conversation_file} 时出错: {e}")
            continue
    
    print(f"\n处理完成:")
    print(f"保留的文件数: {kept_count}")
    print(f"移除的文件数: {removed_count}")
    print(f"总文件数: {kept_count + removed_count}")


def main():
    parser = argparse.ArgumentParser(description="过滤conversation文件夹，移除correctness为wrong的文件")
    parser.add_argument("input_dir", help="输入的conversation文件夹路径")
    parser.add_argument("csv_file", help="evaluation CSV文件路径")
    parser.add_argument("--queries", help="queries.json或queries.jsonl文件路径（可选）")
    parser.add_argument("--output-dir", help="输出文件夹路径（可选，默认为输入文件夹名_correct）")
    
    args = parser.parse_args()
    
    # 检查输入文件/目录是否存在
    if not os.path.exists(args.input_dir):
        print(f"错误: 输入目录不存在: {args.input_dir}")
        return
    
    if not os.path.exists(args.csv_file):
        print(f"错误: CSV文件不存在: {args.csv_file}")
        return
    
    if args.queries and not os.path.exists(args.queries):
        print(f"警告: queries文件不存在: {args.queries}")
    
    # 生成输出目录名
    if args.output_dir:
        output_dir = args.output_dir
    else:
        input_path = Path(args.input_dir)
        output_dir = str(input_path.parent / (input_path.name + "_correct"))
    
    print(f"输入文件夹: {args.input_dir}")
    print(f"输出文件夹: {output_dir}")
    
    filter_conversations(args.input_dir, output_dir, args.csv_file, args.queries)


if __name__ == "__main__":
    main()


# 使用示例:
# python filter_conversations.py ./conversations evaluation.csv --queries queries.json
# 这会创建 ./conversations_correct 文件夹

# 或者指定自定义输出目录:
# python filter_conversations.py ./conversations evaluation.csv --queries queries.json --output-dir ./my_filtered_conversations