import json
import pandas as pd
import os

def load_jsonl(filename):
    """加载JSONL文件"""
    with open(filename, 'r', encoding='utf-8') as f:
        return [json.loads(line.strip()) for line in f if line.strip()]

def save_jsonl(data, filename):
    """保存JSONL文件"""
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

def merge_datasets(dataset_configs, output_dir='datasets/merged'):
    """
    合并多个数据集
    
    Args:
        dataset_configs: 数据集配置列表，每个元素包含：
            {
                'name': '数据集名称',
                'queries_file': 'queries文件路径',
                'train_file': 'train.tsv文件路径', 
                'test_file': 'test.tsv文件路径'
            }
        output_dir: 输出目录
    """
    
    print("开始合并数据集...")
    
    # 存储所有数据
    all_queries = []
    all_train_data = []
    all_test_data = []
    
    # 跟踪ID偏移量
    query_id_offset = 0
    
    for i, config in enumerate(dataset_configs):
        dataset_name = config['name']
        print(f"\n处理数据集 {i+1}: {dataset_name}")
        
        # 加载queries
        print(f"  加载queries: {config['queries_file']}")
        if not os.path.exists(config['queries_file']):
            print(f"  警告: queries文件不存在: {config['queries_file']}")
            continue
            
        queries = load_jsonl(config['queries_file'])
        print(f"     原始queries数量: {len(queries)}")
        
        # 重新分配query ID
        id_mapping = {}  # 原始ID -> 新ID的映射
        for j, query in enumerate(queries):
            # 处理可能的ID格式问题
            original_id = query['_id']
            new_id = query_id_offset + j
            id_mapping[original_id] = new_id
            
            # 更新query的ID
            query['_id'] = str(new_id)
            all_queries.append(query)
        
        print(f"     ID范围: {query_id_offset} - {query_id_offset + len(queries) - 1}")
        
        # 加载并处理train.tsv
        if os.path.exists(config['train_file']):
            print(f"  加载训练数据: {config['train_file']}")
            train_df = pd.read_csv(config['train_file'], sep='\t')
            
            # 更新query-id (这里需要根据你的tsv格式调整)
            # 如果tsv中的query-id与queries中的_id有对应关系，需要映射
            if 'query-id' in train_df.columns:
                train_df['query-id'] = train_df['query-id'] + query_id_offset
            
            all_train_data.append(train_df)
            print(f"     训练数据行数: {len(train_df)}")
        else:
            print(f"  警告: 训练文件不存在: {config['train_file']}")
        
        # 加载并处理test.tsv
        if os.path.exists(config['test_file']):
            print(f"  加载测试数据: {config['test_file']}")
            test_df = pd.read_csv(config['test_file'], sep='\t')
            
            # 更新query-id
            if 'query-id' in test_df.columns:
                test_df['query-id'] = test_df['query-id'] + query_id_offset
            
            all_test_data.append(test_df)
            print(f"     测试数据行数: {len(test_df)}")
        else:
            print(f"  警告: 测试文件不存在: {config['test_file']}")
        
        # 更新偏移量
        query_id_offset += len(queries)
    
    # 合并所有数据
    print(f"\n合并数据...")
    
    # 合并train数据
    if all_train_data:
        merged_train = pd.concat(all_train_data, ignore_index=True)
        merged_train = merged_train.sort_values(['query-id', 'corpus-id']).reset_index(drop=True)
    else:
        merged_train = pd.DataFrame(columns=['query-id', 'corpus-id', 'score'])
    
    # 合并test数据
    if all_test_data:
        merged_test = pd.concat(all_test_data, ignore_index=True)
        merged_test = merged_test.sort_values(['query-id', 'corpus-id']).reset_index(drop=True)
    else:
        merged_test = pd.DataFrame(columns=['query-id', 'corpus-id', 'score'])
    
    # 保存合并后的文件
    print(f"\n保存合并后的数据到: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f"{output_dir}/qrels", exist_ok=True)
    
    # 保存queries
    queries_output = f"{output_dir}/queries.jsonl"
    save_jsonl(all_queries, queries_output)
    print(f"  合并后的queries: {queries_output} ({len(all_queries)} 条)")
    
    # 保存train数据
    train_output = f"{output_dir}/qrels/train.tsv"
    merged_train.to_csv(train_output, sep='\t', index=False)
    print(f"  合并后的训练数据: {train_output} ({len(merged_train)} 行)")
    
    # 保存test数据
    test_output = f"{output_dir}/qrels/test.tsv"
    merged_test.to_csv(test_output, sep='\t', index=False)
    print(f"  合并后的测试数据: {test_output} ({len(merged_test)} 行)")
    
    # 显示统计信息
    print(f"\n合并统计:")
    print(f"  总queries数量: {len(all_queries)}")
    print(f"  总训练样本数: {len(merged_train)}")
    print(f"  总测试样本数: {len(merged_test)}")
    if len(merged_train) > 0:
        print(f"  唯一query数 (训练): {len(merged_train['query-id'].unique())}")
    if len(merged_test) > 0:
        print(f"  唯一query数 (测试): {len(merged_test['query-id'].unique())}")
    
    return all_queries, merged_train, merged_test

def create_corpus_file(output_dir='datasets/merged'):
    """
    创建corpus.jsonl文件（工具描述）
    """
    corpus_data = [
        {"_id": "0", "title": "f_select_column", "text": "Select a subset of columns. A column usually corresponds to an attribute in the table. This operation helps locate necessary attributes to answer the question."},
        {"_id": "1", "title": "f_group_by", "text": "Group the rows by the contents of a specific column and provide the count of each enumeration value in that column."},
        {"_id": "2", "title": "print_table", "text": "Print all rows of the selected/transformed table to see the current state and potentially provide final answer."},
        {"_id": "3", "title": "f_calculate_average", "text": "Calculate the average (mean) of a numeric column, grouped by another column. Perfect for finding which group has the highest/lowest average."},
        {"_id": "4", "title": "f_filter_rows", "text": "Filter rows based on specified conditions. Useful for removing missing data or focusing on specific subsets."},
        {"_id": "5", "title": "f_get_data_info", "text": "Get basic information about the dataset including shape, columns, data types, and missing values."},
        {"_id": "6", "title": "f_select_row", "text": "Return a row-filtered copy of the table. Can filter by explicit indices, single condition, or multiple AND-connected conditions."},
        {"_id": "7", "title": "f_sort_by", "text": "Sort table lexicographically by one or more columns in ascending or descending order."},
        {"_id": "8", "title": "f_aggregate", "text": "Compute aggregation (count, sum, avg, min, max) over a column, optionally grouped by another column."},
        {"_id": "9", "title": "f_compute_column", "text": "Create a new column by applying binary operations between columns or column and scalar. Operations: add, sub, mul, div, ratio."},
        {"_id": "10", "title": "f_distinct_count", "text": "Count the number of unique values in a column."},
        {"_id": "11", "title": "f_final_answer", "text": "Submit the final answer. Use exactly when you are done analysing."},
        {"_id": "12", "title": "f_process_datetime", "text": "Process or transform datetime columns. Use when parsing strings into datetime objects, extracting parts like year, month, day, converting timezone, or formatting datetimes as strings."},
        {"_id": "13", "title": "f_string_operation", "text": "Apply string manipulation to a DataFrame column. Use when standardizing text, extracting substrings, concatenating multiple columns, or replacing substrings or regex patterns."},
        {"_id": "14", "title": "f_undo", "text": "Undo the last operation performed on the table."}
    ]
    
    corpus_file = f"{output_dir}/corpus.jsonl"
    save_jsonl(corpus_data, corpus_file)
    print(f"创建corpus文件: {corpus_file}")
    return corpus_file

# 使用示例 - 修改为处理state-aware文件
if __name__ == "__main__":
    # 配置4个state-aware数据集
    datasets = [
        {
            'name': 'Dataset1_StateAware',
            'queries_file': 'conversations_1_correct_state_aware.jsonl',
            'train_file': 'datasets/train_1_correct.tsv',
            'test_file': 'datasets/test_1_correct.tsv'
        },
        {
            'name': 'Dataset2_StateAware',
            'queries_file': 'conversations_2_gpt2_correct_state_aware.jsonl', 
            'train_file': 'datasets/train_2_correct.tsv',  # 修正文件名
            'test_file': 'datasets/test_2_correct.tsv'     # 修正文件名
        },
        {
            'name': 'Dataset3_StateAware',
            'queries_file': 'conversations_3_ds_correct_state_aware.jsonl',
            'train_file': 'datasets/train_3_correct.tsv',  # 修正文件名
            'test_file': 'datasets/test_3_correct.tsv'     # 修正文件名
        },
        {
            'name': 'Dataset4_StateAware',
            'queries_file': 'conversations_4_correct_state_aware.jsonl',
            'train_file': 'datasets/train_4_correct.tsv',
            'test_file': 'datasets/test_4_correct.tsv'
        }
    ]
    
    # 执行合并
    queries, train_df, test_df = merge_datasets(datasets, output_dir='datasets/WikiTable300_StateAware')
    
    # 创建corpus文件
    create_corpus_file(output_dir='datasets/WikiTable300_StateAware')
    
    print(f"\n数据集合并完成！")
    print(f"输出目录: datasets/WikiTable300_StateAware/")
    print(f"现在可以使用train_sbert.py训练模型了！")