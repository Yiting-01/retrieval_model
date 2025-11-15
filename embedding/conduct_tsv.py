import json
import pandas as pd
import random
import os
import glob

'''71 querieses path
163 output name
260 conversation path
'''

def load_jsonl(filename):
    """加载JSONL文件并返回记录列表"""
    with open(filename, 'r', encoding='utf-8') as f:
        return [json.loads(line.strip()) for line in f if line.strip()]

def load_json(filename):
    """加载JSON文件"""
    with open(filename, 'r', encoding='utf-8') as f:
        return json.load(f)

def extract_actual_tools_from_conversation(conversation):
    """
    从conversation中提取实际调用的工具函数
    返回按调用顺序排列的工具名称列表
    """
    used_tools = []
    
    for message in conversation.get('messages', []):
        if message.get('role') == 'assistant' and 'tool_calls' in message:
            for tool_call in message['tool_calls']:
                function_name = tool_call.get('function', {}).get('name', '')
                if function_name:
                    used_tools.append(function_name)
    
    return used_tools

def map_tool_name_to_corpus_id(tool_name):
    """
    将工具名称映射到corpus中的ID
    """
    tool_mapping = {
        'f_select_column': 0,
        'f_group_by': 1,
        'print_table': 2,
        'f_calculate_average': 3,
        'f_filter_rows': 4,
        'f_get_data_info': 5,
        'f_select_row': 6,
        'f_sort_by': 7,
        'f_aggregate': 8,
        'f_compute_column': 9,
        'f_distinct_count': 10,
        'f_final_answer': 11,
        'f_process_datetime': 12,
        'f_string_operation': 13,
        'f_undo': 14
    }
    return tool_mapping.get(tool_name, None)

def find_query_id_by_question(question, queries):
    """
    根据问题文本找到对应的query_id
    """
    question = question.strip()
    for query in queries:
        if query['text'].strip() == question:
            return int(query['_id'])
    return None

def create_tsv_from_conversations(conversation_files, queries_file='datasets/queries_3_correct.jsonl', 
                                corpus_file='datasets/corpus.jsonl', train_ratio=0.7, random_seed=42):
    """
    从conversation文件中提取实际工具调用记录来创建训练数据
    只使用conversation中真实调用的工具，不做任何预测
    """
    
    # 加载queries和corpus
    print("正在加载数据文件...")
    queries = load_jsonl(queries_file)
    corpus = load_jsonl(corpus_file)
    print(f"已加载 {len(queries)} 个查询和 {len(corpus)} 个语料库文档")
    
    # 从conversation中提取实际工具使用记录
    conversation_data = {}  # query_id -> [actual_tool_ids]
    
    print("正在分析conversation文件...")
    processed_conversations = 0
    
    for conv_file in conversation_files:
        if not os.path.exists(conv_file):
            print(f"警告: 文件 {conv_file} 不存在，跳过")
            continue
            
        # 根据文件扩展名选择加载方式
        if conv_file.endswith('.jsonl'):
            conversations = load_jsonl(conv_file)
        else:
            # 单个JSON对象
            conversations = [load_json(conv_file)]
        
        for conv in conversations:
            # 提取问题和实际使用的工具
            question = conv.get('question', '')
            if not question:
                continue
                
            # 提取实际调用的工具
            used_tools = extract_actual_tools_from_conversation(conv)
            if not used_tools:
                print(f"警告: conversation中没有找到工具调用: {question[:50]}...")
                continue
            
            # 映射工具名到corpus ID
            tool_ids = []
            for tool_name in used_tools:
                tool_id = map_tool_name_to_corpus_id(tool_name)
                if tool_id is not None:
                    tool_ids.append(tool_id)
                else:
                    print(f"警告: 未知工具名 '{tool_name}'")
            
            # 找到对应的query_id
            query_id = find_query_id_by_question(question, queries)
            if query_id is not None:
                conversation_data[query_id] = tool_ids
                processed_conversations += 1
                print(f"✓ 查询 {query_id}: {len(tool_ids)} 个工具调用")
            else:
                print(f"警告: 未找到匹配的query_id: {question[:50]}...")
    
    print(f"\n成功处理了 {processed_conversations} 个conversation记录")
    
    if not conversation_data:
        print("错误: 没有提取到任何有效的工具调用记录！")
        return None, None
    
    # 生成训练数据
    mappings = []
    for query_id, tool_ids in conversation_data.items():
        for tool_id in tool_ids:
            mappings.append([query_id, tool_id, 1])
    
    print(f"生成了 {len(mappings)} 条query-tool映射记录")
    
    # 创建DataFrame
    df = pd.DataFrame(mappings, columns=['query-id', 'corpus-id', 'score'])
    
    # 按查询ID分割训练和测试集
    query_ids = list(conversation_data.keys())
    random.seed(random_seed)
    random.shuffle(query_ids)
    
    split_point = int(len(query_ids) * train_ratio)
    train_query_ids = set(query_ids[:split_point])
    test_query_ids = set(query_ids[split_point:])
    
    # 创建训练集和测试集
    train_df = df[df['query-id'].isin(train_query_ids)].sort_values(['query-id', 'corpus-id']).reset_index(drop=True)
    test_df = df[df['query-id'].isin(test_query_ids)].sort_values(['query-id', 'corpus-id']).reset_index(drop=True)
    
    # 保存文件
    train_df.to_csv('datasets/train_3_correct.tsv', sep='\t', index=False)
    test_df.to_csv('datasets/test_3_correct.tsv', sep='\t', index=False)
    
    # 报告结果
    print(f"\n=== 结果统计 ===")
    print(f"train.tsv: {len(train_df)} 行数据, {len(train_df['query-id'].unique())} 个查询")
    print(f"test.tsv: {len(test_df)} 行数据, {len(test_df['query-id'].unique())} 个查询")
    
    print(f"\n=== train.tsv 示例 ===")
    print(train_df.head(15).to_string(index=False))
    
    print(f"\n=== test.tsv 示例 ===")
    print(test_df.head(10).to_string(index=False))
    
    # 显示实际的工具调用序列
    print(f"\n=== 实际工具调用序列示例 ===")
    sample_queries = list(conversation_data.keys())[:3]
    
    # 创建反向映射字典
    id_to_tool_name = {
        0: 'f_select_column',
        1: 'f_group_by', 
        2: 'print_table',
        3: 'f_calculate_average',
        4: 'f_filter_rows',
        5: 'f_get_data_info',
        6: 'f_select_row',
        7: 'f_sort_by',
        8: 'f_aggregate',
        9: 'f_compute_column',
        10: 'f_distinct_count',
        11: 'f_final_answer',
        12: 'f_process_datetime',
        13: 'f_string_operation',
        14: 'f_undo'
    }
    
    for query_id in sample_queries:
        # 找到对应的query文本
        query_text = next((q['text'] for q in queries if int(q['_id']) == query_id), "未找到")
        tool_ids = conversation_data[query_id]
        
        print(f"查询 {query_id}: '{query_text[:60]}...'")
        print(f"  -> 实际调用工具序列: {tool_ids}")
        
        # 显示工具名称
        tool_names = [id_to_tool_name.get(tid, f"未知工具{tid}") for tid in tool_ids]
        print(f"  -> 工具名称: {' -> '.join(tool_names)}")
        print()
    
    return train_df, test_df

def batch_process_conversations(conversation_dir, pattern='conversation_wikitable-session-*.jsonl'):
    """
    批量处理conversation文件夹中的所有文件
    """
    # 检查文件夹是否存在
    if not os.path.exists(conversation_dir):
        print(f"错误: 文件夹 '{conversation_dir}' 不存在！")
        print(f"当前工作目录: {os.getcwd()}")
        print("请检查文件夹路径是否正确")
        return None, None
    
    # 构建完整的文件路径模式
    full_pattern = os.path.join(conversation_dir, pattern)
    conversation_files = glob.glob(full_pattern)
    
    print(f"在文件夹 '{conversation_dir}' 中找到 {len(conversation_files)} 个conversation文件")
    
    if not conversation_files:
        print(f"错误: 在 '{conversation_dir}' 文件夹中没有找到匹配 '{pattern}' 的文件！")
        print("请检查:")
        print("1. 文件夹路径是否正确")
        print("2. 文件命名格式是否匹配")
        
        # 显示文件夹中实际存在的文件
        try:
            actual_files = os.listdir(conversation_dir)
            print(f"3. 文件夹中实际存在的文件（前10个）:")
            for i, file in enumerate(actual_files[:10]):
                print(f"   - {file}")
            if len(actual_files) > 10:
                print(f"   ... 还有 {len(actual_files) - 10} 个文件")
        except Exception as e:
            print(f"   无法读取文件夹内容: {e}")
            
        return None, None
    
    # 按文件名排序，确保处理顺序一致
    conversation_files.sort()
    print(f"文件范围: {os.path.basename(conversation_files[0])} 到 {os.path.basename(conversation_files[-1])}")
    
    return create_tsv_from_conversations(conversation_files)

# 使用示例
if __name__ == "__main__":
    # 自动处理conversations文件夹中的所有文件
    conversation_dir = 'conversations_3_ds_correct'  #conversation path
    pattern = 'conversation_wikitable-session-*.jsonl'  
    
    train_df, test_df = batch_process_conversations(
        conversation_dir=conversation_dir,
        pattern=pattern
    )