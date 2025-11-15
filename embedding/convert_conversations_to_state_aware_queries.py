#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量转换所有conversation文件夹为state-aware queries
"""

import json
import glob
import os
import re

# 配置所有需要处理的文件夹
DATASETS = [
    {
        'input_dir': 'conversations_1_correct',
        'output_file': 'conversations_1_correct_state_aware.jsonl'
    },
    {
        'input_dir': 'conversations_2_gpt2_correct',
        'output_file': 'conversations_2_gpt2_correct_state_aware.jsonl'
    },
    {
        'input_dir': 'conversations_3_ds_correct',
        'output_file': 'conversations_3_ds_correct_state_aware.jsonl'
    },
    {
        'input_dir': 'conversations_4_correct',
        'output_file': 'conversations_4_correct_state_aware.jsonl'
    }
]

def extract_tools_from_conversation(conversation):
    """从conversation提取工具使用轨迹"""
    tools_used = []
    
    for message in conversation.get('messages', []):
        if message.get('role') == 'assistant' and 'tool_calls' in message:
            for tool_call in message['tool_calls']:
                if 'function' in tool_call and 'name' in tool_call['function']:
                    tool_name = tool_call['function']['name']
                    # 排除最终答案
                    if tool_name != 'f_final_answer':
                        tools_used.append(tool_name)
    
    return tools_used

def extract_table_state(conversation):
    """
    从conversation的tool返回中提取表格状态信息
    """
    # 查找包含表格Shape信息的消息
    for message in conversation.get('messages', []):
        if message.get('role') == 'tool':
            content = message.get('content', '')
            
            # 提取Shape和Columns信息
            shape_match = re.search(r'Shape:\s*\((\d+),\s*(\d+)\)', content)
            cols_match = re.search(r'Columns:\s*(\[.*?\])', content)
            
            if shape_match and cols_match:
                shape = f"Shape: ({shape_match.group(1)}, {shape_match.group(2)})"
                columns = f"Columns: {cols_match.group(1)}"
                return f"{shape} | {columns} | 🔍 Data Types: | ❓ Missing Values: | Has sample data preview"
    
    return "Initial state"

def create_state_aware_text(question, tools_used, state_info="Initial state"):
    """
    创建state-aware格式的query文本
    格式参考：
    Question: how long is four hands in the metric system?
    
    Current Table State: Initial state
    
    Previous Actions: No previous actions
    
    Next Action: f_get_data_info
    """
    # 构建动作轨迹
    if not tools_used:
        actions = "No previous actions"
        next_action = "f_get_data_info"
    else:
        actions = " -> ".join(tools_used)
        # 简单的下一步预测
        if len(tools_used) < 3:
            # 早期阶段
            if 'f_get_data_info' not in tools_used:
                next_action = "f_get_data_info"
            elif 'print_table' not in tools_used:
                next_action = "print_table"
            else:
                next_action = "f_filter_rows"
        else:
            # 后期阶段
            next_action = "f_final_answer"
    
    # 组装文本
    text = f"Question: {question}\n\n"
    text += f"Current Table State: {state_info}\n\n"
    text += f"Previous Actions: {actions}\n\n"
    text += f"Next Action: {next_action}"
    
    return text

def extract_state_from_tool_message(messages, up_to_index):
    """提取到某个步骤为止的表格状态"""
    for i in range(up_to_index, -1, -1):
        message = messages[i]
        if message.get('role') == 'tool':
            content = message.get('content', '')
            
            # 提取Shape和Columns信息
            shape_match = re.search(r'Shape:\s*\((\d+),\s*(\d+)\)', content)
            cols_match = re.search(r'Columns:\s*(\[.*?\])', content)
            
            if shape_match and cols_match:
                shape = f"Shape: ({shape_match.group(1)}, {shape_match.group(2)})"
                columns = f"Columns: {cols_match.group(1)}"
                return f"{shape} | {columns} | 🔍 Data Types: | ❓ Missing Values: | Has sample data preview"
    
    return "Initial state"

def process_dataset(input_dir, output_file):
    """处理单个数据集，为每个解决步骤生成一个query"""
    print(f"\n处理数据集: {input_dir}")
    
    # 查找所有conversation文件
    pattern = os.path.join(input_dir, "conversation_wikitable-session-*.jsonl")
    files = sorted(glob.glob(pattern))
    
    if not files:
        print(f"  警告: 找不到匹配 {pattern} 的文件")
        return 0
    
    print(f"  找到 {len(files)} 个conversation文件")
    
    query_count = 0
    
    with open(output_file, "w", encoding="utf-8") as out_f:
        for fp in files:
            # 从文件名提取session ID
            basename = os.path.basename(fp)
            session_match = re.search(r'session-(\d+)', basename)
            if not session_match:
                print(f"  警告: 无法从文件名提取session ID: {basename}")
                continue
            
            session_id = session_match.group(1)
            
            with open(fp, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    
                    try:
                        convo = json.loads(line)
                    except json.JSONDecodeError as e:
                        print(f"  警告: 无法解析 {fp}: {e}")
                        continue
                    
                    question = convo.get("question", "")
                    if not question:
                        continue
                    
                    messages = convo.get('messages', [])
                    
                    # 为每个工具调用步骤生成一个query
                    tools_used_so_far = []
                    step_num = 0
                    
                    for msg_idx, message in enumerate(messages):
                        if message.get('role') == 'assistant' and 'tool_calls' in message:
                            tool_calls = message.get('tool_calls', [])
                            
                            for tool_idx, tool_call in enumerate(tool_calls):
                                if 'function' not in tool_call:
                                    continue
                                
                                tool_name = tool_call['function'].get('name', '')
                                if not tool_name or tool_name == 'f_final_answer':
                                    continue
                                
                                step_num += 2  # 每个工具调用step_num增加2
                                
                                # 获取当前状态
                                state_info = extract_state_from_tool_message(messages, msg_idx)
                                
                                # 创建state-aware文本
                                if not tools_used_so_far:
                                    actions = "No previous actions"
                                else:
                                    actions = " -> ".join(tools_used_so_far)
                                
                                next_action = tool_name
                                
                                text = f"Question: {question}\n\n"
                                text += f"Current Table State: {state_info}\n\n"
                                text += f"Previous Actions: {actions}\n\n"
                                text += f"Next Action: {next_action}"
                                
                                # 生成ID
                                query_id = f"wikitable-session-{session_id}_step_{step_num}_tool_{tool_idx}"
                                
                                out_obj = {
                                    "_id": query_id,
                                    "text": text
                                }
                                
                                out_f.write(json.dumps(out_obj, ensure_ascii=False) + "\n")
                                query_count += 1
                                
                                # 记录已使用的工具
                                tools_used_so_far.append(tool_name)
    
    print(f"  已写入 {query_count} 条state-aware记录到 {output_file}")
    return query_count

def main():
    print("开始批量转换conversation为state-aware queries...")
    print("="*60)
    
    total_queries = 0
    
    for dataset in DATASETS:
        input_dir = dataset['input_dir']
        output_file = dataset['output_file']
        
        count = process_dataset(input_dir, output_file)
        total_queries += count
    
    print("\n" + "="*60)
    print(f"转换完成！")
    print(f"总共生成 {total_queries} 条state-aware queries")
    print(f"\n生成的文件:")
    for dataset in DATASETS:
        output_file = dataset['output_file']
        if os.path.exists(output_file):
            size = os.path.getsize(output_file)
            print(f"  - {output_file} ({size:,} bytes)")
    
    print(f"\n下一步: 运行 merge_datasets_state_aware.py 合并所有数据集")

if __name__ == "__main__":
    main()