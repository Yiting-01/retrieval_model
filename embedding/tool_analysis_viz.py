#!/usr/bin/env python3
"""
在def main()改model
图1: 工具在标注中的使用频次统计
图2: 应该在Top-5但缺失的工具次数统计
图3: Top-5命中率分布
"""

import json
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from collections import defaultdict, Counter
import os

# 设置中文字体和样式
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']  # 使用英文字体
# plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")

def load_analysis_data(json_file):
    """加载分析结果JSON文件"""
    with open(json_file, 'r', encoding='utf-8') as f:
        return json.load(f)

def analyze_tool_usage(analysis_data):
    """分析工具使用情况"""
    
    # 统计标注中每个工具被使用的次数
    annotated_tool_count = Counter()
    
    # 统计应该在Top-5但没有出现的工具次数
    missing_in_top5_count = Counter()
    
    # 统计Top-5中相关工具的命中情况
    top5_hit_stats = {
        'total_queries': 0,
        'queries_with_all_relevant_in_top5': 0,
        'queries_with_some_relevant_in_top5': 0,
        'queries_with_no_relevant_in_top5': 0
    }
    
    for query_id, query_data in analysis_data.get('query_analysis', {}).items():
        top5_hit_stats['total_queries'] += 1
        
        # 获取标注的相关工具
        relevant_tools = set()
        for tool_perf in query_data.get('relevant_tools_performance', []):
            tool_id = tool_perf['tool_id']
            annotated_tool_count[tool_id] += 1
            relevant_tools.add(tool_id)
        
        # 获取Top-5推荐的工具
        top5_tools = set()
        for tool in query_data.get('top_k_tools', [])[:5]:  # 只看前5个
            if tool['is_relevant']:
                top5_tools.add(tool['tool_id'])
        
        # 计算应该在Top-5但没有出现的工具
        missing_tools = relevant_tools - top5_tools
        for tool_id in missing_tools:
            missing_in_top5_count[tool_id] += 1
        
        # 统计Top-5命中情况
        if len(relevant_tools) == 0:
            continue  # 跳过没有标注相关工具的查询
            
        if relevant_tools.issubset(top5_tools):
            top5_hit_stats['queries_with_all_relevant_in_top5'] += 1
        elif len(relevant_tools & top5_tools) > 0:
            top5_hit_stats['queries_with_some_relevant_in_top5'] += 1
        else:
            top5_hit_stats['queries_with_no_relevant_in_top5'] += 1
    
    return annotated_tool_count, missing_in_top5_count, top5_hit_stats

def get_tool_names():
    """获取工具ID到名称的映射"""
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

def create_visualizations(annotated_count, missing_count, hit_stats, json_filename, output_dir='analysis'):
    """创建可视化图表"""
    
    tool_names = get_tool_names()
    os.makedirs(output_dir, exist_ok=True)
    # 从JSON文件名提取前缀
    file_prefix = json_filename.replace('.json', '')
    
    # 图1: 标注中工具使用次数
    plt.figure(figsize=(14, 8))
    
    # 准备数据
    all_tools = list(tool_names.keys())
    annotated_counts = [annotated_count.get(tool_id, 0) for tool_id in all_tools]
    tool_labels = [tool_names[tool_id].replace('f_', '') for tool_id in all_tools]
    
    # 创建颜色映射
    colors = plt.cm.Set3(np.linspace(0, 1, len(all_tools)))
    
    bars = plt.bar(range(len(all_tools)), annotated_counts, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
    
    plt.title('Tool Usage Frequency in Annotation Data', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Tool Names', fontsize=12)
    plt.ylabel('Usage Count', fontsize=12)
    plt.xticks(range(len(all_tools)), tool_labels, rotation=45, ha='right')
    
    # 在柱子上显示数值
    for i, bar in enumerate(bars):
        height = bar.get_height()
        if height > 0:
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{int(height)}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.grid(axis='y', alpha=0.3)
    plt.savefig(f'{output_dir}/{file_prefix}_tool_annotation_frequency.png', dpi=300, bbox_inches='tight')
    print(f"✅ 图1已保存: {output_dir}/{file_prefix}_tool_annotation_frequency.png")
    plt.show()
    
    # 图2: 应该在Top-5但缺失的工具次数
    plt.figure(figsize=(14, 8))
    
    missing_counts = [missing_count.get(tool_id, 0) for tool_id in all_tools]
    
    # 只显示有缺失的工具
    non_zero_indices = [i for i, count in enumerate(missing_counts) if count > 0]
    if non_zero_indices:
        filtered_labels = [tool_labels[i] for i in non_zero_indices]
        filtered_counts = [missing_counts[i] for i in non_zero_indices]
        filtered_colors = [colors[i] for i in non_zero_indices]
        
        bars = plt.bar(range(len(filtered_labels)), filtered_counts, 
                      color=filtered_colors, alpha=0.8, edgecolor='black', linewidth=0.5)
        
        plt.title('Tools Missing from Top-5 Recommendations', fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Tool Names', fontsize=12)
        plt.ylabel('Missing Count', fontsize=12)
        plt.xticks(range(len(filtered_labels)), filtered_labels, rotation=45, ha='right')
        
        # 在柱子上显示数值
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{int(height)}', ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        plt.grid(axis='y', alpha=0.3)
    else:
        plt.text(0.5, 0.5, 'No tools missing from Top-5!\nPerfect model performance!',
                ha='center', va='center', transform=plt.gca().transAxes, 
                fontsize=20, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen"))
        plt.title('Tools Missing from Top-5 Recommendations', fontsize=16, fontweight='bold')
    
    plt.savefig(f'{output_dir}/{file_prefix}_tool_missing_in_top5.png', dpi=300, bbox_inches='tight')
    print(f"✅ 图2已保存: {output_dir}/{file_prefix}_tool_missing_in_top5.png")
    plt.show()
    
    # 图3: Top-5命中率饼图
    plt.figure(figsize=(10, 8))
    
    labels = ['All Relevant in Top-5', 'Some Relevant in Top-5', 'No Relevant in Top-5']
    sizes = [
        hit_stats['queries_with_all_relevant_in_top5'],
        hit_stats['queries_with_some_relevant_in_top5'], 
        hit_stats['queries_with_no_relevant_in_top5']
    ]
    colors_pie = ['lightgreen', 'orange', 'lightcoral']
    explode = (0.05, 0.05, 0.1)  # 突出显示最后一个
    
    wedges, texts, autotexts = plt.pie(sizes, labels=labels, colors=colors_pie, autopct='%1.1f%%',
                                      explode=explode, shadow=True, startangle=90)
    
    plt.title('Top-5 Recommendation Hit Rate Distribution', fontsize=16, fontweight='bold', pad=20)
    
    # 美化文字
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontsize(12)
        autotext.set_fontweight('bold')
    
    plt.axis('equal')
    plt.savefig(f'{output_dir}/{file_prefix}_top5_hit_rate_distribution.png', dpi=300, bbox_inches='tight')
    print(f"✅ 图3已保存: {output_dir}/{file_prefix}_top5_hit_rate_distribution.png")
    plt.show()
    
# 图4: 工具遗漏率
    plt.figure(figsize=(14, 8))
    
    # 计算每个工具的遗漏率
    miss_rates = []
    miss_rate_labels = []
    miss_rate_colors = []
    
    for i, tool_id in enumerate(all_tools):
        total_needed = annotated_count.get(tool_id, 0)  # 总共需要使用的次数
        missed = missing_count.get(tool_id, 0)          # 遗漏的次数
        
        if total_needed > 0:  # 只显示被标注使用过的工具
            miss_rate = (missed / total_needed) * 100  # 遗漏率百分比
            miss_rates.append(miss_rate)
            miss_rate_labels.append(tool_labels[i])
            miss_rate_colors.append(colors[i])
    
    if miss_rates:
        # 按遗漏率排序
        sorted_data = sorted(zip(miss_rates, miss_rate_labels, miss_rate_colors), reverse=True)
        sorted_rates, sorted_labels, sorted_colors = zip(*sorted_data)
        
        bars = plt.bar(range(len(sorted_rates)), sorted_rates, 
                      color=sorted_colors, alpha=0.8, edgecolor='black', linewidth=0.5)
        
        plt.title('Tool Miss Rate in Top-5 Recommendations', fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Tool Names', fontsize=12)
        plt.ylabel('Miss Rate (%)', fontsize=12)
        plt.xticks(range(len(sorted_labels)), sorted_labels, rotation=45, ha='right')
        
        # 在柱子上显示数值
        for bar, rate in zip(bars, sorted_rates):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{rate:.1f}%', ha='center', va='bottom', fontsize=10)
        
        # 添加参考线
        plt.axhline(y=20, color='orange', linestyle='--', alpha=0.7, label='20% threshold')
        plt.axhline(y=10, color='green', linestyle='--', alpha=0.7, label='10% threshold')
        plt.legend()
        
        plt.tight_layout()
        plt.grid(axis='y', alpha=0.3)
        plt.ylim(0, max(sorted_rates) * 1.1)
    else:
        plt.text(0.5, 0.5, 'Perfect Performance!\nNo tools missed!',
                ha='center', va='center', transform=plt.gca().transAxes, 
                fontsize=20, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen"))
        plt.title('Tool Miss Rate in Top-5 Recommendations', fontsize=16, fontweight='bold')
    
    plt.savefig(f'{output_dir}/{file_prefix}_tool_miss_rate.png', dpi=300, bbox_inches='tight')
    print(f"✅ 图4已保存: {output_dir}/{file_prefix}_tool_miss_rate.png")
    plt.show()

def print_statistics(annotated_count, missing_count, hit_stats):
    """打印详细统计信息"""
    tool_names = get_tool_names()
    
    print("\n" + "="*60)
    print("📊 工具使用统计分析")
    print("="*60)
    
    print(f"\n📋 总查询数: {hit_stats['total_queries']}")
    
    print(f"\n🏆 标注中最常用的工具 (Top 5):")
    top_annotated = annotated_count.most_common(5)
    for i, (tool_id, count) in enumerate(top_annotated, 1):
        print(f"  {i}. {tool_names[tool_id]}: {count}次")
    
    if missing_count:
        print(f"\n❌ Top-5中缺失最多的工具:")
        top_missing = missing_count.most_common(5)
        for i, (tool_id, count) in enumerate(top_missing, 1):
            print(f"  {i}. {tool_names[tool_id]}: 缺失{count}次")
    else:
        print(f"\n✅ 所有相关工具都在Top-5中，没有缺失！")
    
    print(f"\n📈 Top-5命中率统计:")
    total = hit_stats['total_queries']
    if total > 0:
        all_hit_rate = hit_stats['queries_with_all_relevant_in_top5'] / total * 100
        some_hit_rate = hit_stats['queries_with_some_relevant_in_top5'] / total * 100
        no_hit_rate = hit_stats['queries_with_no_relevant_in_top5'] / total * 100
        
        print(f"  ✅ 完全命中: {hit_stats['queries_with_all_relevant_in_top5']}个查询 ({all_hit_rate:.1f}%)")
        print(f"  ⚠️  部分命中: {hit_stats['queries_with_some_relevant_in_top5']}个查询 ({some_hit_rate:.1f}%)")
        print(f"  ❌ 完全未命中: {hit_stats['queries_with_no_relevant_in_top5']}个查询 ({no_hit_rate:.1f}%)")

def main():
    """主函数"""
    print("🔍 工具使用情况分析开始...")
    
    # 查找最新的分析结果文件
    analysis_dir = "analysis"
    json_files = ['ance/train_correct_all/msmarco-roberta-base-ance-firstpWikiTable300-TQA_20250816_120202_bigcorrectquery_train_20250816_141221.json']
    # json_files = [f for f in os.listdir(analysis_dir) if f.endswith('.json') and 'embedding_analysis' in f]
    
    if not json_files:
        print("❌ 未找到分析结果JSON文件，请先运行embedding分析！")
        return
    
    # 使用最新的文件
    latest_file = max(json_files, key=lambda x: os.path.getctime(os.path.join(analysis_dir, x)))
    json_path = os.path.join(analysis_dir, latest_file)
    
    print(f"📁 使用分析文件: {json_path}")
    
    # 加载数据
    analysis_data = load_analysis_data(json_path)
    
    # 分析工具使用情况
    annotated_count, missing_count, hit_stats = analyze_tool_usage(analysis_data)
    
    # 打印统计信息
    print_statistics(annotated_count, missing_count, hit_stats)
    
    # 创建可视化
    print(f"\n🎨 生成可视化图表...")
    create_visualizations(annotated_count, missing_count, hit_stats, latest_file)
    
    print(f"\n🎉 分析完成！图表已保存到 analysis/ 目录")

if __name__ == "__main__":
    main()