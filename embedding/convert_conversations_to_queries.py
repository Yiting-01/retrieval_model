#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
把 conversation_wikitable-session-*.jsonl → queries.jsonl
每条记录格式：{"_id": "<递增编号>", "text": "<question>", "metadata": {}}
"""

import json
import glob
import os

INPUT_PATTERN = "/Users/yiting/Desktop/embedding/conversations_4_correct/conversation_wikitable-session-*.jsonl"
OUTPUT_FILE   = "/Users/yiting/Desktop/embedding/datasets/queries_4_correct.jsonl"

def main():
    files = sorted(glob.glob(INPUT_PATTERN))
    if not files:
        raise FileNotFoundError(f"找不到匹配 {INPUT_PATTERN} 的文件")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as out_f:
        qid = 0
        for fp in files:
            with open(fp, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue  # 跳过空行
                    convo = json.loads(line)
                    query_text = convo["question"]  # 直接取 question 字段
                    
                    out_obj = {
                        "_id": str(qid),
                        "text": query_text,
                        "metadata": {}
                    }
                    out_f.write(json.dumps(out_obj, ensure_ascii=False) + "\n")
                    qid += 1

    print(f"✅ 已写入 {qid} 条记录到 {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
