#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
from pathlib import Path
from typing import List

from enhanced_modular_tools import ENHANCED_MODULAR_TOOLS  

def build_entry(tool, idx: int) -> dict:
    """将单个 Tool 对象转为 corpus 条目。"""
    name = tool.name
    desc = " ".join(tool.description.split())          # 压缩多余空格/换行

    schema = tool.args_schema.schema()
    required = schema.get("required", [])
    optional = [k for k in schema.get("properties", {}) if k not in required]

    text = (
        f"{name}: {desc} "
        f"Required parameters: {', '.join(required) if required else '—'}. "
        f"Optional parameters: {', '.join(optional) if optional else '—'}."
    )

    return {
        "_id": str(idx),
        "title": "",
        "text": text,
        "metadata": {}
    }

def main(out_path: Path = Path("/Users/yiting/Desktop/embedding/datasets/corpus.jsonl")):
    entries: List[dict] = [
        build_entry(tool, idx) for idx, tool in enumerate(ENHANCED_MODULAR_TOOLS)
    ]

    with out_path.open("w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"✅ 已写入 {len(entries)} 条工具描述 → {out_path}")

if __name__ == "__main__":
    main()
