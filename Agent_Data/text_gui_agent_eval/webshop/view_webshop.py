#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pretty viewer for WebShop items_shuffle.json dataset
Usage: python view_webshop.py [index]

完整展示 WebShop 数据集的每条商品记录，格式化输出便于阅读。
由于文件较大 (5.2GB)，使用流式读取方式。
输出保存到 output.txt 文件中。
"""
import json
import sys
import os

# 默认数据路径
DEFAULT_DATA_PATH = '/mnt/petrelfs/liuhaoze/datasets/Agent_Data/webshop/items_shuffle.json'


def format_value(value, indent=0):
    """递归格式化任意值，保持完整内容"""
    prefix = "  " * indent
    
    if value is None:
        return "null"
    elif isinstance(value, bool):
        return "true" if value else "false"
    elif isinstance(value, (int, float)):
        return str(value)
    elif isinstance(value, str):
        if '\n' in value or len(value) > 100:
            lines = value.split('\n')
            if len(lines) > 1:
                result = '"""'
                for line in lines:
                    result += f"\n{prefix}  {line}"
                result += f'\n{prefix}"""'
                return result
            else:
                return f'"{value}"'
        else:
            return f'"{value}"'
    elif isinstance(value, list):
        if len(value) == 0:
            return "[]"
        if all(isinstance(v, (str, int, float, bool)) and (not isinstance(v, str) or len(str(v)) < 50) for v in value):
            formatted_items = [format_value(v, 0) for v in value]
            single_line = "[" + ", ".join(formatted_items) + "]"
            if len(single_line) < 100:
                return single_line
        result = "["
        for i, item in enumerate(value):
            formatted = format_value(item, indent + 1)
            result += f"\n{prefix}  [{i}] {formatted}"
        result += f"\n{prefix}]"
        return result
    elif isinstance(value, dict):
        if len(value) == 0:
            return "{}"
        result = "{"
        for k, v in value.items():
            formatted = format_value(v, indent + 1)
            result += f"\n{prefix}  {k}: {formatted}"
        result += f"\n{prefix}}}"
        return result
    else:
        return str(value)


def read_item_at_index(file_path, target_index):
    """流式读取 JSON 数组中指定索引的元素"""
    import ijson
    
    with open(file_path, 'rb') as f:
        parser = ijson.items(f, 'item')
        for idx, item in enumerate(parser):
            if idx == target_index:
                return item, idx
            if idx > target_index:
                break
    return None, -1


def read_item_simple(file_path, target_index):
    """简单方式读取（备用方案，适用于没有 ijson 的情况）"""
    with open(file_path, 'r', encoding='utf-8') as f:
        # 跳过开头的 [
        f.read(1)
        
        idx = 0
        while True:
            # 跳过空白和逗号
            char = f.read(1)
            while char in ' \n\t\r,':
                char = f.read(1)
            
            if char == ']':
                break
            
            if char != '{':
                continue
            
            # 读取一个完整的 JSON 对象
            depth = 1
            content = '{'
            while depth > 0:
                char = f.read(1)
                if char == '{':
                    depth += 1
                elif char == '}':
                    depth -= 1
                content += char
            
            if idx == target_index:
                return json.loads(content), idx
            
            idx += 1
            
            # 显示进度
            if idx % 10000 == 0:
                print(f"  扫描中... 已跳过 {idx} 条")
    
    return None, -1


def count_items(file_path):
    """统计总条目数（较慢，可选）"""
    try:
        import ijson
        with open(file_path, 'rb') as f:
            count = sum(1 for _ in ijson.items(f, 'item'))
        return count
    except ImportError:
        return -1  # 无法统计


def view_record(record, index, total_count=-1):
    """完整格式化显示一条记录"""
    
    lines = []
    
    def write(s=""):
        lines.append(s)
    
    total_str = f"/ {total_count-1}" if total_count > 0 else ""
    write("=" * 100)
    write(f"WebShop Item #{index} {total_str}")
    write("=" * 100)
    
    # 遍历所有字段，完整显示
    for key, value in record.items():
        write(f"\n{'─' * 100}")
        write(f"【{key}】")
        write("─" * 100)
        
        if isinstance(value, (dict, list)):
            formatted = format_value(value, 1)
            write(formatted)
        else:
            write(f"{value}")
    
    write("\n" + "=" * 100)
    
    return "\n".join(lines)


def print_usage():
    """打印使用说明"""
    print("""
WebShop Dataset Viewer - 商品数据查看器
=======================================

用法: python view_webshop.py [index]

参数:
  index             要查看的记录索引 (默认: 0)

选项:
  --help            显示此帮助信息

输出:
  结果保存到脚本同目录下的 output.txt 文件中

数据源:
  items_shuffle.json (118万+ 商品)

示例:
  python view_webshop.py          # 查看第 0 条
  python view_webshop.py 100      # 查看第 100 条
  python view_webshop.py 10000    # 查看第 10000 条

注意:
  由于文件较大，读取较远的索引可能需要一些时间。
""")


def main():
    # 解析参数
    args = sys.argv[1:]
    
    if '--help' in args or '-h' in args:
        print_usage()
        return
    
    # 默认值
    index = 0
    
    # 解析 index
    for arg in args:
        if arg.isdigit():
            index = int(arg)
            break
    
    data_path = DEFAULT_DATA_PATH
    
    if not os.path.exists(data_path):
        print(f"❌ 文件不存在: {data_path}")
        return
    
    print(f"Loading: {data_path}")
    print(f"查找第 {index} 条记录...")
    
    # 尝试使用 ijson（更快）
    try:
        import ijson
        record, actual_idx = read_item_at_index(data_path, index)
    except ImportError:
        print("  (未安装 ijson，使用备用方案，可能较慢)")
        record, actual_idx = read_item_simple(data_path, index)
    
    if record is None:
        print(f"❌ 未找到索引 {index} 的记录")
        return
    
    print(f"找到记录 #{actual_idx}")
    
    # 生成输出
    output = view_record(record, actual_idx)
    
    # 保存到文件
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, 'output.txt')
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(output)
    
    # 统计信息
    line_count = output.count('\n') + 1
    char_count = len(output)
    
    print(f"\n✅ 输出已保存到: {output_path}")
    print(f"   共 {line_count:,} 行, {char_count:,} 字符")
    
    # 打印摘要
    print(f"\n📋 商品摘要:")
    print(f"   name:     {record.get('name', 'N/A')[:60]}...")
    print(f"   asin:     {record.get('asin', 'N/A')}")
    print(f"   brand:    {record.get('brand', 'N/A')}")
    print(f"   pricing:  {record.get('pricing', 'N/A')}")
    print(f"   category: {record.get('category', 'N/A')}")
    
    images = record.get('images', [])
    print(f"   images:   {len(images)} 张")


if __name__ == "__main__":
    main()
