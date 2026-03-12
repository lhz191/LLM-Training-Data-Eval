"""
浏览器标注服务器

参照 Chameleon miniviewer 的方式，提供浏览器标注界面
"""

import json
import os
from flask import Flask, render_template_string, request, jsonify, send_from_directory
from typing import Dict, Any, Optional


def create_annotation_data(
    generation_file: str,
    ground_truth_file: str,
    output_file: str,
    dataset_root: Optional[str] = None,
):
    """
    创建标注数据文件
    
    Args:
        generation_file: 模型生成结果文件路径
        ground_truth_file: Ground truth 文件路径
        output_file: 输出标注数据文件路径
        dataset_root: 数据集根目录（可选）
    """
    with open(generation_file, "r", encoding="utf-8") as f:
        gen_data = json.load(f)
    
    with open(ground_truth_file, "r", encoding="utf-8") as f:
        gt_data = json.load(f)
    
    # 创建映射
    if isinstance(gt_data, list):
        gt_map = {
            item.get("id") or item.get("image_id"): item.get("caption", "")
            for item in gt_data
        }
    else:
        gt_map = {}
    
    items = []
    for result in gen_data.get("results", []):
        item_id = result.get("id") or result.get("image_id")
        items.append({
            "id": item_id,
            "image_path": result.get("image_path", ""),
            "generated": result.get("generated", ""),
            "ground_truth": gt_map.get(item_id, ""),
            "comparison": None,
        })
    
    annotation_data = {
        "model": gen_data.get("model", "unknown"),
        "dataset_root": dataset_root or "",
        "items": items,
    }
    
    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(annotation_data, f, ensure_ascii=False, indent=2)
    
    print(f"Annotation data created: {output_file} ({len(items)} items)")


app = Flask(__name__)


HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Win Rate Annotation</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #f5f5f5;
            padding: 20px;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            padding: 30px;
        }
        h1 {
            color: #333;
            margin-bottom: 10px;
        }
        .info {
            color: #666;
            margin-bottom: 30px;
            padding: 15px;
            background: #f9f9f9;
            border-radius: 4px;
        }
        .item {
            border: 2px solid #e0e0e0;
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 30px;
            background: #fafafa;
        }
        .item.active {
            border-color: #4CAF50;
            background: white;
        }
        .item-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 15px;
        }
        .item-id {
            font-weight: bold;
            color: #666;
        }
        .progress {
            color: #999;
            font-size: 14px;
        }
        .image-container {
            text-align: center;
            margin: 20px 0;
        }
        .image-container img {
            max-width: 100%;
            max-height: 500px;
            border-radius: 4px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }
        .caption-box {
            background: white;
            border: 1px solid #ddd;
            border-radius: 4px;
            padding: 15px;
            margin: 15px 0;
        }
        .caption-label {
            font-weight: bold;
            color: #555;
            margin-bottom: 8px;
            font-size: 14px;
        }
        .caption-text {
            color: #333;
            line-height: 1.6;
        }
        .buttons {
            display: flex;
            gap: 10px;
            margin-top: 20px;
            justify-content: center;
        }
        .btn {
            padding: 12px 30px;
            border: none;
            border-radius: 4px;
            font-size: 16px;
            font-weight: bold;
            cursor: pointer;
            transition: all 0.2s;
        }
        .btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        }
        .btn-win {
            background: #4CAF50;
            color: white;
        }
        .btn-tie {
            background: #FFC107;
            color: #333;
        }
        .btn-loss {
            background: #F44336;
            color: white;
        }
        .btn.selected {
            box-shadow: 0 0 0 3px rgba(0,0,0,0.2);
        }
        .status {
            text-align: center;
            margin-top: 15px;
            font-weight: bold;
            color: #666;
        }
        .status.win { color: #4CAF50; }
        .status.tie { color: #FFC107; }
        .status.loss { color: #F44336; }
        .navigation {
            display: flex;
            justify-content: space-between;
            margin-top: 30px;
            padding-top: 20px;
            border-top: 1px solid #e0e0e0;
        }
        .nav-btn {
            padding: 10px 20px;
            background: #2196F3;
            color: white;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-size: 14px;
        }
        .nav-btn:hover {
            background: #1976D2;
        }
        .nav-btn:disabled {
            background: #ccc;
            cursor: not-allowed;
        }
        .save-status {
            position: fixed;
            top: 20px;
            right: 20px;
            padding: 10px 20px;
            background: #4CAF50;
            color: white;
            border-radius: 4px;
            display: none;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Win Rate Annotation Tool</h1>
        <div class="info">
            <strong>Model:</strong> {{ model }} | 
            <strong>Total Items:</strong> {{ total }} | 
            <strong>Annotated:</strong> <span id="annotated-count">0</span> / {{ total }}
        </div>
        
        <div id="items-container">
            <!-- Items will be loaded here -->
        </div>
        
        <div class="navigation">
            <button class="nav-btn" id="prev-btn" onclick="prevItem()">← Previous</button>
            <button class="nav-btn" id="next-btn" onclick="nextItem()">Next →</button>
        </div>
    </div>
    
    <div class="save-status" id="save-status">Saved!</div>
    
    <script>
        let items = {{ items|tojson }};
        let datasetRoot = "{{ dataset_root }}";
        let currentIndex = 0;
        
        function renderItem(index) {
            if (index < 0 || index >= items.length) return;
            
            currentIndex = index;
            const item = items[index];
            const imageUrl = `/images/${item.image_path}`;
            
            document.getElementById('items-container').innerHTML = `
                <div class="item active">
                    <div class="item-header">
                        <span class="item-id">Item ${index + 1} / ${items.length} (ID: ${item.id})</span>
                        <span class="progress">${index + 1} / ${items.length}</span>
                    </div>
                    
                    <div class="image-container">
                        <img src="${imageUrl}" alt="Image" onerror="this.src='data:image/svg+xml,%3Csvg xmlns=\\'http://www.w3.org/2000/svg\\' width=\\'400\\' height=\\'300\\'%3E%3Ctext x=\\'50%25\\' y=\\'50%25\\' text-anchor=\\'middle\\'%3EImage not found%3C/text%3E%3C/svg%3E'">
                    </div>
                    
                    <div class="caption-box">
                        <div class="caption-label">Generated Caption:</div>
                        <div class="caption-text">${item.generated || '(empty)'}</div>
                    </div>
                    
                    <div class="caption-box">
                        <div class="caption-label">Ground Truth:</div>
                        <div class="caption-text">${item.ground_truth || '(empty)'}</div>
                    </div>
                    
                    <div class="buttons">
                        <button class="btn btn-win ${item.comparison === 'win' ? 'selected' : ''}" 
                                onclick="setComparison('win')">Win</button>
                        <button class="btn btn-tie ${item.comparison === 'tie' ? 'selected' : ''}" 
                                onclick="setComparison('tie')">Tie</button>
                        <button class="btn btn-loss ${item.comparison === 'loss' ? 'selected' : ''}" 
                                onclick="setComparison('loss')">Loss</button>
                    </div>
                    
                    <div class="status ${item.comparison || ''}">
                        ${item.comparison ? `Current: ${item.comparison.toUpperCase()}` : 'Not annotated'}
                    </div>
                </div>
            `;
            
            updateNavigation();
            updateAnnotatedCount();
        }
        
        function setComparison(value) {
            items[currentIndex].comparison = value;
            saveAnnotations();
            renderItem(currentIndex);
        }
        
        function prevItem() {
            if (currentIndex > 0) {
                renderItem(currentIndex - 1);
            }
        }
        
        function nextItem() {
            if (currentIndex < items.length - 1) {
                renderItem(currentIndex + 1);
            }
        }
        
        function updateNavigation() {
            document.getElementById('prev-btn').disabled = currentIndex === 0;
            document.getElementById('next-btn').disabled = currentIndex === items.length - 1;
        }
        
        function updateAnnotatedCount() {
            const count = items.filter(item => item.comparison).length;
            document.getElementById('annotated-count').textContent = count;
        }
        
        function saveAnnotations() {
            fetch('/save', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({items: items})
            }).then(() => {
                const status = document.getElementById('save-status');
                status.style.display = 'block';
                setTimeout(() => status.style.display = 'none', 2000);
            });
        }
        
        // Keyboard shortcuts
        document.addEventListener('keydown', (e) => {
            if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
            if (e.key === 'ArrowLeft') prevItem();
            if (e.key === 'ArrowRight') nextItem();
            if (e.key === '1') setComparison('win');
            if (e.key === '2') setComparison('tie');
            if (e.key === '3') setComparison('loss');
        });
        
        // Initialize
        renderItem(0);
    </script>
</body>
</html>
"""


@app.route('/')
def index():
    """主页面"""
    annotation_file = app.config.get('ANNOTATION_FILE')
    dataset_root = app.config.get('DATASET_ROOT', '')
    
    with open(annotation_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    return render_template_string(
        HTML_TEMPLATE,
        model=data.get('model', 'unknown'),
        total=len(data.get('items', [])),
        items=data.get('items', []),
        dataset_root=dataset_root
    )


@app.route('/save', methods=['POST'])
def save():
    """保存标注"""
    annotation_file = app.config.get('ANNOTATION_FILE')
    data = request.json
    
    # 加载原文件
    with open(annotation_file, 'r', encoding='utf-8') as f:
        file_data = json.load(f)
    
    # 更新 items
    file_data['items'] = data['items']
    
    # 保存
    with open(annotation_file, 'w', encoding='utf-8') as f:
        json.dump(file_data, f, ensure_ascii=False, indent=2)
    
    return jsonify({'status': 'saved'})


@app.route('/images/<path:filename>')
def serve_image(filename):
    """提供图片服务"""
    dataset_root = app.config.get('DATASET_ROOT', '')
    if dataset_root:
        return send_from_directory(dataset_root, filename)
    return send_from_directory('.', filename)


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description="启动标注服务器")
    parser.add_argument('--annotation-file', required=True, help='标注数据文件路径')
    parser.add_argument('--dataset-root', default='', help='数据集根目录（用于图片路径）')
    parser.add_argument('--port', type=int, default=5000, help='服务器端口')
    parser.add_argument('--host', default='0.0.0.0', help='服务器地址')
    
    args = parser.parse_args()
    
    app.config['ANNOTATION_FILE'] = args.annotation_file
    app.config['DATASET_ROOT'] = args.dataset_root
    
    print(f"Starting annotation server...")
    print(f"Annotation file: {args.annotation_file}")
    print(f"Dataset root: {args.dataset_root}")
    print(f"Open browser at: http://localhost:{args.port}/")
    
    app.run(host=args.host, port=args.port, debug=False)

