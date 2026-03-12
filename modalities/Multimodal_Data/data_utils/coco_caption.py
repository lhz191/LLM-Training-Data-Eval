import os
import json
import torch
from PIL import Image
from typing import Optional


class COCO_Caption(torch.utils.data.Dataset):
    """COCO Caption 数据集加载器
    
    用于加载 COCO 数据集的图片和对应的描述文本，并支持生成评测用的 JSON 文件。
    """
    
    def __init__(self,
                 configs,
                 transform=None):
        """
        初始化 COCO Caption 数据集
        
        Args:
            configs: 配置对象（可选）
            transform: 图片变换函数（可选）
        """
        self.configs = configs
        self.transform = transform
        # 使用相对于当前文件的路径，或从环境变量/配置中获取
        # 获取当前文件所在目录，向上找到 LLMDataBenchmark 根目录
        # 当前文件在: LLMDataBenchmark/Multimodal/data_utils/coco_caption.py
        # 需要向上2级到 LLMDataBenchmark 根目录
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(current_dir, "../.."))
        self.root = os.path.join(project_root, "Datasets", "Multimodal")
        self.img_root = os.path.join(self.root, "train2014")
        
        # 加载 COCO 标注文件
  
        
        # 构建图片 ID 到文件名的映射
        self.id2img = {
            img["id"]: img["file_name"]
            for img in self.coco["images"]
        }
        
        # 构建样本列表
        self.samples = []
        for ann in self.coco["annotations"]:
            self.samples.append({
                "annotation_id": ann["id"],  # annotation ID，用于 JSON 文件中的唯一标识
                "image_id": ann["image_id"],  # 图片 ID
                "caption": ann["caption"],     # 图片描述文本
            })

    def __getitem__(self, index):
        """获取指定索引的数据样本
        
        Args:
            index: 样本索引
            
        Returns:
            dict: 包含 'image' 和 'caption' 的字典
        """
        img_id = self.samples[index]["image_id"]
        caption = self.samples[index]["caption"]
        
        # 加载图片
        img_path = os.path.join(self.img_root, self.id2img[img_id])
        image = Image.open(img_path).convert("RGB")
        
        # 应用变换（如果提供）
        if self.transform is not None:
            image = self.transform(image)
        
        return {
            "image": image,
            "caption": caption
        }

    def __len__(self):
        """返回数据集大小"""
        return len(self.samples)
    
    def generate_evaluation_json(self, output_file: Optional[str] = None) -> str:
        """生成用于评测的 JSON 文件
        
        生成的 JSON 文件包含每条数据的 id、image_id、image_path 和 caption，
        可用于后续的模型评测。
        
        Args:
            output_file: 输出文件路径，如果为 None 则使用默认路径
                        (LLMDataBenchmark/Datasets/Multimodal/coco_caption.json)
            
        Returns:
            str: 生成的 JSON 文件路径
        """
        if output_file is None:
            output_file = os.path.join(self.root, "coco_caption.json")
        
        evaluation_data = []
        
        for sample in self.samples:
            img_id = sample["image_id"]
            img_filename = self.id2img[img_id]
            
            # 构建相对路径（相对于数据集根目录）
            img_path = os.path.join("train2014", img_filename).replace("\\", "/")
            
            # 检查图片文件是否存在
            full_img_path = os.path.join(self.img_root, img_filename)
            if not os.path.exists(full_img_path):
                print(f"Warning: Image file not found: {full_img_path}, skipping...")
                continue
            
            # 构建评测数据项
            evaluation_item = {
                "id": sample["annotation_id"],  # 唯一标识符
                "image_id": img_id,             # 图片 ID
                "image_path": img_path,         # 图片路径（相对路径）
                "caption": sample["caption"],   # 图片描述（ground truth）
            }
            
            evaluation_data.append(evaluation_item)
        
        # 保存 JSON 文件
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(evaluation_data, f, ensure_ascii=False, indent=2)
        
        print(f"✓ 成功生成评测 JSON 文件: {output_file}")
        print(f"  - 总样本数: {len(evaluation_data)}")
        
        return output_file


if __name__ == '__main__':
    """主函数：用于测试数据集加载和生成评测 JSON 文件"""
    import argparse

    parser = argparse.ArgumentParser(description="COCO Caption 数据集测试脚本")
    # 自动检测数据集路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, "../.."))
    default_data_path = os.path.join(project_root, "Datasets", "Multimodal")
    
    parser.add_argument("--data-path", type=str, default=default_data_path,
                       help="数据集路径（当前未使用，保留用于兼容性）")
    args = parser.parse_args()

    # 创建数据集实例
    ds = COCO_Caption(None)
    print(f"数据集加载完成，总样本数: {len(ds)}")
    
    # 生成评测 JSON 文件
    output_file = ds.generate_evaluation_json()
    print(f"\n评测 JSON 文件已生成: {output_file}")
    
    # 测试数据加载
    print("\n测试数据加载...")
    sample = ds[0]
    print(f"样本键: {sample.keys()}")
    print(f"Caption: {sample['caption'][:100]}...")  # 只打印前100个字符