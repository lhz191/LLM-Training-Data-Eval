import json
import torch
import clip
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from tqdm import tqdm 
# ----------------------------
# 1. Dataset
# ----------------------------
class CLIPDataset(Dataset):
    def __init__(self, data, transform):
        self.data = data
        self.transform = transform


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image = Image.open(item["image_path"]).convert("RGB")
        image = self.transform(image)

        text = item["caption"]
        return image, text



def get_prompt_fidelity(data, device):
    # ----------------------------
    # 2. 加载 CLIP
    # ----------------------------
    model, preprocess = clip.load("ViT-B/32", device=device)
    model.eval()

    # ----------------------------
    # 3. DataLoader
    # ----------------------------
    dataset = CLIPDataset(data, preprocess)
    dataloader = DataLoader(dataset, batch_size=1024, shuffle=False)

    # ----------------------------
    # 4. 编码 + 余弦相似度
    # ----------------------------
    all_sims = []

    with torch.no_grad():
        for images, texts in tqdm(dataloader):
            images = images.to(device)
            text_tokens = clip.tokenize(texts).to(device)

            # 编码到统一特征空间
            img_feat = model.encode_image(images)     # [B, D]
            txt_feat = model.encode_text(text_tokens) # [B, D]

            # 归一化
            img_feat = F.normalize(img_feat, dim=-1)
            txt_feat = F.normalize(txt_feat, dim=-1)

            # 计算余弦相似度（逐对）
            cos_sim = (img_feat * txt_feat).sum(dim=-1)  # [B]
            all_sims.extend(cos_sim.cpu().numpy())

    # ----------------------------
    # 5. 输出结果
    # ----------------------------
    for i, sim in enumerate(all_sims):
        print(f"Sample {i}: Cosine Similarity = {sim:.4f}")
        
    return float(np.mean(all_sims))
