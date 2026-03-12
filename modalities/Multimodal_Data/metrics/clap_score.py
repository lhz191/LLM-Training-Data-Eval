import json
import torch
import torchaudio
import numpy as np
from transformers import ClapModel, ClapProcessor
from tqdm import tqdm



# 2. 读取 jsonl 文件
def load_jsonl(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data


# 3. 计算余弦相似度
def cosine_similarity(a, b):
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    return np.dot(a, b)


# 4. 主函数
def compute_clap_score(data, device):
    
        # 1. 加载 CLAP 模型

    model = ClapModel.from_pretrained("laion/clap-htsat-unfused").to(device)
    processor = ClapProcessor.from_pretrained("laion/clap-htsat-unfused")

    model.eval()

    similarities = []

    for item in tqdm(data):
        audio_path = item["audio"]
        text = item["text"]

        # 读取音频
        waveform, sr = torchaudio.load(audio_path)

        # CLAP 预处理
        inputs = processor(
            text=[text],
            audios=[waveform.squeeze().numpy()],
            sampling_rate=sr,
            return_tensors="pt",
            padding=True
        )

        inputs = {k: v.to(device) for k, v in inputs.items()}

        # 提取特征
        with torch.no_grad():
            outputs = model(**inputs)
            audio_emb = outputs.audio_embeds[0].cpu().numpy()
            text_emb = outputs.text_embeds[0].cpu().numpy()

        # 余弦相似度
        sim = cosine_similarity(audio_emb, text_emb)
        similarities.append(sim)

    avg_sim = float(np.mean(similarities))
    # print(f"Average CLAP cosine similarity: {avg_sim:.4f}")
    return avg_sim
