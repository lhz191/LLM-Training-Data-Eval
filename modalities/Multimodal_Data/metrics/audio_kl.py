import json
import torch
import librosa
import numpy as np
from tqdm import tqdm


def load_audio(path, sr=32000):
    wav, _ = librosa.load(path, sr=sr, mono=True)
    return wav

def softmax(x, axis=-1):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / e_x.sum(axis=axis, keepdims=True)

def kl_divergence(p, q, eps=1e-8):
    p = np.clip(p, eps, 1)
    q = np.clip(q, eps, 1)
    return np.sum(p * np.log(p / q))


class PANNClassifier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        from torch.hub import load
        self.model = load("qiuqiangkong/panns_transfer", "Cnn14").eval()

    def forward(self, wav):
        with torch.no_grad():
            # 返回 shape: (num_frames, num_classes)
            # 我们取时间平均
            emb = self.model(wav.unsqueeze(0))[0]  # (frames, classes)
            probs = torch.softmax(emb.mean(dim=0), dim=0)  # (classes,)
            return probs.cpu().numpy()

def compute_kl(data):

    classifier = PANNClassifier()

    num_classes = 527  # AudioSet 类别数量
    gen_probs_list = []
    ref_probs_list = []

    for s in tqdm(data):
        # 读取音频
        gen_wav = torch.tensor(load_audio(s["audio"])).float()
        ref_wav = torch.tensor(load_audio(s["ref_audio"])).float()

        # 获取类别概率
        gen_probs = classifier(gen_wav)
        ref_probs = classifier(ref_wav)

        gen_probs_list.append(gen_probs)
        ref_probs_list.append(ref_probs)

    # 平均概率
    P = np.mean(ref_probs_list, axis=0)
    Q = np.mean(gen_probs_list, axis=0)

    # 计算 KL(P || Q)
    kl_value = float(kl_divergence(P, Q))

    result = {
        "KL_divergence": round(kl_value, 6),
        # "classifier": "PANN-AudioSet",
        # "num_classes": num_classes,
        # "num_samples": len(samples)
    }

    return result

# -----------------------------
# 4. 运行
# -----------------------------

if __name__ == "__main__":
    jsonl_path = "data.jsonl"

    metrics = compute_kl(jsonl_path)

    with open("kl_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("KL Divergence evaluation finished:")
    print(metrics)
