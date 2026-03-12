import json
import torch
import librosa
import numpy as np
from tqdm import tqdm
from scipy import linalg

# -----------------------------
# 1. 工具函数
# -----------------------------

def load_audio(path, sr=16000):
    wav, _ = librosa.load(path, sr=sr, mono=True)
    return wav


def compute_stats(embeddings):
    mu = np.mean(embeddings, axis=0)
    sigma = np.cov(embeddings, rowvar=False)
    return mu, sigma


def frechet_distance(mu1, sigma1, mu2, sigma2):
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1 @ sigma2, disp=False)

    if np.iscomplexobj(covmean):
        covmean = covmean.real

    return diff @ diff + np.trace(sigma1 + sigma2 - 2 * covmean)


# -----------------------------
# 2. VGGish (FAD)
# -----------------------------

class VGGishWrapper(torch.nn.Module):
    def __init__(self):
        super().__init__()
        from torch.hub import load
        self.model = load(
            "harritaylor/torchvggish", "vggish"
        ).eval()

    def forward(self, wav):
        with torch.no_grad():
            return self.model(wav)


def extract_vggish_embedding(model, wav):
    wav = torch.tensor(wav).unsqueeze(0)
    emb = model(wav).mean(dim=0).cpu().numpy()
    return emb


# -----------------------------
# 3. PANN (FD)
# -----------------------------

class PANNWrapper(torch.nn.Module):
    def __init__(self):
        super().__init__()
        from torch.hub import load
        self.model = load(
            "qiuqiangkong/panns_transfer", "Cnn14"
        ).eval()

    def forward(self, wav):
        with torch.no_grad():
            return self.model(wav)[0]


def extract_pann_embedding(model, wav):
    wav = torch.tensor(wav).unsqueeze(0)
    emb = model(wav).mean(dim=0).cpu().numpy()
    return emb


# -----------------------------
# 4. 主流程
# -----------------------------

def compute_fad_fd(data, use_vggish=True, use_pann=True):

    if use_vggish:
        vggish = VGGishWrapper()
        gen_vgg, ref_vgg = [], []

    if use_pann:
        pann = PANNWrapper()
        gen_pann, ref_pann = [], []

    for s in tqdm(data):
        gen_wav = load_audio(s["audio"])
        ref_wav = load_audio(s["ref_audio"])

        if use_vggish:
            gen_vgg.append(extract_vggish_embedding(vggish, gen_wav))
            ref_vgg.append(extract_vggish_embedding(vggish, ref_wav))

        if use_pann:
            gen_pann.append(extract_pann_embedding(pann, gen_wav))
            ref_pann.append(extract_pann_embedding(pann, ref_wav))

    results = {"num_samples": len(data)}

    if use_vggish:
        mu_r, sig_r = compute_stats(ref_vgg)
        mu_g, sig_g = compute_stats(gen_vgg)
        results["FAD_vggish"] = float(frechet_distance(mu_r, sig_r, mu_g, sig_g))

    if use_pann:
        mu_r, sig_r = compute_stats(ref_pann)
        mu_g, sig_g = compute_stats(gen_pann)
        results["FD_pann"] = float(frechet_distance(mu_r, sig_r, mu_g, sig_g))

    return results


# -----------------------------
# 5. 运行入口
# -----------------------------

if __name__ == "__main__":
    jsonl_path = "data.jsonl"

    metrics = compute_fad_fd(
        jsonl_path,
        use_vggish=True,
        use_pann=True
    )

    with open("metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("Evaluation finished:")
    print(metrics)
