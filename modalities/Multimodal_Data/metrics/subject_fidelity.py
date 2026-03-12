"""
Subject Fidelity — CLIP-I Score  (Survey §6 Vision-Language)

Measures visual similarity between a generated image and a reference image
using CLIP image embeddings:

    CLIP-I = cos(CLIP_img(gen), CLIP_img(ref))

High CLIP-I means the generated image preserves the visual identity/subject
of the reference. This complements CLIP-T (text-image alignment) which is
already implemented in image_prompt_fidelity.py.
"""

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from typing import List, Dict, Any, Optional

try:
    import clip
    HAS_CLIP = True
except ImportError:
    HAS_CLIP = False


class ImagePairDataset(Dataset):
    """Dataset yielding (generated_image, reference_image) pairs."""

    def __init__(self, data: List[Dict[str, Any]], transform):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        gen_img = Image.open(item["image_path"]).convert("RGB")
        ref_img = Image.open(item["reference_image_path"]).convert("RGB")
        return self.transform(gen_img), self.transform(ref_img)


def get_subject_fidelity(
    data: List[Dict[str, Any]],
    device: str = None,
    batch_size: int = 256,
    clip_model: str = "ViT-B/32",
) -> Dict[str, Any]:
    """
    Compute CLIP-I score (image-image cosine similarity).

    Args:
        data: list of dicts with "image_path" (generated) and "reference_image_path"
        device: "cuda" or "cpu"
        batch_size: batch size for encoding
        clip_model: CLIP model variant

    Returns:
        dict with mean/median/std CLIP-I scores
    """
    if not HAS_CLIP:
        return {'error': 'clip package not installed (pip install git+https://github.com/openai/CLIP)'}

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    valid_data = [
        item for item in data
        if item.get('image_path') and item.get('reference_image_path')
    ]

    if not valid_data:
        return {
            'clip_i_score': 0.0,
            'note': 'no valid image pairs (need both image_path and reference_image_path)',
        }

    model, preprocess = clip.load(clip_model, device=device)
    model.eval()

    dataset = ImagePairDataset(valid_data, preprocess)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    all_sims = []

    with torch.no_grad():
        for gen_imgs, ref_imgs in tqdm(dataloader, desc="CLIP-I"):
            gen_imgs = gen_imgs.to(device)
            ref_imgs = ref_imgs.to(device)

            gen_feats = model.encode_image(gen_imgs)
            ref_feats = model.encode_image(ref_imgs)

            gen_feats = F.normalize(gen_feats, dim=-1)
            ref_feats = F.normalize(ref_feats, dim=-1)

            cos_sim = (gen_feats * ref_feats).sum(dim=-1)
            all_sims.extend(cos_sim.cpu().numpy().tolist())

    sims = np.array(all_sims)

    return {
        'clip_i_score': float(np.mean(sims)),
        'clip_i_median': float(np.median(sims)),
        'clip_i_std': float(np.std(sims)),
        'clip_i_min': float(np.min(sims)),
        'num_pairs': len(sims),
    }
