from __future__ import annotations

import os
from typing import Any, Dict, List, Tuple, Optional
from urllib.parse import urlparse

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import inception_v3, Inception_V3_Weights


# -------------------------
# Dataset
# -------------------------

_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class ImageListDataset(Dataset):
    def __init__(self, image_paths: List[str], tfm: transforms.Compose):
        self.image_paths = image_paths
        self.tfm = tfm

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        img = Image.open(self.image_paths[idx]).convert("RGB")
        return self.tfm(img)


# -------------------------
# Weights: check-before-download (offline-friendly)
# -------------------------

def _weights_cache_file(weights: Inception_V3_Weights, torch_home: Optional[str]) -> str:
    """
    torchvision weights are cached under:
      <TORCH_HOME or default>/hub/checkpoints/<filename>
    """
    if torch_home:
        os.environ["TORCH_HOME"] = torch_home  # affects torch.hub.get_dir()
    hub_dir = torch.hub.get_dir()
    ckpt_dir = os.path.join(hub_dir, "checkpoints")
    url = weights.url  # e.g. https://download.pytorch.org/models/...
    fname = os.path.basename(urlparse(url).path)
    return os.path.join(ckpt_dir, fname)


def ensure_weights_available(
    weights: Inception_V3_Weights,
    torch_home: Optional[str],
    allow_download: bool,
) -> None:
    cache_path = _weights_cache_file(weights, torch_home)
    if os.path.exists(cache_path):
        return
    if not allow_download:
        raise FileNotFoundError(
            "Inception-v3 weights not found in local cache, and downloading is disabled.\n"
            f"Expected cached file at:\n  {cache_path}\n\n"
            "Fix options:\n"
            "1) Pre-download weights on a machine with internet, then copy the cache directory to this machine.\n"
            "2) Allow download by setting allow_download=True.\n"
            "3) Set torch_home to point to a directory that already contains hub/checkpoints.\n"
        )
    # If downloads are allowed, torchvision will download automatically when model is created.


# -------------------------
# Core: Inception Score
# -------------------------

@torch.no_grad()
def inception_score(
    image_paths: List[str],
    batch_size: int = 64,
    splits: int = 10,
    device: str | None = None,
    num_workers: int = 4,
    seed: int = 0,
    torch_home: Optional[str] = None,
    allow_download: bool = True,
) -> Tuple[float, float, int]:
    """
    Returns: (mean_is, std_is, num_images)
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    torch.manual_seed(seed)
    np.random.seed(seed)

    n = len(image_paths)
    if n == 0:
        raise ValueError("No valid images found from input.")
    if splits < 1:
        raise ValueError("splits must be >= 1")
    if splits > n:
        splits = n

    weights = Inception_V3_Weights.IMAGENET1K_V1

    tfm = transforms.Compose([
        transforms.Resize(299, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(299),
        transforms.ToTensor(),
        transforms.Normalize(mean=weights.transforms().mean, std=weights.transforms().std),
    ])

    ds = ImageListDataset(image_paths, tfm)
    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device == "cuda"),
        drop_last=False,
    )

    # offline-friendly: check local cache first
    ensure_weights_available(weights, torch_home=torch_home, allow_download=allow_download)

    model = inception_v3(weights=weights, transform_input=False)
    model.eval().to(device)

    probs: List[np.ndarray] = []
    for batch in dl:
        batch = batch.to(device)
        logits = model(batch)
        pyx = F.softmax(logits, dim=1)
        probs.append(pyx.detach().cpu().numpy())

    pyx_all = np.concatenate(probs, axis=0)  # [N, 1000]

    split_scores: List[float] = []
    split_size = n // splits
    for i in range(splits):
        start = i * split_size
        end = (i + 1) * split_size if i < splits - 1 else n
        part = pyx_all[start:end]

        py = np.mean(part, axis=0, keepdims=True)
        kl = part * (np.log(part + 1e-16) - np.log(py + 1e-16))
        kl = np.sum(kl, axis=1)
        split_scores.append(float(np.exp(np.mean(kl))))

    mean_is = float(np.mean(split_scores))
    std_is = float(np.std(split_scores, ddof=1)) if len(split_scores) > 1 else 0.0
    return mean_is, std_is, n


# -------------------------
# Public API: benchmark entrypoint
# -------------------------

def get_inception_score(data: List[dict], cfg: Optional[Any] = None) -> Dict[str, Any]:
    """
    Unified entrypoint called by other modules.

    Inputs:
      data: list of dicts, each dict contains `image_path`
            - absolute path OR path relative to Datasets/Multimodal
      cfg: optional config object/dict
        - batch_size: int
        - splits: int
        - device: str|None
        - num_workers: int
        - seed: int
        - torch_home: str|None
        - allow_download: bool
        - return_std: bool (default False)
        - return_n: bool (default False)

    Returns:
      Minimal by default:
        {"Inception_Score": float}
      Optionally:
        {"Inception_Score": float, "Inception_Score_std": float, "n": int}
    """
    # defaults
    batch_size = 64
    splits = 10
    device = None
    num_workers = 4
    seed = 0
    torch_home = None
    allow_download = True

    # output control (默认：只返回最终值)
    return_std = False
    return_n = False

    if cfg is not None:
        try:
            icfg = getattr(cfg, "inception_score", cfg)
        except Exception:
            icfg = cfg

        # dict-like
        if hasattr(icfg, "get"):
            batch_size = int(icfg.get("batch_size", batch_size))
            splits = int(icfg.get("splits", splits))
            device = icfg.get("device", device)
            num_workers = int(icfg.get("num_workers", num_workers))
            seed = int(icfg.get("seed", seed))
            torch_home = icfg.get("torch_home", torch_home)
            allow_download = bool(icfg.get("allow_download", allow_download))
            return_std = bool(icfg.get("return_std", return_std))
            return_n = bool(icfg.get("return_n", return_n))
        else:
            batch_size = int(getattr(icfg, "batch_size", batch_size))
            splits = int(getattr(icfg, "splits", splits))
            device = getattr(icfg, "device", device)
            num_workers = int(getattr(icfg, "num_workers", num_workers))
            seed = int(getattr(icfg, "seed", seed))
            torch_home = getattr(icfg, "torch_home", torch_home)
            allow_download = bool(getattr(icfg, "allow_download", allow_download))
            return_std = bool(getattr(icfg, "return_std", return_std))
            return_n = bool(getattr(icfg, "return_n", return_n))

    # base_dir: ../../Datasets/Multimodal relative to this file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.abspath(os.path.join(current_dir, "../../Datasets/Multimodal"))

    image_paths: List[str] = []
    for it in data:
        if not isinstance(it, dict):
            continue
        rel = it.get("image_path")
        if not rel:
            continue
        p = rel if os.path.isabs(rel) else os.path.join(base_dir, rel)
        ext = os.path.splitext(p)[1].lower()
        if ext in _IMG_EXTS and os.path.exists(p):
            image_paths.append(p)

    image_paths.sort()

    mean_is, std_is, n = inception_score(
        image_paths=image_paths,
        batch_size=batch_size,
        splits=splits,
        device=device,
        num_workers=num_workers,
        seed=seed,
        torch_home=torch_home,
        allow_download=allow_download,
    )

    out: Dict[str, Any] = {"Inception_Score": mean_is}
    if return_std:
        out["Inception_Score_std"] = std_is
    if return_n:
        out["n"] = n
    return out
