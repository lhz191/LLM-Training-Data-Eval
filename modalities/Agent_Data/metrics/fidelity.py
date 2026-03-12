"""
Fidelity — FID, FVD, SSIM, MSE, PSNR, LPIPS, SemAlign

Survey Section 7.2 (Fidelity):

  Distributional (real vs generated set):
    FID  = ‖μX − μY‖² + Tr(ΣX + ΣY − 2(ΣX^½ ΣY ΣX^½)^½)
    FVD  = same formula on video (I3D) embeddings

  Paired (per image/frame):
    SSIM(x,y) = (2μxμy+C1)(2σxy+C2) / ((μx²+μy²+C1)(σx²+σy²+C2))
    MSE  = (1/M) Σ (xi − yi)²
    PSNR = 10 log10(MAX² / MSE)
    LPIPS(x,y) = Σ_l (1/HlWl) Σ_{h,w} ‖wl ⊙ (f̂x − f̂y)‖²

  Reference-free (prompt → outcome):
    SemAlign(p, o) = cos(fT(p), fV(o))   via CLIP
    SemFid = (1/N) Σ SemAlign
    MatchRate(τ) = (1/N) Σ 𝟙{SemAlign ≥ τ}

Libraries (matching cited papers):
    FID:     clean-fid  (Parmar et al.)       — pip install clean-fid
    SSIM:    skimage     (Wang et al., 2004)   — scikit-image
    PSNR:    skimage                           — scikit-image
    LPIPS:   lpips       (Zhang et al., 2018)  — pip install lpips
    CLIP:    transformers (Radford et al., 2021)— pip install transformers
    FVD:     scipy.linalg.sqrtm + I3D features (Unterthiner et al., 2019)
"""

import numpy as np
from typing import List, Dict, Any


# ═══════════════════════════════════════════════════════════════
# FID — Heusel et al., 2018
# Uses clean-fid (standard FID library) for directory-based computation
# ═══════════════════════════════════════════════════════════════

def compute_fid(
    real_dir: str,
    gen_dir: str,
    mode: str = "clean",
    batch_size: int = 64,
    device: str = None,
) -> Dict[str, Any]:
    """
    Fréchet Inception Distance via clean-fid library.

    Args:
        real_dir: path to directory of real images
        gen_dir:  path to directory of generated images
        mode: "clean" (default) or "legacy"
        batch_size: inference batch size
        device: torch device
    """
    import torch
    from cleanfid import fid

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    score = fid.compute_fid(
        real_dir, gen_dir,
        mode=mode,
        batch_size=batch_size,
        device=device,
    )
    return {"fid": float(score), "mode": mode}


# ═══════════════════════════════════════════════════════════════
# FVD — Unterthiner et al., 2019
# Same Fréchet formula on pre-extracted video (I3D) features
# ═══════════════════════════════════════════════════════════════

def compute_frechet_distance(
    mu_real: np.ndarray,
    sigma_real: np.ndarray,
    mu_gen: np.ndarray,
    sigma_gen: np.ndarray,
) -> float:
    """
    Fréchet Distance between two multivariate Gaussians.
    Shared by FID (from Inception features) and FVD (from I3D features).
    """
    from scipy.linalg import sqrtm

    diff = mu_real - mu_gen
    covmean, _ = sqrtm(sigma_real @ sigma_gen, disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    return float(
        diff @ diff
        + np.trace(sigma_real)
        + np.trace(sigma_gen)
        - 2 * np.trace(covmean)
    )


def compute_fvd(
    feats_real: np.ndarray,
    feats_gen: np.ndarray,
) -> Dict[str, Any]:
    """
    Fréchet Video Distance from pre-extracted I3D features.

    Args:
        feats_real: (N, D) I3D feature matrix for real videos
        feats_gen:  (M, D) I3D feature matrix for generated videos
    """
    mu_r = feats_real.mean(axis=0)
    sigma_r = np.cov(feats_real, rowvar=False)
    mu_g = feats_gen.mean(axis=0)
    sigma_g = np.cov(feats_gen, rowvar=False)
    fvd = compute_frechet_distance(mu_r, sigma_r, mu_g, sigma_g)
    return {
        "fvd": fvd,
        "num_real": len(feats_real),
        "num_gen": len(feats_gen),
    }


# ═══════════════════════════════════════════════════════════════
# SSIM — Wang et al., 2004
# Uses skimage.metrics.structural_similarity (standard implementation)
# ═══════════════════════════════════════════════════════════════

def compute_ssim(
    images_real: List[np.ndarray],
    images_gen: List[np.ndarray],
) -> Dict[str, Any]:
    """
    Structural Similarity Index over paired images.

    Args:
        images_real: list of reference images (H,W) or (H,W,C) as numpy
        images_gen:  list of generated images, same shapes
    """
    from skimage.metrics import structural_similarity

    scores = []
    for real, gen in zip(images_real, images_gen):
        is_multichannel = real.ndim == 3
        s = structural_similarity(
            real, gen,
            channel_axis=2 if is_multichannel else None,
            data_range=real.max() - real.min(),
        )
        scores.append(float(s))

    return {
        "ssim_mean": float(np.mean(scores)),
        "ssim_std": float(np.std(scores)),
        "num_pairs": len(scores),
    }


# ═══════════════════════════════════════════════════════════════
# MSE & PSNR
# Uses skimage.metrics (standard implementation)
# ═══════════════════════════════════════════════════════════════

def compute_mse_psnr(
    images_real: List[np.ndarray],
    images_gen: List[np.ndarray],
) -> Dict[str, Any]:
    """
    Mean Squared Error and Peak Signal-to-Noise Ratio over paired images.

    Args:
        images_real: list of reference images
        images_gen:  list of generated images
    """
    from skimage.metrics import mean_squared_error, peak_signal_noise_ratio

    mse_scores = []
    psnr_scores = []

    for real, gen in zip(images_real, images_gen):
        mse = float(mean_squared_error(real, gen))
        mse_scores.append(mse)
        if mse > 0:
            psnr = float(peak_signal_noise_ratio(real, gen,
                                                  data_range=real.max() - real.min()))
            psnr_scores.append(psnr)
        else:
            psnr_scores.append(float("inf"))

    return {
        "mse_mean": float(np.mean(mse_scores)),
        "psnr_mean": float(np.mean(psnr_scores)),
        "psnr_std": float(np.std(psnr_scores)),
        "num_pairs": len(mse_scores),
    }


# ═══════════════════════════════════════════════════════════════
# LPIPS — Zhang et al., 2018
# Uses official lpips package (https://github.com/richzhang/PerceptualSimilarity)
# ═══════════════════════════════════════════════════════════════

def compute_lpips(
    images_real: List[np.ndarray],
    images_gen: List[np.ndarray],
    net: str = "alex",
    device: str = None,
) -> Dict[str, Any]:
    """
    Learned Perceptual Image Patch Similarity.

    Args:
        images_real: list of reference images (H, W, 3), uint8 or float [0,1]
        images_gen:  list of generated images
        net: backbone ("alex" | "vgg" | "squeeze"), "alex" is default in paper
        device: torch device
    """
    import torch
    import lpips as lpips_lib

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    loss_fn = lpips_lib.LPIPS(net=net).to(device).eval()

    scores = []
    for real, gen in zip(images_real, images_gen):
        r = _img_to_tensor(real).to(device)
        g = _img_to_tensor(gen).to(device)
        with torch.no_grad():
            d = loss_fn(r, g).item()
        scores.append(d)

    return {
        "lpips_mean": float(np.mean(scores)),
        "lpips_std": float(np.std(scores)),
        "num_pairs": len(scores),
        "net": net,
    }


def _img_to_tensor(img: np.ndarray):
    """Convert (H,W,C) uint8/float image → (1,C,H,W) tensor in [-1, 1]."""
    import torch

    if img.dtype == np.uint8:
        img = img.astype(np.float32) / 255.0
    t = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float()
    return t * 2.0 - 1.0


# ═══════════════════════════════════════════════════════════════
# SemAlign / SemFid / MatchRate — via CLIP (Radford et al., 2021)
# Uses transformers CLIPModel
# ═══════════════════════════════════════════════════════════════

def compute_semantic_alignment(
    prompts: List[str],
    images: list,
    model_name: str = "openai/clip-vit-base-patch32",
    threshold: float = 0.25,
    device: str = None,
    batch_size: int = 32,
) -> Dict[str, Any]:
    """
    Prompt–outcome semantic alignment via CLIP cosine similarity.

    SemAlign(p, o) = cos(fT(p), fV(o))
    SemFid         = (1/N) Σ SemAlign(pᵢ, oᵢ)
    MatchRate(τ)   = (1/N) Σ 𝟙{SemAlign(pᵢ, oᵢ) ≥ τ}

    Args:
        prompts: list of text prompts / specifications
        images: list of generated images, (H,W,3) uint8 numpy or PIL.Image
        model_name: HuggingFace CLIP model identifier
        threshold: τ for MatchRate
        device: torch device
        batch_size: inference batch size
    """
    import torch
    from PIL import Image
    from transformers import CLIPModel, CLIPProcessor

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    processor = CLIPProcessor.from_pretrained(model_name)
    model = CLIPModel.from_pretrained(model_name).to(device).eval()

    all_scores = []

    for i in range(0, len(prompts), batch_size):
        batch_p = prompts[i:i + batch_size]
        batch_img = [
            Image.fromarray(im) if isinstance(im, np.ndarray) else im
            for im in images[i:i + batch_size]
        ]

        inputs = processor(
            text=batch_p, images=batch_img,
            return_tensors="pt", padding=True, truncation=True,
        ).to(device)

        with torch.no_grad():
            out = model(**inputs)
            t_emb = out.text_embeds
            v_emb = out.image_embeds
            t_emb = t_emb / t_emb.norm(dim=-1, keepdim=True)
            v_emb = v_emb / v_emb.norm(dim=-1, keepdim=True)
            sims = (t_emb * v_emb).sum(dim=-1)

        all_scores.extend(sims.cpu().tolist())

    scores = np.array(all_scores)
    return {
        "sem_fid": float(scores.mean()),
        "sem_align_std": float(scores.std()),
        "match_rate": float((scores >= threshold).mean()),
        "threshold": threshold,
        "num_pairs": len(scores),
        "model": model_name,
    }
