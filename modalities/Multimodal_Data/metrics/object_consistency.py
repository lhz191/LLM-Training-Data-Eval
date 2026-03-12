import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from transformers import CLIPProcessor, CLIPModel
from tqdm import tqdm
from torch.utils.data import DataLoader, Subset, Dataset
import random
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

class Video_dataset(Dataset):
    def __init__(self, data):
        super().__init__()
        
        self.data = data
        self.target_video_len = 16
        # CLIP 图像预处理
        self.transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711]
            )
        ])

    
    def __getitem__(self, index, ):
        path = self.data[index]['video']
        # class_name = path.split('/')[-2]
        # class_index = self.class_to_idx[class_name]

        vframes, _, _ = torchvision.io.read_video(filename=path, pts_unit='sec', output_format='TCHW')
        # print("vframes",vframes.shape)
        total_frames = len(vframes)
        # Sampling video frames
        start_frame_ind, end_frame_ind = 0, total_frames-1
        assert end_frame_ind - start_frame_ind >= self.target_video_len, f"Video has too few frames: {total_frames} < {self.target_video_len}, sample:{path}"
        
        frame_indice = np.linspace(start_frame_ind, end_frame_ind-1, self.target_video_len, dtype=int)
        # print(frame_indice)
        video = vframes[frame_indice].to(torch.float32) / 255.0 # T C H W
        video = torch.stack([self.transform(frame) for frame in video])
        

        return {'video': video}

    def __len__(self):
        return len(self.data)
    

def calculate_fidelity(dataset, model, device):
    """
    Calculate FID (Fidelity) based on object feature similarity between adjacent frames.

    Parameters:
    - video_frames (list of torch.Tensor): List of frames as tensors, shape [F, C, H, W]
    - model (CLIPModel): Pre-trained CLIP model for feature extraction

    Returns:
    - float: Fidelity score (FID)
    """
    # k = int(0.1*len(dataset))
    # indices = list(range(len(dataset)))
    # random.shuffle(indices)
    # selected = indices[:k]
    # dataset = Subset(dataset, selected)

    N = len(dataset) # Total number

    dataloader = DataLoader(dataset, batch_size=1024, shuffle=False, drop_last=False, num_workers=8)
    
    # Extract features for each frame
    features = []
    model = model.to(device)
    model.eval()
    
    with torch.no_grad():

        for x in tqdm(dataloader, desc="Extracting features"):
            frame = x['video'] # [B, T, C, H, W]
            frame = frame.to(device)
            B, T, C, H, W = frame.shape
            frame = frame.reshape(B*T, C, H, W)
            output = model.get_image_features(frame)
            features.append(output)  # Shape: [C]
    
    features = torch.cat(features, dim=0)
    features = features.reshape(N, T, -1)
    # Compute similarity between adjacent frames
    # L2-normalize for stable cosine
    features = F.normalize(features, p=2, dim=-1)

    # compute cosine similarity between adjacent frames
    sims = (features[:, :-1] * features[:, 1:]).sum(dim=-1)  # equivalent to cosine if normalized

    # per-video fidelity: mean over time
    fid_per_video = sims.mean(dim=1)  # [B]
    
    return fid_per_video.mean(dim=0).item()

    


# def get_object_consistency(dataset, device):

#     model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    
#     processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

#     return calculate_fidelity(dataset, model, processor, device)



# 视频内容是否连贯

def get_object_consistency(data, device):

    dataset = Video_dataset(data)
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    
    # processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    return calculate_fidelity(dataset, model, device)
        