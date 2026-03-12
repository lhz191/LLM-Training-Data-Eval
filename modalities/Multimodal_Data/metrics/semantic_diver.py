import torch
import torch.nn.functional as F
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
import torchvision


class Inception_dataset(Dataset):
    def __init__(self, data):
        super().__init__()
        
        self.data = data
        self.target_video_len = 16
        # CLIP 图像预处理
        self.transform = transforms.Compose([
            transforms.Resize(299),          # 最短边 299
            transforms.CenterCrop(299),      # 中心裁剪到 299x299
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],  # ImageNet 均值
                std=[0.229, 0.224, 0.225]    # ImageNet 标准差
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
        video = vframes[frame_indice].float() / 255.0 # T C H W
        video = torch.stack([self.transform(frame) for frame in video])
        

        return {'video': video}

    def __len__(self):
        return len(self.data)



def load_inception(device):
    model = models.inception_v3(pretrained=True)
    model.eval()
    model.to(device)
    return model


def semantic_diversity(preds):
    # preds: [B, T, C]   softmax outputs
    py = preds.mean(dim=1, keepdim=True)# p(y) [B, 1, C]
    kl = preds * (preds.log() - py.log())  # KL divergence for each sample [B, T, C]
    kl = kl.sum(dim=2) # sum over classes [B, T]
    kl = kl.mean(dim=1) # [B]
    return torch.exp(kl)


def get_semantic_diversity(data, device):
    
    model = load_inception(device)
    
    dataset = Inception_dataset(data)
    
    dataloader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=8)
    
    sd_list = []
    
    with torch.no_grad():
        for x in tqdm(dataloader):
            v = x["video"] # [B,T,C,H,W]
            B, T, C, H, W = v.shape
            v = v.to(device)
            frames = v.reshape(B*T, C, H, W)
            preds = model(frames)
            preds = F.softmax(preds, -1)
            preds = preds.reshape(B, T, -1)
            
            sd_list.append(semantic_diversity(preds))
            # print(preds)
    sd_list = torch.cat(sd_list, dim=0).cpu().numpy()
    return np.mean(sd_list)