import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from internvid.viclip import ViCLIP
from tqdm import tqdm
from torch.utils.data import DataLoader, Subset, Dataset
import random
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import cv2 
from internvid.simple_tokenizer import SimpleTokenizer as _Tokenizer
class ViCLIP_dataset(Dataset):
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
    

def calculate_cmc(dataset, model, device):
    """
    Calculate FID (Fidelity) based on object feature similarity between adjacent frames.

    Parameters:
    - video_frames (list of torch.Tensor): List of frames as tensors, shape [F, C, H, W]
    - model (ViCLIP): Pre-trained ViCLIP model for feature extraction

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





clip_candidates = {'viclip':None, 'clip':None}

def get_clip(name='viclip'):
    global clip_candidates
    m = clip_candidates[name]
    if m is None:
        if name == 'viclip':
            tokenizer = _Tokenizer()
            vclip = ViCLIP(tokenizer)
            # m = vclip
            m = (vclip, tokenizer)
        else:
            raise Exception('the target clip model is not found.')
    
    return m

def get_text_feat_dict(texts, clip, tokenizer, text_feat_d={}):
    for t in texts:
        feat = clip.get_text_features(t, tokenizer, text_feat_d)
        text_feat_d[t] = feat
    return text_feat_d

def get_vid_feat(frames, clip):
    return clip.get_vid_features(frames)

def _frame_from_video(video):
    while video.isOpened():
        success, frame = video.read()
        if success:
            yield frame
        else:
            break

v_mean = np.array([0.485, 0.456, 0.406]).reshape(1,1,3)
v_std = np.array([0.229, 0.224, 0.225]).reshape(1,1,3)
def normalize(data):
    return (data/255.0-v_mean)/v_std

def frames2tensor(vid_list, fnum=8, target_size=(224, 224), device=torch.device('cuda')):
    assert(len(vid_list) >= fnum)
    step = len(vid_list) // fnum
    vid_list = vid_list[::step][:fnum]
    vid_list = [cv2.resize(x[:,:,::-1], target_size) for x in vid_list]
    vid_tube = [np.expand_dims(normalize(x), axis=(0, 1)) for x in vid_list]
    vid_tube = np.concatenate(vid_tube, axis=1)
    vid_tube = np.transpose(vid_tube, (0, 1, 4, 2, 3))
    vid_tube = torch.from_numpy(vid_tube).to(device, non_blocking=True).float()
    return vid_tube

def retrieve_text(frames, texts, name='viclip', topk=5, device=torch.device('cuda')):
    clip, tokenizer = get_clip(name)
    clip = clip.to(device)
    # frames_tensor = frames2tensor(frames, device=device)
    vid_feat = get_vid_feat(frames, clip)

    text_feat_d = {}
    text_feat_d = get_text_feat_dict(texts, clip, tokenizer, text_feat_d)
    
    text_feats = [text_feat_d[t] for t in texts]
    text_feats_tensor = torch.cat(text_feats, 0)
    
    # probs, idxs = clip.get_predict_label(vid_feat, text_feats_tensor, top=topk)

    # ret_texts = [texts[i] for i in idxs.numpy()[0].tolist()]
    # return ret_texts, probs.numpy()[0]
    return F.cosine_similarity(vid_feat, text_feats_tensor, dim=1)


def get_cmc(data, device):
    dataset = ViCLIP_dataset(data)
    
    dataloader = DataLoader(dataset, batch_size=256, shuffle=False, drop_last=False, num_workers=8)
    
    cos_sims = []
    with torch.no_grad():    
        for x in tqdm(dataloader):
            v, class_name = x["video"], x["video_name"]
            # label_idx = label_idx.numpy().astype(int)
            # print(f'label_idx: {label_idx}, type: {type(label_idx)}')
            v = v.to(device) #[B, T, C, H, W]
            
            # class_names = classes[label_idx] #[B,]
            cos_sim = retrieve_text(v, class_name, device=device) # [B, 1]
            cos_sim = cos_sim.mean(dim=0)
            cos_sims.append(cos_sim.item())
    
    res = np.mean(cos_sims)
    
    print(res)
    return res
    
