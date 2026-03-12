import os
import re
import json
import torch
import decord
import torchvision
import numpy as np
import cv2
from datasets import load_dataset
from PIL import Image
from einops import rearrange
from typing import Dict, List, Tuple

class_labels_map = None
cls_sample_cnt = None

class_labels_map = None
cls_sample_cnt = None


def temporal_sampling(frames, start_idx, end_idx, num_samples):
    """
    Given the start and end frame index, sample num_samples frames between
    the start and end with equal interval.
    Args:
        frames (tensor): a tensor of video frames, dimension is
            `num video frames` x `channel` x `height` x `width`.
        start_idx (int): the index of the start frame.
        end_idx (int): the index of the end frame.
        num_samples (int): number of frames to sample.
    Returns:
        frames (tersor): a tensor of temporal sampled video frames, dimension is
            `num clip frames` x `channel` x `height` x `width`.
    """
    index = torch.linspace(start_idx, end_idx, num_samples)
    index = torch.clamp(index, 0, frames.shape[0] - 1).long()
    frames = torch.index_select(frames, 0, index)
    ds = load_dataset("ucf101")
    
    return frames


def get_filelist(file_path):
    Filelist = []
    for home, dirs, files in os.walk(file_path):
        for filename in files:
            # 文件名列表，包含完整路径
            Filelist.append(os.path.join(home, filename))
            # # 文件名列表，只包含文件名
            # Filelist.append( filename)
    return Filelist


def load_annotation_data(data_file_path):
    with open(data_file_path, 'r') as data_file:
        return json.load(data_file)


def get_class_labels(num_class, anno_pth='./k400_classmap.json'):
    global class_labels_map, cls_sample_cnt
    
    if class_labels_map is not None:
        return class_labels_map, cls_sample_cnt
    else:
        cls_sample_cnt = {}
        class_labels_map = load_annotation_data(anno_pth)
        for cls in class_labels_map:
            cls_sample_cnt[cls] = 0
        return class_labels_map, cls_sample_cnt


def load_annotations(ann_file, num_class, num_samples_per_cls):
    dataset = []
    class_to_idx, cls_sample_cnt = get_class_labels(num_class)
    with open(ann_file, 'r') as fin:
        for line in fin:
            line_split = line.strip().split('\t')
            sample = {}
            idx = 0
            # idx for frame_dir
            frame_dir = line_split[idx]
            sample['video'] = frame_dir
            idx += 1
                                
            # idx for label[s]
            label = [x for x in line_split[idx:]]
            assert label, f'missing label in line: {line}'
            assert len(label) == 1
            class_name = label[0]
            class_index = int(class_to_idx[class_name])
            
            # choose a class subset of whole dataset
            if class_index < num_class:
                sample['label'] = class_index
                if cls_sample_cnt[class_name] < num_samples_per_cls:
                    dataset.append(sample)
                    cls_sample_cnt[class_name]+=1

    return dataset


def find_classes(directory: str) -> Tuple[List[str], Dict[str, int]]:
    """Finds the class folders in a dataset.

    See :class:`DatasetFolder` for details.
    """
    classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())
    if not classes:
        raise FileNotFoundError(f"Couldn't find any class folder in {directory}.")

    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    return classes, class_to_idx


class DecordInit(object):
    """Using Decord(https://github.com/dmlc/decord) to initialize the video_reader."""

    def __init__(self, num_threads=1):
        self.num_threads = num_threads
        self.ctx = decord.cpu(0)
        
    def __call__(self, filename):
        """Perform the Decord initialization.
        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        reader = decord.VideoReader(filename,
                                    ctx=self.ctx,
                                    num_threads=self.num_threads)
        return reader

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'sr={self.sr},'
                    f'num_threads={self.num_threads})')
        return repr_str


class UCF101(torch.utils.data.Dataset):
    """Load the UCF101 video files
    
    Args:
        target_video_len (int): the number of video frames will be load.
        align_transform (callable): Align different videos in a specified size.
        temporal_sample (callable): Sample the target length of a video.
    """

    def __init__(self,
                 configs,
                 transform=None,
                 temporal_sample=None):
        self.configs = configs
        self.data_path = configs.data_path
        self.video_lists = get_filelist(configs.data_path)
        self.transform = transform
        self.temporal_sample = temporal_sample
        self.target_video_len = self.configs.num_frames
        self.v_decoder = DecordInit()
        self.classes, self.class_to_idx = find_classes(self.data_path)
        
        with open("ucf101.jsonl", 'w', encoding='utf-8') as f:
            for idx in range(len(self.video_lists)):
                path = self.video_lists[idx]
                class_name = path.split('/')[-2]
                dump_data = {
                    "idx" : idx, 
                    "video" : path, 
                    "label" : class_name,
                    "prompt" : "generate a video"
                }
                f.write(json.dumps(dump_data, ensure_ascii=False) + "\n")
        # print(self.class_to_idx)
        # exit()

    def __getitem__(self, index):
        path = self.video_lists[index]
        class_name = path.split('/')[-2]
        class_index = self.class_to_idx[class_name]

        vframes, aframes, info = torchvision.io.read_video(filename=path, pts_unit='sec', output_format='TCHW')
        # print("vframes",vframes.shape)
        total_frames = len(vframes)
        # print("total_frames: ", total_frames)
        
        # Sampling video frames
        # start_frame_ind, end_frame_ind = self.temporal_sample(total_frames)
        start_frame_ind, end_frame_ind = 0, total_frames-1
        assert end_frame_ind - start_frame_ind >= self.target_video_len, f"Video has too few frames: {total_frames} < {self.target_video_len}, sample:{path}"
    
        frame_indice = np.linspace(start_frame_ind, end_frame_ind-1, self.target_video_len, dtype=int)
        # print(frame_indice)
        video = vframes[frame_indice] #
        video = self.transform(video) # T C H W

        return {'video': video, 'video_name': class_name}

    def __len__(self):
        return len(self.video_lists)
    
    
    def get_OpticalFlowFarneback(self, index, stride=5):
        path = self.video_lists[index]
        cap = cv2.VideoCapture(path)
        
        # 获取第一帧
        ret, prev_frame = cap.read()

        # 转为灰度图
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        
        
        motion_scores = []
        
        frame_cnt = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_cnt += 1
            
            if frame_cnt%stride != 0:
                continue
            # 转为灰度图
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # 计算光流（使用Farneback算法）
            flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            
            # 计算光流的幅值作为运动评分
            mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            motion_score = np.mean(mag)  # 运动评分是所有光流幅值的平均值
            
            motion_scores.append(motion_score)
            
            # 更新上一帧图像
            prev_gray = gray

        cap.release()
        
        return np.mean(motion_scores)
    



if __name__ == '__main__':

    import argparse
    import video_transforms
    import torch.utils.data as Data
    import torchvision.transforms as transforms
    
    from PIL import Image

    parser = argparse.ArgumentParser()
    parser.add_argument("--num_frames", type=int, default=16)
    parser.add_argument("--frame_interval", type=int, default=1)
    # parser.add_argument("--data-path", type=str, default="/nvme/share_data/datasets/UCF101/videos")
    parser.add_argument("--data-path", type=str, default="/mnt/petrelfs/wangjiedong/LLMDataBenchmark/Datasets/Multimodal/UCF-101")
    config = parser.parse_args()

    UCF101(configs=config)
    
    # get_filelist("/mnt/petrelfs/wangjiedong/LLMDataBenchmark/Datasets/Multimodal/UCF-101")

    # temporal_sample = video_transforms.TemporalRandomCrop(config.num_frames)

    # transform_ucf101 = transforms.Compose([
    #         video_transforms.ToTensorVideo(), # TCHW
    #         # video_transforms.RandomHorizontalFlipVideo(),
    #         video_transforms.UCFCenterCropVideo(256),
    #         transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    #     ])


    # ffs_dataset = UCF101(config, transform=transform_ucf101, temporal_sample=temporal_sample)
    # ffs_dataloader = Data.DataLoader(dataset=ffs_dataset, batch_size=6, shuffle=False, num_workers=1)

    # # for i, video_data in enumerate(ffs_dataloader):
    # for video_data in ffs_dataloader:
    #     print(type(video_data))
    #     video = video_data['video']
    #     video_name = video_data['video_name']
    #     print(video.shape)
    #     print(video_name)
    #     # print(video_data[2])

    #     # for i in range(16):
    #     #     img0 = rearrange(video_data[0][0][i], 'c h w -> h w c')
    #     #     print('Label: {}'.format(video_data[1]))
    #     #     print(img0.shape)
    #     #     img0 = Image.fromarray(np.uint8(img0 * 255))
    #     #     img0.save('./img{}.jpg'.format(i))
    #     exit()