from datasets import load_dataset
from torch.utils.data import Dataset


class InternVid(Dataset):
    def __init__(self,
                 configs,
                 transform=None,
                 temporal_sample=None):
        self.ds = load_dataset("liuhuanjim013/kinetics400")
        print(self.ds['train'][0])
        exit()
        
        self.transform = self.transform
        
        # print(self.class_to_idx)
        # exit()

    # def __getitem__(self, index):
    #     path = self.video_lists[index]
    #     class_name = path.split('/')[-2]
    #     class_index = self.class_to_idx[class_name]

    #     vframes, aframes, info = torchvision.io.read_video(filename=path, pts_unit='sec', output_format='TCHW')
    #     # print("vframes",vframes.shape)
    #     total_frames = len(vframes)
    #     # print("total_frames: ", total_frames)
        
    #     # Sampling video frames
    #     # start_frame_ind, end_frame_ind = self.temporal_sample(total_frames)
    #     start_frame_ind, end_frame_ind = 0, total_frames-1
    #     assert end_frame_ind - start_frame_ind >= self.target_video_len, f"Video has too few frames: {total_frames} < {self.target_video_len}, sample:{path}"
    
    #     frame_indice = np.linspace(start_frame_ind, end_frame_ind-1, self.target_video_len, dtype=int)
    #     # print(frame_indice)
    #     video = vframes[frame_indice] #
    #     video = self.transform(video) # T C H W

    #     return {'video': video, 'video_name': class_index}

    def __len__(self):
        return len(self.video_lists)
    
    
if __name__ == '__main__':
    InternVid(None)