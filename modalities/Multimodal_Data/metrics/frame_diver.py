import numpy as np 
from tqdm import tqdm
import cv2

def get_flow_farn(dataset):
    flowFarn = []
    for i in tqdm(range(len(dataset))):
        # print(dataset.get_OpticalFlowFarneback(i))
        flowFarn.append(dataset.get_OpticalFlowFarneback(i))
        
    return np.mean(flowFarn)




def get_frame_diversity(video_data):
    res = []
    for data in video_data:
        video_path = data['video']
        cap = cv2.VideoCapture(video_path)
        
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
            
            if frame_cnt% 10 != 0:
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
        
        res.append(np.mean(motion_scores))
    return np.mean(res)