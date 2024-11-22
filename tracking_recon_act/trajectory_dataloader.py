import os
import torch
from torch.utils.data import Dataset
import json
import random
import cv2
from tqdm import tqdm
class TrajectoryDataset(Dataset):
    def __init__(self, ori_data_path, seq_len, pred_len, split="train", train_ratio=0.7, val_ratio=0.15, step_size=1):
        """
        Args:
            ori_data_path (str): 数据集的根路径。
            seq_len (int): 输入轨迹的长度。
            pred_len (int): 预测轨迹的长度。
            split (str): "train", "val", or "test" 用于指定训练、验证、或测试集。
            train_ratio (float): 训练集所占比例。
            val_ratio (float): 验证集所占比例。
            step_size (int): 滑动窗口的步长。
        """
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.step_size = step_size
        self.trajectories = []
        self.behaviors = []  # 行为标签
        self.future_trajectories = []  # 未来的轨迹数据
        self.images = [] # 对应轨迹的图像帧

        # 划分数据集
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.split = split

        # 预处理并加载数据
        self.preprocess(ori_data_path)

    def __len__(self):
        return len(self.trajectories)
    
    def __getitem__(self, idx):
        trajectory = torch.tensor(self.trajectories[idx], dtype=torch.float32).view(self.seq_len, -1)
        future_trajectory = torch.tensor(self.future_trajectories[idx], dtype=torch.float32).view(self.pred_len, -1)
        behavior = torch.tensor(self.behaviors[idx], dtype=torch.long)
        images = torch.stack(self.images[idx], dim=0)
        return trajectory, behavior, future_trajectory, images
    
    def preprocess(self, ori_data_path):
        all_trajectories = []
        all_behaviors = []
        
        for dir_name in tqdm(os.listdir(ori_data_path)):
            dir_path = os.path.join(ori_data_path, dir_name)
            if os.path.isdir(dir_path):
                annotated_keypoints_path = os.path.join(dir_path, 'annotated_videos')
                if os.path.exists(annotated_keypoints_path):
                    for label_folder in os.listdir(annotated_keypoints_path):
                        label_folder_path = os.path.join(annotated_keypoints_path, label_folder)
                        if os.path.isdir(label_folder_path):
                            label = int(label_folder)
                            for subfolder in os.listdir(label_folder_path):
                                subfolder_path = os.path.join(label_folder_path, subfolder)
                                for keypoints_file in os.listdir(subfolder_path):
                                    if keypoints_file.endswith(".json"):
                                        keypoints_path = os.path.join(subfolder_path, keypoints_file)
                                        frame_index_list, keypoints_trajectory = self.get_keypoint_position(keypoints_path)
                                        video_path = keypoints_path.replace("_keypoints.json", ".mp4")
                                        video_frames = self.get_images(video_path)
                                        # 生成滑动窗口样本
                                        self.generate_samples(frame_index_list, keypoints_trajectory, label, video_frames)
        self.split_data(self.trajectories, self.behaviors, self.future_trajectories, self.images)
    def generate_samples(self, frame_index_list, keypoints_trajectory, label, video_frames):
        """使用滑动窗口从轨迹中生成 (seq_len, pred_len) 对"""
        if not frame_index_list:
            print(f"警告：视频帧索引列表为空，跳过该样本。")
            return
        num_frames = len(keypoints_trajectory)
        if max(frame_index_list) >= len(video_frames):
            raise ValueError("frame_index_list index is out of range")
        
        aligned_frames = [video_frames[idx] for idx in frame_index_list]
        for i in range(0, num_frames - self.seq_len - self.pred_len + 1, self.step_size):
            seq = keypoints_trajectory[i:i + self.seq_len]  # 输入序列
            future = keypoints_trajectory[i + self.seq_len:i + self.seq_len + self.pred_len]  # 预测序列
            seq_images = aligned_frames[i:i + self.seq_len]
            self.trajectories.append(seq)
            self.behaviors.append(label)
            self.future_trajectories.append(future)
            self.images.append(seq_images)

    def get_keypoint_position(self, json_path):
        """从 JSON 文件加载关键点数据，输出 [frame_len, 33, 3] 的数据"""
        keypoints_trajectory = []
        frame_index_list = []
        with open(json_path, "r") as f:
            data = json.load(f)
        for frame in data:
            tmp_keypoints = []
            frame_index = frame["frame_index"]
            for joint in frame["keypoints"]:
                keypoints_x = joint["x"]
                keypoints_y = joint["y"]
                keypoints_z = joint["z"]
                tmp_keypoints.append([keypoints_x, keypoints_y, keypoints_z])
            frame_index_list.append(frame_index)
            keypoints_trajectory.append(tmp_keypoints)
        return frame_index_list, keypoints_trajectory
    
    def get_images(self, video_path):
        """读取视频帧并返回 [frame_len, C, H, W] 的图像张量"""
        cap = cv2.VideoCapture(video_path)
        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            # 转换为 RGB 格式并调整图像大小
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (64, 64))
            frame_tensor = torch.tensor(frame, dtype=torch.float32).permute(2, 0, 1) / 255.0  # [C, H, W]
            frames.append(frame_tensor)
            del frame
        cap.release()
        return frames
    

    def split_data(self, trajectories, behaviors, future_trajectories, images):
        combined = list(zip(trajectories, behaviors, future_trajectories, images))
        random.shuffle(combined)
        trajectories[:], behaviors[:], future_trajectories[:], images[:] = zip(*combined)
        
        # 计算划分比例的索引
        train_idx = int(len(trajectories) * self.train_ratio)
        val_idx = int(len(trajectories) * (self.train_ratio + self.val_ratio))

        # 划分数据集
        if self.split == "train":
            self.trajectories = trajectories[:train_idx]
            self.behaviors = behaviors[:train_idx]
            self.future_trajectories = future_trajectories[:train_idx]
            self.images = images[:train_idx]
        elif self.split == "val":
            self.trajectories = trajectories[train_idx:val_idx]
            self.behaviors = behaviors[train_idx:val_idx]
            self.future_trajectories = future_trajectories[train_idx:val_idx]
            self.images = images[train_idx:val_idx]
        elif self.split == "test":
            self.trajectories = trajectories[val_idx:]
            self.behaviors = behaviors[val_idx:]
            self.future_trajectories = future_trajectories[val_idx:]
            self.images = images[val_idx:]
        else:
            raise ValueError("split 参数必须是 'train', 'val' 或 'test'")

if __name__ == "__main__":
    dataset_path = r"F:\video_rec_new\data\rec_728"
    seq_len = 40
    pred_len = 12
    step_size = 20
    
    train_dataset = TrajectoryDataset(dataset_path, seq_len, pred_len, split="train", step_size=step_size)
    val_dataset = TrajectoryDataset(dataset_path, seq_len, pred_len, split="val", step_size=step_size)
    test_dataset = TrajectoryDataset(dataset_path, seq_len, pred_len, split="test", step_size=step_size)
    
    print(f"训练集大小: {len(train_dataset)}")
    print(f"验证集大小: {len(val_dataset)}")
    print(f"测试集大小: {len(test_dataset)}")
    
    trajectory, behavior, future_trajectory, images = train_dataset[0]

    print(f"轨迹形状: {trajectory.shape}, 行为标签: {behavior}, 未来轨迹形状: {future_trajectory.shape}, 图像形状: {images.shape}")
