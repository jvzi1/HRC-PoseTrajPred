import os
import torch
from torch.utils.data import Dataset
import json
import random

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
        self.camera_names = [] # 相机名称标签

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
        camera_name = torch.tensor(self.camera_names[idx], dtype=torch.long)
        return trajectory, behavior, future_trajectory, camera_name
    
    def preprocess(self, ori_data_path):
        for dir_name in os.listdir(ori_data_path):
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
                                        camera_name = int(keypoints_file.split("_")[1]) - 1
                                        keypoints_path = os.path.join(subfolder_path, keypoints_file)
                                        keypoints_trajectory = self.get_keypoint_position(keypoints_path)
                                        # 生成滑动窗口样本
                                        self.generate_samples(keypoints_trajectory, label, camera_name)
        self.split_data(self.trajectories, self.behaviors, self.future_trajectories, self.camera_names)
    def generate_samples(self, keypoints_trajectory, label, camera_name):
        """使用滑动窗口从轨迹中生成 (seq_len, pred_len) 对"""
        num_frames = len(keypoints_trajectory)
        for i in range(0, num_frames - self.seq_len - self.pred_len + 1, self.step_size):
            seq = keypoints_trajectory[i:i + self.seq_len]  # 输入序列
            future = keypoints_trajectory[i + self.seq_len:i + self.seq_len + self.pred_len]  # 预测序列
            self.trajectories.append(seq)
            self.behaviors.append(label)
            self.future_trajectories.append(future)
            self.camera_names.append(camera_name)

    def get_keypoint_position(self, json_path):
        """从 JSON 文件加载关键点数据，输出 [frame_len, 33, 3] 的数据"""
        keypoints_trajectory = []
        with open(json_path, "r") as f:
            data = json.load(f)
        for frame in data:
            tmp = []
            for joint in frame["keypoints"]:
                keypoints_x = joint["x"]
                keypoints_y = joint["y"]
                keypoints_z = joint["z"]
                tmp.append([keypoints_x, keypoints_y, keypoints_z])
            keypoints_trajectory.append(tmp)
        return keypoints_trajectory

    def split_data(self, trajectories, behaviors, future_trajectories, camera_names):
        combined = list(zip(trajectories, behaviors, future_trajectories, camera_names))
        random.shuffle(combined)
        trajectories[:], behaviors[:], future_trajectories[:], camera_names[:] = zip(*combined)
        
        # 计算划分比例的索引
        train_idx = int(len(trajectories) * self.train_ratio)
        val_idx = int(len(trajectories) * (self.train_ratio + self.val_ratio))

        # 划分数据集
        if self.split == "train":
            self.trajectories = trajectories[:train_idx]
            self.behaviors = behaviors[:train_idx]
            self.future_trajectories = future_trajectories[:train_idx]
            self.camera_names = camera_names[:train_idx]
        elif self.split == "val":
            self.trajectories = trajectories[train_idx:val_idx]
            self.behaviors = behaviors[train_idx:val_idx]
            self.future_trajectories = future_trajectories[train_idx:val_idx]
            self.camera_names = camera_names[train_idx:val_idx]
        elif self.split == "test":
            self.trajectories = trajectories[val_idx:]
            self.behaviors = behaviors[val_idx:]
            self.future_trajectories = future_trajectories[val_idx:]
            self.camera_names = camera_names[val_idx:]
        else:
            raise ValueError("split 参数必须是 'train', 'val' 或 'test'")

if __name__ == "__main__":
    dataset_path = r"F:\video_rec_new\data\rec_728"
    seq_len = 80
    pred_len = 20
    step_size = 5
    
    train_dataset = TrajectoryDataset(dataset_path, seq_len, pred_len, split="train", step_size=step_size)
    val_dataset = TrajectoryDataset(dataset_path, seq_len, pred_len, split="val", step_size=step_size)
    test_dataset = TrajectoryDataset(dataset_path, seq_len, pred_len, split="test", step_size=step_size)
    
    print(f"训练集大小: {len(train_dataset)}")
    print(f"验证集大小: {len(val_dataset)}")
    print(f"测试集大小: {len(test_dataset)}")
    
    trajectory, behavior, future_trajectory = train_dataset[0]

    print(f"轨迹形状: {trajectory.shape}, 行为标签: {behavior}, 未来轨迹形状: {future_trajectory.shape}")
