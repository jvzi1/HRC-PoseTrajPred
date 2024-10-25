# update 1021:更新了train，val，test数据集


import torch
from torch.utils.data import Dataset, DataLoader
import os
import json
import random
class TrajectoryDataset(Dataset):
    def __init__(self, ori_data_path, seq_len, pred_len, split="train", train_ratio=0.7, val_ratio=0.15):
        """
        Args:
            ori_data_path (str): 数据集的根路径。
            seq_len (int): 输入轨迹的长度。
            pred_len (int): 预测轨迹的长度。
            split (str): "train", "val", or "test" 用于指定训练、验证、或测试集。
            train_ratio (float): 训练集所占比例。
            val_ratio (float): 验证集所占比例。
        """
        self.seq_len = seq_len # 过去轨迹长度
        self.pred_len = pred_len # 未来轨迹长度
        self.trajectories = []
        self.behaviors = []  # 行为标签
        self.future_trajectories = []  # 未来的轨迹数据
        
        # 划分数据集
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.split = split

        # 预处理并加载数据
        self.preprocess(ori_data_path)

        # self.dataset_path = dataset_path  # 数据集的地址
        # self.split = images_path  # 训练集，测试集，验证集的名字
        # self.clip_len = clip_len  # 生成的数据的的深度的值

    
    def __len__(self):
        return len(self.trajectories)
    
    def __getitem__(self, idx):
        # 获取输入的轨迹序列
        trajectory = self.trajectories[idx][:self.seq_len]
        behavior = self.behaviors[idx]
        future_trajectory = self.trajectories[idx][self.seq_len:self.seq_len+self.pred_len]

        # 将数据转换为 tensor
        trajectory = torch.tensor(trajectory, dtype=torch.float32)
        future_trajectory = torch.tensor(future_trajectory, dtype=torch.float32)
        behavior = torch.tensor(behavior, dtype=torch.long)
        
        return trajectory, behavior, future_trajectory
    
    def preprocess(self, ori_data_path):
        all_trajectories = []
        all_behaviors = []
        all_future_trajectories = []
        # import pdb;pdb.set_trace()
        for dir_name in os.listdir(ori_data_path):
            dir_path = os.path.join(ori_data_path, dir_name)
            if os.path.isdir(dir_path):
                annotated_keypoints_path = os.path.join(dir_path, 'annotated_videos')
                if os.path.exists(annotated_keypoints_path):
                    for label_folder in os.listdir(annotated_keypoints_path):
                        label_folder_path = os.path.join(annotated_keypoints_path, label_folder)
                        if os.path.isdir(label_folder_path):
                            # label 文件夹名即为标签
                            label = int(label_folder)  # 转换为整数标签
                            for subfolder in os.listdir(label_folder_path):
                                subfolder_path = os.path.join(label_folder_path, subfolder)
                                for keypoints_file in os.listdir(subfolder_path):
                                    if keypoints_file.endswith(".json"):
                                        keypoints_path = os.path.join(subfolder_path, keypoints_file)
                                        keypoints_trajectory = self.get_keypoint_position(keypoints_path)  # [frame_len, 33, 3]

                                        # 检查轨迹是否足够长以满足 seq_len + pred_len
                                        if len(keypoints_trajectory) >= self.seq_len + self.pred_len:
                                            all_trajectories.append(keypoints_trajectory)
                                            all_behaviors.append(label)
                                            # TODO
                                            all_future_trajectories.append(keypoints_trajectory)
        
        self.split_data(all_trajectories, all_behaviors, all_future_trajectories)

    def split_data(self, trajectories, behaviors, future_trajectories):
        combined = list(zip(trajectories, behaviors, future_trajectories))
        random.shuffle(combined)
        trajectories[:], behaviors[:], future_trajectories[:] = zip(*combined)
        
        # 计算划分比例的索引
        train_idx = int(len(trajectories) * self.train_ratio)
        val_idx = int(len(trajectories) * (self.train_ratio + self.val_ratio))

        # 划分数据集
        if self.split == "train":
            self.trajectories = trajectories[:train_idx]
            self.behaviors = behaviors[:train_idx]
            self.future_trajectories = future_trajectories[:train_idx]
        elif self.split == "val":
            self.trajectories = trajectories[train_idx:val_idx]
            self.behaviors = behaviors[train_idx:val_idx]
            self.future_trajectories = future_trajectories[train_idx:val_idx]
        elif self.split == "test":
            self.trajectories = trajectories[val_idx:]
            self.behaviors = behaviors[val_idx:]
            self.future_trajectories = future_trajectories[val_idx:]
        else:
            raise ValueError("split 参数必须是 'train', 'val' 或 'test'")
        
    def get_keypoint_position(self, json_path):
        """
        输出一个维度为[frame_len, 33, 3]的数据
        """
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
    
if __name__ == "__main__":
    dataset_path = r"F:\video_rec_new\dataset\rec_728"
    seq_len = 10  # 输入轨迹的长度
    pred_len = 5  # 预测轨迹的长度
    
    # 创建训练集、验证集、测试集数据集
    train_dataset = TrajectoryDataset(dataset_path, seq_len, pred_len, split="train")
    val_dataset = TrajectoryDataset(dataset_path, seq_len, pred_len, split="val")
    test_dataset = TrajectoryDataset(dataset_path, seq_len, pred_len, split="test")
    
    # 打印数据集大小
    print(f"训练集大小: {len(train_dataset)}")
    print(f"验证集大小: {len(val_dataset)}")
    print(f"测试集大小: {len(test_dataset)}")
    
    # 获取训练集第一个样本
    trajectory, behavior, future_trajectory = train_dataset[0]
    print(f"轨迹形状: {trajectory.shape}, 行为标签: {behavior}, 未来轨迹形状: {future_trajectory.shape}")