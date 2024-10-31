import torch
from torch.utils.data import Dataset, DataLoader

class TrajectoryDataset(Dataset):
    def __init__(self, trajectories, behaviors, future_trajectories, seq_len, pred_len):
        """
        Args:
            trajectories (list of lists): 轨迹数据，每个样本是 [[x1, y1], [x2, y2], ...]
            behaviors (list of int): 行为标签
            future_trajectories (list of lists): 未来轨迹数据
            seq_len (int): 输入轨迹的长度
            pred_len (int): 预测轨迹的长度
        """
        self.trajectories = trajectories
        self.behaviors = behaviors
        self.future_trajectories = future_trajectories
        self.seq_len = seq_len
        self.pred_len = pred_len
    
    def __len__(self):
        return len(self.trajectories)
    
    def __getitem__(self, idx):
        # 输入的轨迹序列
        trajectory = self.trajectories[idx][:self.seq_len]
        behavior = self.behaviors[idx]
        future_trajectory = self.future_trajectories[idx][:self.pred_len]

        # 将数据转换为 tensor
        trajectory = torch.tensor(trajectory, dtype=torch.float32)
        future_trajectory = torch.tensor(future_trajectory, dtype=torch.float32)
        behavior = torch.tensor(behavior, dtype=torch.long)
        
        return trajectory, behavior, future_trajectory

# 示例数据
trajectories = [[[100, 200], [105, 205], [110, 210]], [[120, 130], [125, 135], [130, 140]]]  # 两个样本
behaviors = [0, 1]  # 两个行为标签
future_trajectories = [[[115, 215], [120, 220]], [[135, 145], [140, 150]]]  # 对应的未来轨迹

dataset = TrajectoryDataset(trajectories, behaviors, future_trajectories, seq_len=3, pred_len=2)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
