import torch.optim as optim
import torch.nn as nn
from trajectory_dataloader import TrajectoryDataset
from transformer import TrajectoryTransformer
from torch.utils.data import DataLoader

# 初始化模型参数
num_joints = 17  # 人体关键点的数量
embed_size = 128  # 轨迹嵌入的大小
num_heads = 8  # 自注意力头的数量
num_layers = 4  # Transformer 层数
behavior_vocab_size = 5  # 行为标签数量
behavior_embed_size = 64  # 行为嵌入向量的大小
pred_length = 2  # 预测未来轨迹的长度

# 初始化模型
model = TrajectoryTransformer(num_joints, embed_size, num_heads, num_layers, behavior_vocab_size, behavior_embed_size, pred_length).to('cuda')
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

dataset = TrajectoryDataset(trajectories, behaviors, future_trajectories, seq_len=3, pred_len=2)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# 训练循环
num_epochs = 20
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for trajectory, behavior, future_trajectory in dataloader:
        trajectory = trajectory.to('cuda')
        behavior = behavior.to('cuda')
        future_trajectory = future_trajectory.to('cuda')

        # 前向传播
        predicted_trajectory = model(trajectory, behavior)
        loss = criterion(predicted_trajectory, future_trajectory)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * trajectory.size(0)
    
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss / len(dataloader.dataset)}')
