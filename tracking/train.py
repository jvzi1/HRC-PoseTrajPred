import torch.optim as optim
import torch.nn as nn
from trajectory_dataset import TrajectoryDataset
from transformer import TrajectoryTransformer
from torch.utils.data import DataLoader
import torch
# 初始化模型参数
num_joints = 33  # 人体关键点的数量
embed_size = 128  # 轨迹嵌入的大小
num_heads = 8  # 自注意力头的数量
num_layers = 4  # Transformer 层数
behavior_vocab_size = 6  # 行为标签数量
behavior_embed_size = 64  # 行为嵌入向量的大小
pred_length = 2  # 预测未来轨迹的长度
dataset_path = r"F:\video_rec_new\dataset\rec_728"
seq_len = 10  # 输入轨迹的长度
pred_len = 5  # 预测轨迹的长度
lstm_hidden_size = 256 # LSTM隐藏层
lstm_num_layers = 2 # LSTM的层数
lr = 1e-3

# 初始化模型
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = TrajectoryTransformer(num_joints, embed_size, num_heads, num_layers, behavior_vocab_size, behavior_embed_size, pred_length).to('cuda')
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

train_dataset = TrajectoryDataset(dataset_path, seq_len, pred_len, split="train")
val_dataset = TrajectoryDataset(dataset_path, seq_len, pred_len, split="val")
test_dataset = TrajectoryDataset(dataset_path, seq_len, pred_len, split="test")
dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True)
dataloader = DataLoader(val_dataset, batch_size=2, shuffle=True)
dataloader = DataLoader(test_dataset, batch_size=2, shuffle=True)


model.to(device=0)
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
