import torch.optim as optim
import torch.nn as nn
from trajectory_dataloader import TrajectoryDataset
from transformer import TrajectoryTransformer
from torch.utils.data import DataLoader
import torch

import os
# 初始化模型参数
num_joints = 33  # 人体关键点的数量
embed_size = 128  # 轨迹嵌入的大小
num_heads = 8  # 自注意力头的数量
num_layers = 4  # Transformer 层数
behavior_vocab_size = 6  # 行为标签数量
behavior_embed_size = 64  # 行为嵌入向量的大小
# pred_length = 5  # 预测未来轨迹的长度
dataset_path = r"F:\video_rec_new\data\rec_728"
seq_len = 10  # 输入轨迹的长度
pred_len = 5  # 预测轨迹的长度
lstm_hidden_size = 256 # LSTM隐藏层
lstm_num_layers = 2 # LSTM的层数
lr = 1e-3


# 初始化模型
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = TrajectoryTransformer(
    num_joints=num_joints,
    embed_size=embed_size,
    num_heads=num_heads,
    num_layers=num_layers,
    behavior_vocab_size=behavior_vocab_size,
    behavior_embed_size=behavior_embed_size,
    pred_length=pred_len,
    lstm_hidden_size = lstm_hidden_size,
    lstm_num_layers = lstm_num_layers,
).to("cuda")

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

train_dataset = TrajectoryDataset(dataset_path, seq_len, pred_len, split="train")
val_dataset = TrajectoryDataset(dataset_path, seq_len, pred_len, split="val")
test_dataset = TrajectoryDataset(dataset_path, seq_len, pred_len, split="test")
train_loader  = DataLoader(train_dataset, batch_size=2, shuffle=True)
val_loader  = DataLoader(val_dataset, batch_size=2, shuffle=True)
test_loader  = DataLoader(test_dataset, batch_size=2, shuffle=True)


model.to(device=0)

def evaluate(model, val_loader):
    model.eval()
    total_loss = 0.0
    total_samples = 0
    with torch.no_grad():
        for trajectory, behavior, future_trajectory in val_loader:
            trajectory = trajectory.to(device)
            behavior = behavior.to(device)
            future_trajectory = future_trajectory.to(device)

            predicted_trajectory = model(trajectory, behavior)
            loss = compute_loss(predicted_trajectory, future_trajectory)
            total_loss += loss.item() * trajectory.size(0)
            total_samples += trajectory.size(0)
    assert total_samples == len(val_loader.dataset)
    return total_loss / total_samples

def compute_loss(predicted, target):
    """
    计算 3D 轨迹的欧氏距离损失 (L2 Loss) 和 Smooth L1 Loss 的组合。
    """
    # 预测值和真实值的形状: [batch_size, pred_len, num_joints * 3]
    
    # 1. 计算欧氏距离损失 (L2 Loss)
    # 对最后一个维度分成 (x, y, z) 坐标
    predicted = predicted.view(predicted.size(0), predicted.size(1), -1, 3)
    target = target.view(target.size(0), target.size(1), -1, 3)
    
    # 计算每个关键点的 3D 欧氏距离
    l2_loss = torch.sqrt(((predicted - target) ** 2).sum(dim=-1)).mean()
    
    # 2. 平滑 L1 损失 (Smooth L1 Loss)
    smooth_l1_loss = nn.SmoothL1Loss()
    l1_loss = smooth_l1_loss(predicted, target)
    
    # 3. 返回组合损失
    return l2_loss + 0.5 * l1_loss
# 创建保存模型的文件夹
os.makedirs("model_result/trajectory", exist_ok=True)

best_val_loss = float('inf')
best_model_path = "model_result/trajectory/best_model.pth"
last_model_path = "model_result/trajectory/last_model.pth"
    
# 训练循环
num_epochs = 50
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for trajectory, behavior, future_trajectory in train_loader:
        trajectory = trajectory.to('cuda')
        behavior = behavior.to('cuda')
        future_trajectory = future_trajectory.to('cuda')

        # 前向传播
        predicted_trajectory = model(trajectory, behavior)
        loss = compute_loss(predicted_trajectory, future_trajectory)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * trajectory.size(0)
    train_loss = running_loss / len(train_loader.dataset)
    val_loss = evaluate(model, val_loader)
    
    # print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss / len(train_loader.dataset)}')
    print(f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

    # 保存最佳模型
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': val_loss
        }, best_model_path)
        print(f"Best model saved at epoch {epoch + 1} with Val Loss: {val_loss:.4f}")

    # 保存最后一个模型
    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': val_loss
    }, last_model_path)
# 测试模型
test_loss = evaluate(model, test_loader)
print(f'Test Loss: {test_loss:.4f}')

