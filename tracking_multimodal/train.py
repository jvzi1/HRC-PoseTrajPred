import torch.optim as optim
import torch.nn as nn
from trajectory_dataset import TrajectoryDataset
from torch.utils.data import DataLoader
import torch
from loguru import logger 
import os
import time
from tqdm import tqdm
from multimodal_models import MultiModalTrajectoryPredictor

# 初始化模型参数
num_joints = 33  # 人体关键点的数量
embed_size = 128  # 轨迹嵌入的大小
num_heads = 8  # 自注意力头的数量
num_layers = 4  # Transformer 层数
behavior_vocab_size = 6  # 行为标签数量
behavior_embed_size = 64  # 行为嵌入向量的大小
# pred_length = 5  # 预测未来轨迹的长度
dataset_path = r"F:\video_rec_new\data\rec_728"
seq_len = 20  # 输入轨迹的长度
pred_len = 8  # 预测轨迹的长度
lstm_hidden_size = 256 # LSTM隐藏层
lstm_num_layers = 2 # LSTM的层数
lr = 1e-4
step_size = 10

# 初始化模型
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = MultiModalTrajectoryPredictor(seq_len=seq_len, pred_len=pred_len, behavior_num_classes=behavior_vocab_size, hidden_dim=256)


criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=lr) # 尝试修改optim参数，尝试使用AdamW优化器

logger.info("start data process")
train_dataset = TrajectoryDataset(dataset_path, seq_len, pred_len, split="train", step_size=step_size)
val_dataset = TrajectoryDataset(dataset_path, seq_len, pred_len, split="val", step_size=step_size)
test_dataset = TrajectoryDataset(dataset_path, seq_len, pred_len, split="test", step_size=step_size)
train_loader  = DataLoader(train_dataset, batch_size=2, shuffle=True)
val_loader  = DataLoader(val_dataset, batch_size=2, shuffle=True)
test_loader  = DataLoader(test_dataset, batch_size=2, shuffle=True)
logger.info("finish data process")

model.to(device)

def compute_loss(predicted, target):
    """
    计算 3D 轨迹的欧氏距离损失 (L2 Loss) 和 Smooth L1 Loss 的组合。
    """
    # 预测值和真实值的形状: [batch_size, pred_len, num_joints * 3]
    
    # 1.L2 Loss
    predicted = predicted.view(predicted.size(0), predicted.size(1), -1, 3)
    target = target.view(target.size(0), target.size(1), -1, 3)
    l2_loss = torch.sqrt(((predicted - target) ** 2).sum(dim=-1)).mean()
    
    # 2.Smooth L1 Loss
    smooth_l1_loss = nn.SmoothL1Loss()
    l1_loss = smooth_l1_loss(predicted, target)
    
    return l2_loss + 0.5 * l1_loss
# 创建保存模型的文件夹
os.makedirs("model_result/trajectory", exist_ok=True)

best_val_loss = float('inf')
best_model_path = "model_result/trajectory/best_model.pth"
last_model_path = "model_result/trajectory/last_model.pth"

logger.info("start training")
# 训练循环
num_epochs = 30
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    train_loader_tqdm = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}", unit="batch")
    for trajectory, behavior, future_trajectory, images in train_loader_tqdm:
        trajectory = trajectory.to(device)
        behavior = behavior.to(device)
        future_trajectory = future_trajectory.to(device)
        images = images.to(device)
        # 前向传播
        predicted_trajectory = model(trajectory, images, behavior)
        loss = criterion(predicted_trajectory, future_trajectory)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * trajectory.size(0)
    train_loss = running_loss / len(train_loader.dataset) * 100
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for trajectory, behavior, future_trajectory, images in val_loader:
            trajectory = trajectory.to(device)
            behavior = behavior.to(device)
            future_trajectory = future_trajectory.to(device)
            images = images.to(device)

            predicted_trajectory = model(trajectory, images, behavior)
            loss = criterion(predicted_trajectory, future_trajectory)

            running_loss += loss.item() * trajectory.size(0)
        
    val_loss = running_loss / len(val_loader.dataset) * 100

    
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
model.eval()
running_loss = 0.0
with torch.no_grad():
    for trajectory, behavior, future_trajectory, images in test_loader:
        trajectory = trajectory.to(device)
        behavior = behavior.to(device)
        future_trajectory = future_trajectory.to(device)
        images = images.to(device)

        predicted_trajectory = model(trajectory, images, behavior)
        loss = criterion(predicted_trajectory, future_trajectory)

        running_loss += loss.item() * trajectory.size(0)
    
test_loss = running_loss / len(test_loader.dataset) * 100
print(f'Test Loss: {test_loss:.4f}')

