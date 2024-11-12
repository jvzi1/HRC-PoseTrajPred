import torch.optim as optim
import torch.nn as nn
from trajectory_dataloader import TrajectoryDataset
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
# pred_length = 5  # 预测未来轨迹的长度
dataset_path = r"F:\video_rec_new\data\rec_728"
seq_len = 10  # 输入轨迹的长度
pred_len = 5  # 预测轨迹的长度
lstm_hidden_size = 256 # LSTM隐藏层
lstm_num_layers = 2 # LSTM的层数
lr = 1e-3

trajectory_input = torch.randn(32, 10, num_joints * 2)  # [batch_size, seq_len, num_joints * 2]
behavior_input = torch.randint(0, behavior_vocab_size, (32,))  # [batch_size]

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

def evaluate(model, val_loader, criterion):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for trajectory, behavior, future_trajectory in val_loader:
            trajectory = trajectory.to(device)
            behavior = behavior.to(device)
            future_trajectory = future_trajectory.to(device)

            predicted_trajectory = model(trajectory, behavior)
            loss = criterion(predicted_trajectory, future_trajectory)
            total_loss += loss.item()
    
    return total_loss / len(val_loader)

# 训练循环
num_epochs = 20
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for trajectory, behavior, future_trajectory in train_loader:
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
    val_loss = evaluate(model, val_loader, criterion)
    
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss / len(train_loader.dataset)}')
    print(f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {running_loss / len(train_loader.dataset):.4f}, Val Loss: {val_loss:.4f}')

# 测试模型
test_loss = evaluate(model, test_loader, criterion)
print(f'Test Loss: {test_loss:.4f}')


# if __name__ == "__main__":
#     num_joints = 33
#     embed_size = 128
#     num_heads = 8
#     num_layers = 4
#     behavior_vocab_size = 10
#     behavior_embed_size = 32
#     pred_length = 4
#     lstm_hidden_size = 256
#     lstm_num_layers = 2

#     # 假设输入是一个批次的历史轨迹和行为标签
#     trajectory_input = torch.randn(32, 10, num_joints * 2)  # [batch_size, seq_len, num_joints * 2]
#     behavior_input = torch.randint(0, behavior_vocab_size, (32,))  # [batch_size]

#     # 初始化模型（确保包含所有参数）
#     model = TrajectoryTransformer(
#         num_joints=num_joints,
#         embed_size=embed_size,
#         num_heads=num_heads,
#         num_layers=num_layers,
#         behavior_vocab_size=behavior_vocab_size,
#         behavior_embed_size=behavior_embed_size,
#         pred_length=pred_length,
#         lstm_hidden_size=lstm_hidden_size,
#         lstm_num_layers=lstm_num_layers
#     ).to('cuda')

#     # 前向传播，预测未来轨迹
#     future_trajectory = model(trajectory_input.to('cuda'), behavior_input.to('cuda'))

#     print(future_trajectory.shape)  # 输出形状: [batch_size, pred_length, num_joints, 2]