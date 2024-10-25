import matplotlib.pyplot as plt
import torch
from trajectory_dataloader import TrajectoryDataset
from torch.utils.data import DataLoader
from transformer import TrajectoryTransformer
def plot_trajectory(real_traj, predicted_traj):
    real_traj = real_traj.cpu().numpy()
    predicted_traj = predicted_traj.cpu().detach().numpy()
    
    plt.plot(real_traj[:, 0], real_traj[:, 1], label='Real Trajectory', marker='o', color='b')
    plt.plot(predicted_traj[:, 0], predicted_traj[:, 1], label='Predicted Trajectory', marker='x', color='r')
    
    plt.legend()
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Trajectory Prediction')
    plt.show()

dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
model = TrajectoryTransformer(num_joints, embed_size, num_heads, num_layers, behavior_vocab_size, behavior_embed_size, pred_length).to('cuda')
# 示例：推理并可视化轨迹
model.eval()
with torch.no_grad():
    for trajectory, behavior, future_trajectory in dataloader:
        trajectory = trajectory.to('cuda')
        behavior = behavior.to('cuda')

        predicted_trajectory = model(trajectory, behavior)[0]  # 只可视化第一个样本
        plot_trajectory(future_trajectory[0], predicted_trajectory)
