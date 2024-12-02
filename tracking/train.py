import torch.optim as optim
import torch.nn as nn
from trajectory_dataset import TrajectoryDataset
from trajectory_model import TrajectoryTransformerModel
from torch.utils.data import DataLoader
import torch
from loguru import logger 
import os
import time
from tqdm import tqdm
import argparse
import numpy as np
from time import time


def get_parse_arguments():
    parser = argparse.ArgumentParser()
    # 模型参数
    parser.add_argument('--num_joints', type=int, default=33, help='Number of joints in the input')
    parser.add_argument('--embed_size', type=int, default=128, help='Embedding size for trajectories')
    parser.add_argument('--num_heads', type=int, default=8, help='Number of attention heads in Transformer')
    parser.add_argument('--num_layers', type=int, default=4, help='Number of Transformer layers')
    parser.add_argument('--behavior_vocab_size', type=int, default=6, help='Vocabulary size for behaviors')
    parser.add_argument('--behavior_embed_size', type=int, default=32, help='Embedding size for behaviors')
    parser.add_argument('--camera_name_vocab_size', type=int, default=3, help='Vocabulary size for camera names')
    parser.add_argument('--camera_name_embed_size', type=int, default=32, help='Embedding size for camera names')
    parser.add_argument('--pred_length', type=int, default=8, help='Length of predicted trajectory')
    parser.add_argument('--lstm_hidden_size', type=int, default=256, help='Hidden size of LSTM')
    parser.add_argument('--lstm_num_layers', type=int, default=2, help='Number of LSTM layers')
    
    # 训练参数
    parser.add_argument('--dataset_path', type=str, default="data/rec_728", help='Path to the dataset')
    parser.add_argument('--seq_len', type=int, default=40, help='Length of input sequence')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size for training and evaluation')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--num_epochs', type=int, default=80, help='Number of training epochs')
    parser.add_argument('--step_size', type=int, default=5, help='Step size for data sampling')
    parser.add_argument('--save_model_path', type=str, default='best_model_1129.pth', help='Save model checkpoint')
    # 中断训练
    parser.add_argument('--resume_train', action='store_true', help='Resume training from checkpoint')
    parser.add_argument('--checkpoint_path', type=str, default="model_result/trajectory/best_model_1128.pth", help='Path to the checkpoint file')
    parser.add_argument('--output_dir', type=str, default="model_result/trajectory",
                        help='Directory to save model checkpoints')
    
    return parser.parse_args()

def compute_loss(predicted, target):
    """
    计算 3D 轨迹的欧氏距离损失 (L2 Loss) 和 Smooth L1 Loss 的组合。
    """
    predicted = predicted.view(predicted.size(0), predicted.size(1), -1, 3)
    target = target.view(target.size(0), target.size(1), -1, 3)
    l2_loss = torch.sqrt(((predicted - target) ** 2).sum(dim=-1)).mean()
    smooth_l1_loss = nn.SmoothL1Loss()
    l1_loss = smooth_l1_loss(predicted, target)
    
    return l2_loss + 0.5 * l1_loss

def init_model(num_joints, embed_size, num_heads, 
               num_layers, behavior_vocab_size, camera_name_vocab_size, 
               behavior_embed_size, camera_name_embed_size, pred_length, 
               lstm_hidden_size, lstm_num_layers, device):
    
    model = TrajectoryTransformerModel(
        num_joints=num_joints,
        embed_size=embed_size,
        num_heads=num_heads,
        num_layers=num_layers,
        behavior_vocab_size=behavior_vocab_size,
        camera_name_vocab_size=camera_name_vocab_size,
        behavior_embed_size=behavior_embed_size,
        camera_name_embed_size=camera_name_embed_size,
        pred_length=pred_length,
        lstm_hidden_size=lstm_hidden_size,
        lstm_num_layers=lstm_num_layers
    ).to(device)

    return model

def load_data(dataset_path, seq_len, pred_length, step_size, batch_size):
    logger.info("Loading datasets...")
    train_dataset = TrajectoryDataset(dataset_path, seq_len, pred_length, split="train",
                                      step_size=step_size)
    val_dataset = TrajectoryDataset(dataset_path, seq_len, pred_length, split="val",
                                      step_size=step_size)
    test_dataset = TrajectoryDataset(dataset_path, seq_len, pred_length, split="test",
                                      step_size=step_size)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    logger.info("finish data process")
    return train_loader, val_loader, test_loader

def train(model, train_loader, optimizer, device, epoch, num_epochs):
    model.train()
    running_loss = 0.0
    train_loader_tqdm = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}", unit="batch")
    for trajectory, behavior, future_trajectory, camera_name in train_loader_tqdm:
        trajectory = trajectory.to(device)
        behavior = behavior.to(device)
        future_trajectory = future_trajectory.to(device)
        camera_name = camera_name.to(device)
        # 前向传播
        predicted_trajectory = model(trajectory, behavior, camera_name)
        loss = compute_loss(predicted_trajectory, future_trajectory) * 10
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * trajectory.size(0)
    train_loss = running_loss / len(train_loader.dataset)
    return train_loss
def evaluate(model, val_loader, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for trajectory, behavior, future_trajectory, camera_name in val_loader:
            trajectory = trajectory.to(device)
            behavior = behavior.to(device)
            future_trajectory = future_trajectory.to(device)
            camera_name = camera_name.to(device)

            predicted_trajectory = model(trajectory, behavior, camera_name)
            loss = compute_loss(predicted_trajectory, future_trajectory) * 10
            total_loss += loss.item() * trajectory.size(0)
    return total_loss / len(val_loader.dataset)

def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = init_model(args.num_joints, args.embed_size, args.num_heads, 
                       args.num_layers, args.behavior_vocab_size, args.camera_name_vocab_size, 
                       args.behavior_embed_size, args.camera_name_embed_size, args.pred_length, 
                       args.lstm_hidden_size, args.lstm_num_layers, device)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr) # 尝试修改optim参数，尝试使用AdamW优化器
    train_loader, val_loader, test_loader = load_data(args.dataset_path, args.seq_len, args.pred_length, args.step_size, args.batch_size)

    os.makedirs(args.output_dir, exist_ok=True)
    best_model_path = os.path.join(args.output_dir, args.save_model_path)
    last_model_path = os.path.join(args.output_dir, args.save_model_path.replace('best', 'last'))
    logger.info(f"check {args.save_model_path.replace('best', 'last')}")
    start_epoch = 0
    best_val_loss = float('inf')
    # 中断训练
    if args.resume_train and os.path.exists(args.checkpoint_path):
        logger.info(f"Resuming training from checkpoint: {args.checkpoint_path}")
        checkpoint = torch.load(args.checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint['loss']

    logger.info("Starting training...")
    for epoch in range(start_epoch, args.num_epochs):
        train_loss = train(model, train_loader, optimizer, device, epoch, args.num_epochs)
        val_loss = evaluate(model, val_loader, device)
        logger.info(f'Epoch {epoch + 1}/{args.num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': val_loss
            }, best_model_path)
            logger.info(f"Best model saved at epoch {epoch + 1} with Val Loss: {val_loss:.4f}")

        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': val_loss
        }, last_model_path)
    test_loss = evaluate(model, test_loader, device)
    logger.info(f"Test Loss: {test_loss:.4f}")

    metrics = evaluate_metrics(model, test_loader, device)
    logger.info(f"Test MPJPE: {metrics['mpjpe']:.4f}")
    logger.info(f"Test PCK: {metrics['pck']:.4f}")
    logger.info(f"Test OKS: {metrics['oks']:.4f}")
    logger.info(f"Model Parameter Count: {metrics['num_params']}")
    logger.info(f"Model Inference Speed (FPS): {metrics['fps']:.2f}")

def evaluate_metrics(model, test_loader, device):

    model.eval()
    predicted_list = []
    ground_truth_list = []
    inference_times = []

    sigmas = np.ones((33,)) * 0.1  # 假设的关节标准差
    areas = np.ones((len(test_loader.dataset),))  # 假设目标区域大小为1

    with torch.no_grad():
        for trajectory, _, future_trajectory, _ in test_loader:
            trajectory = trajectory.to(device)
            future_trajectory = future_trajectory.to(device)

            # 测量推理时间
            start_time = time()
            predicted = model(trajectory)
            inference_times.append(time() - start_time)

            predicted = predicted.cpu().numpy()
            future_trajectory = future_trajectory.cpu().numpy()

            predicted_list.append(predicted)
            ground_truth_list.append(future_trajectory)

    predicted = np.concatenate(predicted_list, axis=0)
    ground_truth = np.concatenate(ground_truth_list, axis=0)

    # MPJPE
    mpjpe = compute_mpjpe(predicted, ground_truth)

    # PCK
    threshold = 0.5
    scale = np.ones((len(test_loader.dataset),)) 
    pck = compute_pck(predicted, ground_truth, threshold, scale)

    # OKS
    oks = np.mean([compute_oks(predicted[i], ground_truth[i], areas[i], sigmas) for i in range(len(predicted))])

    # 推理速度 (FPS)
    fps = len(test_loader.dataset) / sum(inference_times)

    # 模型参数量
    num_params = sum(p.numel() for p in model.parameters())

    return {
        "mpjpe": mpjpe,
        "pck": pck,
        "oks": oks,
        "fps": fps,
        "num_params": num_params
    }

def compute_mpjpe(predicted, ground_truth):
    """
    计算平均关节位置误差MPJPE
    
    参数：
    predicted (numpy.ndarray): 预测的关节点位置，形状为 (N, K, 3)
    ground_truth (numpy.ndarray): 真实的关节点位置，形状为 (N, K, 3)
    
    返回：
    float: 平均关节位置误差
    """
    print(predicted.shape, ground_truth.shape)
    assert predicted.shape == ground_truth.shape, "预测和真实数据的形状应相同"
    return np.mean(np.linalg.norm(predicted - ground_truth, axis=-1))

# 正确关节点百分比（PCK）

def compute_pck(predicted, ground_truth, threshold, scale):
    """
    计算正确关节点百分比PCK
    
    参数：
    predicted (numpy.ndarray): 预测的关节点位置，形状为 (N, K, 2)
    ground_truth (numpy.ndarray): 真实的关节点位置，形状为 (N, K, 2)
    threshold (float): 阈值比例，例如0.5表示50%
    scale (numpy.ndarray): 用于归一化的尺度，例如头部尺寸，形状为 (N,)
    
    返回：
    float: 正确关节点百分比
    """
    assert predicted.shape == ground_truth.shape, "预测和真实数据的形状应相同"
    N, K, _ = predicted.shape
    correct = 0
    total = N * K
    for i in range(N):
        for j in range(K):
            distance = np.linalg.norm(predicted[i, j] - ground_truth[i, j])
            if distance <= threshold * scale[i]:
                correct += 1
    return correct / total

# 目标关键点相似度（OKS）

def compute_oks(predicted, ground_truth, area, sigmas):
    """
    计算目标关键点相似度OKS
    
    参数：
    predicted (numpy.ndarray): 预测的关节点位置，形状为 (K, 3)
    ground_truth (numpy.ndarray): 真实的关节点位置，形状为 (K, 3)
    area (float): 目标区域的面积
    sigmas (numpy.ndarray): 关键点的标准差，形状为 (K,)
    
    返回：
    float: 目标关键点相似度
    """
    vars = (sigmas * 2) ** 2
    xg = ground_truth[:, 0]
    yg = ground_truth[:, 1]
    vg = ground_truth[:, 2]
    xd = predicted[:, 0]
    yd = predicted[:, 1]
    dx = xd - xg
    dy = yd - yg
    e = (dx ** 2 + dy ** 2) / vars / (area + np.spacing(1)) / 2
    if np.count_nonzero(vg > 0) > 0:
        e = e[vg > 0]
    return np.sum(np.exp(-e)) / e.shape[0]


if __name__ == '__main__':
    args = get_parse_arguments()
    main(args)

