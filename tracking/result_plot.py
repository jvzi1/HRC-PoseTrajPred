import numpy as np
import torch
from trajectory_model import TrajectoryTransformerModel
from loguru import logger
import os
import json

# config
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_joints = 33
embed_size = 128
num_heads = 8
num_layers = 4
behavior_vocab_size = 6
behavior_embed_size = 16
camera_name_vocab_size = 3
camera_name_embed_size = 16
pred_len = 8
seq_len = 40
lstm_hidden_size = 256
lstm_num_layers = 2


config = {
    "num_joints":33,
    "embed_size" : 128,
    "num_heads" : 8,
    "num_layers" : 4,
    "behavior_vocab_size" : 6,
    "behavior_embed_size" : 64,
    "pred_len" : 8,
    "seq_len" : 40,
    "lstm_hidden_size" : 256,
    "lstm_num_layers" : 2,
}
# 平均关节位置误差（MPJPE）
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

def initialize_model(model_path):
    """初始化并加载轨迹预测模型"""
    model = TrajectoryTransformerModel(
        num_joints=num_joints,
        embed_size=embed_size,
        num_heads=num_heads,
        num_layers=num_layers,
        behavior_vocab_size=behavior_vocab_size,
        behavior_embed_size=behavior_embed_size,
        camera_name_vocab_size=camera_name_vocab_size,
        camera_name_embed_size=camera_name_embed_size,
        pred_length=pred_len,
        lstm_hidden_size=lstm_hidden_size,
        lstm_num_layers=lstm_num_layers,
    ).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model
def model_result():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def load_keypoints():
    pass

def extract_keypoints_from_json(json_data):
    """从 JSON 数据中提取关键点"""
    all_frames = []
    for frame_info in json_data:
        keypoints = frame_info['keypoints']
        keypoints_array = []
        for kp in keypoints:
            # 提取 (x, y, z) 数据并展平
            keypoints_array.extend([kp['x'], kp['y'], kp['z']])
        all_frames.append(keypoints_array)  # 每帧应为 [num_joints * 3]
    return np.array(all_frames)

def preprocess_keypoints(keypoints_list):
    """预处理关键点数据以输入模型"""
    keypoints_array = np.array(keypoints_list)  # [num_frames, num_joints * 3]
    
    # 保证序列长度为 seq_len
    if len(keypoints_array) < seq_len:
        padding = np.zeros((seq_len - len(keypoints_array), num_joints * 3))
        keypoints_array = np.vstack((padding, keypoints_array))
    else:
        keypoints_array = keypoints_array[-seq_len:]
    
    # 转换为 (batch_size, seq_len, num_joints * 3)
    keypoints_array = keypoints_array.reshape(1, seq_len, num_joints * 3)
    return torch.tensor(keypoints_array, dtype=torch.float32).to(device)

def predict_trajectory(model, trajectory_input):
    """使用模型预测未来轨迹"""
    behavior_input = torch.tensor([3]).to(device)
    camera_name_input = torch.tensor([0]).to(device)
    with torch.no_grad():
        predicted_trajectory = model(trajectory_input, behavior_input, camera_name_input).cpu().numpy()
    return predicted_trajectory

def get_ground_truth_and_predictions(json_data, model):
    """
    提取 Ground Truth 和预测结果，用于计算指标
    Args:
        json_data (list): 包含关键点数据的 JSON 数据
        model (nn.Module): 轨迹预测模型

    Returns:
        tuple: Ground Truth 和模型预测的关键点数据
    """
    ground_truth = extract_keypoints_from_json(json_data)
    seq_buffer = ground_truth[:seq_len].tolist()
    predictions = []

    # 滑动窗口生成预测
    for i in range(len(ground_truth) - seq_len):
        trajectory_input = preprocess_keypoints(seq_buffer)
        predicted_trajectory = predict_trajectory(model, trajectory_input)
        predictions.append(predicted_trajectory[0])
        seq_buffer = seq_buffer[1:] + [ground_truth[i + seq_len]]
    
    # Ground Truth 截取以对齐预测
    ground_truth = ground_truth[seq_len:len(predictions) + seq_len]
    return np.array(ground_truth), np.array(predictions)

if __name__ == '__main__':
    json_path = r"F:\video_rec_new\data\rec_728\20240728141330\annotated_videos\3\1\video_1_3_1_keypoints.json"
    model_path = "model_result/trajectory/best_model_1129.pth"
    model = initialize_model(model_path)
    with open(json_path, 'r') as f:
        json_data = json.load(f)
    
    ground_truth, predictions = get_ground_truth_and_predictions(json_data, model)

    # MPJPE 计算
    mpjpe = compute_mpjpe(predictions, ground_truth)
    print(f"MPJPE: {mpjpe:.4f}")

    # PCK 计算
    pck = compute_pck(predictions, ground_truth, threshold=0.5)
    print(f"PCK@0.5: {pck:.4f}")

    # OKS 计算
    sigmas = np.ones(num_joints) * 0.1  # 假设每个关节的标准差为 0.1
    area = 1.0  # 假设目标区域面积为 1.0
    oks = compute_oks(predictions, ground_truth, area, sigmas)
    print(f"OKS: {oks:.4f}")