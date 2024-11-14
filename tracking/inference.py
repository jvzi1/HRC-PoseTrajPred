import os
import json
import torch
import numpy as np
import cv2
from transformer import TrajectoryTransformer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_joints = 33
embed_size = 128
num_heads = 8
num_layers = 4
behavior_vocab_size = 6
behavior_embed_size = 64
pred_len = 5
seq_len = 10
lstm_hidden_size = 256
lstm_num_layers = 2

def initialize_model():
    """初始化并加载轨迹预测模型"""
    model = TrajectoryTransformer(
        num_joints=num_joints,
        embed_size=embed_size,
        num_heads=num_heads,
        num_layers=num_layers,
        behavior_vocab_size=behavior_vocab_size,
        behavior_embed_size=behavior_embed_size,
        pred_length=pred_len,
        lstm_hidden_size=lstm_hidden_size,
        lstm_num_layers=lstm_num_layers,
    ).to(device)
    checkpoint = torch.load("model_result/trajectory/best_model.pth", map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model

def preprocess_keypoints(keypoints_list):
    """预处理关键点数据以输入模型"""
    keypoints_array = np.array(keypoints_list)[:, :, :3]  # 只保留 (x, y, z) 数据
    return torch.tensor(keypoints_array, dtype=torch.float32).unsqueeze(0).to(device)

def extract_keypoints_from_json(json_data):
    """从 JSON 数据中提取关键点"""
    all_frames = []
    for frame_info in json_data:
        keypoints = frame_info['keypoints']
        keypoints_array = []
        for kp in keypoints:
            keypoints_array.append([kp['x'], kp['y'], kp['z'], kp['visibility']])
        all_frames.append(keypoints_array)
    return all_frames

def predict_trajectory(model, seq_buffer):
    """使用模型预测未来轨迹"""
    trajectory_input = preprocess_keypoints(seq_buffer)
    behavior_input = torch.randint(0, behavior_vocab_size, (1,)).to(device)
    with torch.no_grad():
        predicted_trajectory = model(trajectory_input, behavior_input).cpu().numpy()
    return predicted_trajectory

def visualize_result(frame, keypoints, predicted_trajectory):
    """在视频帧上绘制关键点和预测轨迹"""
    # 绘制实际关键点
    for x, y, z, visibility in keypoints:
        if visibility > 0.5:  # 仅绘制可见性较高的关键点
            cv2.circle(frame, (int(x * frame.shape[1]), int(y * frame.shape[0])), 5, (0, 255, 0), -1)
    
    # 绘制预测的轨迹点
    for t in range(predicted_trajectory.shape[1]):
        for j in range(predicted_trajectory.shape[2]):
            x, y, z = predicted_trajectory[0, t, j]
            cv2.circle(frame, (int(x * frame.shape[1]), int(y * frame.shape[0])), 3, (0, 0, 255), -1)
    return frame

def infer_from_json(model, json_path, video_path, output_path):
    """从 JSON 文件加载关键点数据并进行轨迹预测"""
    # 读取 JSON 文件
    with open(json_path, 'r') as f:
        json_data = json.load(f)

    # 从 JSON 数据中提取关键点
    keypoint_data = extract_keypoints_from_json(json_data)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video file {video_path}")
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

    seq_buffer = []
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx < len(keypoint_data):
            keypoints = keypoint_data[frame_idx]
            seq_buffer.append(keypoints)

            if len(seq_buffer) >= seq_len:
                # 保持序列长度为 seq_len
                seq_buffer = seq_buffer[-seq_len:]
                predicted_trajectory = predict_trajectory(model, seq_buffer)
                frame = visualize_result(frame, keypoints, predicted_trajectory)

            out.write(frame)
            cv2.imshow('Prediction', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        frame_idx += 1

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"预测结果已保存到 {output_path}")

if __name__ == "__main__":
    json_path = r"F:\video_rec_new\data\rec_728\20240728141124\video_1_keypoints.json"
    video_path = r"F:\video_rec_new\data\rec_728\20240728141124\video_1.mp4"
    output_path = r"F:\video_rec_new\data\test\output_prediction.mp4"
    
    model = initialize_model()
    infer_from_json(model, json_path, video_path, output_path)
    print(f"预测结果已保存到 {output_path}")
