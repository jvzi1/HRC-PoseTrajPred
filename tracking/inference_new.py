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
pred_len = 8
seq_len = 40
lstm_hidden_size = 256
lstm_num_layers = 2

def initialize_model(model_path):
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
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model

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

def predict_trajectory(model, trajectory_input):
    """使用模型预测未来轨迹"""
    behavior_input = torch.tensor([0]).to(device)
    with torch.no_grad():
        predicted_trajectory = model(trajectory_input, behavior_input).cpu().numpy()
    return predicted_trajectory


def visualize_result(frame, keypoints, predicted_trajectory, history_buffer=None):
    """
    在视频帧上绘制关键点和预测轨迹，同时显示历史轨迹与未来预测融合。
    
    Args:
        frame: 当前帧图像。
        keypoints: 原始关键点数据，形状为 [num_joints, 3]。
        predicted_trajectory: 模型预测的未来轨迹，形状为 [pred_len, num_joints, 3]。
        history_buffer: 历史关键点数据，用于绘制历史轨迹。
        
    Returns:
        带可视化结果的图像帧。
    """
    if keypoints.shape[1] != 3:
        raise ValueError("关键点数据应为 [num_joints, 3] 形状")

    skeleton = [
        (1, 2), (2, 3), (4, 5), (5, 6), (9, 10), (11, 12), (11, 13), 
        (11, 23), (12, 14), (12, 24), (13, 15), (14, 16), (15, 17),
        (15, 19), (15, 21), (16, 18), (16, 20), (16, 22), (23, 24),
        (23, 25), (24, 26), (25, 27), (26, 28), (27, 29), (27, 31),
        (28, 30), (28, 32)
    ]

    # 绘制当前帧关键点
    for x, y, z in keypoints:
        cv2.circle(frame, (int(x * frame.shape[1]), int(y * frame.shape[0])), 2, (0, 255, 0), -1)

    # 绘制历史轨迹（如果提供）
    if history_buffer is not None:
        for t, history_keypoints in enumerate(history_buffer):
            alpha = 1.0 - t / len(history_buffer)  # 随时间渐变透明
            color = (0, int(255 * alpha), int(255 * alpha))  # 淡蓝色
            for x, y, z in history_keypoints:
                cv2.circle(frame, (int(x * frame.shape[1]), int(y * frame.shape[0])), 2, color, -1)

    # 绘制预测轨迹
    if predicted_trajectory is not None:
        predicted_trajectory = predicted_trajectory.reshape(-1, 33, 3)
        num_predictions = predicted_trajectory.shape[0]
        for t in range(num_predictions):
            predicted_keypoints = predicted_trajectory[t]
            r = 255
            g = int(200 * (t / num_predictions))
            b = int(200 * (t / num_predictions))
            color = (b, g, r)

            # 绘制每个预测帧的关键点
            for idx, predicted_keypoint in enumerate(predicted_keypoints):
                x, y, z = predicted_keypoint
                cv2.circle(frame, (int(x * frame.shape[1]), int(y * frame.shape[0])), 2, color, -1)

            # 绘制骨架连接
            for start, end in skeleton:
                start_point = predicted_keypoints[start]
                end_point = predicted_keypoints[end]
                cv2.line(frame, (int(start_point[0] * frame.shape[1]), int(start_point[1] * frame.shape[0])),
                         (int(end_point[0] * frame.shape[1]), int(end_point[1] * frame.shape[0])), color, 1)

    return frame



def infer_from_json(model, json_path, video_path, output_path, speed=100):
    """从 JSON 文件加载关键点数据并进行轨迹预测"""
    with open(json_path, 'r') as f:
        json_data = json.load(f)
    keypoint_data = extract_keypoints_from_json(json_data)

    # 检查输出目录
    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"输出目录已创建: {output_dir}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video file {video_path}")
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    seq_buffer = keypoint_data[:seq_len].tolist()
    history_buffer = []  # 保存历史轨迹
    frame_idx = 0

    while frame_idx < len(keypoint_data):
        if len(seq_buffer) < seq_len:
            break  # 如果剩余帧不足 seq_len，则停止

        trajectory_input = preprocess_keypoints(seq_buffer)
        predicted_trajectory = predict_trajectory(model, trajectory_input)

        ret, frame = cap.read()
        if not ret:
            break

        current_keypoints = np.array(keypoint_data[frame_idx])
        if current_keypoints.shape == (num_joints * 3,):
            current_keypoints = current_keypoints.reshape(num_joints, 3)

        # 滑动窗口：更新 seq_buffer，融合预测结果
        predicted_trajectory_flat = predicted_trajectory.reshape(pred_len, -1)  # [pred_len, num_joints * 3]
        seq_buffer = seq_buffer[-seq_len // 2:] + predicted_trajectory_flat.tolist()[:seq_len // 2]

        # 更新历史轨迹缓冲区
        history_buffer.append(current_keypoints)
        if len(history_buffer) > 10:  # 保留最近 10 帧的历史轨迹
            history_buffer.pop(0)

        # 可视化并写入结果
        frame = visualize_result(frame, current_keypoints, predicted_trajectory, history_buffer)
        out.write(frame)

        frame_idx += 1

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    # 校验输出视频
    if os.path.exists(output_path):
        test_cap = cv2.VideoCapture(output_path)
        if test_cap.isOpened():
            print(f"预测结果已保存到 {output_path}，视频完整性校验通过。")
            test_cap.release()
        else:
            print(f"预测结果保存到 {output_path}，但视频无法打开，请检查保存逻辑！")
    else:
        print(f"预测结果未能正确保存到 {output_path}，请检查路径和写入逻辑。")






if __name__ == "__main__":
    json_path = r"F:\video_rec_new\data\rec_728\20240728141330\annotated_videos\0\1\video_1_0_1_keypoints.json"
    video_path = r"F:\video_rec_new\data\rec_728\20240728141330\annotated_videos\0\1\video_1_0_1.mp4"
    output_path = r"F:\video_rec_new\data\test\output_prediction.mp4"
    model_path = "model_result/trajectory/last_model_8.pth"
    model = initialize_model(model_path)

    infer_from_json(model, json_path, video_path, output_path)
    print(f"预测结果已保存到 {output_path}")
