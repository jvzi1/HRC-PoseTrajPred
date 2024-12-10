import os
import json
import torch
import numpy as np
import cv2
from trajectory_model import TrajectoryTransformerModel
from loguru import logger 

# mediapipe
import mediapipe as mp
mp_pose = mp.solutions.pose

# C3D
import C3D_model
from torch.autograd import Variable
from torch import nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_joints = 33
embed_size = 256
num_heads = 8
num_layers = 4
behavior_vocab_size = 6
behavior_embed_size = 64
pred_len = 8
seq_len = 40
lstm_hidden_size = 256
lstm_num_layers = 2
camera_name_vocab_size = 3
camera_name_embed_size = 64

behavior_dict = {
    0: "walk",
    1: "operate",
    2: "crouch",
    3: "carry",
    4: "playphone",
    5: "measure"
}

def center_crop(frame):
    return frame[8:120, 30:142, :]
def initialize_trajectory_model(model_path):
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

def initialize_c3d_model(c3d_model_path):
    """初始化并加载C3D模型"""
    model_c3d = C3D_model.C3D(num_classes=6)
    checkpoint = torch.load(c3d_model_path, map_location=device)
    model_c3d.load_state_dict(checkpoint['state_dict'])
    model_c3d.to(device)
    model_c3d.eval()
    return model_c3d

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


def predict_trajectory(model, trajectory_input, behavior, camera_name):
    """使用模型预测未来轨迹"""
    behavior_input = torch.tensor([behavior]).to(device)
    camera_name_input = torch.tensor([camera_name]).to(device)
    with torch.no_grad():
        predicted_trajectory = model(trajectory_input, behavior_input, camera_name_input).cpu().numpy()
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

    # 绘制历史轨迹(暂不需要)
    if history_buffer is not None:
        for t, history_keypoints in enumerate(history_buffer):
            alpha = 1.0 - t / len(history_buffer)  # 随时间渐变透明
            color = (0, int(255 * alpha), int(255 * alpha))  # 淡蓝色
            for x, y, z in history_keypoints:
                cv2.circle(frame, (int(x * frame.shape[1]), int(y * frame.shape[0])), 2, color, -1)

    # 绘制预测轨迹
    if predicted_trajectory is not None:
        predicted_trajectory = np.array(predicted_trajectory)
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


from time import time 

def c3d_preprocess_frame(frame):
    """对单帧进行C3D模型所需预处理"""
    resized = cv2.resize(frame, (171, 128))
    cropped = center_crop(resized)
    tmp = cropped - np.array([[[90.0, 98.0, 102.0]]])
    return tmp

def infer_online(trajectory_model_path, c3d_model_path, camera_index=0, speed=100, iterations = 5):
    trajectory_model = initialize_trajectory_model(trajectory_model_path)
    c3d_model = initialize_c3d_model(c3d_model_path)

    pose_estimator = mp_pose.Pose(static_image_mode=False, 
                                model_complexity=1,
                                smooth_landmarks=True, 
                                enable_segmentation=False, 
                                min_detection_confidence=0.5, 
                                min_tracking_confidence=0.5)
    
    # TODO 修改相机接口
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print("无法打开摄像头")
        return

    seq_buffer = []
    history_buffer = []
    predicted_trajectories = None

    # C3D行为预测相关
    c3d_clip = []
    behavior = 4  # 默认行为为playphone，之后修改playphone标签为others,为负样本动作标签
    camera_name = camera_index # 摄像头名称标签
    
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        original_frame = frame.copy()
        frame_height, frame_width = frame.shape[:2]

        # MediaPipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose_estimator.process(frame_rgb)

        if results.pose_landmarks:
            # Mediapipe返回33个关键点，分别具有x,y,z，x,y为归一化到[0,1], z为相对深度
            landmarks = results.pose_landmarks.landmark
            keypoints_array = []
            for lm in landmarks:
                keypoints_array.extend([lm.x, lm.y, lm.z])
            keypoints_array = np.array(keypoints_array) # [num_joints*3]
        else:
            seq_buffer.append([0.0] * (num_joints * 3))  
        seq_buffer.append(keypoints_array)

        # 更新历史轨迹，用于可视化(可根据需求限制长度)
        # history_buffer.append(keypoints_array.reshape(num_joints, 3))
        # if len(history_buffer) > seq_len:
        #     history_buffer.pop(0)


        c3d_frame = frame.copy()
        height, width = c3d_frame.shape[:2]
        new_width = 171
        new_height = 128
        c3d_frame = cv2.resize(c3d_frame, (new_width, new_height))
        tmp = c3d_preprocess_frame(c3d_frame)
        c3d_clip.append(tmp)

        # 每16帧用C3D更新一次behavior
        if len(c3d_clip) == 16:
            inputs = np.array(c3d_clip).astype(np.float32)
            inputs = np.expand_dims(inputs, axis=0)
            inputs = np.transpose(inputs, (0,4,1,2,3))
            inputs = torch.from_numpy(inputs).to(device)
            with torch.no_grad():
                outputs = c3d_model(inputs)
            probs = torch.nn.Softmax(dim=1)(outputs)
            label = torch.argmax(probs, dim=1).item()
            behavior = label
            c3d_clip.pop(0)

        if len(seq_buffer) >= seq_len:
            trajectory_input = preprocess_keypoints(seq_buffer)

            # 多次迭代预测
            predicted_trajectories = []
            current_input = trajectory_input
            for _ in range(iterations):
                predicted_trajectory = predict_trajectory(trajectory_model, current_input, behavior, camera_name)
                # 保存本次预测结果
                predicted_trajectories.append(predicted_trajectory[0].reshape(pred_len, num_joints, 3))
                # 将预测结果拼接进输入用于下一次迭代
                predicted_trajectory_flat = predicted_trajectory.reshape(pred_len, -1)
                combined = np.vstack([seq_buffer[-(seq_len - pred_len):], predicted_trajectory_flat])
                current_input = preprocess_keypoints(combined)

            # 保持seq_buffer长度不变或者滑动更新，这里保留最新seq_len-1帧加上本帧
            seq_buffer = seq_buffer[-(seq_len-1):]
            
        # 可视化结果并写入视频
        if len(seq_buffer) > 0:
            current_keypoints = seq_buffer[-1].reshape(num_joints, 3)
            frame = visualize_result(frame, current_keypoints, predicted_trajectories)
        behavior_label = behavior_dict.get(behavior, "Unknown")
        cv2.putText(frame, f"Behavior: {behavior_label}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        cv2.imshow('Trajectory Prediction', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        frame_count += 1
    cap.release()
    cv2.destroyAllWindows()
    pose_estimator.close() 


if __name__ == "__main__":
    trajectory_model_path = "model_result/trajectory/best_model_1202.pth"
    c3d_model_path = "model_result/best_model_0811/C3D_best_epoch-71.pth.tar"
    infer_online(trajectory_model_path, c3d_model_path)
