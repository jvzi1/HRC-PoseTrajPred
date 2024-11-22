import cv2
import torch
import numpy as np
from act_models import MultiModalTrajectoryPredictor  
import json
class BehaviorAndTrajectoryVisualizer:
    def __init__(self, model_path, device='cuda'):
        """
        初始化可视化器
        
        Args:
            model_path (str): 训练好的模型路径。
            device (str): 'cuda' 或 'cpu'。
        """
        self.device = torch.device(device)
        self.model = MultiModalTrajectoryPredictor(seq_len=20, pred_len=8, behavior_num_classes=6, hidden_dim=256, nhead=8, num_layers=4)
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
    def preprocess_keypoints(self, keypoints_list):
        """
        预处理关键点数据以输入模型
        
        Args:
            keypoints_list (list): 包含所有帧关键点的列表，每帧为 [num_joints * 3]。
        
        Returns:
            torch.Tensor: 模型输入的关键点张量，形状为 [1, seq_len, num_joints * 3]。
        """
        seq_len = 20  # 模型的输入序列长度
        num_joints = 33

        keypoints_array = np.array(keypoints_list)  # [num_frames, num_joints * 3]
        if len(keypoints_array) < seq_len:
            padding = np.zeros((seq_len - len(keypoints_array), num_joints * 3))
            keypoints_array = np.vstack((padding, keypoints_array))
        else:
            keypoints_array = keypoints_array[-seq_len:]

        keypoints_array = keypoints_array.reshape(1, seq_len, num_joints * 3)
        return torch.tensor(keypoints_array, dtype=torch.float32).to(self.device)
    
    def preprocess_images(self, image_buffer):
        """
        预处理图像帧数据以输入模型
        
        Args:
            image_buffer (list): 包含所有帧图像的列表，每帧为 [H, W, C]。
        
        Returns:
            torch.Tensor: 模型输入的图像张量，形状为 [1, seq_len, 3, 64, 64]。
        """
        seq_len = 20  # 模型的输入序列长度
        processed_images = []
        for img in image_buffer[-seq_len:]:
            resized_img = cv2.resize(img, (64, 64))  # 调整图像大小为 (64, 64)
            processed_images.append(resized_img)

        processed_images = np.stack(processed_images)  # [seq_len, 64, 64, 3]
        processed_images = processed_images.transpose(0, 3, 1, 2)  # 转为 [seq_len, 3, 64, 64]
        return torch.tensor(processed_images, dtype=torch.float32).unsqueeze(0).to(self.device)

    def extract_keypoints_from_json(self, json_data):
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
    def visualize_result(self, frame, keypoints, predicted_trajectory):
        """
        在视频帧上绘制关键点和预测轨迹
        
        Args:
            frame (np.ndarray): 当前视频帧。
            keypoints (np.ndarray): 当前帧关键点，形状为 [num_joints, 3]。
            predicted_trajectory (np.ndarray): 预测的轨迹，形状为 [pred_len, num_joints, 3]。
        
        Returns:
            np.ndarray: 带有绘制结果的视频帧。
        """
        skeleton = [
            (1, 2), (2, 3), (4, 5), (5, 6), (9, 10), (11, 12), (11, 13),
            (11, 23), (12, 14), (12, 24), (13, 15), (14, 16), (15, 17),
            (15, 19), (15, 21), (16, 18), (16, 20), (16, 22), (23, 24),
            (23, 25), (24, 26), (25, 27), (26, 28), (27, 29), (27, 31),
            (28, 30), (28, 32)
        ]

        for x, y, z in keypoints:
            cv2.circle(frame, (int(x * frame.shape[1]), int(y * frame.shape[0])), 2, (0, 255, 0), -1)

        if predicted_trajectory is not None:
            predicted_trajectory = predicted_trajectory.reshape(-1, 33, 3)
            num_predictions = predicted_trajectory.shape[0]
            print(num_predictions)
            for t in range(num_predictions):
                predicted_keypoints = predicted_trajectory[t]
                r = 255
                g = int(200 * (t / num_predictions))
                b = int(200 * (t / num_predictions))
                color = (b, g, r)

                for idx, predicted_keypoint in enumerate(predicted_keypoints):
                    x, y, z = predicted_keypoint
                    cv2.circle(frame, (int(x * frame.shape[1]), int(y * frame.shape[0])), 2, color, -1)
                for start, end in skeleton:
                    start_point = predicted_keypoints[start]
                    end_point = predicted_keypoints[end]
                    cv2.line(frame, (int(start_point[0] * frame.shape[1]), int(start_point[1] * frame.shape[0])),
                             (int(end_point[0] * frame.shape[1]), int(end_point[1] * frame.shape[0])), color, 1)

        return frame
    def visualize(self, json_path, video_path, output_path, speed=100):
        """
        可视化行为识别与轨迹预测结果
        
        Args:
            video_path (str): 输入视频路径。
            output_path (str): 输出视频路径，可选。
        """
        with open(json_path, 'r') as f:
            json_data = json.load(f)
        keypoint_data = self.extract_keypoints_from_json(json_data)
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"无法打开视频文件 {video_path}")
            return
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))
        seq_buffer = keypoint_data[:30].tolist()
        image_buffer = []
        frame_idx = 0

        while frame_idx < len(keypoint_data):
            if len(seq_buffer) < 30 or len(image_buffer) < 30:
                ret, frame = cap.read()
                if not ret:
                    break
                image_buffer.append(frame)
                seq_buffer.append(keypoint_data[frame_idx])
                frame_idx += 1
                continue 
            trajectory_input = self.preprocess_keypoints(seq_buffer)
            images_input = self.preprocess_images(image_buffer)

            with torch.no_grad():
                predicted_trajectory = self.model(trajectory_input, images_input, torch.tensor([0]).to(self.device)).cpu().numpy()

            ret, frame = cap.read()
            if not ret:
                break

            current_keypoints = np.array(keypoint_data[frame_idx])
            if current_keypoints.shape == (33 * 3,):
                current_keypoints = current_keypoints.reshape(33, 3)

            image_buffer.append(frame)
            predicted_trajectory = predicted_trajectory[0]
            frame = self.visualize_result(frame, current_keypoints, predicted_trajectory)
            out.write(frame)
            delay = int(speed)
            cv2.imshow('Prediction', frame)
            if cv2.waitKey(delay) & 0xFF == ord('q'):
                break

            seq_buffer = seq_buffer[1:]
            seq_buffer.append(keypoint_data[frame_idx])
            frame_idx += 1

        cap.release()
        out.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    model_path = "model_result/trajectory/best_model.pth"
    json_path = r"F:\video_rec_new\data\rec_728\20240728141330\annotated_videos\0\1\video_1_0_1_keypoints.json"
    video_path = r"F:\video_rec_new\data\rec_728\20240728141330\annotated_videos\0\1\video_1_0_1.mp4"
    output_path = r"F:\video_rec_new\data\test\output_prediction.mp4"

    visualizer = BehaviorAndTrajectoryVisualizer(model_path, device="cuda")
    visualizer.visualize(json_path, video_path, output_path)