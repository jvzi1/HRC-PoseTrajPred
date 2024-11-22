class BehaviorAndTrajectoryVisualizer:
    def __init__(self, model_path, device='cuda'):
        """
        初始化可视化器
        
        Args:
            model_path (str): 训练好的模型路径。
            device (str): 'cuda' 或 'cpu'。
        """
        self.device = torch.device(device)
        self.model = MultiModalTrajectoryPredictorTransformer(
            seq_len=30, pred_len=8, behavior_num_classes=6, hidden_dim=256, nhead=8, num_layers=4
        )
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
        seq_len = 30  # 模型的输入序列长度
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
        seq_len = 30  # 模型的输入序列长度
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
        （同原代码）
        """
        # 保留原实现
        pass

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
                predicted_trajectory = self.model(
                    trajectory_input, images_input, torch.tensor([0]).to(self.device)
                ).cpu().numpy()

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