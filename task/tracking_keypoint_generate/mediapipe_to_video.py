import cv2
import mediapipe as mp
import json
import os
import glob
# 初始化MediaPipe姿态估计
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)
# 关节点信息
POSE_LANDMARKS = [
    "nose", "left_eye_inner", "left_eye", "left_eye_outer", "right_eye_inner", "right_eye", "right_eye_outer",
    "left_ear", "right_ear", "mouth_left", "mouth_right", "left_shoulder", "right_shoulder", "left_elbow",
    "right_elbow", "left_wrist", "right_wrist", "left_pinky", "right_pinky", "left_index", "right_index",
    "left_thumb", "right_thumb", "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle",
    "left_heel", "right_heel", "left_foot_index", "right_foot_index"
]
def process_video(video_path, output_dir):
    cap = cv2.VideoCapture(video_path)
    frame_data = []

    frame_index = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # 转换BGR为RGB
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 进行姿态估计
        result = pose.process(image_rgb)
        
        # 获取关键点
        if result.pose_landmarks:
            landmarks = result.pose_landmarks.landmark
            keypoints = []
            for idx, lm in enumerate(landmarks):
                keypoints.append({
                    'name': POSE_LANDMARKS[idx],
                    'x': lm.x,
                    'y': lm.y,
                    'z': lm.z,
                    'visibility': lm.visibility
                })
            # 存储每一帧的关键点和时序信息（帧号）
            frame_data.append({
                'frame_index': frame_index,
                'keypoints': keypoints
            })
        frame_index += 1

    cap.release()

    # 输出为JSON文件
    output_filename = os.path.join(output_dir, os.path.basename(video_path).replace('.mp4', '_keypoints.json'))
    with open(output_filename, 'w') as f:
        json.dump(frame_data, f, indent=4)

    print(f'关键点数据已保存到 {output_filename}')

# 处理一批视频
def process_batch_videos(folder_path):
    video_files = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.mp4'):
                video_path = os.path.join(root, file)
                video_files.append(video_path)
    
    for video_path in video_files:
        # 生成关键点文件在 视频文件同级目录下
        output_dir = os.path.dirname(video_path)
        process_video(video_path, output_dir)

if __name__ == '__main__':

    folder = r'F:\video_rec_new\rec_728'
    process_batch_videos(folder)
