import os
from sklearn.model_selection import train_test_split
import cv2
import numpy as np
from ultralytics import YOLO
from tqdm import tqdm
def process_video_with_roi(dir_name, ori_data_path, video_path, save_dir, model_yolo, resize_dims=(171, 128), padding=10):
    # 初始化变量
    video_basename = os.path.basename(video_path).split('.')[0]
    frame_dir_name = f"{dir_name}_{video_basename}"
    target_dir = os.path.join(save_dir, frame_dir_name)
    os.makedirs(target_dir, exist_ok=True)

    capture = cv2.VideoCapture(os.path.join(ori_data_path, video_path))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    retaining = True

    frame_index = 0
    while retaining:
        retaining, frame = capture.read()
        if not retaining or frame is None:
            break

        # YOLO检测生成ROI
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = model_yolo.predict(frame_rgb, verbose=False)

        roi_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.float32)
        for result in results:
            for box in result.boxes.xyxy:
                x1, y1, x2, y2 = box.int().tolist()
                x1 = max(x1 - padding, 0)  
                y1 = max(y1 - padding, 0)  
                x2 = min(x2 + padding, frame.shape[1])  
                y2 = min(y2 + padding, frame.shape[0])
                roi_mask[y1:y2, x1:x2] = 1.0 # 将ROI区域标记为1
                roi_mask[roi_mask == 0] = 0.2 # 环境区域标记为0.2
                break  # 只处理第一个检测目标

        # 调整尺寸
        resized_frame = cv2.resize(frame, resize_dims)
        resized_roi = cv2.resize(roi_mask, resize_dims)

        # 合并帧与ROI
        combined_frame = np.concatenate([resized_frame, resized_roi[:, :, np.newaxis] * 255], axis=-1)

        # 保存结果
        save_path = os.path.join(target_dir, f"{frame_index:04d}.png")
        cv2.imwrite(save_path, combined_frame)
        frame_index += 1

    capture.release()

def preprocess_with_roi(ori_data_path, output_data_path, model_yolo):
    if not os.path.exists(output_data_path):
        os.mkdir(output_data_path)
        os.mkdir(os.path.join(output_data_path, 'train'))
        os.mkdir(os.path.join(output_data_path, 'val'))
        os.mkdir(os.path.join(output_data_path, 'test'))
    for dir_name in tqdm(os.listdir(ori_data_path), desc="Processing Directories"):
        dir_path = os.path.join(ori_data_path, dir_name)
        if os.path.isdir(dir_path):
            annotated_videos_path = os.path.join(dir_path, 'annotated_videos')
            if os.path.exists(annotated_videos_path):
                for action_name in tqdm(os.listdir(annotated_videos_path), desc=f"Processing {dir_name}"):
                    action_path = os.path.join(annotated_videos_path, action_name)
                    if os.path.isdir(action_path):
                        video_folders = []
                        for subfolder in os.listdir(action_path):
                            subfolder_path = os.path.join(action_path, subfolder)
                            if os.path.isdir(subfolder_path):
                                video_folders.extend(
                                    [os.path.join(subfolder, video) for video in os.listdir(subfolder_path) if ".mp4" in video])
                        # 划分数据集
                        train_and_valid, test = train_test_split(video_folders, test_size=0.1, random_state=42)
                        train, val = train_test_split(train_and_valid, test_size=0.1, random_state=42)

                        train_dir = os.path.join(output_data_path, 'train', action_name)
                        val_dir = os.path.join(output_data_path, 'val', action_name)
                        test_dir = os.path.join(output_data_path, 'test', action_name)

                        if not os.path.exists(train_dir):
                            os.mkdir(train_dir)
                        if not os.path.exists(val_dir):
                            os.mkdir(val_dir)
                        if not os.path.exists(test_dir):
                            os.mkdir(test_dir)

                        # 处理每个划分的数据集
                        for video in tqdm(train, desc=f"Processing Train Videos in {action_name}"):
                            process_video_with_roi(dir_name, action_path, video, train_dir, model_yolo)
                        for video in tqdm(val, desc=f"Processing Validation Videos in {action_name}"):
                            process_video_with_roi(dir_name, action_path, video, val_dir, model_yolo)
                        for video in tqdm(test, desc=f"Processing Test Videos in {action_name}"):
                            process_video_with_roi(dir_name, action_path, video, test_dir, model_yolo)
                        print(f'{action_name}类别下的数据处理完成')

    print('所有数据划分完成')

if __name__ == "__main__":
    ori_data_path = "data/rec_728"
    output_data_path = "data/rec_728_frame_with_roi"
    yolo_model = YOLO('yolov8l.pt')  # 初始化YOLO模型

    preprocess_with_roi(ori_data_path, output_data_path, yolo_model)
