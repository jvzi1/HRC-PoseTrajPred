import os
import cv2
import torch
import numpy as np
import json
import C3D_model
from loguru import logger
from smoother import ActionSegmentSmoother
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
"""
update:
1.滑动窗口
2.加权平滑,并且当
分成两个py文件

update:
问题：为什么切出视频过少
更新：标签更新名称√
"""


class ActionRecognizer:
    def __init__(self, model_path, video_path, num_classes=6):
        self.model_path = model_path
        self.video_path = video_path
        self.num_classes = num_classes
        self.smoother = ActionSegmentSmoother(alpha=0.9, min_segment_length=16, delay_threshold=5)
    # def center_crop(self, frame):
    #     frame = frame[8:120, 30:142, :]
    #     return np.array(frame).astype(np.uint8)
    def center_crop(self, frame):
        return frame[8:120, 30:142, :]
     
    def interface(self, video_path):
        logger.info("start interface ")
        start_time = time.time()
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        with open("./data/label_custom.txt", 'r') as f:
            class_names = [line.strip() for line in f.readlines()]
        model = C3D_model.C3D(num_classes=self.num_classes)
        checkpoint = torch.load(self.model_path, map_location=device)
        model.load_state_dict(checkpoint['state_dict'])
        model.to(device)
        model.eval()
        cap = cv2.VideoCapture(video_path)
        retaining = True
        current_label = None
        frame_start = 0
        frame_count = 0
        results = []
        frame_probs = []

        clip = []
        while retaining:
            retaining, frame = cap.read()
            if not retaining and frame is None:
                continue
            height, width = frame.shape[:2]
            new_width = int(width / 2)
            new_height = int(height / 2)
            frame = cv2.resize(frame, (new_width, new_height))
            tmp_ = self.center_crop(cv2.resize(frame, (171, 128)))  # 是否需要--TODO 不需要，后续删除
            tmp = tmp_ - np.array([[[90.0, 98.0, 102.0]]])
            clip.append(tmp)

            if len(clip) == 16:
                inputs = np.array(clip).astype(np.float32)
                inputs = np.expand_dims(inputs, axis=0)
                inputs = np.transpose(inputs, (0, 4, 1, 2, 3))
                inputs = torch.from_numpy(inputs)
                inputs = torch.autograd.Variable(inputs, requires_grad=False).to(device)

                with torch.no_grad():
                    outputs = model.forward(inputs)
                # probs = torch.nn.Softmax(dim=1)(outputs) 
                probs = torch.nn.Softmax(dim=1)(outputs).cpu().numpy()[0]# TODO 修改激活函数
                # label_num = torch.max(probs, 1)[1].detach().cpu().numpy()[0]
                # label = class_names[label_num]
                # if current_label is None:
                #     current_label = label
                #     frame_start = frame_count
                #     print(f'started recording video: {frame_start}')
                # elif label != current_label:
                #     results.append((frame_start, frame_count - 1, current_label))
                #     current_label = label
                #     frame_start = frame_count
                frame_probs.append(probs)
                clip = []
            frame_count += 1
        # if current_label is not None:
        #     results.append((frame_start, frame_count - 1, current_label))
        cap.release()
        end_time = time.time()
        logger.info(f"Interface task cost {end_time - start_time}s")

        # 计算置信度
        labeled_segments = self.calculate_segment_confidence(results, frame_probs, class_names)
        # smoother
        self.vis_category_dist(labeled_segments, class_names)

        smoothed_results = self.smoother.smooth_and_segment(frame_probs, class_names)
        labeled_segments = self.calculate_segment_confidence(smoothed_results, frame_probs, class_names)
        self.vis_category_dist2(labeled_segments, class_names)

        return smoothed_results

    def save_video_info(self, results, video_path, json_path):
        video_info = {
            "video_path": video_path,
            "labels": {}
        }
        for start_frame, end_frame, label in results:
            if label not in video_info["labels"]:
                video_info["labels"][label] = []
            video_info["labels"][label].append({"start_frame": start_frame, "end_frame": end_frame})

        with open(json_path, 'w') as json_file:
            json.dump(video_info, json_file, indent=4)
        return video_info

    def split_videos(self, results, save_dir):
        cap = cv2.VideoCapture(self.video_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        for start_frame, end_frame, label in tqdm(results):
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            out_path = os.path.join(save_dir, f"{label}_{start_frame}_{end_frame}.mp4")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
            for frame_num in range(start_frame, end_frame + 1):
                ret, frame = cap.read()
                if ret:
                    out.write(frame)
                else:
                    break

            out.release()

        cap.release()

    def calculate_segment_confidence(self, results, frame_probs, class_names):
        # 计算每个片段的平均置信度
        labeled_segments = []
        for start_frame, end_frame, label in results:
            segment_probs = frame_probs[start_frame:end_frame + 1]
            # 计算每帧的置信度
            confidences = [max(probs) for probs in segment_probs]
            avg_confidence = np.mean(confidences)
            labeled_segments.append((start_frame, end_frame, label, avg_confidence))
        return labeled_segments
    def vis_category_dist2(self, labeled_segments, class_names):
        # 统计每个类别的片段数量
        category_counts = {label: [] for label in class_names}
        for _, _, label, confidence in labeled_segments:
            category_counts[label].append(confidence)

        # 准备绘图数据
        labels = list(category_counts.keys())
        counts = [len(category_counts[label]) for label in labels]
        avg_confidences = [np.mean(category_counts[label]) if category_counts[label] else 0 for label in labels]

        # 创建分组柱状图
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(labels, counts, color=plt.cm.Blues(avg_confidences))

        # 添加置信度标签
        for bar, confidence in zip(bars, avg_confidences):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, height, f"{confidence:.2f}", ha='center', va='bottom')

        ax.set_xlabel('Action Categories')
        ax.set_ylabel('Number of Segments')
        ax.set_title('Distribution of Video Segments by Category with Confidence smoothed')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    def vis_category_dist(self, labeled_segments, class_names):
        # 统计每个类别的片段数量
        category_counts = {label: [] for label in class_names}
        for _, _, label, confidence in labeled_segments:
            category_counts[label].append(confidence)

        # 准备绘图数据
        labels = list(category_counts.keys())
        counts = [len(category_counts[label]) for label in labels]
        avg_confidences = [np.mean(category_counts[label]) if category_counts[label] else 0 for label in labels]

        # 创建分组柱状图
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(labels, counts, color=plt.cm.Blues(avg_confidences))

        # 添加置信度标签
        for bar, confidence in zip(bars, avg_confidences):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, height, f"{confidence:.2f}", ha='center', va='bottom')

        ax.set_xlabel('Action Categories')
        ax.set_ylabel('Number of Segments')
        ax.set_title('Distribution of Video Segments by Category with Confidence')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    def multi_video_split(self, main_video_path, save_dir):
        pass


if __name__ == "__main__":
    model_path = "./model_result/train36/C3D_last_epoch-200.pth.tar"
    video_dir = './dataset/20240728150616'
    video_names = [
        "video_1.mp4",
        "video_2.mp4",
        "video_3.mp4"
    ]

    main_video_path = os.path.join(video_dir, video_names[0])
    logger.info("start action recognize")
    action_recognizer = ActionRecognizer(model_path, main_video_path)
    results = action_recognizer.interface(main_video_path) # 只需要用第一个视频的输出-- TODO 是否需要添加其他视频的输出做一个对照
    for video_name in video_names:
        video_path = os.path.join(video_dir, video_name)
        logger.info("start action recognize")
        action_recognizer.save_video_info(results, video_path, os.path.join(video_dir, video_name + ".json"))
        action_recognizer.split_videos(results, os.path.join(video_dir, video_name + "_split"))