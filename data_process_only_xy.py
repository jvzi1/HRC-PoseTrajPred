import os
from sklearn.model_selection import train_test_split
import cv2
import numpy as np
from ultralytics import YOLO

def process_video(dir_name,ori_data_path, video_path, save_dir):
    resize_height = 144
    resize_width = 144

    # 生成唯一的视频文件夹名称
    video_basename = os.path.basename(video_path).split('.')[0]
    video_folder = os.path.basename(os.path.dirname(video_path))
    frame_dir_name = f"{dir_name}_{video_folder}_{video_basename}"

    target_dir = os.path.join(save_dir, frame_dir_name)
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)  # 使用 os.makedirs 递归创建目录

    capture = cv2.VideoCapture(os.path.join(ori_data_path, video_path))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    EXTRACT_FREQUENCY = 4
    if frame_count // EXTRACT_FREQUENCY <= 16:
        EXTRACT_FREQUENCY -= 1
        if frame_count // EXTRACT_FREQUENCY <= 16:
            EXTRACT_FREQUENCY -= 1
            if frame_count // EXTRACT_FREQUENCY <= 16:
                EXTRACT_FREQUENCY -= 1

    count = 0
    i = 0
    retaining = True
    model = YOLO('yolov8l.pt')
    while (count < frame_count and retaining):
        retaining, frame = capture.read()
        if frame is None:
            continue

        if count % EXTRACT_FREQUENCY == 0:
            color_image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = model.predict(color_image_rgb)
            for r in results:
                for box, cls_index in zip(r.boxes.xywh, r.boxes.cls):
                    x, y, w, h = box[:4]
                    x, y, w, h = int(x - w/2), int(y - h/2), int(w), int(h)  # 转换为左上角坐标
                    x = max(0, x - 15)
                    y = max(0, y - 15)
                    w = min(frame_width - x, w + 30)
                    h = min(frame_height - y, h + 30)
                    # 裁剪检测到的区域
                    cropped_frame = frame[y:y+h, x:x+w]

                    # 如果裁剪后的帧不为空，则调整大小并保存
                    if cropped_frame.size != 0:
                        cropped_frame = cv2.resize(cropped_frame, (resize_width, resize_height))
                        cv2.imwrite(filename=os.path.join(target_dir, f'0000{i}.jpg'), img=cropped_frame)
                        i += 1
                    break
        count += 1

    capture.release()


def preprocess(ori_data_path, output_data_path):
    # 查看是否存在输出文件地址，如果没有则创建，同时创建，train， val， test文件夹
    if not os.path.exists(output_data_path):
        os.mkdir(output_data_path)
        os.mkdir(os.path.join(output_data_path, 'train'))
        os.mkdir(os.path.join(output_data_path, 'val'))
        os.mkdir(os.path.join(output_data_path, 'test'))
    for dir_name in ["20240728141330",
"20240728141852",
"20240728142532",
"20240728143133",
"20240728143915"]:
        print(dir_name)
        dir_path = os.path.join(ori_data_path, dir_name)
        if os.path.isdir(dir_path):
            annotated_videos_path = os.path.join(dir_path, 'annotated_videos')
            if os.path.exists(annotated_videos_path):
                for action_name in os.listdir(annotated_videos_path):
                    action_path = os.path.join(annotated_videos_path, action_name)
                    if os.path.isdir(action_path):
                        video_folders = []
                        for subfolder in os.listdir(action_path):
                            subfolder_path = os.path.join(action_path, subfolder)
                            if os.path.isdir(subfolder_path):
                                video_folders.extend(
                                    [os.path.join(subfolder, video) for video in os.listdir(subfolder_path)])
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
                        for video in train:
                            process_video(dir_name,action_path, video, train_dir)
                        for video in val:
                            process_video(dir_name,action_path, video, val_dir)
                        for video in test:
                            process_video(dir_name,action_path, video, test_dir)
                        print(f'{action_name}类别下的数据处理完成')

    print('所有数据划分完成')


def label_text_write(ori_data_path, out_label_path):
    folder = ori_data_path
    fnames, labels = [], []
    for label in sorted(os.listdir(folder)):
        for fname in os.listdir(os.path.join(folder, label)):
            fnames.append(os.path.join(folder, label, fname))
            labels.append(label)

    label2index = {label: index for index, label in enumerate(sorted(set(labels)))}
    if not os.path.exists(out_label_path + '/labels.txt'):
        with open(out_label_path + '/labels.txt', 'w') as f:
            for id, label in enumerate(sorted(label2index)):
                f.writelines(str(id+1) + ' ' + label + '\n')



if __name__ == "__main__":
    ori_data_path = 'rec_728'
    out_label_path = 'data'
    output_data_path = 'rec_728_frame_only_xy'

    # 生成标签文档
    label_text_write(ori_data_path, out_label_path)

    # 划分数据集，生成对应的图片数据集
    preprocess(ori_data_path, output_data_path)


