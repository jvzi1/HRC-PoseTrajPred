import os
from sklearn.model_selection import train_test_split
import cv2
import numpy as np

def process_video(identifier, video_path, save_dir):
    resize_height = 128
    resize_width = 171

    # 生成唯一的视频文件夹名称
    video_basename = os.path.basename(video_path).split('.')[0]
    frame_dir_name = f"{identifier}_{video_basename}"
    target_dir = os.path.join(save_dir, frame_dir_name)

    # 检查并处理重复的 frame_dir_name
    counter = 1
    while os.path.exists(target_dir):
        frame_dir_name = f"{identifier}_{video_basename}_{counter}"
        target_dir = os.path.join(save_dir, frame_dir_name)
        counter += 1

    os.makedirs(target_dir)  # 使用 os.makedirs 递归创建目录

    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        print(f"Error opening video file: {video_path}")
        return

    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    EXTRACT_FREQUENCY = 4
    while frame_count // EXTRACT_FREQUENCY <= 16 and EXTRACT_FREQUENCY > 1:
        EXTRACT_FREQUENCY -= 1

    count = 0
    i = 0
    retaining = True

    while count < frame_count and retaining:
        retaining, frame = capture.read()
        if frame is None:
            continue

        if count % EXTRACT_FREQUENCY == 0:
            if frame_height != resize_height or frame_width != resize_width:
                frame = cv2.resize(frame, (resize_width, resize_height))
            cv2.imwrite(filename=os.path.join(target_dir, f'0000{i}.jpg'), img=frame)
            i += 1
        count += 1

    capture.release()

def preprocess(ori_data_path, output_data_path):
    # 查看是否存在输出文件地址，如果没有则创建，同时创建，train， val， test文件夹
    if not os.path.exists(output_data_path):
        os.mkdir(output_data_path)
        os.mkdir(os.path.join(output_data_path, 'train'))
        os.mkdir(os.path.join(output_data_path, 'val'))
        os.mkdir(os.path.join(output_data_path, 'test'))
    video_folder_dict = {}
    for dir_name in os.listdir(ori_data_path):
        dir_path = os.path.join(ori_data_path, dir_name)
        if os.path.isdir(dir_path):
            annotated_videos_path = os.path.join(dir_path, 'annotated_videos')
            if os.path.exists(annotated_videos_path):
                for action_name in os.listdir(annotated_videos_path):
                    action_path = os.path.join(annotated_videos_path, action_name)
                    if os.path.isdir(action_path):
                        for subfolder in os.listdir(action_path):
                            subfolder_path = os.path.join(action_path, subfolder)
                            if os.path.isdir(subfolder_path):
                                videos = [os.path.join(subfolder_path, video) for video in os.listdir(subfolder_path) if "video_2" in video]
                                if videos:
                                    if action_name not in video_folder_dict:
                                        video_folder_dict[action_name] = []
                                    video_folder_dict[action_name].extend(videos)

    if not video_folder_dict:
        print("No videos found for processing.")
        return

    for identifier, video_folders in video_folder_dict.items():
        train_and_valid, test = train_test_split(video_folders, test_size=0.2, random_state=42)
        train, val = train_test_split(train_and_valid, test_size=0.2, random_state=42)

        train_dir = os.path.join(output_data_path, 'train', identifier)
        val_dir = os.path.join(output_data_path, 'val', identifier)
        test_dir = os.path.join(output_data_path, 'test', identifier)

        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(val_dir, exist_ok=True)
        os.makedirs(test_dir, exist_ok=True)

        # 处理每个划分的数据集
        for video in train:
            process_video(identifier, video, train_dir)
        for video in val:
            process_video(identifier, video, val_dir)
        for video in test:
            process_video(identifier, video, test_dir)

        print(f'{identifier} 类别下的数据处理完成')


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
    output_data_path = 'rec_728_frame_only_video2'

    # 生成标签文档
    label_text_write(ori_data_path, out_label_path)

    # 划分数据集，生成对应的图片数据集
    preprocess(ori_data_path, output_data_path)


