import json
import numpy as np
import os

def get_keypoint_position(json_path):
    """
    输出一个维度为[frame_len, 33, 3]的数据
    """
    keypoints_trajectory = []
    with open(json_path, "r") as f:
        data = json.load(f)
    for frame in data:
        tmp = []
        for joint in frame["keypoints"]:
            keypoints_x = joint["x"]
            keypoints_y = joint["y"]
            keypoints_z = joint["z"]
            tmp.append([keypoints_x, keypoints_y, keypoints_z])
        keypoints_trajectory.append(tmp)
    return keypoints_trajectory

def preprocess(ori_data_path):
    all_trajectory = []
    for dir_name in os.listdir(ori_data_path):
        dir_path = os.path.join(ori_data_path, dir_name)
        if os.path.isdir(dir_path):
            annotated_keypoints_path = os.path.join(dir_path, 'annotated_videos')
            if os.path.exists(annotated_keypoints_path):
                for action_name in os.listdir(annotated_keypoints_path):
                    action_path = os.path.join(annotated_keypoints_path, action_name)
                    if os.path.isdir(action_path):
                        video_folders = []
                        for subfolder in os.listdir(action_path):
                            subfolder_path = os.path.join(action_path, subfolder)
                            if os.path.isdir(subfolder_path):
                                for keypoints_path in [os.path.join(action_path, subfolder, video) for video in os.listdir(subfolder_path) if ".json" in video]:
                                    keypoints_trajectory = get_keypoint_position(keypoints_path)
                                    all_trajectory.append(keypoints_trajectory)
                                # video_folders.extend(
                                #     [os.path.join(action_path, subfolder, video) for video in os.listdir(subfolder_path) if ".mp4" in video])
    return  all_trajectory 

if __name__ == "__main__":
    all_trajectory = preprocess(r"F:\video_rec_new\dataset\rec_728")
    max_length = max(len(sublist) for sublist in all_trajectory)
    padded_trajectory = [sublist + [0] * (max_length - len(sublist)) for sublist in all_trajectory]
    all_trajectory_array = np.array(padded_trajectory)
    print(all_trajectory.shape)