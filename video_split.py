import os

import cv2
import torch
import numpy as np
import json
import C3D_model


class ActionRecognizer:
    def __init__(self, model_path, video_path):
        self.model_path = model_path
        self.video_path = video_path

    def center_crop(self, frame):
        frame = frame[8:120, 30:142, :]
        return np.array(frame).astype(np.uint8)

    def interface(self, video_path):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        with open("./data/labels.txt", 'r') as f:
            class_names = f.readlines()
            f.close()
        model = C3D_model.C3D(num_classes=101)
        checkpoint = torch.load('model_result/train2/C3D_last_epoch-30.pth.tar')
        model.load_state_dict(checkpoint['state_dict'])
        model.to(device)
        model.eval()
        cap = cv2.VideoCapture(video_path)
        retaining = True
        current_label = None
        frame_start = 0
        frame_count = 0
        results = []

        clip = []
        while retaining:
            retaining, frame = cap.read()
            if not retaining and frame is None:
                continue
            height, width = frame.shape[:2]
            new_width = int(width / 2)
            new_height = int(height / 2)
            frame = cv2.resize(frame, (new_width, new_height))
            tmp_ = self.center_crop(cv2.resize(frame, (171, 128)))  # 是否需要
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
                probs = torch.nn.Softmax(dim=1)(outputs)
                label_num = torch.max(probs, 1)[1].detach().cpu().numpy()[0]
                label = class_names[label_num]
                if current_label is None:
                    current_label = label
                    frame_start = frame_count
                    print(f'started recording video: {frame_start}')
                elif label != current_label:
                    results.append((frame_start, frame_count - 1, current_label))
                    current_label = label
                    frame_start = frame_count
                clip = []
            frame_count += 1
        if current_label is not None:
            results.append((frame_start, frame_count - 1, current_label))
        cap.release()

        # detect
        return results

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
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        for start_frame, end_frame, label in results:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            out_path = os.path.join(save_dir, f"{label}_{start_frame}_{end_frame}.mp4")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

            for frame_num in range(start_frame, end_frame + 1):
                ret, frame = cap.read()
                if ret:
                    out.write(frame)
                else:
                    break

            out.release()

        cap.release()

    def multi_video_split(self, main_video_path, save_dir):
        pass


if __name__ == "__main__":
    model_path = "./model_result/C3D_last_epoch-30.pth.tar"
    video_dir = './recording_sessions/rec_716/'
    video_names = [
        "video_1.mp4",
        "video_2.mp4",
        "video_3.mp4"
    ]

    main_video_path = os.path.join(video_dir, video_names[0])
    action_recognizer = ActionRecognizer(model_path, main_video_path)
    results = action_recognizer.interface(main_video_path)
    for video_name in video_names:
        video_path = os.path.join(video_dir, video_name)
        action_recognizer.save_video_info(results, video_path, os.path.join(video_dir, video_name + ".json"))
        action_recognizer.split_videos(results, os.path.join(video_dir, video_name + "_split"))