import numpy as np
import torch
import cv2
import C3D_model
import math
from torch.autograd import Variable
from torch import nn

from ultralytics import YOLO


def center_crop(frame):
    frame = frame[8:120, 30:142, :]
    return np.array(frame).astype(np.uint8)

def draw_label(img, text, pos, bg_color):
    font_face = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.6
    color = (255, 255, 255)  # 白色字体
    thickness = cv2.FILLED
    margin = 2

    txt_size = cv2.getTextSize(text, font_face, scale, thickness)

    end_x = pos[0] + txt_size[0][0] + margin
    end_y = pos[1] - txt_size[0][1] - margin

    cv2.rectangle(img, pos, (end_x, end_y), bg_color, thickness)
    cv2.putText(img, text, (pos[0], pos[1] - margin), font_face, scale, color, 1, lineType=cv2.LINE_AA)

def inference():

    # 定义模型训练的设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # 加载数据集标签
    with open("./data/labels.txt", 'r') as f :
        class_names = f.readlines()
        # print(class_names)
        f.close()

    # 加载模型，并将模型参数加载到模型中
    model = C3D_model.C3D(num_classes=3)
    checkpoint = torch.load('model_result/models/C3D_best_epoch-11.pth.tar')
    model.load_state_dict(checkpoint['state_dict'])

    # 将模型放入到设备中，并设置验证模式
    model.to(device)
    model.eval()

    video = "./data/collaboration/tight/VID_20231023_191235.mp4"
    # video_path = r'F:\yolov8-main\VID_20231023_191516.mp4'
    # model = YOLO(r'F:\yolov8-main\yolov8x-pose-p6.pt')
    # result = model.predict(video_path, save=False, imgsz=320, conf=0.5)
    # print(type(result))
    cap = cv2.VideoCapture(video)
    retaining = True


    clip = []
    while retaining:
        retaining, frame = cap.read() # 读取视频帧
        if not retaining and frame is None:
            continue
        # 缩小尺寸
        height, width = frame.shape[:2]
        new_width = int(width / 2)
        new_height = int(height / 2)
        frame = cv2.resize(frame, (new_width, new_height))
        tmp_ = center_crop(cv2.resize(frame, (171, 128)))
        tmp = tmp_ - np.array([[[90.0, 98.0, 102.0]]])
        clip.append(tmp)

        if len(clip) == 16:
            inputs = np.array(clip).astype(np.float32)
            inputs = np.expand_dims(inputs, axis=0)
            inputs = np.transpose(inputs, (0, 4, 1, 2, 3))
            inputs = torch.from_numpy(inputs)
            inputs = Variable(inputs, requires_grad=False).to(device)

            with torch.no_grad():
                outputs = model.forward(inputs)
            probs = torch.nn.Softmax(dim=1)(outputs)
            label = torch.max(probs, 1)[1].detach().cpu().numpy()[0]

            label_text = class_names[label].split(' ')[-1].strip()
            prob_text = f"prob: {probs[0][label]:.4f}"
            draw_label(frame, label_text, (20, 20), (0, 0, 255))  # Red background for label
            draw_label(frame, prob_text, (20, 40), (0, 0, 255))  # Red background for probability

            clip.pop(0)

        cv2.imshow('result', frame)
        cv2.waitKey(30)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    inference()