# kill center_crop

import numpy as np
import torch
import cv2
import C3D_model_with_roi
from torch.autograd import Variable
from torch import nn
from ultralytics import YOLO

def draw_label(frame, label_text, prob_text, position=(20, 20)):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.0
    font_thickness = 2
    bg_color = (0, 0, 255)  # 红色背景
    text_color = (255, 255, 255)  # 白色文字
    margin = 5

    # 获取文本尺寸
    label_size = cv2.getTextSize(label_text, font, font_scale, font_thickness)[0]
    prob_size = cv2.getTextSize(prob_text, font, font_scale, font_thickness)[0]

    # 计算背景框的尺寸和位置
    box_width = max(label_size[0], prob_size[0]) + 2 * margin
    box_height = label_size[1] + prob_size[1] + 3 * margin
    box_x, box_y = position[0], position[1]
    box_end_x, box_end_y = box_x + box_width, box_y + box_height

    # 绘制背景框
    cv2.rectangle(frame, (box_x, box_y), (box_end_x, box_end_y), bg_color, thickness=-1)

    # 绘制标签文字
    text_x = box_x + margin
    text_y = box_y + label_size[1] + margin
    cv2.putText(frame, label_text, (text_x, text_y), font, font_scale, text_color, font_thickness)

    # 绘制概率文字
    text_y = text_y + prob_size[1] + margin
    cv2.putText(frame, prob_text, (text_x, text_y), font, font_scale, text_color, font_thickness)

def inference(padding=10, resize_dims=(256, 256)):
    # 定义模型训练的设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # 加载数据集标签
    with open("./data/label_custom.txt", 'r') as f :
        class_names = f.readlines()
        # print(class_names)
        f.close()

    # 加载模型，并将模型参数加载到模型中
    model = C3D_model_with_roi.C3D_with_roi(num_classes=6)
    checkpoint = torch.load('model_result/action_recognize/C3D_last_epoch-100.pth.tar')
    model.load_state_dict(checkpoint['state_dict'])

    # 将模型放入到设备中，并设置验证模式
    model.to(device)
    model.eval()

    model_yolo = YOLO('yolov8l.pt')
    video = r"F:\video_rec_new\data\rec_728\20240728141330\video_1.mp4"
    cap = cv2.VideoCapture(video)
    retaining = True

    clip = []
    while retaining:
        retaining, frame = cap.read() # 读取视频帧
        if not retaining and frame is None:
            continue
        roi_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.float32)
        color_image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = model_yolo.predict(color_image_rgb, verbose=False)
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
        if roi_mask.any() :
            continue
        resized_frame = cv2.resize(frame, resize_dims)
        resized_roi = cv2.resize(roi_mask, resize_dims)
        combined_frame = np.concatenate([resized_frame, resized_roi[:, :, np.newaxis] * 255], axis=-1)
        # height, width = frame.shape[:2]
        # new_width = int(width / 2)
        # new_height = int(height / 2)
        # frame = cv2.resize(frame, (new_width, new_height))
        tmp = combined_frame - np.array([[[90.0, 98.0, 102.0, 0.5]]])  # 处理4通道的均值减去
        
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
            label = torch.max(probs, 1)[1].detach().cpu().numpy()[0]

            label_text = class_names[label].strip()
            prob_text = f"Prob: {probs[0][label]:.4f}"

            draw_label(frame, label_text, prob_text, position=(20, 20))


            clip.pop(0)

        cv2.imshow('result', frame)
        cv2.waitKey(1)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    inference()