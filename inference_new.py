import numpy as np
import torch
import cv2
import C3D_model
from torch.autograd import Variable
from torch import nn
from ultralytics import YOLO
def center_crop(frame):
    frame = frame[8:120, 30:142, :]
    return np.array(frame).astype(np.uint8)

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

def inference():
    # 定义模型训练的设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # 加载数据集标签
    with open("./data/label_custom.txt", 'r') as f :
        class_names = f.readlines()
        # print(class_names)
        f.close()

    # 加载模型，并将模型参数加载到模型中
    model = C3D_model.C3D(num_classes=6)
    checkpoint = torch.load('model_result/best_model_0811/C3D_best_epoch-71.pth.tar')
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
        cropped_frame = None
        color_image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = model_yolo.predict(color_image_rgb, verbose=False)
        for r in results:
            if len(r.boxes) > 0:
                box = r.boxes.xywh[0]
                x, y, w, h = box[:4]
                x, y, w, h = int(x - w/2), int(y - h/2), int(w), int(h)
                
                # 可适当扩大检测框范围
                x = max(0, x - 15)
                y = max(0, y - 15)
                # 获取原始帧宽高
                frame_height, frame_width = frame.shape[:2]
                w = min(frame_width - x, w + 30)
                h = min(frame_height - y, h + 30)

                cropped_frame = frame[y:y+h, x:x+w]
                break
        if cropped_frame is None:
            continue
        # height, width = frame.shape[:2]
        # new_width = int(width / 2)
        # new_height = int(height / 2)
        # frame = cv2.resize(frame, (new_width, new_height))
        tmp_ = center_crop(cv2.resize(cropped_frame, (171, 128)))
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