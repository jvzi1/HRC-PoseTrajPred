import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from dataset import VideoDataset
from config import CONFIG
import os
import numpy as np
from tqdm import tqdm
import C3D_model

# 加载模型
def load_model(model_path, num_classes):
    model = C3D_model.C3D(num_classes, pretrained=False)
    checkpoint = torch.load(model_path, map_location=CONFIG.DEVICE)
    model.load_state_dict(checkpoint['state_dict'])
    model = model.to(CONFIG.DEVICE)
    model.eval()
    return model

# 进行不确定性采样
def uncertainty_sampling(model, unlabeled_dataloader, sample_size=100):
    uncertainties = []
    with torch.no_grad():
        for i, (inputs, _) in enumerate(tqdm(unlabeled_dataloader)):
            inputs = inputs.to(CONFIG.DEVICE)
            outputs = model(inputs)
            probs = F.softmax(outputs, dim=1)
            # 计算最大概率，置信度越低表示模型越不确定
            batch_uncertainties = 1 - torch.max(probs, dim=1)[0]
            uncertainties.extend(zip(range(i * CONFIG.BATCH_SIZE, (i + 1) * CONFIG.BATCH_SIZE), batch_uncertainties.cpu().numpy()))

    # 根据不确定性排序并选择前 sample_size 个样本
    uncertainties = sorted(uncertainties, key=lambda x: x[1], reverse=True)
    selected_indices = [idx for idx, _ in uncertainties[:sample_size]]
    return selected_indices

if __name__ == "__main__":
    # 加载未标注数据集
    unlabeled_dataset = VideoDataset(dataset_path=CONFIG.DATASET_PATH, images_path=CONFIG.TRAIN_IMAGES_PATH, clip_len=CONFIG.CLIP_LEN)
    _, unlabeled_dataset = random_split(unlabeled_dataset, [int(CONFIG.TRAIN_SIZE_RATIO * len(unlabeled_dataset)), len(unlabeled_dataset) - int(CONFIG.TRAIN_SIZE_RATIO * len(unlabeled_dataset))])
    unlabeled_dataloader = DataLoader(unlabeled_dataset, batch_size=CONFIG.BATCH_SIZE, shuffle=False, num_workers=2)

    # 加载模型
    model_path = os.path.join(CONFIG.SAVE_DIR, "init_model", 'train1', 'C3D_best_epoch-1.pth.tar')  # 确保路径正确
    model = load_model(model_path, CONFIG.NUM_CLASSES)

    # 执行不确定性采样
    selected_indices = uncertainty_sampling(model, unlabeled_dataloader, sample_size=100)

    # 保存选中的样本索引，供标注和迭代使用
    np.save('selected_indices.npy', selected_indices)
    print(f"Saved selected indices for labeling in selected_indices.npy")
