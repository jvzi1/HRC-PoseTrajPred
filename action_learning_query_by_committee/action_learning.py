import torch
import numpy as np
from torch.utils.data import Subset, DataLoader
from dataset import VideoDataset
from model import C3DModel
from query_strategy import select_samples_by_entropy
from train import train_model
from config import CONFIG

def active_learning_loop():
    # 加载数据集
    full_dataset = VideoDataset(CONFIG.DATASET_PATH, CONFIG.CLIP_LEN)
    
    # 初始化训练集与未标注数据集
    train_size = int(CONFIG.INITIAL_TRAIN_SIZE * len(full_dataset))
    unlabeled_indices = list(range(train_size, len(full_dataset)))
    train_dataset = Subset(full_dataset, list(range(train_size)))
    unlabeled_dataset = Subset(full_dataset, unlabeled_indices)

    train_dataloader = DataLoader(train_dataset, batch_size=CONFIG.BATCH_SIZE, shuffle=True)
    val_dataloader = DataLoader(Subset(full_dataset, list(range(len(full_dataset)))), batch_size=CONFIG.BATCH_SIZE)

    # 初始化委员会
    committee = [C3DModel(CONFIG.NUM_CLASSES) for _ in range(CONFIG.COMMITTEE_SIZE)]
    
    # 训练每个委员会成员
    committee_predictions = []
    for model in committee:
        model = train_model(model, train_dataloader, val_dataloader)
        model.eval()
        predictions = []
        for inputs, _ in DataLoader(unlabeled_dataset, batch_size=CONFIG.BATCH_SIZE):
            inputs = inputs.to(CONFIG.DEVICE)
            outputs = model(inputs)
            preds = torch.softmax(outputs, dim=1).cpu().numpy()
            predictions.append(preds)
        committee_predictions.append(np.vstack(predictions))

    # 基于Vote Entropy选择样本进行标注
    selected_indices = select_samples_by_entropy(committee_predictions, CONFIG.QUERY_SIZE)

    # 从未标注数据集中选择并加入训练集
    new_train_indices = [unlabeled_indices[i] for i in selected_indices]
    train_dataset = Subset(full_dataset, list(range(train_size)) + new_train_indices)

    print(f"Selected {len(new_train_indices)} samples for labeling.")

if __name__ == "__main__":
    active_learning_loop()
