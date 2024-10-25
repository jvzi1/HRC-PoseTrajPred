import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from datetime import datetime
import numpy as np
from torch.autograd import Variable
from tqdm import tqdm
from tensorboardX import SummaryWriter
from dataset import VideoDataset
import C3D_model
from config import CONFIG
import os

def iterate_training(train_dataset, val_dataloader, num_epochs, model_save_path):
    model = C3D_model.C3D(CONFIG.NUM_CLASSES, pretrained=False)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=CONFIG.LEARNING_RATE, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    # 将模型和损失函数放入设备
    model = model.to(CONFIG.DEVICE)
    criterion = criterion.to(CONFIG.DEVICE)

    # 日志记录
    writer = SummaryWriter(log_dir=os.path.join(CONFIG.SAVE_DIR, 'logs', datetime.now().strftime('%b%d_%H-%M-%S')))

    # 创建数据加载器
    train_dataloader = DataLoader(train_dataset, batch_size=CONFIG.BATCH_SIZE, shuffle=True, num_workers=2)

    # 训练模型
    for epoch in range(num_epochs):
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            dataloader = train_dataloader if phase == 'train' else val_dataloader
            for inputs, labels in tqdm(dataloader):
                inputs = Variable(inputs, requires_grad=True).to(CONFIG.DEVICE)
                labels = Variable(labels).to(CONFIG.DEVICE)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / len(dataloader.dataset)
            epoch_acc = running_corrects.double() / len(dataloader.dataset)

            writer.add_scalar(f'data/{phase}_loss_epoch', epoch_loss, epoch)
            writer.add_scalar(f'data/{phase}_acc_epoch', epoch_acc, epoch)

            print(f"[{phase}] Epoch: {epoch + 1}/{num_epochs} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

    # 保存迭代后的模型
    torch.save({'epoch': num_epochs, 'state_dict': model.state_dict(), 'opt_dict': optimizer.state_dict()}, model_save_path)
    print(f"Saved updated model at {model_save_path}")

    writer.close()

if __name__ == "__main__":
    # 加载未标注数据集
    full_dataset = VideoDataset(dataset_path=CONFIG.DATASET_PATH, images_path=CONFIG.TRAIN_IMAGES_PATH, clip_len=CONFIG.CLIP_LEN)

    # 加载选中的样本索引
    selected_indices = np.load('selected_indices.npy')

    # 构建被标注的数据集（假设我们已经手动标注完毕）
    labeled_subset = Subset(full_dataset, selected_indices)

    # 使用已标注的数据和初始训练集合并
    initial_train_size = int(CONFIG.TRAIN_SIZE_RATIO * len(full_dataset))
    train_dataset = Subset(full_dataset, list(range(initial_train_size))) + labeled_subset

    # 加载验证集
    val_dataloader = DataLoader(
        VideoDataset(dataset_path=CONFIG.DATASET_PATH, images_path=CONFIG.VAL_IMAGES_PATH, clip_len=CONFIG.CLIP_LEN),
        batch_size=CONFIG.BATCH_SIZE, num_workers=2
    )

    # 重新训练模型
    model_save_path = os.path.join(CONFIG.SAVE_DIR, 'C3D_iterated_model.pth.tar')
    iterate_training(train_dataset, val_dataloader, CONFIG.NUM_EPOCHS, model_save_path)
