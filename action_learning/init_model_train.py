import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from datetime import datetime
import os
from torch.autograd import Variable
from tqdm import tqdm
# from tensorboardX import SummaryWriter
from loguru import logger
from datatime import datatime
from dataset import VideoDataset 
import C3D_model  
from config import CONFIG
# 创建训练目录
def create_train_subfolder(base_dir='model_result'):
    train_dir_base = 'train{}'
    i = 1
    while True:
        train_dir = os.path.join(base_dir, train_dir_base.format(i))
        if not os.path.exists(train_dir):
            os.makedirs(train_dir)
            return train_dir
        i += 1

# 模型训练函数
def train_model(num_epochs, num_classes, lr, device, save_dir, train_dataloader, val_dataloader):
    # 初始化最佳验证精度
    best_val_acc = 0.0
    best_epoch = 0

    # 创建保存模型的文件夹
    train_subfolder = create_train_subfolder(base_dir=save_dir)

    # 初始化 C3D 模型
    model = C3D_model.C3D(num_classes, pretrained=False)

    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)

    # 学习率调度器
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    # 将模型和损失函数放入设备中
    model = model.to(device)
    criterion = criterion.to(device)

    # 日志记录
    log_filename = f"logs/run_{datetime.now():%Y-%m-%d_%H-%M-%S}.log"
    logger.add(log_filename, format="{time} - {level} - {message}", level="INFO", retention="7 days")

    # 开始训练
    for epoch in range(num_epochs):
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            # 训练或验证的循环
            for inputs, labels in tqdm(train_dataloader if phase == 'train' else val_dataloader):
                inputs = Variable(inputs, requires_grad=True).to(device)
                labels = Variable(labels).to(device)

                optimizer.zero_grad()

                # 前向传播
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)

                    # 训练阶段：反向传播和优化
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # 统计损失和准确率
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            # 学习率调度器更新
            if phase == 'train':
                scheduler.step()

            # 计算一个 epoch 的损失和准确率
            epoch_loss = running_loss / len(train_dataloader.dataset)
            epoch_acc = running_corrects.double() / len(train_dataloader.dataset)

            # 记录日志
            logger.info(f'data/{phase}_loss_epoch{epoch_loss} {epoch}')
            logger.info(f'data/{phase}_acc_epoch {epoch_acc} {epoch}')

            logger.info(f"[{phase}] Epoch: {epoch + 1}/{num_epochs} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

            # 保存最佳模型
            if phase == 'val' and epoch_acc > best_val_acc:
                best_val_acc = epoch_acc
                best_epoch = epoch
                best_model_path = os.path.join(train_subfolder, f'C3D_best_epoch-{best_epoch + 1}.pth.tar')
                torch.save({'epoch': best_epoch + 1, 'state_dict': model.state_dict(), 'opt_dict': optimizer.state_dict()}, best_model_path)
                logger.info(f"Save best model at {best_model_path}")



    # 保存最后一个 epoch 的模型
    last_model_path = os.path.join(train_subfolder, f'C3D_last_epoch-{num_epochs}.pth.tar')
    torch.save({'epoch': num_epochs, 'state_dict': model.state_dict(), 'opt_dict': optimizer.state_dict()}, last_model_path)
    logger.info(f"Save last model at {last_model_path}")

if __name__ == "__main__":
    # 配置训练参数
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    num_epochs = 50  # 训练轮次
    num_classes = 6   # 类别数量
    lr = 1e-4  # 学习率
    save_dir = os.path.join(CONFIG.SAVE_DIR,"init_model")

    # 创建全量数据集
    full_dataset = VideoDataset(dataset_path=CONFIG.DATASET_PATH, images_path='train', clip_len=CONFIG.CLIP_LEN)

    # 拆分数据集：10% 作为训练集，90% 作为未标注集
    train_size = int(0.1 * len(full_dataset))
    unlabeled_size = len(full_dataset) - train_size
    train_dataset, _ = random_split(full_dataset, [train_size, unlabeled_size])

    # 创建数据加载器
    train_dataloader = DataLoader(train_dataset, batch_size=CONFIG.BATCH_SIZE, shuffle=True, num_workers=2)
    val_dataloader = DataLoader(
        VideoDataset(dataset_path='rec_728_frame', images_path='val', clip_len=CONFIG.CLIP_LEN),
        batch_size=CONFIG.BATCH_SIZE, num_workers=2
    )

    # 训练模型
    train_model(num_epochs, num_classes, lr, device, save_dir, train_dataloader, val_dataloader)
