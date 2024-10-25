"""
后续改进：
1.修改训练策略，早停。。。
2.修改损失函数和激活函数，改网络中relu
3.修改网络层，添加attention
4.使用公共数据集，添加自采数据集
5.
"""
import torch
from torch import nn, optim
import C3D_model
from tensorboardX import SummaryWriter
import os
from datetime import datetime
import socket
import timeit
from tqdm import tqdm
from torch.autograd import Variable
from torch.utils.data import DataLoader
from dataset import VideoDataset


def create_train_subfolder(base_dir='model_result'):
    train_dir_base = 'train{}'
    i = 1
    while True:
        train_dir = os.path.join(base_dir, train_dir_base.format(i))
        if not os.path.exists(train_dir):
            os.makedirs(train_dir)
            return train_dir
        i += 1

def train_model(num_epochs, num_classes, lr, device, save_dir, train_dataloader, val_dataloader, test_dataloader):
    # 记录最佳的精度和epoch
    best_val_acc = 0.0
    best_epoch = 0
    # 在每一次训练创建一个文件夹
    train_subfolder = create_train_subfolder()
    # C3D模型实例化
    model = C3D_model.C3D(num_classes, pretrained=True)

    # 定义模型的损失函数
    criterion = nn.CrossEntropyLoss()

    # 定义优化器
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)

    # 定义学习率的更新策略
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    # 将模型和损失函数放入到训练设备中
    model.to(device)
    criterion.to(device)

    # 日志记录
    log_dir = os.path.join(save_dir, 'models', datetime.now().strftime('%b%d_%H-%M-%S') + '_' + socket.gethostname())
    writer = SummaryWriter(log_dir=log_dir)

    # 开始模型的训练
    trainval_loaders = {'train': train_dataloader, 'val': val_dataloader}  # 将验证集和训练集以字典的形式报存
    trainval_sizes = {x: len(trainval_loaders[x].dataset) for x in
                      ['train', 'val']}  # 计算训练集和验证的大小 {'train': 8460, 'val': 2159}
    test_size = len(test_dataloader.dataset)  # 计算测试集的大小test_size:2701

    # 开始训练
    for epoch in range(num_epochs):
        for phase in ['train', 'val']:
            if phase == 'val':
                if phase == 'val' and epoch_acc > best_val_acc:
                    best_val_acc = epoch_acc
                    best_epoch = epoch

            start_time = timeit.default_timer()  # 计算训练开始时间
            running_loss = 0.0  # 初始化loss值
            running_corrects = 0.0  # 初始化准确率值

            if phase == 'train':
                model.train()
            else:
                model.eval()
            for inputs, labels in tqdm(trainval_loaders[phase]):
                # 将数据和标签放入到设备中
                inputs = Variable(inputs, requires_grad=True).to(device)
                labels = Variable(labels).to(device)
                optimizer.zero_grad()  # 清除梯度

                if phase == "train":
                    outputs = model(inputs)
                else:
                    with torch.no_grad():
                        outputs = model(inputs)

                # 计算softmax的输出概率
                probs = nn.Softmax(dim=1)(outputs)
                # 计算最大概率值的标签
                preds = torch.max(probs, 1)[1]
                labels = labels.long()   # 计算最大概率值的标签
                loss = criterion(outputs, labels) # 计算损失函数

                if phase == "train":
                    loss.backward()
                    optimizer.step()

                # 计算该轮次所有loss值的累加
                running_loss += loss.item() * inputs.size(0)
                # 计算该轮次所有预测正确值的累加
                running_corrects += torch.sum(preds == labels.data)

            scheduler.step()
            epoch_loss = running_loss / trainval_sizes[phase]  # 计算该轮次的loss值，总loss除以样本数量
            epoch_acc = running_corrects.double() / trainval_sizes[phase]  # 计算该轮次的准确率值，总预测正确值除以样本数量

            if phase == "train":
                writer.add_scalar('data/train_loss_epoch', epoch_loss, epoch)
                writer.add_scalar('data/train_acc_epoch', epoch_acc, epoch)
            else:
                writer.add_scalar('data/val_loss_epoch', epoch_loss, epoch)
                writer.add_scalar('data/val_acc_epoch', epoch_acc, epoch)

            # 计算停止的时间戳
            stop_time = timeit.default_timer()

            print("[{}] Epoch: {}/{} Loss: {} Acc: {}".format(phase, epoch + 1, num_epochs, epoch_loss, epoch_acc))
            print("Execution time: " + str(stop_time - start_time) + "\n")
    best_model_path = os.path.join(train_subfolder, 'C3D_best_epoch-{}.pth.tar'.format(best_epoch + 1))
    torch.save(
        {'epoch': best_epoch + 1, 'state_dict': model.state_dict(), 'opt_dict': optimizer.state_dict()},
        best_model_path)
    print("Save best model at {}\n".format(best_model_path))
    last_model_path = os.path.join(train_subfolder, 'C3D_last_epoch-{}.pth.tar'.format(num_epochs))
    torch.save(
        {'epoch': num_epochs, 'state_dict': model.state_dict(), 'opt_dict': optimizer.state_dict()},
        last_model_path)
    writer.close()

    # # 保存训练的好权重
    # torch.save({'epoch': epoch + 1, 'state_dict': model.state_dict(), 'opt_dict': optimizer.state_dict(), },
    #            os.path.join(save_dir, 'models', 'C3D' + '_epoch-' + str(epoch) + '.pth.tar'))
    # print("Save model at {}\n".format(os.path.join(save_dir, 'models', 'C3D' + '_epoch-' + str(epoch) + '.pth.tar')))


    # 开始模型的测试
    model.eval()
    running_corrects = 0.0 # 初始化准确率的值
    # 循环推理测试集中的数据，并计算准确率
    for inputs, labels in tqdm(test_dataloader):
        # 将数据和标签放入到设备中
        inputs = inputs.to(device)
        labels = labels.long()
        labels = labels.to(device)
        with torch.no_grad():
            outputs = model(inputs)

        # 计算softmax的输出概率
        probs = nn.Softmax(dim=1)(outputs)
        # 计算最大概率值的标签
        preds = torch.max(probs, 1)[1]

        running_corrects += torch.sum(preds == labels.data)
    epoch_acc = running_corrects.double() / test_size  # 计算该轮次的准确率值，总预测正确值除以样本数量
    print("test Acc: {}".format(epoch_acc))



if __name__ == "__main__":
    # 定义模型训练的设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    num_epochs = 200  # 训练轮次
    num_classes = 6   # 模型使用的数据集和网络最后一层输出参数
    lr = 1e-4  # 学习率
    save_dir = 'model_result'


    train_dataloader = DataLoader(VideoDataset(dataset_path='rec_728_frame_only_body_new', images_path='train', clip_len=16), batch_size=16, shuffle=True, num_workers=2)
    val_dataloader = DataLoader(VideoDataset(dataset_path='rec_728_frame_only_body_new', images_path='val', clip_len=16), batch_size=16, num_workers=2)
    test_dataloader = DataLoader(VideoDataset(dataset_path='rec_728_frame_only_body_new', images_path='test', clip_len=16), batch_size=16, num_workers=2)

    train_model(num_epochs, num_classes, lr, device, save_dir, train_dataloader, val_dataloader, test_dataloader)
