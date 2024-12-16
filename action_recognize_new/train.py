# train.py: 训练脚本更新以支持 ROI
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset_with_roi import VideoDatasetWithROI
from C3D_model_with_roi import C3D_with_roi
from torch.optim.lr_scheduler import StepLR
import argparse
import timeit
from tqdm import tqdm

    
def get_parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of joints in the input')
    parser.add_argument("--num_classes", type=int, default=6 , help="Number of output classes")
    parser.add_argument('--dataset_path', type=str, default="data/rec_728", help='Path to the dataset')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training and evaluation')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    # parser.add_argument('--save_model_name', type=str, default='best_model.pth', help='Save model checkpoint')
    parser.add_argument("--optimizer", type=str, default="sgd", choices=["adam", "sgd"], help="Optimizer type")
    # 中断训练
    parser.add_argument('--resume_train', action='store_true', help='Resume training from checkpoint')
    parser.add_argument('--checkpoint_path', type=str, default=None, help='Path to the checkpoint file')
    parser.add_argument('--output_dir', type=str, default="model_result/action_recognize_new",
                        help='Directory to save model checkpoints')
    parser.add_argument("--scheduler_step_size", type=int, default=10, help="Step size for learning rate scheduler")
    parser.add_argument("--scheduler_gamma", type=float, default=0.1, help="Gamma for learning rate scheduler")

    return parser.parse_args()

def create_train_subfolder(base_dir='model_result/action_recognize_new'):
    train_dir_base = 'train{}'
    i = 1
    while True:
        train_dir = os.path.join(base_dir, train_dir_base.format(i))
        if not os.path.exists(train_dir):
            os.makedirs(train_dir)
            return train_dir
        i += 1

def train_model(args):
    best_val_acc = 0.0
    best_epoch = 0
    train_subfolder = create_train_subfolder(args.output_dir)
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 数据集路径
    train_data_path = os.path.join(args.dataset_path, "train")
    val_data_path = os.path.join(args.dataset_path, "val")

    # 加载数据集
    train_dataset = VideoDatasetWithROI(train_data_path, clip_len=16)
    val_dataset = VideoDatasetWithROI(val_data_path, clip_len=16)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # 加载模型
    model = C3D_with_roi(num_classes=args.num_classes)
    model.to(device)

    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    if args.optimizer == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    else:
        raise ValueError("Unsupported optimizer type")
    
    # 定义学习率的更新策略
    scheduler = StepLR(optimizer, step_size=args.scheduler_step_size, gamma=args.scheduler_gamma)

    # 中断训练
    if args.resume_train:
        checkpoint = torch.load(args.resume_train)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resuming training from epoch {start_epoch}")
    else:
        start_epoch = 0

    best_val_acc = 0.0
    best_epoch = 0
    # 训练循环
    for epoch in range(args.num_epochs):
        for phase in ['train', 'val']:
            start_time = timeit.default_timer()
            running_loss = 0.0  # 初始化loss值
            running_corrects = 0.0  # 初始化准确率值

            if phase == 'train':
                model.train()
                data_loader = train_loader
            else:
                model.eval()
                data_loader = val_loader

            for inputs, labels in tqdm(data_loader, desc=f"{phase} Epoch {epoch + 1}/{args.num_epochs}"):
                inputs, labels = inputs.to(device), labels.to(device)

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

            epoch_loss = running_loss / len(data_loader.dataset)
            epoch_acc = running_corrects.double() / len(data_loader.dataset)
            
            print(f"[{phase}] Epoch {epoch + 1}/{args.num_epochs} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")
            print(f"Execution time for {phase}: {timeit.default_timer() - start_time:.2f} seconds")

            if phase == 'val' and epoch_acc > best_val_acc:
                best_val_acc = epoch_acc
                best_epoch = epoch
                best_model_path = os.path.join(train_subfolder, 'best_model-{}.pth'.format(best_epoch + 1))
                torch.save(
                    {'epoch': best_epoch + 1, 'state_dict': model.state_dict(), 'opt_dict': optimizer.state_dict(),
                     'scheduler_state_dict': scheduler.state_dict()},
                    best_model_path)
                print(f"Best model saved at epoch {best_epoch + 1} with accuracy {best_val_acc:.4f}")

    # 保存最后一轮模型
    last_model_path = os.path.join(train_subfolder, 'C3D_last_epoch-{}.pth.tar'.format(args.num_epochs))
    torch.save(
        {'epoch': args.num_epochs, 'state_dict': model.state_dict(), 'opt_dict': optimizer.state_dict(),
         'scheduler_state_dict': scheduler.state_dict()},
        last_model_path)
    print(f"Last model saved at epoch {args.num_epochs}")

if __name__ == "__main__":
    args = get_parse_arguments()
    train_model(args)