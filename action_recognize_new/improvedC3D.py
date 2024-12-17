import torch.nn.functional as F
import torch.nn as nn

class Conv3dWithROI(nn.Conv3d):
    def forward(self, input):
        # 分离 RGB 和 ROI
        rgb_input = input[:, :-1, :, :, :]  # 前 3 通道为 RGB
        roi_input = input[:, -1:, :, :, :]  # 第 4 通道为 ROI

        # 分别计算卷积
        rgb_output = F.conv3d(rgb_input, self.weight[:, :-1, :, :, :], self.bias,
                              self.stride, self.padding, self.dilation, self.groups)
        roi_output = F.conv3d(roi_input, self.weight[:, -1:, :, :, :], None,
                              self.stride, self.padding, self.dilation, self.groups)

        # 加权融合
        output = rgb_output + 0.5 * roi_output
        return output

class ImprovedC3D(nn.Module):
    def __init__(self, num_classes):
        super(ImprovedC3D, self).__init__()

        self.conv1 = Conv3dWithROI(4, 64, kernel_size=(3, 3, 3), padding=1)  # 4 通道输入
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.conv2 = Conv3dWithROI(64, 128, kernel_size=(3, 3, 3), padding=1)
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv3a = Conv3dWithROI(128, 256, kernel_size=(3, 3, 3), padding=1)
        self.conv3b = Conv3dWithROI(256, 256, kernel_size=(3, 3, 3), padding=1)
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv4a = Conv3dWithROI(256, 512, kernel_size=(3, 3, 3), padding=1)
        self.conv4b = Conv3dWithROI(512, 512, kernel_size=(3, 3, 3), padding=1)
        self.pool4 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv5a = Conv3dWithROI(512, 512, kernel_size=(3, 3, 3), padding=1)
        self.conv5b = Conv3dWithROI(512, 512, kernel_size=(3, 3, 3), padding=1)
        self.pool5 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.fc6 = nn.Linear(512 * 2 * 4 * 4, 4096)
        self.fc7 = nn.Linear(4096, 4096)
        self.fc8 = nn.Linear(4096, num_classes)

        self.dropout = nn.Dropout(p=0.5)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool1(x)

        x = self.relu(self.conv2(x))
        x = self.pool2(x)

        x = self.relu(self.conv3a(x))
        x = self.relu(self.conv3b(x))
        x = self.pool3(x)

        x = self.relu(self.conv4a(x))
        x = self.relu(self.conv4b(x))
        x = self.pool4(x)

        x = self.relu(self.conv5a(x))
        x = self.relu(self.conv5b(x))
        x = self.pool5(x)

        x = x.view(x.size(0), -1)
        x = self.relu(self.fc6(x))
        x = self.dropout(x)
        x = self.relu(self.fc7(x))
        x = self.dropout(x)
        x = self.fc8(x)

        return x
