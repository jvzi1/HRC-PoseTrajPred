import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer, TransformerDecoder, TransformerDecoderLayer

class MultiModalTrajectoryPredictor(nn.Module):
    def __init__(self, seq_len, pred_len, behavior_num_classes=6, hidden_dim=256, nhead=8, num_layers=4):
        """
        多模态轨迹预测模型，融合轨迹、图像和行为标签输入。

        Args:
            seq_len (int): 输入的历史序列长度。
            pred_len (int): 输出的预测轨迹序列长度。
            behavior_num_classes (int): 行为标签类别数。
            hidden_dim (int): 中间隐藏层的维度。
        """
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len

        # 图像编码器
        self.image_encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.image_fc = nn.Linear(128 * 16 * 16, hidden_dim)

        # 轨迹编码器
        self.trajectory_fc = nn.Linear(33 * 3, hidden_dim)

        # 行为标签嵌入
        self.behavior_embedding = nn.Embedding(behavior_num_classes, hidden_dim)

        # Transformer 编码器和解码器
        encoder_layer = TransformerEncoderLayer(d_model=hidden_dim, nhead=nhead)
        decoder_layer = TransformerDecoderLayer(d_model=hidden_dim, nhead=nhead)
        self.encoder = TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.decoder = TransformerDecoder(decoder_layer, num_layers=num_layers)
        # 位置编码
        self.positional_encoding = nn.Parameter(torch.randn(seq_len + pred_len, hidden_dim))

        # 输出层
        self.future_output_layer = nn.Linear(hidden_dim, 33 * 3)
    
    def _calculate_flatten_size(self):
        dummy_input = torch.randn(1, 3, 64, 64)  # 假设输入图像尺寸为 (3, 64, 64)
        with torch.no_grad():
            output = self.image_encoder(dummy_input)
        self.flatten_size = output.view(-1).size(0)

    def forward(self, trajectory, images, behavior):
        """
        前向传播函数。

        Args:
            trajectory (torch.Tensor): 历史轨迹，形状为 [batch_size, seq_len, 33, 3]。
            images (torch.Tensor): 历史图像帧，形状为 [batch_size, seq_len, 3, 224, 224]。
            behavior (torch.Tensor): 行为标签，形状为 [batch_size]。

        Returns:
            predicted_trajectory (torch.Tensor): 预测的未来轨迹，形状为 [batch_size, pred_len, 33*3]。
        """
        batch_size, seq_len, _, _, _ = images.shape

        # 图像编码
        images = images.view(batch_size * seq_len, 3, 64, 64)  # 展平批次
        image_features = self.image_encoder(images)              # [batch_size * seq_len, 128, 56, 56]
        image_features = image_features.contiguous().view(batch_size * seq_len, -1)  # 展平特征
        image_features = self.image_fc(image_features)           # [batch_size * seq_len, hidden_dim]
        image_features = image_features.view(batch_size, seq_len, -1)  # 恢复时间序列

        # 轨迹编码
        trajectory = trajectory.view(batch_size, seq_len, -1)  # 展平关键点
        trajectory_features = self.trajectory_fc(trajectory)  # [batch_size, seq_len, hidden_dim]

        # 行为标签嵌入
        behavior_features = self.behavior_embedding(behavior)  # [batch_size, hidden_dim]

        # 融合特征
        combined_features = trajectory_features + image_features
        combined_features += behavior_features.unsqueeze(1)  # 添加行为嵌入

        # 加入位置编码
        position_enc = self.positional_encoding[:seq_len, :].unsqueeze(0)  # [1, seq_len, hidden_dim]
        combined_features += position_enc  # [batch_size, seq_len, hidden_dim]

        # 编码历史轨迹
        combined_features = combined_features.permute(1, 0, 2)  # [seq_len, batch_size, hidden_dim]
        memory = self.encoder(combined_features)  # [seq_len, batch_size, hidden_dim]

        # 解码未来轨迹
        future_queries = torch.zeros(self.pred_len, batch_size, memory.size(2), device=memory.device)  # [pred_len, batch_size, hidden_dim]
        future_queries += self.positional_encoding[seq_len:seq_len + self.pred_len].unsqueeze(1)  # 添加未来位置编码
        future_trajectory = self.decoder(future_queries, memory)  # [pred_len, batch_size, hidden_dim]

        # 输出未来轨迹
        future_trajectory = future_trajectory.permute(1, 0, 2)  # [batch_size, pred_len, hidden_dim]
        future_trajectory = self.future_output_layer(future_trajectory)  # [batch_size, pred_len, 33 * 3]
        future_trajectory = future_trajectory.view(batch_size, self.pred_len, 33 * 3)

        return future_trajectory
