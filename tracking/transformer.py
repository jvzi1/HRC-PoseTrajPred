import torch
import torch.nn as nn

class DynamicTrajectoryHead(nn.Module):
    def __init__(self, embed_size, num_joints, pred_length):
        super(DynamicTrajectoryHead, self).__init__()
        
        # 自注意力层，用于计算每个时间步的注意力权重
        self.attention = nn.MultiheadAttention(embed_size, num_heads=4, batch_first=True)
        
        # 用于从注意力加权后的特征中预测未来轨迹
        self.fc_out = nn.Linear(embed_size, num_joints * 2 * pred_length)
        
    def forward(self, x):
        """
        x: [batch_size, seq_len, embed_size] 输入历史轨迹的特征
        """
        # 对输入的时序特征应用自注意力机制
        attention_output, _ = self.attention(x, x, x)  # 自注意力机制
        attention_output = attention_output.mean(dim=1)  # 对时间维度求平均，得到全局的轨迹特征
        
        # 动态生成未来轨迹
        output = self.fc_out(attention_output)  # [batch_size, num_joints * 2 * pred_length]
        return output.view(output.size(0), pred_length, -1, 2)  # [batch_size, pred_length, num_joints, 2]


class TrajectoryTransformer(nn.Module):
    def __init__(self, num_joints, embed_size, num_heads, num_layers, 
                 behavior_vocab_size, behavior_embed_size, pred_length, lstm_hidden_size, lstm_num_layers):
        super(TrajectoryTransformer, self).__init__()
        
        # 轨迹数据输入线性层，将输入关键点位置映射到 embed_size
        self.trajectory_embedding = nn.Linear(num_joints * 2, embed_size)  # 每个关键点 (x, y)
        
        # 行为标签嵌入层
        self.behavior_embedding = nn.Embedding(behavior_vocab_size, behavior_embed_size)
        
        # Transformer 编码器
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_size + behavior_embed_size, nhead=num_heads)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 主干网络--lstm
        self.lstm = nn.LSTM(input_size=embed_size + behavior_embed_size, 
                            hidden_size=lstm_hidden_size, 
                            num_layers=lstm_num_layers, 
                            batch_first=True)
        
        # Head--输出tracking轨迹
        self.dynamic_head = DynamicTrajectoryHead(lstm_hidden_size, num_joints, pred_length)
    
    
    def forward(self, trajectory, behavior):
        # 轨迹数据 embedding
        # trajectory: [batch_size, seq_len, num_joints * 2]
        trajectory_embedded = self.trajectory_embedding(trajectory)  # [batch_size, seq_len, embed_size]
        
        # 行为标签 embedding
        # behavior: [batch_size]
        behavior_embedded = self.behavior_embedding(behavior).unsqueeze(1)  # [batch_size, 1, behavior_embed_size]
        behavior_embedded = behavior_embedded.repeat(1, trajectory_embedded.size(1), 1)  # 行为标签扩展到与轨迹长度匹配
        
        # 将行为嵌入与轨迹嵌入拼接
        x = torch.cat((trajectory_embedded, behavior_embedded), dim=2)  # [batch_size, seq_len, embed_size + behavior_embed_size]
        
        # Transformer 编码器
        x = self.transformer_encoder(x)  # [batch_size, seq_len, embed_size + behavior_embed_size]
        
        # x = x[:, -1, :]  # 只取最后时间步的输出 [batch_size, embed_size + behavior_embed_size]
        
        # LSTM 主干网络
        x, _ = self.lstm(x)  # LSTM 输出 [batch_size, seq_len, lstm_hidden_size]
        
        # 使用dyhead来预测未来轨迹
        output = self.dynamic_head(x)  # [batch_size, pred_length, num_joints, 2]
        
        return output
    

if __name__ == "__main__":
    num_joints = 33
    embed_size = 128
    num_heads = 8
    num_layers = 4
    behavior_vocab_size = 10
    behavior_embed_size = 32
    pred_length = 5
    lstm_hidden_size = 256
    lstm_num_layers = 2

    # 假设输入是一个批次的历史轨迹和行为标签
    trajectory_input = torch.randn(32, 10, num_joints * 2)  # [batch_size, seq_len, num_joints * 2]
    behavior_input = torch.randint(0, behavior_vocab_size, (32,))  # [batch_size]

    # 模型实例
    model = TrajectoryTransformer(num_joints=num_joints, embed_size=embed_size, num_heads=num_heads, num_layers=num_layers, 
                                  behavior_vocab_size=behavior_vocab_size, behavior_embed_size=behavior_embed_size, 
                                  pred_length=pred_length, lstm_hidden_size=lstm_hidden_size, lstm_num_layers=lstm_num_layers)

    # 前向传播，预测未来轨迹
    future_trajectory = model(trajectory_input, behavior_input)

    print(future_trajectory.shape)  # 输出形状: [batch_size, pred_length, num_joints, 2]