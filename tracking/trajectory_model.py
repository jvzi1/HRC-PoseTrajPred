import torch
import torch.nn as nn

class DynamicTrajectoryHead(nn.Module):
    def __init__(self, embed_size, num_joints, pred_length):
        super(DynamicTrajectoryHead, self).__init__()

        self.pred_length = pred_length
        self.num_joints = num_joints
        
        # 自注意力层，用于计算每个时间步的注意力权重
        self.attention = nn.MultiheadAttention(embed_size, num_heads=4, batch_first=True)
        
        # 用于从注意力加权后的特征中预测未来轨迹
        self.fc_out = nn.Linear(embed_size, num_joints * 3 * pred_length)
        
    def forward(self, x):
        """
        x: [batch_size, seq_len, embed_size] 输入历史轨迹的特征
        """
        # 对输入的时序特征应用自注意力机制
        attention_output, _ = self.attention(x, x, x)  # 自注意力机制
        attention_output = attention_output.mean(dim=1)  # 对时间维度求平均，得到全局的轨迹特征
        
        # 动态生成未来轨迹
        output = self.fc_out(attention_output)  # [batch_size, num_joints * 3 * pred_length]
        return output.view(output.size(0), self.pred_length, self.num_joints * 3)  # [batch_size, pred_length, num_joints * 3]


class TrajectoryTransformerModel(nn.Module):
    def __init__(self, num_joints, embed_size, num_heads, num_layers, 
                 behavior_vocab_size, camera_name_vocab_size,behavior_embed_size, camera_name_embed_size, pred_length, lstm_hidden_size, lstm_num_layers):
        super(TrajectoryTransformerModel, self).__init__()
        


        # 轨迹数据输入线性层，将输入关键点位置映射到 embed_size
        self.trajectory_embedding = nn.Linear(num_joints * 3, embed_size)  # 每个关键点 (x, y, z)
        
        # 行为标签嵌入层
        self.behavior_embedding = nn.Embedding(behavior_vocab_size, behavior_embed_size)

        # 相机标签嵌入层
        self.camera_embedding = nn.Embedding(camera_name_vocab_size, camera_name_embed_size)
        
        # Transformer 编码器
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_size + behavior_embed_size + camera_name_embed_size, nhead=num_heads)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 主干网络--lstm
        self.lstm = nn.LSTM(input_size=embed_size + behavior_embed_size + camera_name_embed_size, 
                            hidden_size=lstm_hidden_size, 
                            num_layers=lstm_num_layers, 
                            batch_first=True)
        
        # Head--输出tracking轨迹
        self.dynamic_head = DynamicTrajectoryHead(lstm_hidden_size, num_joints, pred_length)
    
    
    def forward(self, trajectory, behavior, camera_name):
        # 轨迹数据 embedding
        # trajectory: [batch_size, seq_len, num_joints * 3]
        trajectory_embedded = self.trajectory_embedding(trajectory)  # [batch_size, seq_len, embed_size]
        
        # 行为标签 embedding
        # behavior: [batch_size]
        behavior_embedded = self.behavior_embedding(behavior).unsqueeze(1)  # [batch_size, 1, behavior_embed_size]
        behavior_embedded = behavior_embedded.repeat(1, trajectory_embedded.size(1), 1)  # 行为标签扩展到与轨迹长度匹配
        camera_embedded = self.camera_embedding(camera_name).unsqueeze(1)  # [batch_size, 1, behavior_embed_size]
        camera_embedded = camera_embedded.repeat(1, trajectory_embedded.size(1), 1)  # 行为标签扩展到与轨迹长度匹配
        # 将行为嵌入与轨迹嵌入拼接
        x = torch.cat((trajectory_embedded, behavior_embedded, camera_embedded), dim=2)  # [batch_size, seq_len, embed_size + behavior_embed_size]
        
        # Transformer 编码器
        x = self.transformer_encoder(x)  # [batch_size, seq_len, embed_size + behavior_embed_size]
        
        # x = x[:, -1, :]  # 只取最后时间步的输出 [batch_size, embed_size + behavior_embed_size]
        
        # LSTM 主干网络
        x, _ = self.lstm(x)  # LSTM 输出 [batch_size, seq_len, lstm_hidden_size]
        
        # 使用dyhead来预测未来轨迹
        output = self.dynamic_head(x)  # [batch_size, pred_length, num_joints, 2]
        
        return output
    
import torch
import torch.nn as nn

class LSTMTrajectoryModel(nn.Module):
    def __init__(self, num_joints, embed_size, pred_length, lstm_hidden_size, lstm_num_layers):
        super(LSTMTrajectoryModel, self).__init__()
        
        self.pred_length = pred_length
        self.num_joints = num_joints
        
        # 轨迹数据输入线性层，将输入关键点位置映射到 embed_size
        self.trajectory_embedding = nn.Linear(num_joints * 3, embed_size)
        
        # 主干网络 -- LSTM
        self.lstm = nn.LSTM(input_size=embed_size, 
                            hidden_size=lstm_hidden_size, 
                            num_layers=lstm_num_layers, 
                            batch_first=True)
        
        # 用于从 LSTM 的隐藏状态中预测未来轨迹
        self.fc_out = nn.Linear(lstm_hidden_size, num_joints * 3 * pred_length)
    
    def forward(self, trajectory):
        """
        trajectory: [batch_size, seq_len, num_joints * 3]
        """
        # 轨迹数据 embedding
        trajectory_embedded = self.trajectory_embedding(trajectory)  # [batch_size, seq_len, embed_size]
        
        # LSTM 主干网络
        lstm_output, _ = self.lstm(trajectory_embedded)  # [batch_size, seq_len, lstm_hidden_size]
        
        # 只取最后一个时间步的 LSTM 输出
        lstm_output_last = lstm_output[:, -1, :]  # [batch_size, lstm_hidden_size]
        
        # 预测未来轨迹
        output = self.fc_out(lstm_output_last)  # [batch_size, num_joints * 3 * pred_length]
        
        return output.view(output.size(0), self.pred_length, self.num_joints * 3)  # [batch_size, pred_length, num_joints * 3]


if __name__ == "__main__":
    num_joints = 33
    embed_size = 128
    pred_length = 8
    lstm_hidden_size = 256
    lstm_num_layers = 2

    model = LSTMTrajectoryModel(
        num_joints=num_joints,
        embed_size=embed_size,
        pred_length=pred_length,
        lstm_hidden_size=lstm_hidden_size,
        lstm_num_layers=lstm_num_layers
    )

    print(model)


    num_joints = 33
    embed_size = 128
    num_heads = 8
    num_layers = 4
    behavior_vocab_size = 6
    camera_name_vocab_size = 3
    behavior_embed_size = 32
    camera_name_embed_size = 32
    pred_length = 8
    lstm_hidden_size = 256
    lstm_num_layers = 2

    model = TrajectoryTransformerModel(
        num_joints=num_joints,
        embed_size=embed_size,
        num_heads=num_heads,
        num_layers=num_layers,
        behavior_vocab_size=behavior_vocab_size,
        camera_name_vocab_size=camera_name_vocab_size,
        behavior_embed_size=behavior_embed_size,
        camera_name_embed_size=camera_name_embed_size,
        pred_length=pred_length,
        lstm_hidden_size=lstm_hidden_size,
        lstm_num_layers=lstm_num_layers
    )

    print(model)