import torch
import torch.nn as nn
from act_module.backbone import build_backbone_v2
from act_module.transformer import build_transformer_v2, TransformerEncoder, TransformerEncoderLayer

class DETRVAEBuilder:
    def __init__(self):
        pass

    class Network(nn.Module):
        def __init__(self, params, **kwargs):
            super().__init__()
            self.state_dim = params['state_dim']
            self.num_queries = params['num_queries']
            self.camera_names = params['camera_names']

            self.backbones = [build_backbone_v2(params)] if params.get('use_image') else None
            self.transformer = build_transformer_v2(params)
            self.encoder = self._build_encoder(params)

            hidden_dim = self.transformer.d_model
            self.action_head = nn.Linear(hidden_dim, self.state_dim)
            self.is_pad_head = nn.Linear(hidden_dim, 1)
            self.query_embed = nn.Embedding(self.num_queries, hidden_dim)
            self.input_proj_robot_state = nn.Linear(14, hidden_dim)
            self.latent_dim = 32
            self.latent_proj = nn.Linear(hidden_dim, self.latent_dim * 2)
            self.latent_out_proj = nn.Linear(self.latent_dim, hidden_dim)

        def _build_encoder(self, params):
            d_model = params['hidden_dim']
            nhead = params['nheads']
            dim_feedforward = params['dim_feedforward']
            num_layers = params['enc_layers']
            activation = 'relu'
            dropout = params['dropout']

            encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation)
            encoder_norm = nn.LayerNorm(d_model)
            encoder = TransformerEncoder(encoder_layer, num_layers, encoder_norm)
            return encoder

        def forward(self, qpos, image, env_state=None, actions=None, is_pad=None):
            bs, _ = qpos.shape

            if actions is not None:
                action_embed = self.encoder_action_proj(actions)
                qpos_embed = self.encoder_joint_proj(qpos).unsqueeze(1)
                cls_embed = self.cls_embed.weight.unsqueeze(0).repeat(bs, 1, 1)
                encoder_input = torch.cat([cls_embed, qpos_embed, action_embed], dim=1)
                encoder_input = encoder_input.permute(1, 0, 2)
                pos_embed = self.pos_table.clone().detach().permute(1, 0, 2)
                encoder_output = self.encoder(encoder_input, pos=pos_embed, src_key_padding_mask=is_pad)
                encoder_output = encoder_output[0]
                latent_info = self.latent_proj(encoder_output)
                mu, logvar = latent_info[:, :self.latent_dim], latent_info[:, self.latent_dim:]
                latent_sample = self.reparametrize(mu, logvar)
                latent_input = self.latent_out_proj(latent_sample)
            else:
                mu = logvar = None
                latent_input = torch.zeros([bs, self.latent_dim], dtype=torch.float32).to(qpos.device)

            proprio_input = self.input_proj_robot_state(qpos)
            features, pos = self.backbones[0](image[:, 0]) if self.backbones else (None, None)
            src = self.input_proj(features) if features is not None else proprio_input
            hs = self.transformer(src, None, self.query_embed.weight, pos, latent_input)[0]
            a_hat = self.action_head(hs)
            is_pad_hat = self.is_pad_head(hs)

            return a_hat, is_pad_hat, [mu, logvar]

        def reparametrize(self, mu, logvar):
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std

    def build(self, name, **kwargs):
        return DETRVAEBuilder.Network(kwargs)
