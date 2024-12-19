import torch
import torch.nn as nn
import torch.nn.functional as F
import re
import math
import importlib
import spacy
import torch
import torch.nn as nn
from torch.autograd import Variable
import copy
import torch.nn.functional as F
from datetime import datetime

def exists(val):
    return val is not None


def prob_mask_like(shape, prob, device):
    if prob == 1:
        return torch.ones(shape, device = device, dtype = torch.bool)
    elif prob == 0:
        return torch.zeros(shape, device = device, dtype = torch.bool)
    else:
        return torch.zeros(shape, device = device).float().uniform_(0, 1) < prob 
        
# classifier-free condition dropout
def conditional_dropout(text_embeds, cond_drop_prob):
    # 获取text_embeds的形状和设备
    batch_size, seq_len, d_feature = text_embeds.shape
    device = text_embeds.device
    
    # 生成与text_embeds形状相同的概率掩码
    keep_mask = prob_mask_like((batch_size, seq_len), 1 - cond_drop_prob, device)
    
    # 扩展keep_mask的维度以匹配text_embeds的维度
    keep_mask_expanded = keep_mask.unsqueeze(-1).expand(batch_size, seq_len, d_feature)
    
    # 使用掩码进行dropout，将不保留的部分设置为0
    text_embeds_with_dropout = torch.where(keep_mask_expanded, text_embeds, torch.zeros_like(text_embeds))
    
    return text_embeds_with_dropout

class Unet_1d(nn.Module):
    def __init__(self,
                 self_cond=False,
                 seq_len=62,
                 condition_feature=256,
                 time_in_feature=1,
                 time_hidden_feature=512,
                 time_out_feature=512,
                 dropout=0.1,
                 fc_h=1024,
                 fc_out=1024,
                 num_blocks=5,  # transformer的encoder数量5+1=6
                 num_heads=8,
                 d_encoder_ff=2048,  # 前馈网络隐藏层维度
                 in_hidden_channel=1024,  # 噪音张量的隐藏层特征数量
                 in_feature=1024,  # 噪音张量输出特征数量
                 lstm_hidden_size=512,
                 lstm_num_layers=6,
                 batch_first=True,
                 bidirectional=True,
                 num_decoder_block=7,
                 d_decoder_ff=1024,
                 classifier_free=True，
                 cond_drop_prob=0.1
                 ):
        super().__init__()
        self.self_cond = self_cond
        self.lowres_cond = False
        self.dummy_parameter = nn.Parameter(torch.tensor([0.]))
        self.classifier_free=classifier_free

        # 噪音张量(64,)>(64, in_feature)
        self.Embedding = SeqInput(time_out_feature, in_hidden_channel, in_feature)  # 512, 1024, 1024
        self.input_PE = RotaryPositionEmbedding_learned(in_feature)  # 1024
        self.lstm_down = BiDirectLSTMEncoderBlockInitial(
            in_feature,  # 1024
            lstm_hidden_size,  # 512
            lstm_num_layers,  # 6
            batch_first,  # True
            bidirectional
        )

        # self.condition_PE = RotaryPositionEmbedding_learned(in_feature) # 128

        self.time_fusion = time_fusion(
            seq_len,
            condition_feature,  # 256
            time_in_feature,  # 1
            time_hidden_feature,  # 512,
            time_out_feature,  # 512
            dropout=0.1
        )
        self.fc = nn.Sequential(
            nn.Linear(time_out_feature, fc_h),
            nn.Linear(fc_h, fc_out),
            nn.SELU(inplace=True),
        )

        self.encoder = TranformerEncoder(
            num_blocks,  # 5+1(default)
            fc_out,  # 1024
            num_heads,  # 8
            d_encoder_ff,  # 2048
            dropout,  # 0.1
        )

        self.decoder = TransformerDecoder(
            num_decoder_block,  # 6
            lstm_hidden_size * 2,  # 512*2=1024
            num_heads,  # 8
            d_decoder_ff,  # 1024
            dropout,  # 0.1
        )

        self.lstm_up = nn.LSTM(input_size=d_decoder_ff,  # 1024
                               hidden_size=lstm_hidden_size,  # 512
                               num_layers=lstm_num_layers,  # 6
                               batch_first=batch_first,  # True
                               bidirectional=bidirectional,  # True
                               )
        self.dense = nn.Sequential(
            nn.Linear(lstm_hidden_size * 2, lstm_hidden_size),  # 1024， 512
            nn.Linear(lstm_hidden_size, 1),  # 512, 1
            nn.SELU(inplace=True),
        )
        self.cond_drop_prob = cond_drop_prob

    def forward_with_cond_scale(
            self,
            *args,
            cond_scale=1.,
            **kwargs
    ):
        logits = self.forward(*args, **kwargs)

        if cond_scale == 1:
            return logits

        null_logits = self.forward(*args, cond_drop_prob=1., **kwargs)
        return null_logits + (logits - null_logits) * cond_scale

    def forward(self,
                x,
                time,
                *,
                lowres_cond_img=None,
                lowres_noise_times=None,
                text_embeds=None,
                text_mask=None,
                cond_images=None,
                self_cond=None,
                cond_drop_prob=0.
                ):
        '''

        :param x: noise tensor - (batch, seq,)
        :param text_embeds: condition of secondary structure seq - (batch, seq, d_gcn + d_condition)
        :param time: noise level - float
        :return: denoise tensor - (batch, seq,)
        '''
        if exists(text_embeds) and self.classifier_free:
            text_embeds = conditional_dropout(text_embeds, cond_drop_prob)

        # print(x.shape)
        # print(c.shape)
        # print(edge_index.shape)
        cond = self.time_fusion(text_embeds, time)  # (batch, seq, 1) + (batch, seq, d_con=256) > (batch, seq, 512)
        condition = self.fc(cond) # (batch, seq, 512)>(batch, seq, 1024)

        # print('condition shape:', condition.shape)
        condition = self.encoder(condition)  # condition
        # (batch, seq, 1024) > (batch, seq, 1024)

        # print('condition shape:', condition.shape)

        x = self.Embedding(x, cond)  # (batch, seq, ) > (batch, seq, 1+d_cond=512) > (batch, seq, 1024)
        x = self.input_PE(x)
        # print('x.shape:', x.shape)
        x, (hn, cn) = self.lstm_down(x, time)  # hn - (num_layers * num_directs, batch, hidden_size) = (12, batch, 512)
        # print('lstm_down', x.shape, hn.shape, cn.shape)

        output = self.decoder(x, condition) # (batch, seq, 1024)
        # print('output:', output.shape)

        output, (_, _) = self.lstm_up(output, (hn, cn)) # (batch, seq, 2*512)
        output = self.dense(output)
        output = output.squeeze(dim=-1)
        # print('output:', output.shape)

        return output

    def cast_model_parameters(self, *args, **kwargs):
        return self


class SeqInput(nn.Module):
    def __init__(self, d_condition=512, d_emb=1024, d_model=1024):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(1 + d_condition, d_emb),
            nn.Linear(d_emb, d_model),
            nn.SELU(inplace=True)
        )

    def forward(self, src, cond):
        src = src.unsqueeze(dim=-1)
        src = torch.cat((src, cond), dim=-1)
        output = self.ffn(src)

        return output


# 自定义初始化的双向lstm
class BiDirectLSTMEncoderBlockInitial(nn.Module):
    # 自定义初始化h0 c0
    def __init__(self, input_size=512, hidden_size=512, num_layers=6, batch_first=True, bidirectional=True):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first

        # 可学习的初始单元状态c0
        self.h0 = nn.Parameter(torch.randn(num_layers * 2, 1, hidden_size), requires_grad=True)

        # 初始化双向LSTM层
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=batch_first, bidirectional=bidirectional)

    def forward(self, x, t):
        # 因为是双向的，所以隐藏状态和细胞状态的维度会是单向的两倍
        # 可学习的隐藏状态初始化
        h0 = self.h0.repeat(1, x.size(0), 1).to(x.device)

        # 根据时间条件进行细胞状态初始化
        c0 = t * torch.ones((self.num_layers * 2, x.size(0), self.hidden_size), dtype=torch.float32).to(x.device)

        # 前向传播通过LSTM
        out, (hn, cn) = self.lstm(x, (h0, c0))
        # 求平均保持output维度一致
        # out_avg = out.view(out.size(0), out.size(1), 2, self.hidden_size).mean(dim=2)
        return out, (hn, cn)


class CrossAttention(nn.Module):
    # 条件张量用于key, value
    def __init__(self, d_model=1024, num_heads=8):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model, num_heads)
        self.d_model = d_model
        self.dropout = nn.Dropout(0.1)
        self.layer_norm = nn.LayerNorm(d_model)

        # 前馈神经网络，用于增加模型的表达能力
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.Linear(d_model * 2, d_model),
            nn.ReLU()
        )

    def forward(self, query, key, value, mask=None):
        """
        query: 输入张量，形状为 (batch_size, seq_len_q, d_model)
        key: 条件张量（用作key），形状为 (batch_size, seq_len_k, d_model)
        value: 条件张量（用作value），形状为 (batch_size, seq_len_v, d_model)
        mask: 可选的注意力掩码，形状为 (batch_size, seq_len_q, seq_len_k)
        """
        # 使用多头注意力机制计算cross-attention
        attn_output, _ = self.multihead_attn(query, key, value, attn_mask=mask)

        # 残差连接和层归一化
        attn_output = self.dropout(attn_output)
        attn_output = attn_output + query
        attn_output = self.layer_norm(attn_output)

        # 前馈神经网络
        attn_output = self.ffn(attn_output)

        # 返回cross-attention的输出
        return attn_output


class SelfAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(SelfAttention, self).__init__()
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)

        self.out_linear = nn.Linear(d_model, d_model)

        self.softmax = nn.Softmax(dim=-1)

    def split_heads(self, x, batch_size):
        x = x.reshape(batch_size, -1, self.num_heads, self.d_k)
        return x.permute(0, 2, 1, 3)

    def forward(self, v, k, q, mask):
        batch_size = q.size(0)

        q = self.q_linear(q)  # (batch_size, seq_len, d_model)
        k = self.k_linear(k)  # (batch_size, seq_len, d_model)
        v = self.v_linear(v)  # (batch_size, seq_len, d_model)

        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, d_k)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, d_k)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, d_k)

        scaled_attention, attention_weights = self.scaled_dot_product_attention(q, k, v, mask)
        scaled_attention = scaled_attention.permute(0, 2, 1, 3).contiguous()

        new_context_layer = scaled_attention.reshape(batch_size, -1, self.d_k * self.num_heads)
        output = self.out_linear(new_context_layer)
        return output, attention_weights

    def scaled_dot_product_attention(self, q, k, v, mask):
        matmul_qk = torch.matmul(q, k.transpose(-2, -1))  # (batch_size, num_heads, seq_len_q, seq_len_k)
        dk = torch.tensor(self.d_k, dtype=torch.float32)
        scaled_attention_logits = matmul_qk / dk

        if mask is not None:
            scaled_attention_logits += (mask * -1e9)

        attention_weights = self.softmax(scaled_attention_logits)  # (batch_size, num_heads, seq_len_q, seq_len_k)
        output = torch.matmul(attention_weights, v)  # (batch_size, num_heads, seq_len_q, d_k)
        return output, attention_weights


class double_GCN(nn.Module):
    def __init__(self, num_features, hidden_channels, num_classes):
        super().__init__()
        self.conv1 = GCNConv(num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, num_classes)

    def forward(self, x, edge_index):
        # 第一层GCN
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)

        # 第二层GCN
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)

        return F.log_softmax(x, dim=1)


class RotaryPositionEmbedding_learned(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        assert d_model % 2 == 0, "d_model must be an even number"
        self.d_model = d_model
        # 随机初始化
        # self.theta_raw = nn.Parameter(torch.randn(self.d_model // 2), requires_grad=True)
        # pi/2初始化
        self.theta_raw = nn.Parameter(torch.pi / 2 * torch.ones(self.d_model // 2), requires_grad=True)

    def forward(self, x):
        # 计算角度
        device = x.device
        seq_len = x.shape[1]
        theta_raw = self.theta_raw.unsqueeze(0).expand(seq_len, -1).to(device)  # theta的尺寸 [seq_len, d_model // 2]
        seq_pos = torch.arange(1, seq_len + 1, dtype=torch.float32).unsqueeze(-1).expand(-1, self.d_model // 2).to(
            device)
        theta = theta_raw * seq_pos

        # 生成复数表示
        sin_theta = torch.sin(theta)
        cos_theta = torch.cos(theta)
        # theta_vals 的形状将变成 [seq_len, d_model // 2, 2]
        # .stack沿最后一维将cos张量和sin张量堆积起来
        theta_vals = torch.stack([cos_theta, sin_theta], dim=-1)

        # 对嵌入向量进行旋转
        # x [batch_size, seq_len, d_model]
        x_r = x[:, :, 0::2]  # 取偶数维度  0::2 表示从索引0开始，每隔一个元素取一个，即提取偶数索引的元素。
        x_i = x[:, :, 1::2]  # 取奇数维度  1::2 表示从索引1开始，每隔一个元素取一个，即提取奇数索引的元素。

        # 应用旋转
        # ...是>3.5python的新索引语法，表示前面所有维度保持不变
        x_r_new = x_r * theta_vals[..., 0] - x_i * theta_vals[..., 1]
        x_i_new = x_r * theta_vals[..., 1] + x_i * theta_vals[..., 0]

        # 合并回原始形状
        x = torch.stack([x_r_new, x_i_new], dim=-1)
        x = x.reshape(-1, seq_len, self.d_model)  # [batch_size, seq_len, d_model]

        return x


class condition_fusion(nn.Module):
    '''
    融合边索引/接触矩阵,特征张量,之后进行旋转位置编码，以及融合噪音强度条件(时间)

    输入：
    time：标量，当前扩散时间的噪音强度

    gcn：
    两层的gcn网络
    c: 特征张量(batch * seq, )
    edge_index: shape(2, batch * num_edge)张量，第一行节点起点，第二行节点终点
    output: shape(seq, out_d_feature)

    PE:
    角度可学习的旋转位置编码
    c: 特征张量(seq, out_d_feature)

    噪音强度标量扩维，再拼接到条件张量特征维度上
    通过线性层处理

    '''

    def __init__(self,
                 seq_len=62,
                 gcn_in_features=32,
                 gcn_hidden_channels=64,
                 gcn_out_feature=128,
                 time_in_feature=1,
                 time_hidden_feature=128,
                 time_out_feature=256,
                 dropout=0.1
                 ):
        super().__init__()
        self.seq_len = seq_len
        self.gcn_in_features = gcn_in_features
        self.gcn_out_feature = gcn_out_feature
        self.embedding = nn.Sequential(
            nn.Linear(1, gcn_in_features),
            nn.Linear(gcn_in_features, gcn_in_features)
        )
        self.gcn = double_GCN(gcn_in_features, gcn_hidden_channels, gcn_out_feature)
        self.PE = RotaryPositionEmbedding_learned(self.gcn_out_feature + self.gcn_in_features)
        self.time_in_feature = time_in_feature
        self.fc = nn.Sequential(
            nn.Linear(self.time_in_feature + self.gcn_out_feature + self.gcn_in_features, time_hidden_feature),
            nn.Linear(time_hidden_feature, time_out_feature),
            nn.ReLU(inplace=True),
        )
        self.norm = nn.LayerNorm(time_out_feature)
        self.dropout = nn.Dropout(dropout)

    def forward(self, c, edge_index, time):
        # 扩维和嵌入
        c = c.unsqueeze(-1)
        c = self.embedding(c)
        c_gcn = self.gcn(c, edge_index)  # (batch*seq, 128)
        c = torch.cat((c, c_gcn), dim=-1).reshape(-1, self.seq_len, self.gcn_out_feature + self.gcn_in_features)
        c = self.PE(c)

        # 条件融合
        batch = c.shape[0]
        time_tensor = time * torch.ones(batch, self.seq_len, self.time_in_feature)
        condition = torch.cat((c, time_tensor), dim=-1)
        condition = self.norm(self.fc(condition))
        output = self.dropout(condition)

        # output - shape(batch, seq_len, d_out_feature)
        return output


class condition_process(nn.Module):
    '''
    特征张量进行旋转位置编码

    c: 特征张量(batch , seq )
    output: shape(seq, out_d_feature)

    PE:
    角度可学习的旋转位置编码
    c: 特征张量(seq, out_d_feature)

    噪音强度标量扩维，再拼接到条件张量特征维度上
    通过线性层处理

    '''

    def __init__(self,
                 seq_len=400,
                 in_features=1,
                 hidden_channels=512,
                 out_feature =512,
                 dropout=0.1
                 ):
        super().__init__()
        self.seq_len = seq_len
        self.embedding = nn.Sequential(
            nn.Linear(in_features, hidden_channels),
            nn.Linear(hidden_channels, out_feature),
            nn.LayerNorm(out_feature),
            nn.Dropout(dropout),
            nn.Sigmoid()
        )
        self.PE = RotaryPositionEmbedding_learned(out_feature)

    def forward(self, c):
        # 扩维和嵌入
        c = c.unsqueeze(-1)
        c = self.embedding(c)
        c = self.PE(c)
        return c


class time_fusion(nn.Module):
    def __init__(self,
                 seq_len=300,
                 condition_feature=256,  #128, 128
                 time_in_feature=1,
                 time_hidden_feature=512,
                 time_out_feature=1024,
                 dropout=0.1
                 ):
        super().__init__()
        self.seq_len = seq_len
        self.time_in_feature = time_in_feature
        self.fc = nn.Sequential(
            nn.Linear(condition_feature + self.time_in_feature, time_hidden_feature),
            nn.Linear(time_hidden_feature, time_out_feature),
            nn.SELU(inplace=True),
        )
        self.norm = nn.LayerNorm(time_out_feature)
        self.dropout = nn.Dropout(dropout)

    def forward(self, c, time):
        # 条件融合
        batch, seq_len, device = c.shape[0], c.shape[1], c.device
        time_tensor = time * torch.ones(batch, seq_len, self.time_in_feature).to(device)

        condition = torch.cat((c, time_tensor), dim=-1)
        condition = self.norm(self.fc(condition))
        output = self.dropout(condition)

        # output - shape(batch, seq_len, d_out_feature)
        return output


class SelfAttentionBlock(nn.Module):
    def __init__(self, d_model=256, num_heads=8, d_ff=512, dropout=0.1):
        super().__init__()

        self.self_attention = SelfAttention(d_model, num_heads)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

        # 前馈网络定义
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(inplace=True),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, src, mask=None):
        # 假设src的形状是(batch, seq_len, d_model)
        x, _ = self.self_attention(src, src, src, mask)  # x (batch, seq_len, d_model)
        x = x + self.dropout(src)
        x = self.norm(x)
        x = self.ffn(x)

        return x


class LastSelfAttentionBlock(nn.Module):
    def __init__(self, d_model=256, num_heads=8, d_ff=512, dropout=0.1):
        super().__init__()

        self.self_attention = SelfAttention(d_model, num_heads)
        self.dropout = nn.Dropout(dropout)
        self.norm_1 = nn.LayerNorm(d_model)

        # 前馈网络定义
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(inplace=True),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        self.norm_2 = nn.LayerNorm(d_model)

    def forward(self, src, mask=None):
        # 假设src的形状是(batch, seq_len, d_model)
        x, _ = self.self_attention(src, src, src, mask)  # x (batch, seq_len, d_model)
        x = x + self.dropout(src)
        x = self.norm_1(x)
        x = x + self.ffn(x)
        x = self.norm_2(x)

        return x


class MultiSelfAttentionBlock(nn.Module):
    def __init__(self, num_layers=6, d_model=256, num_heads=8, d_ff=1024, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            SelfAttentionBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, src, mask=None):
        for layer in self.layers:
            src = layer(src, mask)
        return src


class TranformerEncoder(nn.Module):
    def __init__(self, num_layers=5, d_model=1024, num_heads=8, d_ff=2048, dropout=0.1):
        super().__init__()
        self.multi_SA = MultiSelfAttentionBlock(num_layers, d_model, num_heads, d_ff, dropout)
        self.last_SA = LastSelfAttentionBlock(d_model, num_heads, d_ff, dropout)

    def forward(self, condition):
        condition = self.multi_SA(condition)
        condition = self.last_SA(condition)

        return condition


class DecoderBlock(nn.Module):
    def __init__(self, d_model, nhead=8, dim_feedforward=1024, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # 实现时可能需要为encoder-decoder attention传入encoder的输出和mask
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, x, condition, condition_mask=None, src_key_padding_mask=None):
        # 自注意力
        q = k = v = self.dropout1(x)
        x_attn = self.self_attn(q, k, v, attn_mask=condition_mask, key_padding_mask=src_key_padding_mask)[0]
        x = x + self.dropout2(x_attn)
        x = self.norm1(x)

        # 交叉注意力（encoder-decoder attention）
        q = self.dropout1(x)
        x_attn = \
            self.multihead_attn(q, condition, condition, attn_mask=condition_mask,
                                key_padding_mask=src_key_padding_mask)[0]
        x = x + self.dropout2(x_attn)
        x = self.norm2(x)

        # 前馈网络
        x2 = self.linear2(self.dropout(F.relu(self.linear1(x))))
        x = x + self.dropout3(x2)
        x = self.norm3(x)
        return x


class TransformerDecoder(nn.Module):
    def __init__(self, num_block, d_model, nhead=8, dim_feedforward=1024, dropout=0.1):
        super().__init__()
        self.decoder = nn.ModuleList([DecoderBlock(d_model, nhead, dim_feedforward, dropout) for _ in range(num_block)])

    def forward(self, x, condition, condition_mask=None, src_key_padding_mask=None):
        for layer in self.decoder:
            x = layer(x, condition, condition_mask=condition_mask,
                      src_key_padding_mask=src_key_padding_mask)
        return x


def count_parameters(model, count_all=False):
    if count_all == False:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())


if __name__ == '__main__':
    # self_cond = False,
    # seq_len = 64,
    # gcn_in_features = 32,
    # gcn_hidden_channels = 64,
    # gcn_out_feature = 64,
    # time_in_feature = 1,
    # time_hidden_feature = 128,
    # time_out_feature = 256,
    # dropout = 0.1,
    # num_blocks = 5,  # transformer的encoder数量5+1=6
    # num_heads = 8,
    # d_encoder_ff = 1024,  # 前馈网络隐藏层维度
    # in_hidden_channel = 64,  # 噪音张量的隐藏层特征数量
    # in_feature = 128,  # 噪音张量输出特征数量
    # lstm_hidden_size = 128,
    # lstm_num_layers = 4,
    # batch_first = True,
    # bidirectional = True,
    # num_decoder_block = 6,
    # d_decoder_ff = 1024,

    num_features = 32  # 节点特征的维度
    hidden_channels = 64  # 隐藏层的维度
    num_classes = 64  # 分类的类别数
    num_nodes = 64  # 节点的个数
    seq_len = 62  # 最大的seq长度
    time_in_feature = 1  # 时间信息维度
    time_out_feature = 256  # 整合时间信息后线性层的输出维度
    time_hidden_feature = 128  # 整合时间信息后线性隐藏层的维度

    # model = denoiser()
    model = Unet_1d()

    # model = condition_fusion(seq_len=seq_len,
    #                          gcn_in_features=num_features,
    #                          gcn_hidden_channels=hidden_channels,
    #                          gcn_out_feature=num_classes,
    #                          time_in_feature=time_in_feature,
    #                          time_hidden_feature=time_hidden_feature,
    #                          time_out_feature=time_out_feature
    #                          )
    print(count_parameters(model))
    # 假设你已经有以下数据
    node_features = torch.randn((num_nodes,))  # 节点特征，大小为 (64, d_feature)
    edge_index = torch.tensor([[1, 2, 5, 6], [2, 3, 7, 9]], dtype=torch.long)  # 边索引，大小为 (2, E)，其中E是边的数量
    node_labels = torch.randn((num_nodes,))  # 节点标签，假设为单标签分类，大小为 (64,)
    time = 0.5

    # 创建Data对象
    data_list = [Data(x=node_features, edge_index=edge_index, y=node_labels) for _ in range(16)]

    # 使用DataLoader进行批处理（这里假设只有一个图）
    # 如果你有多个图，你需要使用data_list代替单个data对象
    dataloader = DataLoader(data_list, batch_size=4, shuffle=True)
    for b in dataloader:
        # b.y.shape (batch*seq)
        # print('b.x.shape', b.x.shape)
        # print('b.y.shape', b.y.shape)
        # print('b.edge_index.shape', b.edge_index.shape)
        b.y = b.y.reshape(4, -1)
        out = model(b.y, b.x, b.edge_index, time)

        # 输出尺寸(batch * seq, d_feature) 需要重新reshape
        # print(out)
        print(out.shape)
        break

