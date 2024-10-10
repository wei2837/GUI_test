import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn import MultiheadAttention
from torch.autograd import Variable
import math

class PositionalEncoder(nn.Module):
    def __init__(self, d_model, max_seq_len = 501):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        # create constant 'pe' matrix with values dependant on 
        # pos and i
        pe = torch.zeros(max_seq_len, d_model)
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = \
                math.sin(pos / (10000 ** ((2 * i)/d_model)))
                pe[pos, i + 1] = \
                math.cos(pos / (10000 ** ((2 * (i + 1))/d_model)))
                
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        # make embeddings relatively larger
        # add constant to embedding
        
        seq_len = x.size(1)
        #print(seq_len)
        #print('pe', self.pe.shape)
        #assert seq_len<=self.max_seq_len
        if self.training:
            assert seq_len<=self.max_seq_len
            x = x + Variable(self.pe[:,:seq_len], requires_grad=False).cuda()
        else:
            if seq_len >= self.max_seq_len:
                pe = F.interpolate(
                    self.pe.transpose(1,2), seq_len, mode='linear', align_corners=False)
                #print('pe', self.pe.shape)
                #print('x', x.shape)
                x = x.transpose(1,2) + Variable(pe, requires_grad=False).cuda()
                x = x.transpose(1,2)
        return x

class attentionLayer(nn.Module):

    def __init__(self, d_model, nhead, positional_emb_flag, dropout=0.1):
        super(attentionLayer, self).__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        self.positional_emb_flag = positional_emb_flag
        if positional_emb_flag:
            self.positional_emb = PositionalEncoder(d_model)
        
        self.linear1 = nn.Linear(d_model, d_model * 4)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_model * 4, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = F.relu

    def forward(self, src, tar):
        # type: (Tensor, Optional[Tensor], Optional[Tensor]) -> Tensor
        if self.positional_emb_flag:
            src = self.positional_emb(src)
            tar = self.positional_emb(tar)
        src = src.transpose(0, 1) # B, T, C -> T, B, C
        tar = tar.transpose(0, 1) # B, T, C -> T, B, C
        src2 = self.self_attn(tar, src, src, attn_mask=None,
                              key_padding_mask=None)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        src = src.transpose(0, 1) # T, B, C -> B, T, C
        return src

class attentionLayer_mask(nn.Module):

    def __init__(self, d_model, nhead, positional_emb_flag, dropout=0.1):
        super(attentionLayer_mask, self).__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        self.positional_emb_flag = positional_emb_flag
        if positional_emb_flag:
            self.positional_emb = PositionalEncoder(d_model)
        
        self.linear1 = nn.Linear(d_model, d_model * 4)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_model * 4, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = F.relu

    def forward(self, src, tar, mask):
        # type: (Tensor, Optional[Tensor], Optional[Tensor]) -> Tensor
        mask = ~mask
        if self.positional_emb_flag:
            src = self.positional_emb(src)
            tar = self.positional_emb(tar)
        src = src.transpose(0, 1) # B, T, C -> T, B, C
        tar = tar.transpose(0, 1) # B, T, C -> T, B, C
        src2 = self.self_attn(tar, src, src, attn_mask=None,
                              key_padding_mask=mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        src = src.transpose(0, 1) # T, B, C -> B, T, C
        return src

