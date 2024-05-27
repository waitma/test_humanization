import torch
import torch.nn as nn
from inspect import isfunction

import torch
import torch.nn.functional as F
# from einops import rearrange, repeat
# from torch import nn, einsum

def exists(val):
    return val is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

# feedforward
class GEGLU(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


class FeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4, glu=False, dropout=0.):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)
        project_in = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU()
        ) if not glu else GEGLU(dim, inner_dim)

        self.net = nn.Sequential(
            project_in,
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim_out)
        )

    def forward(self, x):
        return self.net(x)

class AttLayer(nn.Module):
    def __init__(self, d_model, att_model, nhead):
        super().__init__()
        self.nhead = nhead
        self.dk = (att_model / nhead) ** 0.5

        self.query = nn.Linear(d_model, att_model)
        self.key = nn.Linear(d_model, att_model)
        self.value = nn.Linear(d_model, att_model)
        self.softmax = nn.Softmax(dim=-1)

        self.out_put = nn.Linear(att_model, d_model)

    def forward(self, x, context=None, mask=None):
        Q = self.query(x)
        if context is None:
            K = self.key(x)
            V = self.value(x)
        else:
            K = self.key(context)
            V = self.key(context)

        Q = Q.view(Q.shape[0], Q.shape[1], self.nhead, -1).permute(0, 2, 1, 3)
        K = K.view(K.shape[0], K.shape[1], self.nhead, -1).permute(0, 2, 1, 3)
        V = V.view(V.shape[0], V.shape[1], self.nhead, -1).permute(0, 2, 1, 3)

        attn_weights = torch.matmul(Q, K.transpose(-2, -1)) / self.dk
        # if mask is not None:
        #     attn_weights = attn_weights.masked_fill(mask == 0, float('-inf'))
        attn_weights = self.softmax(attn_weights)
        output = torch.matmul(attn_weights, V)
        output = output.permute(0, 2, 1, 3).contiguous().view(Q.shape[0], -1, Q.shape[1]*Q.shape[3])
        output = self.out_put(output)
        return output


class TransformerBlock(nn.Module):
    def __init__(self, d_model, nhead, att_model, dim_feedforward):
        super(TransformerBlock, self).__init__()
        self.attention = AttLayer(d_model, att_model, nhead)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, d_model)
        )

    def forward(self, x, mask=None):
        attn_output = self.attention(x, mask)
        x = x + attn_output
        x = self.norm1(x)

        ffn_output = self.ffn(x)
        x = x + ffn_output
        x = self.norm2(x)
        return x


class TransformerNet(nn.Module):
    def __init__(self, d_model, att_model, nhead, num_layers, dim_feedforward):
        super(TransformerNet, self).__init__()
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, nhead, att_model, dim_feedforward) for _ in range(num_layers)
        ])

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        return x


class CrossAttBlock(nn.Module):

    def __init__(self, d_model, att_model, dim_feedforward, nhead):
        super().__init__()
        self.attnh = AttLayer(d_model, att_model, nhead)
        self.attn_hc = AttLayer(d_model, att_model, nhead)
        self.attnl = AttLayer(d_model, att_model, nhead)
        self.attn_lc = AttLayer(d_model, att_model, nhead)

        self.normh1 = nn.LayerNorm(d_model)
        self.normh2 = nn.LayerNorm(d_model)

        self.norml1 = nn.LayerNorm(d_model)
        self.norml2 = nn.LayerNorm(d_model)

        self.ffh = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, d_model)
        )
        self.ffl = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, d_model)
        )

    def forward(self, h, l, mask=None):
        """

        :param h:
        :param l:
        :param mask:
        :return:
        """
        at_h = h + self.attnh(h)
        at_l = l + self.attnl(l)

        at_h = at_h + self.attn_hc(self.normh1(h), l)
        at_l = at_l + self.attn_lc(self.norml1(l), h)

        h = self.ffh(self.normh2(at_h)) + at_h
        l = self.ffl(self.norml2(at_l)) + at_l
        return h, l


class CrossAttNet(nn.Module):

    def __init__(self, d_model, att_model, dim_feedforward, nhead, num_cross_layers):
        super().__init__()
        self.layers = nn.ModuleList(
            [CrossAttBlock(d_model, att_model, dim_feedforward, nhead) for _ in range(num_cross_layers)]
        )
        # self.reshape_lin = nn.Linear(d_model//2, d_model)

    def forward(self, h, l, pos_emb, mask=None):
        """

        :param h:
        :param l:
        :param mask:
        :return:
        """
        h = h + pos_emb[:h.size(0)]
        l = l + pos_emb[h.size(0):]
        for layer in self.layers:
            h, l = layer(h, l)
        return h, l

# class CrossAttNet(nn.Module):
#
#     def __init__(self, d_model, att_model, dim_feedforward, nhead, num_cross_layers):
#         super().__init__()
#         self.layers = nn.ModuleList(
#             [CrossAttBlock(d_model, att_model, dim_feedforward, nhead) for _ in range(num_cross_layers)]
#         )
#
#     def forward(self, h, l, mask=None):
#         """
#
#         :param h:
#         :param l:
#         :param mask:
#         :return:
#         """
#         for layer in self.layers:
#             h, l = layer(h, l)
#         return h, l


