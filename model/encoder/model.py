import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
from sequence_models.layers import PositionFeedForward, DoubleEmbedding
from sequence_models.convolutional import ByteNetBlock, MaskedConv1d
from .cross_attention import CrossAttNet, TransformerNet


class MLP(nn.Module):

    def __init__(self, n_embd):
        super().__init__()
        self.ln1 = nn.Linear(n_embd, 2 * n_embd)
        self.gelu = nn.GELU()
        self.ln2 = nn.Linear(2 * n_embd, n_embd)
        self.dropout = nn.Dropout()

    def forward(self, x):
        x = self.ln1(x)
        x = self.gelu(x)
        x = self.ln2(x)
        x = self.dropout(x)
        return x

class PositionalEncoding1D(nn.Module):
    def __init__(self, d_model=8, length=500):
        super().__init__()
        self.d_model = d_model
        self.length = length

    def forward(self, x):
        """
        Used for encoding timestep in diffusion models

        :param d_model: dimension of the model
        :param length: length of positions
        :return: length*d_model position matrix
        """
        if self.d_model % 2 != 0:
            raise ValueError("Cannot use sin/cos positional encoding with "
                             "odd dim (got dim={:d})".format(self.d_model))
        pe = torch.zeros(self.length, self.d_model)
        position = torch.arange(0, self.length).unsqueeze(1)
        div_term = torch.exp((torch.arange(0, self.d_model, 2, dtype=torch.float) * -(np.log(10000.0) / self.d_model)))
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)
        device = x.device
        pe = pe.to(device)
        return pe[x] # .to(x.device)

class PositionalEncoding(nn.Module):

    """
    2D Positional encoding for transformer
    :param d_model: dimension of the model
    :param max_len: max number of positions
    """

    def __init__(self, d_model, max_len=152):
        super().__init__()

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        y = self.pe[:x.size(1)]
        x = x + y.reshape(y.shape[1], y.shape[0], y.shape[2])
        return x

class ByteNetTime(nn.Module):
    """Stacked residual blocks from ByteNet paper defined by n_layers

         Shape:
            Input: (N, L,)
            input_mask: (N, L, 1), optional
            Output: (N, L, d)
    """

    def __init__(self, n_tokens, d_embedding, d_model, n_layers, kernel_size, r, rank=None, n_frozen_embs=None,
                 padding_idx=None, causal=False, dropout=0.0, slim=True, activation='relu', down_embed=True,
                 timesteps=None):
        """
        :param n_tokens: number of tokens in token dictionary
        :param d_embedding: dimension of embedding
        :param d_model: dimension to use within ByteNet model, //2 every layer
        :param n_layers: number of layers of ByteNet block
        :param kernel_size: the kernel width
        :param r: used to calculate dilation factor
        :padding_idx: location of padding token in ordered alphabet
        :param causal: if True, chooses MaskedCausalConv1d() over MaskedConv1d()
        :param rank: rank of compressed weight matrices
        :param n_frozen_embs: number of frozen embeddings
        :param slim: if True, use half as many dimensions in the NLP as in the CNN
        :param activation: 'relu' or 'gelu'
        :param down_embed: if True, have lower dimension for initial embedding than in CNN layers
        :param timesteps: None or int providing max timesteps in DM model
        """
        super().__init__()
        self.timesteps = timesteps
        self.time_encoding = PositionalEncoding1D(d_embedding, timesteps) # Timestep encoding
        if n_tokens is not None:
            if n_frozen_embs is None:
                self.embedder = nn.Embedding(n_tokens, d_embedding, padding_idx=padding_idx)
            else:
                self.embedder = DoubleEmbedding(n_tokens - n_frozen_embs, n_frozen_embs,
                                                d_embedding, padding_idx=padding_idx)
        else:
            self.embedder = nn.Identity()
        if down_embed:
            self.up_embedder = PositionFeedForward(d_embedding, d_model)
        else:
            self.up_embedder = nn.Identity()
            assert n_tokens == d_embedding
        log2 = int(np.log2(r)) + 1
        dilations = [2 ** (n % log2) for n in range(n_layers)]
        d_h = d_model
        if slim:
            d_h = d_h // 2
        layers = [
            ByteNetBlock(d_model, d_h, d_model, kernel_size, dilation=d, causal=causal, rank=rank,
                         activation=activation)
            for d in dilations
        ]
        self.layers = nn.ModuleList(modules=layers)
        self.dropout = dropout

    def forward(self, x, input_mask=None):
        """
        :param x: (batch, length)
        :param y: (batch)
        :param input_mask: (batch, length, 1)
        :return: (batch, length,)
        """
        e = self._embed(x, timesteps=self.timesteps)
        return self._convolve(e, input_mask=input_mask)

    def _embed(self, x, timesteps=None):
        e = self.embedder(x)
        # if timesteps is not None:
        #     e2 = self.time_encoding(y)
        #     # expand dim of e2 to match e1
        #     e2 = e2.expand(e.shape[1], e2.shape[0], e2.shape[1])
        #     e2 = e2.reshape(e.shape[0], e.shape[1], e.shape[2])
        #     e = torch.add(e2, e)
        e = self.up_embedder(e)
        return e

    def _convolve(self, e, input_mask=None):
        for layer in self.layers:
            e = layer(e, input_mask=input_mask)
            if self.dropout > 0.0:
                e = F.dropout(e, self.dropout)
        return e


class ByteNetLMTime(nn.Module):

    def __init__(self, n_tokens, d_embedding, d_model, n_layers, kernel_size, r, nhead, rank=None, n_frozen_embs=None,
                 padding_idx=None, causal=False, dropout=0.0, final_ln=False, slim=True, activation='relu',
                 tie_weights=False, down_embed=True, timesteps=None):
        super().__init__()
        self.H_embedder = ByteNetTime(n_tokens, d_embedding, d_model, n_layers, kernel_size, r,
                                padding_idx=padding_idx, causal=causal, dropout=dropout, down_embed=down_embed,
                                slim=slim, activation=activation, rank=rank, n_frozen_embs=n_frozen_embs,
                                timesteps=timesteps)
        self.L_embedder = ByteNetTime(n_tokens, d_embedding, d_model, n_layers, kernel_size, r,
                                padding_idx=padding_idx, causal=causal, dropout=dropout, down_embed=down_embed,
                                slim=slim, activation=activation, rank=rank, n_frozen_embs=n_frozen_embs,
                                timesteps=timesteps)
        # self.H_att = SelfAttention(d_model, nhead=nhead)
        # self.L_att = SelfAttention(d_model, nhead=nhead)
        # self.L_att = CrossAttention(d_model, context_dim=d_model, inner_dim=d_model//2)
        if tie_weights:
            self.H_decoder = nn.Linear(d_model, n_tokens, bias=False)
            # self.H_decoder.weight = self.H_embedder.embedder.weight
            self.L_decoder = nn.Linear(d_model, n_tokens, bias=False)
            # self.L_decoder.weight = self.L_embedder.embedder.weight
        else:
            self.H_decoder = PositionFeedForward(d_model, n_tokens)
            self.L_decoder = PositionFeedForward(d_model, n_tokens)
        if final_ln:
            self.H_last_norm = nn.LayerNorm(d_model)
            self.L_last_norm = nn.LayerNorm(d_model)
        else:
            self.H_last_norm = nn.Identity()
            self.L_last_norm = nn.Identity()

    def forward(self, H_x, H_y, L_x, L_y, H_input_mask=None, L_input_mask=None):
        H_e = self.H_embedder(H_x, H_y, input_mask=H_input_mask.unsqueeze(-1))
        H_e = self.H_last_norm(H_e)
        L_e = self.L_embedder(L_x, L_y, input_mask=L_input_mask.unsqueeze(-1))
        L_e = self.H_last_norm(L_e)

        # Cross attention
        H_e = self.H_att(H_e, L_e, H_input_mask.unsqueeze(-1), L_input_mask.unsqueeze(-1))
        L_e = self.L_att(L_e, H_e, L_input_mask.unsqueeze(-1), H_input_mask.unsqueeze(-1))

        return self.H_decoder(H_e), self.L_decoder(L_e)


class SideEmbedder(nn.Module):

    def __init__(self, n_side, s_embedding, d_side):
        super().__init__()
        self.side_embeddinng = nn.Embedding(n_side, s_embedding)
        self.side_mlp = nn.Sequential(
            nn.Linear(s_embedding, d_side),
            nn.LayerNorm(d_side),
            nn.ReLU(),
            nn.Linear(d_side, d_side),
        )

    def forward(self, side, a_seq_length, mask=None):
        emb_side = self.side_embeddinng(side.view(-1, 1))
        emb_side = self.side_mlp(emb_side).repeat(1, a_seq_length, 1)
        return emb_side


class RegionEmbedder(nn.Module):

    def __init__(self, r_pos, r_embedding, r_model, rank=None):
        super().__init__()
        self.region_embedding = nn.Embedding(r_pos, r_embedding)
        self.region_layer1 = nn.Sequential(
            nn.LayerNorm(r_embedding),
            nn.ReLU(),
            PositionFeedForward(r_embedding, r_model, rank=rank),
            nn.LayerNorm(r_model),
            nn.ReLU()
        )
        # padding_idx = (kernel_size - 1) // 2
        # self.conv1 = nn.Conv1d(r_model, r_model, kernel_size=kernel_size, padding=padding_idx)
        # self.conv2 = MaskedConv1d(r_model, r_model, kernel_size=kernel_size)

    def forward(self, pos_seq, mask=None):
        """
        :param pos_seq:
        :param mask:
        :return:
        """
        x = self.region_embedding(pos_seq)
        x = self.region_layer1(x)
        # x = x + self.conv2(self.conv1(x.transpose(2, 1)).transpose(1, 2))
        return x


class PosEmbedder(nn.Module):

    def __init__(self, p_emb, max_len):
        super().__init__()
        self.pos_embedding = PositionalEncoding(p_emb, max_len)
        self.pos_lin = MLP(n_embd=p_emb)

    def forward(self, H_L_region_emb):
        x = self.pos_embedding(H_L_region_emb)
        pos_emb = self.pos_lin(x)
        x = x + pos_emb
        return x

# class PosEmbedder(nn.Module):
#
#     def __init__(self, n_pos, p_embedding, p_model, kernel_size, rank=None):
#         super().__init__()
#         self.pos_embedding = nn.Embedding(n_pos, p_embedding)
#         self.pos_layer1 = nn.Sequential(
#             nn.LayerNorm(p_embedding),
#             nn.ReLU(),
#             PositionFeedForward(p_embedding, p_model, rank=rank),
#             nn.LayerNorm(p_model),
#             nn.ReLU()
#         )
#         padding_idx = (kernel_size - 1) // 2
#         self.conv1 = nn.Conv1d(p_model, p_model, kernel_size=kernel_size, padding=padding_idx)
#         self.conv2 = MaskedConv1d(p_model, p_model, kernel_size=kernel_size)
#
#     def forward(self, pos_seq, mask=None):
#         """
#         :param pos_seq:
#         :param mask:
#         :return:
#         """
#         x = self.pos_embedding(pos_seq)
#         x = self.pos_layer1(x)
#         x = x + self.conv2(self.conv1(x.transpose(2, 1)).transpose(1, 2))
#         return x

class DualConv(nn.Module):

    def __init__(self, d_model, n_layers, kernel_size, r, rank=None,
                causal=False, dropout=0.0, slim=True, activation='relu', timesteps=None):
        super().__init__()

        log2 = int(np.log2(r)) + 1
        dilations = [2 ** (n % log2) for n in range(n_layers)]
        d_h = d_model
        if slim:
            d_h = d_h // 2
        h_layers = [
            ByteNetBlock(d_model, d_h, d_model, kernel_size, dilation=d, causal=causal, rank=rank,
                         activation=activation)
            for d in dilations
        ]
        l_layers = [
            ByteNetBlock(d_model, d_h, d_model, kernel_size, dilation=d, causal=causal, rank=rank,
                         activation=activation)
            for d in dilations
        ]
        self.h_layers = nn.ModuleList(modules=h_layers)
        self.l_layers = nn.ModuleList(modules=l_layers)

    def forward(self, s, batch, mask=None):
        """

        :param s:
        :param batch:
        :param mask:
        :return:
        """
        h_s = s[batch==0]
        l_s = s[batch!=0]

        h_s = self._hconv(h_s)
        l_s = self._lconv(l_s)
        return h_s, l_s

    def _hconv(self, h_s):
        for layer in self.h_layers:
            h_s = layer(h_s)
        return h_s

    def _lconv(self, l_s):
        for layer in self.l_layers:
            l_s = layer(l_s)
        return l_s


class TransformerEncoder(nn.Module):

    def __init__(self, n_tokens, d_embedding, d_model, att_model, nhead, num_layers, dim_feedforward):
        super().__init__()
        self.embed = nn.Embedding(n_tokens, d_embedding)
        self.up_embedder = PositionFeedForward(d_embedding, d_model)
        self.att_net = TransformerNet(d_model, att_model, nhead, num_layers=num_layers, dim_feedforward=dim_feedforward)

    def forward(self, x):
        emb_x = self.embed(x)
        up_emb_x = self.up_embedder(emb_x)
        x = self.att_net(up_emb_x)
        return x


class AntiTFNet(nn.Module):

    def __init__(self, n_tokens, d_embedding, d_model, n_encoder_layers, aa_kernel_size, r,
                 n_side, s_embedding, s_model,
                 n_region, r_embedding, r_model,
                 n_pos_model, max_len,
                 sum_d_model, dual_layers,
                 att_model, dim_feedforward, nhead, cs_layers,
                 rank=None, n_frozen_embs=None,
                 padding_idx=None, causal=False, dropout=0.0, slim=True, activation='relu',
                 down_embed=True, timesteps=None):
        super().__init__()

        self.aa_encoder = ByteNetTime(n_tokens, d_embedding, d_model, n_encoder_layers, aa_kernel_size, r,
                                padding_idx=padding_idx, causal=causal, dropout=dropout, down_embed=down_embed,
                                slim=slim, activation=activation, rank=rank, n_frozen_embs=n_frozen_embs,
                                timesteps=timesteps)
        # self.aa_encoder = TransformerEncoder(n_tokens, d_embedding, d_model, att_model,
        #                                      nhead, n_encoder_layers, d_model//2)
        self.side_encoder = SideEmbedder(n_side, s_embedding, s_model)
        self.region_encoder = RegionEmbedder(n_region, r_embedding, r_model)
        self.pos_encoder = PosEmbedder(n_pos_model, max_len)
        self.dual_conv_block = DualConv(sum_d_model, dual_layers, aa_kernel_size, r)
        self.cross_at = CrossAttNet(sum_d_model, att_model, dim_feedforward, nhead, num_cross_layers=cs_layers)
        self.last_norm = nn.LayerNorm(sum_d_model)
        self.pos_reshape_lin = nn.Linear(n_pos_model, sum_d_model)
        self.decoder = nn.Linear(sum_d_model, n_tokens)

    def _encoder(self, h_l_aa_seq, h_l_chn_type, h_l_region_type):
        H_L_emb = self.aa_encoder(h_l_aa_seq)
        aa_seq_length = H_L_emb.size(1)
        H_L_chn_emb = self.side_encoder(h_l_chn_type, aa_seq_length)
        H_L_region_emb = self.region_encoder(h_l_region_type)
        H_L_pos_emb = self.pos_encoder(H_L_region_emb)
        H_L_emb = H_L_emb + H_L_pos_emb
        h_l_feature = torch.cat((H_L_emb, H_L_chn_emb), dim=-1)
        H_L_pos_emb = self.pos_reshape_lin(H_L_pos_emb)
        return h_l_feature, H_L_pos_emb

    def _att(self, h, l, pos_emb):
        h, l = self.cross_at(h, l, pos_emb)
        h_l = torch.cat((h, l), dim=0)
        return h_l

    def forward(self, H_L_seq, H_L_region_type, H_L_chn_type, H_L_batch, type):
        """

        :param H_L_seq: (Batch, length);
        :param H_L_pos_type: (Batch, length); distinguish the different region of Chain.
        :param H_L_chn_type: (Batch); gene ?
        :param H_L_batch: (Batch); distinguish the type of Chain.
        :param H_L_mask: None
        :return: (Batch, length, feature)
        """
        h_l_feature, H_L_pos_emb = self._encoder(
            h_l_aa_seq=H_L_seq.int(),
            h_l_chn_type=H_L_chn_type,
            h_l_region_type=H_L_region_type.int(),
        )
        h, l = self.dual_conv_block(h_l_feature, H_L_batch)
        if type == 'pair':
            h_l = self._att(h=h, l=l, pos_emb=H_L_pos_emb)
        else:
            h_l = torch.cat((h, l), dim=0)
            h_l = h_l + H_L_pos_emb
        h_l = self.decoder(self.last_norm(h_l))
        return h_l

# class AntiTFNet(nn.Module):
#
#     def __init__(self, n_tokens, d_embedding, d_model, n_encoder_layers, aa_kernel_size, r,
#                  n_side, s_embedding, s_model,
#                  n_pos, p_embedding, p_model, p_kernel_size,
#                  sum_d_model, dual_layers,
#                  att_model, dim_feedforward, nhead, cs_layers,
#                  rank=None, n_frozen_embs=None,
#                  padding_idx=None, causal=False, dropout=0.0, slim=True, activation='relu',
#                  down_embed=True, timesteps=None):
#         super().__init__()
#
#         self.aa_encoder = ByteNetTime(n_tokens, d_embedding, d_model, n_encoder_layers, aa_kernel_size, r,
#                                 padding_idx=padding_idx, causal=causal, dropout=dropout, down_embed=down_embed,
#                                 slim=slim, activation=activation, rank=rank, n_frozen_embs=n_frozen_embs,
#                                 timesteps=timesteps)
#         # self.aa_encoder = TransformerEncoder(n_tokens, d_embedding, d_model, att_model,
#         #                                      nhead, n_encoder_layers, d_model//2)
#         self.side_encoder = SideEmbedder(n_side, s_embedding, s_model)
#         self.pos_encoder = PosEmbedder(n_pos, p_embedding, p_model, p_kernel_size)
#         self.dual_conv_block = DualConv(sum_d_model, dual_layers, aa_kernel_size, r)
#         self.cross_at = CrossAttNet(sum_d_model, att_model, dim_feedforward, nhead, num_cross_layers=cs_layers)
#         self.last_norm = nn.LayerNorm(sum_d_model)
#         self.decoder = nn.Linear(sum_d_model, n_tokens)
#
#     def _encoder(self, h_l_aa_seq, h_l_chn_type, h_l_pos_type):
#         H_L_emb = self.aa_encoder(h_l_aa_seq)
#         aa_seq_length = H_L_emb.size(1)
#         H_L_chn_emb = self.side_encoder(h_l_chn_type, aa_seq_length)
#         H_L_pos_emb = self.pos_encoder(h_l_pos_type)
#         h_l_feature = torch.cat((H_L_emb, H_L_chn_emb, H_L_pos_emb), dim=-1)
#         return h_l_feature
#
#     def _att(self, h, l):
#         h, l = self.cross_at(h, l)
#         h_l = torch.cat((h, l), dim=0)
#         return h_l
#
#     def forward(self, H_L_seq, H_L_pos_type, H_L_chn_type, H_L_batch, type):
#         """
#
#         :param H_L_seq: (Batch, length);
#         :param H_L_pos_type: (Batch, length); distinguish the different region of Chain.
#         :param H_L_chn_type: (Batch); gene ?
#         :param H_L_batch: (Batch); distinguish the type of Chain.
#         :param H_L_mask: None
#         :return: (Batch, length, feature)
#         """
#         h_l_feature = self._encoder(
#             h_l_aa_seq=H_L_seq,
#             h_l_chn_type=H_L_chn_type,
#             h_l_pos_type=H_L_pos_type
#         )
#         h, l = self.dual_conv_block(h_l_feature, H_L_batch)
#         if type == 'pair':
#             h_l = self._att(h=h, l=l)
#         else:
#             h_l = torch.cat((h, l), dim=0)
#         h_l = self.decoder(self.last_norm(h_l))
#         return h_l