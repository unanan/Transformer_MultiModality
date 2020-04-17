from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F

import copy
import math
import numpy as np
import collections

from model.srcembed.resnet import ResNet50_MB, ResNet50_WD, ResNet18_BN
# from drn import drn_d_22
# from unet import UNet_Nested,DenseUNet_Nested


def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
    # print("attention: ",mask)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
        # print(scores)
    p_attn = F.softmax(scores, dim=-1)
    
    if dropout is not None:
        p_attn = dropout(p_attn)
    
    return torch.matmul(p_attn, value), p_attn


class Generator(nn.Module):
    "Define standard linear + softmax generation step."
    
    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)
    
    def forward(self, x):
        # return F.log_softmax(self.proj(x), dim=-1)
        out = F.softmax(self.proj(x), dim=-1)
        return out
        # return  #TODO


class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."
    
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps
    
    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
    
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"
    
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = nn.ModuleList(
            [SublayerConnection(size, dropout) for _ in range(2)])  # clones(SublayerConnection(size, dropout), 2)
        self.size = size
    
    def forward(self, x, mask):
        "Follow Figure 1 (left) for connections."
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


class DecoderLayer(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"
    
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = nn.ModuleList(
            [SublayerConnection(size, dropout) for _ in range(3)])  # clones(SublayerConnection(size, dropout), 3)
    
    def forward(self, x, memory, src_mask, tgt_mask):
        "Follow Figure 1 (right) for connections."
        m = memory
        
        # print("DecoderLayer:",tgt_mask.shape)
        
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = nn.ModuleList(
            [nn.Linear(d_model, d_model) for _ in range(4)])  # clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
    
    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        # print("MultiHeadedAttention:", mask)
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        
        nbatches = query.size(0)
        
        # 1) Do all the linear projections in batch from d_model => h x d_k
        # for l, x in zip(self.linears, (query, key, value)):
        #     print(l(x).shape)
        
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]
        
        # 2) Apply attention on all the projected vectors in batch.
        # print("before2 attention calculate",query.shape, key.shape, value.shape)
        x, self.attn = attention(query, key, value, mask=mask,
                                 dropout=self.dropout)
        
        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.h * self.d_k)
        
        return self.linears[-1](x)


class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."
    
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model
    
    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


class PositionalEncoding2d(nn.Module):
    
    def __init__(self, d_model, dropout, height, width):
        """
        :param d_model: dimension of the model
        :param height: height of the positions
        :param width: width of the positions
        :return: d_model*height*width position matrix
        """
        super(PositionalEncoding2d, self).__init__()
        # self.d_model = d_model
        if d_model % 4 != 0:
            raise ValueError("Cannot use sin/cos positional encoding with "
                             "odd dimension (got dim={:d})".format(d_model))
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(d_model, height, width)
        
        # Each dimension use half of d_model
        d_model = int(d_model / 2)
        div_term = torch.exp(torch.arange(0., d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pos_w = torch.arange(0., width).unsqueeze(1)
        pos_h = torch.arange(0., height).unsqueeze(1)
        pe[0:d_model:2, :, :] = torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
        pe[1:d_model:2, :, :] = torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
        pe[d_model::2, :, :] = torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
        pe[d_model + 1::2, :, :] = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        # print(x.shape, self.pe.shape)
        x = x + self.pe[:, :, :x.size(2), :x.size(3)]
        x = x.view(x.size(0), x.size(1), -1)
        x = x.permute([0, 2, 1]).contiguous()
        return self.dropout(x)


class PositionalEncoding(nn.Module):
    "Implement the PE function."
    
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        # print(x.shape, self.pe.shape)
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


# =======================================================================================================================
# ======================================================= Wrappers ======================================================
# =======================================================================================================================
class Encoder(nn.Module):
    "Core encoder is a stack of N layers"
    
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList([layer for _ in range(N)])  # clones(layer, N)
        self.norm = LayerNorm(layer.size)
    
    def forward(self, x, mask):
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class Decoder(nn.Module):
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList([layer for _ in range(N)])  # clones(layer, N)
        self.norm = LayerNorm(layer.size)
    
    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        
        return self.norm(x)


class Embedder(nn.Module):
    def __init__(self, layers=None, debug=False):
        super(Embedder, self).__init__()
        self.debug = debug
        
        if layers == None:  # default settings
            self.layers = [ResNet50_WD(pretrained=False), self.position2d]
        else:
            assert isinstance(layers, collections.Iterable), f"type of layers is invalid: {type(layers)}"
            self.layers = list(layers)
        
        if not self.debug:
            self.layers = nn.Sequential(*layers)
    
    def forward(self, x):
        if self.debug:
            for layer in self.layers:
                x = layer(x)
                # print debug info
        else:
            x = self.layers(x)
        return x


# =========================================================================================
# =============================== United Transformer Classes ==============================
# =========================================================================================
class TransformerCNN(nn.Module):
    def __init__(self, trg_vocab, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1):
        
        super(TransformerCNN, self).__init__()
        self.d_model = d_model
        self.attn_e1 = MultiHeadedAttention(h, d_model)
        self.attn_d1 = MultiHeadedAttention(h, d_model)
        self.attn_d2 = MultiHeadedAttention(h, d_model)
        
        self.ff_e1 = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.ff_d1 = PositionwiseFeedForward(d_model, d_ff, dropout)
        
        self.position1d = PositionalEncoding(d_model, dropout)
        self.position2d = PositionalEncoding2d(d_model=d_model, dropout=dropout, height=192, width=192)
        
        self.encoder = Encoder(EncoderLayer(d_model, self.attn_e1, self.ff_e1, dropout), N)
        self.decoder = Decoder(DecoderLayer(d_model, self.attn_d1, self.attn_d2, self.ff_d1, dropout), N)
        self.src_embed = Embedder([ResNet50_WD(pretrained=False), self.position2d])
        self.trg_embed = Embedder([Embeddings(d_model, trg_vocab), self.position1d])
        self.generator = Generator(d_model, trg_vocab)
        
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, src, trg, trg_mask, src_mask=None):
        return self.decode(self.encode(src, src_mask), src_mask,
                           trg, trg_mask)
    
    def encode(self, src, src_mask):
        srcembed = self.src_embed(src)
        assert len(srcembed.shape) == 3, "output of src_embed must be dim-3"
        
        return self.encoder(srcembed, src_mask)
    
    def decode(self, memory, src_mask, trg, trg_mask):
        return self.decoder(self.trg_embed(trg), memory, src_mask, trg_mask)


if __name__ == '__main__':
    pass
    # For Test
    # transformer = TransformerModelWithCNN()
    # model = transformer.make_model(10, 10, 2)
    # print(model)