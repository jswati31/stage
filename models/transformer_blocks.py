import torch
import torch.nn as nn
import math
import copy
from .common import Conv1D, MLP, LayerNorm


class SelfAttention(nn.Module):
    def __init__(self, nx, n_ctx, dk, n_head, scale=False):
        super(SelfAttention, self).__init__()
        self.register_buffer("bias", (torch.tril(torch.ones(n_ctx, n_ctx)).view(1, 1, n_ctx, n_ctx)))
        self.n_head = n_head
        self.scale = scale
        self.split_size = dk * n_head
        self.c_attn = Conv1D(dk * n_head * 3, nx)

        self.c_proj = Conv1D(nx, dk * n_head)

        self.dropout = nn.Dropout(p=0.1)

    def _attn(self, q, k, v, mask=None):
        w = torch.matmul(q, k)
        if self.scale:
            w = w / math.sqrt(v.size(-1))
        nd, ns = w.size(-2), w.size(-1)
        b = self.bias[:, :, ns-nd:ns, :ns]
        if mask is not None:
            b = mask & b.bool()
            b = b.float()
        w = w * b - 1e10 * (1 - b)
        w = nn.Softmax(dim=-1)(w)
        w = self.dropout(w)
        return w

    def merge_heads(self, x):
        x = x.permute(0, 2, 1, 3).contiguous()
        new_x_shape = x.size()[:-2] + (x.size(-2) * x.size(-1),)
        return x.view(*new_x_shape)

    def split_heads(self, x, k=False):
        new_x_shape = x.size()[:-1] + (self.n_head, x.size(-1) // self.n_head)
        x = x.view(*new_x_shape)
        if k:
            return x.permute(0, 2, 3, 1)  # (batch, head, head_features, seq_length)
        else:
            return x.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)

    def forward(self, x, layer_past=None, mask=None):

        x_attn = self.c_attn(x)
        query_x, key_x, value_x = x_attn.split(self.split_size, dim=2)
        value_x = self.split_heads(value_x)
        key_x = self.split_heads(key_x, k=True)
        query_x = self.split_heads(query_x)

        if layer_past is not None:
            past_key, past_value = layer_past[0].transpose(-2, -1), layer_past[1]  # transpose back cf below
            key_x = torch.cat((past_key, key_x), dim=-1)
            value_x = torch.cat((past_value, value_x), dim=-2)

        present = torch.stack((key_x.transpose(-2, -1), value_x))  # transpose to have same shapes for stacking

        attn = self._attn(query_x, key_x, value_x, mask)
        ###
        ###
        a = torch.matmul(attn, value_x)
        a = self.merge_heads(a)
        a = self.c_proj(a)
        return a, present

class SelfAttentionBlock(nn.Module):
    def __init__(self, nx, n_ctx, dk, n_head, scale=False):
        super(SelfAttentionBlock, self).__init__()
        self.ln_1 = LayerNorm(nx)
        self.attn = SelfAttention(nx, n_ctx, dk, n_head, scale)
        self.ln_2 = LayerNorm(nx)
        self.mlp = MLP(4 * nx, nx)

    def forward(self, x, layer_past=None, mask=None):
        a, present = self.attn(self.ln_1(x), layer_past=layer_past, mask=mask)
        x = x + a
        m = self.mlp(self.ln_2(x))
        x = x + m
        return x, present


class GPT2Model(nn.Module):
    def __init__(self, num_frames, embedding_dim, n_layers, dk, n_head, scale_emb=True, args=None):
        super(GPT2Model, self).__init__()
        self.n_layer = n_layers
        self.n_embd = embedding_dim
        self.num_frames = num_frames

        self.args = args

        self.pos_embedding = nn.Embedding(self.num_frames, embedding_dim)

        block = SelfAttentionBlock(embedding_dim, self.num_frames, dk, n_head, scale=scale_emb)
        self.h = nn.ModuleList([copy.deepcopy(block) for _ in range(n_layers)])
        self.ln_f = LayerNorm(embedding_dim)

    def forward(self, input_embeddings, past=None):

        if past is None:
            past_length = 0
            past = [None] * len(self.h)
        else:
            past_length = past[0][0].size(-2)

        b, n, _ = input_embeddings.shape

        pos = torch.arange(0, self.num_frames, dtype=torch.long, device=input_embeddings.device).unsqueeze(0) # shape (1, n)
        pos_emb = self.pos_embedding(pos)

        hidden_states = input_embeddings + pos_emb

        presents = []
        for block, layer_past in zip(self.h, past):
            hidden_states, present = block(hidden_states, layer_past)
            presents.append(present)
        hidden_states = self.ln_f(hidden_states)

        return hidden_states, presents


