########################################################################################################
# AI人工智障写作 - https://github.com/BlinkDL/AI-Writer
########################################################################################################

import math
import logging
import torch
import torch.nn as nn
from torch.nn import functional as F
logger = logging.getLogger(__name__)


class RWKV_TimeMix(nn.Module):
    def __init__(self, config, layer_id):
        super().__init__()
        assert config.n_attn % config.n_head == 0
        self.layer_id = layer_id
        self.ctx_len = config.ctx_len
        self.n_head = config.n_head
        self.head_size = config.n_attn // config.n_head

        self.time_ww = nn.Parameter(
            torch.ones(config.n_head, config.ctx_len, config.ctx_len))
        self.time_gamma = nn.Parameter(torch.ones(config.ctx_len, 1))

        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))

        self.key = nn.Linear(config.n_embd, config.n_attn)
        self.value = nn.Linear(config.n_embd, config.n_attn)
        self.receptance = nn.Linear(config.n_embd, config.n_attn)

        self.output = nn.Linear(config.n_attn, config.n_embd)

        self.key.scale_init = 0
        self.receptance.scale_init = 0
        self.output.scale_init = 0

    def forward(self, x):
        B, T, C = x.size()

        x = torch.cat(
            [self.time_shift(x[:, :, :C//2]), x[:, :, C//2:]], dim=-1)

        k = self.key(x)
        v = self.value(x)
        r = self.receptance(x)

        k = torch.clamp(k, max=30, min=-60)
        k = torch.exp(k)
        sum_k = torch.cumsum(k, dim=1)

        kv = (k * v).view(B, T, self.n_head, self.head_size)

        wkv = (torch.einsum('htu,buhc->bthc', self.time_ww[:,:T,:T], kv)
               ).contiguous().view(B, T, -1)

        rwkv = torch.sigmoid(r) * wkv / sum_k

        rwkv = self.output(rwkv)
        return rwkv * self.time_gamma[:T, :]

class RWKV_ChannelMix(nn.Module):
    def __init__(self, config, layer_id):
        super().__init__()
        self.layer_id = layer_id
        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))

        hidden_sz = 5 * config.n_ffn // 2
        self.key = nn.Linear(config.n_embd, hidden_sz)
        self.value = nn.Linear(config.n_embd, hidden_sz)
        self.weight = nn.Linear(hidden_sz, config.n_embd)
        self.receptance = nn.Linear(config.n_embd, config.n_embd)

        self.receptance.scale_init = 0
        self.weight.scale_init = 0

    def forward(self, x):
        B, T, C = x.size()

        x = torch.cat(
            [self.time_shift(x[:, :, :C//2]), x[:, :, C//2:]], dim=-1)
        k = self.key(x)
        v = self.value(x)
        r = self.receptance(x)

        wkv = self.weight(F.mish(k) * v)

        rwkv = torch.sigmoid(r) * wkv

        return rwkv


class GPTConfig:
    def __init__(self, vocab_size, ctx_len, **kwargs):
        self.vocab_size = vocab_size
        self.ctx_len = ctx_len
        for k, v in kwargs.items():
            setattr(self, k, v)


class Block(nn.Module):
    def __init__(self, config, layer_id):
        super().__init__()
        self.config = config

        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)

        self.attn = RWKV_TimeMix(config, layer_id)
        self.mlp = RWKV_ChannelMix(config, layer_id)

    def forward(self, x):

        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))

        return x


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)

        self.blocks = nn.Sequential(*[Block(config, i)
                                    for i in range(config.n_layer)])

        self.ln_f = nn.LayerNorm(config.n_embd)
        self.time_out = nn.Parameter(torch.ones(1, config.ctx_len, 1))
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.head_q = nn.Linear(config.n_embd, 256)
        self.head_k = nn.Linear(config.n_embd, 256)
        self.register_buffer("copy_mask", torch.tril(torch.ones(config.ctx_len, config.ctx_len)))

        self.ctx_len = config.ctx_len

        logger.info("number of parameters: %e", sum(p.numel()
                    for p in self.parameters()))

    def get_ctx_len(self):
        return self.ctx_len

    def forward(self, idx, targets=None):
        B, T = idx.size()
        assert T <= self.ctx_len, "Cannot forward, because len(input) > model ctx_len."

        x = self.tok_emb(idx)

        x = self.blocks(x)

        x = self.ln_f(x)
        q = self.head_q(x)[:,:T,:]
        k = self.head_k(x)[:,:T,:]
        c = (q @ k.transpose(-2, -1)) * (1.0 / 256)
        c = c.masked_fill(self.copy_mask[:T,:T] == 0, 0)
        c = c @ F.one_hot(idx, num_classes = self.config.vocab_size).float()       
        x = x * self.time_out[:, :T, :]
        x = self.head(x) + c

        loss = None
        if targets is not None:
            loss = F.cross_entropy(x.view(-1, x.size(-1)), targets.view(-1))

        return x, loss
