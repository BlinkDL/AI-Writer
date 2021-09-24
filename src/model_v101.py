########################################################################################################
# AI人工智障写作 - https://github.com/BlinkDL/AI-Writer
########################################################################################################

import math
import logging
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

logger = logging.getLogger(__name__)

class GPTConfig:
    embd_pdrop = 0.1
    resid_pdrop = 0.1
    attn_pdrop = 0.1

    def __init__(self, vocab_size, block_size, **kwargs):
        self.vocab_size = vocab_size
        self.block_size = block_size
        for k,v in kwargs.items():
            setattr(self, k, v)

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.block_size = config.block_size

        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)

        self.register_buffer("mask", torch.tril(torch.ones(config.block_size, config.block_size))           # causal mask
                                     .view(1, 1, config.block_size, config.block_size))

        self.head_mix = nn.Conv2d(self.n_head, self.n_head, kernel_size=1, bias=False)                      # talking heads

        self.time_shift = nn.ZeroPad2d((0,0,1,0))                                                           # time-mixing
        
        self.time_weighting = nn.Parameter(torch.ones(self.n_head, config.block_size, config.block_size))   # time-weighting

        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        
        self.proj = nn.Linear(config.n_embd, config.n_embd)                                                 # output projection

    def forward(self, x):
        B, T, C = x.size()

        x = torch.cat([self.time_shift(x)[:, :T, :C//2], x[:, :T, C//2:]], dim = -1)    # time-mixing

        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)     # (B, T, C) -> (B, nh, T, hs)
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)       # (B, T, C) -> (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)     # (B, T, C) -> (B, nh, T, hs)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))                 # self-attention: (B, nh, T, hs) * (B, nh, hs, T) -> (B, nh, T, T)

        att = att.masked_fill(self.mask[:,:,:T,:T] == 0, float('-inf'))                 # causal mask

        att = F.softmax(att, dim = -1)                                                  # softmax
        att = att * self.time_weighting[:,:T,:T]                                        # time-weighting

        att = self.head_mix(att)                                                        # talking heads

        att = self.attn_drop(att)
        x = att @ v                                                                     # (B, nh, T, T) * (B, nh, T, hs) -> (B, nh, T, hs)
        x = x.transpose(1, 2).contiguous().view(B, T, C)                                # (B, nh, T, hs) -> (B, T, nh, hs) -> (B, T, C)

        x = self.resid_drop(self.proj(x))                                               # output projection
        return x

class SelfGate(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x = self.proj(x)                                                                # (B, T, C*6)
        x, gate = x.chunk(2, dim = -1)                                                  # (B, T, C*3), (B, T, C*3)
        x = x * F.mish(gate)                                                            # (B, T, C*3)
        return x

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.ln1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)

        self.ln2 = nn.LayerNorm(config.n_embd)

        self.mlp = nn.Sequential(
            SelfGate(config.n_embd, 3 * config.n_embd),
            nn.Linear(3 * config.n_embd, config.n_embd),
            nn.Dropout(config.resid_pdrop),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_emb = nn.Parameter(torch.zeros(1, config.block_size, config.n_embd))
        self.drop = nn.Dropout(config.embd_pdrop)

        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])

        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.block_size = config.block_size
        self.apply(self._init_weights)

        logger.info("number of parameters: %e", sum(p.numel() for p in self.parameters()))

    def get_block_size(self):
        return self.block_size

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, idx, targets=None):
        B, T = idx.size()
        assert T <= self.block_size, "Cannot forward, model block size is exhausted."

        # forward the GPT model
        token_embeddings = self.tok_emb(idx)                        # each index maps to a (learnable) vector
        x = self.drop(token_embeddings) + self.pos_emb[:, :T, :]    # don't drop position embedding

        x = self.blocks(x)
        
        x = self.ln_f(x)
        logits = self.head(x)

        # if we are given some desired targets also calculate the loss
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss
