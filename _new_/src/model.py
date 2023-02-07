import numpy as np
import types
import os
import torch
import array
from torch.nn import functional as F
import torch.nn as nn

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True

def to_float(x):
    return x.cpu().detach().squeeze().numpy()

################################################################################

class RWKV_CFG():
    def __init__(self, type, MODEL_NAME):
        self.b_eps = 1e-9
        self.k_max = 60
        self.RUN_DEVICE = 'cpu'
        self.HEAD_QK = False
        self.FFN_PRE = False
        self.HEAD_QK_DIM = 256
        self.MODEL_NAME = MODEL_NAME

        if type == 'L6_CHN':
            self.ctx_len = 256*6
            self.n_layer = 6
            self.n_embd = 512
            self.b_eps = 1e-16
            self.HEAD_QK = True
            self.FFN_PRE = True
        if type == 'L24_CHN':
            self.ctx_len = 1024
            self.n_layer = 24
            self.n_embd = 1024
            self.HEAD_QK = True
            self.FFN_PRE = True
        if type == 'L12_ENG':
            self.ctx_len = 768
            self.n_layer = 12
            self.n_embd = 768
            self.HEAD_QK = False
            self.FFN_PRE = False

class RWKV_ChannelMix(nn.Module):
    def __init__(self, rnn, name):
        super().__init__()
        self.rnn = rnn
        cfg = rnn.cfg
        self.cfg = cfg
        self.name = name

        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
        self.time_mix_k = nn.Parameter(torch.ones(1, 1, cfg.n_embd))
        self.time_mix_r = nn.Parameter(torch.ones(1, 1, cfg.n_embd))

        hidden_sz = 4 * cfg.n_embd
        self.key = nn.Linear(cfg.n_embd, hidden_sz, bias=False)
        self.receptance = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
        self.value = nn.Linear(hidden_sz, cfg.n_embd, bias=False)

    def forward(self, x):
        self.rnn.write(self.name + 'xx', x[0][-1])

        xx = self.time_shift(x)
        xk = x * self.time_mix_k + xx * (1 - self.time_mix_k)
        xr = x * self.time_mix_r + xx * (1 - self.time_mix_r)
        k = self.key(xk)
        k = torch.square(torch.relu(k))
        kv = self.value(k)

        rkv = torch.sigmoid(self.receptance(xr)) * kv
        return rkv


class RWKV_TimeMix(nn.Module):
    def __init__(self, rnn, name):
        super().__init__()
        self.rnn = rnn
        cfg = rnn.cfg
        self.cfg = cfg
        self.name = name

        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
        self.time_mix_k = nn.Parameter(torch.ones(1,1,cfg.n_embd))
        self.time_mix_v = nn.Parameter(torch.ones(1,1,cfg.n_embd))
        self.time_mix_r = nn.Parameter(torch.ones(1,1,cfg.n_embd))

        self.time_w = nn.Parameter(torch.zeros(cfg.n_embd, 1, cfg.ctx_len))
        self.time_ww = nn.Parameter(torch.zeros(cfg.n_embd, 1, cfg.ctx_len))

        self.key = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
        self.value = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
        self.receptance = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)

        self.output = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)

    def forward(self, x):
        B, T, C = x.size()

        self.rnn.write(self.name + 'xx', x[0][-1])
        xx = self.time_shift(x)
        xk = x * self.time_mix_k + xx * (1 - self.time_mix_k)
        xv = x * self.time_mix_v + xx * (1 - self.time_mix_v)
        xr = x * self.time_mix_r + xx * (1 - self.time_mix_r)

        k = self.key(xk).transpose(-1, -2)
        v = self.value(xv).transpose(-1, -2)
        r = self.receptance(xr)

        k = torch.clamp(k, max=self.cfg.k_max)
        k = torch.exp(k)

        kv = k * v

        w = self.time_w[:, :, -T:]
        wkv = F.conv1d(nn.ZeroPad2d((T-1, 0, 0, 0))(kv), w, groups=C)
        wk = F.conv1d(nn.ZeroPad2d((T-1, 0, 0, 0))(k),
                      w, groups=C) + self.cfg.b_eps

        ww = self.time_ww[:, :, -T:]
        self.rnn.write(self.name + 'aa', F.conv1d(
            nn.ZeroPad2d((T-1, 0, 0, 0))(kv), ww, groups=C)[0, :, -1])
        self.rnn.write(self.name + 'bb', (F.conv1d(nn.ZeroPad2d(
            (T-1, 0, 0, 0))(k), ww, groups=C) + self.cfg.b_eps)[0, :, -1])

        rwkv = torch.sigmoid(r) * (wkv / wk).transpose(-1, -2)

        rwkv = self.output(rwkv)
        return rwkv


class Block(nn.Module):
    def __init__(self, rnn, layer_id):
        super().__init__()
        cfg = rnn.cfg
        self.cfg = cfg
        self.layer_id = layer_id

        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.ln2 = nn.LayerNorm(cfg.n_embd)

        if self.layer_id == 0 and cfg.FFN_PRE:
            self.ffnPre = RWKV_ChannelMix(rnn, f'ffnPre.{layer_id}')
        else:
            self.att = RWKV_TimeMix(rnn, f'att.{layer_id}')
        self.ffn = RWKV_ChannelMix(rnn, f'ffn.{layer_id}')

    def forward(self, x):
        x = self.ln1(x)
        if self.layer_id == 0 and self.cfg.FFN_PRE:
            x = x + self.ffnPre(x)
        else:
            x = x + self.att(x)
        x = self.ln2(x)
        x = x + self.ffn(x)
        return x


class RWKV_GPT(nn.Module):
    def __init__(self, rnn):
        super().__init__()
        cfg = rnn.cfg
        self.rnn = rnn
        self.cfg = cfg
        self.emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)

        self.blocks = nn.Sequential(*[Block(rnn, i)
                                      for i in range(cfg.n_layer)])

        self.ln_out = nn.LayerNorm(cfg.n_embd)
        self.head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)

        if self.cfg.HEAD_QK:
            self.head_q = nn.Linear(cfg.n_embd, cfg.HEAD_QK_DIM, bias=False)
            self.head_k = nn.Linear(cfg.n_embd, cfg.HEAD_QK_DIM, bias=False)
            self.register_buffer("copy_mask", torch.tril(
                torch.ones(cfg.ctx_len, cfg.ctx_len)))

        self.ctx_len = cfg.ctx_len

    def forward(self, idx, shall_save=False):
        with torch.no_grad():
            B, T = idx.size()
            # print(f'[GPT {T}]', end='')
            assert T <= self.ctx_len, "Cannot forward, because len(input) > model ctx_len."

            x = self.emb(idx)
            x = self.blocks(x)
            x = self.ln_out(x)

            if self.cfg.HEAD_QK:
                q = self.head_q(x)[:, :T, :]
                k = self.head_k(x)[:, :T, :]
                self.rnn.hk = k[0]
                c = (q @ k.transpose(-2, -1)) * (1.0 / self.cfg.HEAD_QK_DIM)
                c = c.masked_fill(self.copy_mask[:T, :T] == 0, 0)
                c = c @ F.one_hot(idx,
                                  num_classes=self.cfg.vocab_size).float()
                x = self.head(x) + c
            else:
                x = self.head(x)

            x = x[0][-1].detach().cpu().flatten().tolist()

            self.rnn.out = x
            ctx = idx[0].detach().cpu().tolist()
            self.rnn.code_last = self.rnn.ctx_to_key(ctx)

            if shall_save:
                self.rnn.save_stat()

            return x.copy()

################################################################################


class RWKV_RNN():
    def __init__(self, cfg):
        self.cfg = cfg
        print('\nloading RWKV-RNN', self.cfg.MODEL_NAME)

        self.w = types.SimpleNamespace()

        w = torch.load(self.cfg.MODEL_NAME + '.pth',
                       map_location=torch.device(self.cfg.RUN_DEVICE))

        time_curve = torch.tensor([-(cfg.ctx_len - 2 - i)
                                  for i in range(cfg.ctx_len-1)]).unsqueeze(0).to(self.cfg.RUN_DEVICE)
        time_curve_ww = torch.tensor([-(cfg.ctx_len - 1 - i)
                                      for i in range(cfg.ctx_len)]).unsqueeze(0).to(self.cfg.RUN_DEVICE)
        w_keys = list(w.keys())
        for x in w_keys:
            if 'time_decay' in x:
                nnn = x.strip('time_decay')
                w[nnn + 'time_w'] = torch.exp(
                    torch.cat([torch.exp(w[x]) * time_curve, w[nnn + 'time_first']], dim=-1)).unsqueeze(1)
                w[nnn + 'time_ww'] = torch.exp(
                    torch.exp(w[x]) * time_curve_ww).unsqueeze(1)
        if cfg.HEAD_QK:
            if w['copy_mask'].shape[0] != cfg.ctx_len:
                print(
                    f"\nWARNING: ctx_len {w['copy_mask'].shape[0]} --> {cfg.ctx_len}\n")
                w['copy_mask'] = torch.tril(torch.ones(
                    cfg.ctx_len, cfg.ctx_len)).to(self.cfg.RUN_DEVICE)

        for x in w.keys():
            if '.time_' in x and 'time_w' not in x:
                w[x] = w[x].squeeze()
            if '.time_decay' in x:
                w[x] = torch.exp(-torch.exp(w[x]))
            if '.time_first' in x:
                w[x] = torch.exp(w[x])

            xx = x.split('.')
            here = self.w
            for i in range(len(xx)):
                if xx[i].isdigit():
                    ii = int(xx[i])
                    if ii not in here:
                        here[ii] = types.SimpleNamespace()
                    here = here[ii]
                else:
                    if i == len(xx) - 1:
                        setattr(here, xx[i], w[x])
                    elif not hasattr(here, xx[i]):
                        if xx[i+1].isdigit():
                            setattr(here, xx[i], {})
                        else:
                            setattr(here, xx[i], types.SimpleNamespace())
                    here = getattr(here, xx[i])

        w_keys = list(w.keys())
        for x in w_keys:
            if 'time_mix' in x:
                w[x] = w[x].unsqueeze(0).unsqueeze(0)
            if 'time_decay' in x or 'time_first' in x:
                del w[x]

        self.gpt = RWKV_GPT(self).to(cfg.RUN_DEVICE)
        self.gpt.load_state_dict(w)

        self.gpt.eval()

        self.index = {}
        for i in range(cfg.n_layer):
            if cfg.FFN_PRE:
                if i == 0:
                    self.index[f'ffnPre.{i}xx'] = 0
                    self.index[f'ffn.{i}xx'] = 1
                else:
                    self.index[f'att.{i}xx'] = i * 4 - 2
                    self.index[f'att.{i}aa'] = i * 4 - 1
                    self.index[f'att.{i}bb'] = i * 4
                    self.index[f'ffn.{i}xx'] = i * 4 + 1

        self.cache = {}
        self.clear_stat()
        self.save_stat()

    def read(self, name):
        return self.xab[self.index[name]]

    def write(self, name, value):
        self.xab[self.index[name]] = value

    def ctx_to_key(self, ctx):
        return array.array('H', ctx[-self.cfg.ctx_len:]).tobytes()

    def clear_stat(self):
        self.code_last = self.ctx_to_key([])
        if self.cfg.FFN_PRE:
            self.xab = torch.zeros(
                (4 * self.cfg.n_layer - 2, self.cfg.n_embd), device=self.cfg.RUN_DEVICE)
        else:
            self.xab = torch.zeros(
                (4 * self.cfg.n_layer, self.cfg.n_embd), device=self.cfg.RUN_DEVICE)
        self.out = []
        if self.cfg.HEAD_QK:
            self.hk = torch.zeros((0, self.cfg.HEAD_QK_DIM),
                                  device=self.cfg.RUN_DEVICE)

    def save_stat(self):
        self.cache[self.code_last] = types.SimpleNamespace()
        target = self.cache[self.code_last]
        target.xab = self.xab.clone()
        target.out = self.out.copy()
        if self.cfg.HEAD_QK:
            target.hk = self.hk.clone()

    def load_stat(self, code):
        if code == self.code_last:
            return True
        if code not in self.cache:
            return False
        target = self.cache[code]
        self.xab = target.xab.clone()
        self.out = target.out.copy()
        if self.cfg.HEAD_QK:
            self.hk = target.hk.clone()
        self.code_last = code
        return True

    def LN(self, xx, w):
        return F.layer_norm(xx, (self.cfg.n_embd,), weight=w.weight, bias=w.bias)

    def FF(self, xx, w, name):
        xk = xx * w.time_mix_k + self.read(name + 'xx') * (1 - w.time_mix_k)
        xr = xx * w.time_mix_r + self.read(name + 'xx') * (1 - w.time_mix_r)
        
        self.write(name + 'xx', xx)

        r = torch.sigmoid(w.receptance.weight @ xr)
        k = torch.square(torch.relu(w.key.weight @ xk))
        kv = w.value.weight @ k

        return r * kv

    def SA(self, xx, w, name):
        xk = xx * w.time_mix_k + self.read(name + 'xx') * (1 - w.time_mix_k)
        xv = xx * w.time_mix_v + self.read(name + 'xx') * (1 - w.time_mix_v)
        xr = xx * w.time_mix_r + self.read(name + 'xx') * (1 - w.time_mix_r)

        self.write(name + 'xx', xx)

        r = torch.sigmoid(w.receptance.weight @ xr)
        k = torch.exp(torch.clamp(w.key.weight @ xk, max=self.cfg.k_max))
        v = w.value.weight @ xv
        kv = k * v

        aa = self.read(name + 'aa')
        bb = self.read(name + 'bb')
        a = aa + w.time_first * kv
        b = bb + w.time_first * k
        self.write(name + 'aa', w.time_decay * aa + kv)
        self.write(name + 'bb', w.time_decay * bb + k)

        rwkv = r * a / (b + self.cfg.b_eps)

        return w.output.weight @ rwkv

    def test_gpt(self, ctx):
        return self.gpt.forward(torch.tensor(ctx, device=self.cfg.RUN_DEVICE).unsqueeze(0))

    def run(self, ctx_extra, shall_save=False):
        with torch.no_grad():
            ctx = ctx_extra[-self.cfg.ctx_len:]
            code_now = self.ctx_to_key(ctx)
            if self.load_stat(code_now):
                return self.out.copy()

            code_last = self.ctx_to_key(ctx_extra[:-1])
            if not self.load_stat(code_last):
                self.gpt.forward(torch.tensor(
                    ctx, device=self.cfg.RUN_DEVICE).unsqueeze(0), shall_save=True)
                return self.out.copy()

            w = self.w
            x = w.emb.weight[ctx[-1]]

            for i in range(self.cfg.n_layer):
                x = self.LN(x, w.blocks[i].ln1)
                if i == 0 and self.cfg.FFN_PRE:
                    x = x + self.FF(x, w.blocks[0].ffnPre, f'ffnPre.{i}')
                else:
                    x = x + self.SA(x, w.blocks[i].att, f'att.{i}')
                x = self.LN(x, w.blocks[i].ln2)
                x = x + self.FF(x, w.blocks[i].ffn, f'ffn.{i}')

            x = self.LN(x, w.ln_out)

            if self.cfg.HEAD_QK:
                self.hk = torch.cat(
                    [self.hk, (w.head_k.weight @ x).unsqueeze(0)], dim=0)
                if self.hk.shape[0] > self.cfg.ctx_len:
                    self.hk = self.hk[-self.cfg.ctx_len:, :]

                q = w.head_q.weight @ x
                c = ((self.hk @ q) / self.cfg.HEAD_QK_DIM).tolist()

            x = w.head.weight @ x
            x = x.tolist()

            if self.cfg.HEAD_QK:
                for i in range(len(c)):
                    x[ctx[i]] += c[i]

            self.out = x

            self.code_last = code_now
            if shall_save:
                self.save_stat()

            return x.copy()

################################################################################
