# -*- coding: utf-8 -*-
########################################################################################################
# AI人工智障写作 - https://github.com/BlinkDL/AI-Writer
########################################################################################################

import numpy as np
import math, json
import torch
import torch.nn as nn
from torch.nn import functional as F

import src.utils
from src.model import GPT, GPTConfig

# src.utils.set_seed(42) # 是否固定随机数（固定后每次运行的生成结果都一样）

print('\nAI人工智障写作 https://github.com/BlinkDL/AI-Writer')
print('请关注我的知乎 https://zhuanlan.zhihu.com/p/394766831')
print('\n声明：模型的训练数据全部来自网文，缺乏生活常识。生成的文字仅供娱乐。请遵守法律法规。')

RUN_DEVICE = 'gpu' # gpu 或 cpu

MODEL_NAME = 'model/xuanhuan-2021-10-26'
WORD_NAME = 'model/xuanhuan-2021-10-26'

NUM_OF_RUNS = 9999
LENGTH_OF_EACH = 200 # 每次写多少字

min_p_ratio = 0.02 # 这个数字的范围是 0 到 1。数字越大，生成效果越规矩。数字越小，变化越多。

# 开头这样输入：
# context = "我"
# context = "他"
# context = "她"
# context = "魔法"
# context = "三体舰队"
context = "这是一颗"
# context = "众人一惊，没想到这林黛玉的剑法竟如此精妙，只见在那剑影下，剑尖朝着伏地魔的脖子探去，眼见避无可避，伏地魔情急，大喊"

# 多行的开头这样输入：
# context = """
# 这几天心里颇不宁静。今晚在院子里坐着乘凉，忽然想起日日走过的荷塘，在这满月的光里，总该另有一番样子吧。月亮渐渐地升高了，墙外马路上孩子们的欢笑，已经听不见了；妻在屋里拍着闰儿，迷迷糊糊地哼着眠歌。我悄悄地披了大衫，带上门出去。
# 沿着荷塘，是一条曲折的小煤屑路。这是一条幽僻的路；白天也少人走，夜晚更加寂寞。荷塘四面，长着许多树，蓊蓊郁郁的。路的一旁，是些杨柳，和一些不知道名字的树。没有月光的晚上，这路上阴森森的，有些怕人。今晚却很好，虽然月光也还是淡淡的。
# 路上只我一个人，背着手踱着。这一
# """

##############################################################################

ctx_len = 512
n_layer = 12
n_head = 12
n_embd = n_head * 64
n_attn = n_embd
n_ffn = n_embd

context = context.strip().split('\n')
for c in range(len(context)):
    context[c] = context[c].strip().strip('\u3000')
context = '\n' + ('\n'.join(context)).strip()
print('您输入的开头有 ' + str(len(context)) + ' 个字。注意，模型只会看最后 ' + str(ctx_len) + ' 个字。')

with open(WORD_NAME + '.json', "r", encoding="utf-16") as result_file:
    word_table = json.load(result_file)   

vocab_size = len(word_table)

train_dataset = lambda: None
train_dataset.stoi = {v: int(k) for k, v in word_table.items()}
train_dataset.itos = {int(k): v for k, v in word_table.items()}
UNKNOWN_CHAR = train_dataset.stoi['\ue083']

print('\nLoading model...', end=' ')
model = GPT(GPTConfig(vocab_size, ctx_len, n_layer=n_layer, n_head=n_head, n_embd=n_embd, n_attn=n_attn, n_ffn=n_ffn))
if RUN_DEVICE == 'gpu':
    model = model.cuda()
    model.load_state_dict(torch.load(MODEL_NAME + '.pth').state_dict())
else:
    model.load_state_dict(torch.load(MODEL_NAME + '.pth', map_location='cpu').state_dict())

print('done:', MODEL_NAME, '&', WORD_NAME)

##############################################################################

for run in range(NUM_OF_RUNS):

    x = np.array([train_dataset.stoi.get(s, UNKNOWN_CHAR) for s in context], dtype=np.int64)

    real_len = len(x)
    print_begin = 0
        
    for i in range(LENGTH_OF_EACH):

        if i == 0:

            print(('-' * 60) + '\n' + context.replace('\n', '\n  ').strip('\n'), end = '')
            print_begin = real_len

        with torch.no_grad():
            xxx = torch.tensor(x[-ctx_len:], dtype=torch.long)[None,...]
            if RUN_DEVICE == 'gpu':
                xxx = xxx.cuda()
            out, _ = model(xxx)
            out[:, :, UNKNOWN_CHAR] = -float('Inf')
        pos = -1 if real_len >= ctx_len else real_len - 1

        if train_dataset.itos[int(x[real_len-1])] == '\n':
            char = src.utils.sample_logits(out, pos, temperature=1.0, top_p=0.995)
        else:
            char = src.utils.sample_logits(out, pos, temperature=0.9, min_p_pow=2.0, min_p_ratio=min_p_ratio)
    
        x = np.append(x, char)
        real_len += 1

        if i % 10 == 9 or i == LENGTH_OF_EACH-1 or i < 10 or RUN_DEVICE != 'gpu':
            completion = ''.join([train_dataset.itos[int(i)] for i in x[print_begin:real_len]])
            print(completion.replace('\n', '\n  '), end = '')
            print_begin = real_len
    print()
