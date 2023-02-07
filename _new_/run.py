# -*- coding:utf-8 -*-
print('\nAI人工智障写作 https://github.com/BlinkDL/AI-Writer')
print('如果觉得好用，欢迎选购我们的护眼灯 https://withablink.taobao.com')
print('我的知乎是 https://zhuanlan.zhihu.com/p/423646620')
print('\n声明：模型的训练数据来自网文，缺乏生活常识。生成的文字仅供娱乐。请遵守法律法规。')

import sys, os, time, datetime
import numpy as np
import time
import numpy as np
import torch
from src.model import RWKV_RNN, RWKV_CFG
from src.utils import CHN_TOKENIZER
np.set_printoptions(precision=4, suppress=True, linewidth=200)

##############################################################################

cfg = RWKV_CFG('L24_CHN', 'all-44704')
cfg.RUN_DEVICE = 'cuda' # 使用 cpu 还是 cuda？

print(f'\n正在使用 {cfg.RUN_DEVICE} 模式\n')

NUM_OF_RUNS = 33
LENGTH_OF_EACH = 333

TEMPERATURE = 1.0
top_p_newline = 0.9
top_p = 0.7
allow_p = 0.01

# 开头非常重要。开头越长，续写效果越好。开头需创造剧情点。开头文笔越好，续写就越好。开头乱写，续写也乱写。
# 开头这样输入：
# context = "魔法"
context = "“区区"
# context = "三体舰队"
# context = "这是一颗"
# context = "众人一惊，没想到这林黛玉的剑法竟如此精妙，只见在那剑影下，剑尖朝着伏地魔的脖子探去，眼见避无可避，伏地魔情急，大喊"

# 多行的开头这样输入：
# context = """
# 这几天心里颇不宁静。今晚在院子里坐着乘凉，忽然想起日日走过的荷塘，在这满月的光里，总该另有一番样子吧。月亮渐渐地升高了，墙外马路上孩子们的欢笑，已经听不见了；妻在屋里拍着闰儿，迷迷糊糊地哼着眠歌。我悄悄地披了大衫，带上门出去。
# 沿着荷塘，是一条曲折的小煤屑路。这是一条幽僻的
# """

##############################################################################

if len(context) + LENGTH_OF_EACH >= cfg.ctx_len:
    print('总长度过长，请缩短续写长度和原文长度。')
    exit(0)
tokenizer = CHN_TOKENIZER(WORD_NAME='word-2022-02-16', UNKNOWN_CHAR='\ue083')
cfg.vocab_size = tokenizer.vocab_size
model = RWKV_RNN(cfg)
model.tokenizer = tokenizer

def ai_print(str):
    print(str, end = '')

def run(context):
    context = '\n' + tokenizer.refine_context(context).strip()
    print(f'开头有 {str(len(context))} 字，写 {NUM_OF_RUNS} 次，每次 {LENGTH_OF_EACH} 字。\n')
    for run in range(NUM_OF_RUNS):
        print('-' * 20 + '\x1B[93m' + context.replace('\n', '\n　'), end = '\x1B[37m')
        t_begin = time.time_ns()
        x = np.array([tokenizer.stoi.get(s, tokenizer.UNKNOWN_CHAR) for s in context], dtype=np.int64)

        real_len = len(x)
        print_begin = 0
            
        for i in range(LENGTH_OF_EACH):

            if i == 0:
                print_begin = real_len

            with torch.no_grad():
                xin = x[-cfg.ctx_len-1:]
                out = model.run(xin)

            char = tokenizer.sample_logits(out, xin.tolist(), cfg.ctx_len, temperature=TEMPERATURE, top_p_usual=top_p, top_p_newline=top_p_newline, allow_p=allow_p)
            
            x = np.append(x, char)
            real_len += 1

            if i % 2 == 0 or i == LENGTH_OF_EACH-1 or i < 10:
                completion = ''.join([tokenizer.itos[int(i)] for i in x[print_begin:real_len]])
                ai_print(completion.replace('\n', '\n　'))
                print_begin = real_len

        print()
        t_end = time.time_ns()
        print(f"---------- {round((t_end - t_begin) / (10 ** 9), 2)} s, 第 {run+1}/{NUM_OF_RUNS} 次", end='')

run(context)
print('\n')
