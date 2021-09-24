# -*- coding: utf-8 -*-
import numpy as np
import math, json
import torch
import torch.nn as nn
from torch.nn import functional as F

import src.utils
from src.model_v101 import GPT, GPTConfig

# src.utils.set_seed(42) # 是否固定随机数（固定后每次运行的生成结果都一样）

print('\nAI-writer demo https://github.com/BlinkDL/AI-Writer')
print('\n声明：模型的训练数据全部来自网文，缺乏生活常识。生成的文字仅供娱乐。请遵守法律法规。')

RUN_DEVICE = 'gpu' # gpu 或 cpu

MODEL_NAME = 'model/ww-101-L12-H12-C768-T256-20210723'
WORD_NAME = 'model/ww-20210723'

NUM_OF_RUNS = 9999
LENGTH_OF_EACH = 400

min_p_ratio = 0.1 # 这个数字的范围是 0 到 1。数字越大，生成效果越规矩。数字越小，变化越多。

# 开头这样输入：
# context = "我"
# context = "他"
# context = "她"
# context = "魔法"
# context = "魔皇"
# context = "总裁"
# context = "都城"
# context = "龙傲天"
# context = "星际旅行"
# context = "三体舰队"
# context = "乾坤混元一气鼎！这是"

# 多行的开头这样输入（注意模型只会看最后256个字！）：
context = """
安柏：愿风神护佑你，陌生人！
安柏：我是西风骑士团侦察骑士，安柏。
安柏：你不是蒙德市民吧？那么，请说明自己的身份！
派蒙：冷静一下，我们不是可疑人员——
安柏：可疑人员都会这么说。
旅行者：你好，我是旅行者。
安柏：……听着不像是本地人的名字。
安柏：还有这只……吉祥物，又是怎么回事？
旅行者：是应急食品。
派蒙：完全不对！怎么还不如吉祥物啊！
安柏：总而言之，是旅行者对吧。
安柏：最近蒙德周围有巨龙出没，你们还是尽快进城比较好。
"""
# context = """
# 金载圭喝了一口酒，说道：“前辈，你不来支持阁下，还有谁来做啊！阁下，和这种虫豸在一起，怎么能搞好政治呢？”说完向车智澈开枪。
# 阁下怒吼：“要造反啊！”
# """
# context = """
# 我与父亲不相见已二年余了，我最不能忘记的是他的背影。
# 那年冬天，祖母死了，父亲的差使也交卸了，正是祸不单行的日子。我从北京到徐州，打算跟着父亲奔丧回家。到徐州见着父亲，看见满院狼藉的东西，又想起祖母，不禁簌簌地流下眼泪。父亲说：“事已如此，不必难过，好在天无绝人之路！”
# 回家变卖典质，父亲还了亏空；又借钱办了丧事。这些日子，家中光景很是惨淡，一半为了丧事，一半为了父亲赋闲。丧事完毕，父亲要到南京谋事，我也要回北京念书，我们便同行。
# 到南京时，有朋友约去游逛，勾留了一日；第二日上午便
# """
# context = """
# 这几天心里颇不宁静。今晚在院子里坐着乘凉，忽然想起日日走过的荷塘，在这满月的光里，总该另有一番样子吧。月亮渐渐地升高了，墙外马路上孩子们的欢笑，已经听不见了；妻在屋里拍着闰儿，迷迷糊糊地哼着眠歌。我悄悄地披了大衫，带上门出去。
# 沿着荷塘，是一条曲折的小煤屑路。这是一条幽僻的路；白天也少人走，夜晚更加寂寞。荷塘四面，长着许多树，蓊蓊郁郁的。路的一旁，是些杨柳，和一些不知道名字的树。没有月光的晚上，这路上阴森森的，有些怕人。今晚却很好，虽然月光也还是淡淡的。
# 路上只我一个人，背着手踱着。这一
# """

##############################################################################

nLayers = 12
nHead = 12
nEmb = 768
block_size = 256

context = context.strip().split('\n')
for c in range(len(context)):
    context[c] = context[c].strip()
context = '\n' + '\n'.join(context)

print('您输入的开头有 ' + str(len(context)) + ' 个字。注意，模型只能看到最后 ' + str(block_size) + ' 个字。')

with open(WORD_NAME + '.json', "r", encoding="utf-16") as result_file:
    word_table = json.load(result_file)   

vocab_size = len(word_table)

train_dataset = lambda: None
train_dataset.stoi = {v: int(k) for k, v in word_table.items()}
train_dataset.itos = {int(k): v for k, v in word_table.items()}
UNKNOWN_CHAR = train_dataset.stoi['\ue083']

print('\nLoading model...', end=' ')
model = GPT(GPTConfig(vocab_size, block_size, n_layer=nLayers, n_head=nHead, n_embd=nEmb))
if RUN_DEVICE == 'gpu':
    model = model.cuda()
    model.load_state_dict(torch.load(MODEL_NAME + '.pth'))
else:
    model.load_state_dict(torch.load(MODEL_NAME + '.pth', map_location='cpu'))

print('done:', MODEL_NAME, '&', WORD_NAME)

##############################################################################

for run in range(NUM_OF_RUNS):

    x = np.array([train_dataset.stoi.get(s, UNKNOWN_CHAR) for s in context], dtype=np.int64)

    real_len = len(x)
    if real_len < block_size:
        x = np.pad(x, (0, block_size - real_len))
    print_begin = 0
        
    for i in range(LENGTH_OF_EACH):

        if i == 0:

            print(('-' * 60) + '\n' + context.replace('\n', '\n  ').strip('\n'), end = '')
            print_begin = real_len

        with torch.no_grad():
            xxx = torch.tensor(x[-block_size:], dtype=torch.long)[None,...]
            if RUN_DEVICE == 'gpu':
                xxx = xxx.cuda()
            out, _ = model(xxx)
        pos = -1 if real_len >= block_size else real_len - 1

        if train_dataset.itos[int(x[real_len-1])] == '\n':
            char = src.utils.sample_logits(out, pos, temperature=1.0, top_p=0.99)
        else:
            char = src.utils.sample_logits(out, pos, temperature=1.0, min_p_pow=2.0, min_p_ratio=min_p_ratio, top_k=150, top_p=0.99)
    
        if real_len < block_size:
            x[real_len] = char
        else:
            x = np.append(x, char)
        real_len += 1

        if i % 10 == 9 or i == LENGTH_OF_EACH-1 or i < 10 or RUN_DEVICE != 'gpu':
            completion = ''.join([train_dataset.itos[int(i)] for i in x[print_begin:real_len]])
            print(completion.replace('\n', '\n  '), end = '')
            print_begin = real_len
    print()
