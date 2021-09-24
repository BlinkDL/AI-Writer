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
print('\nLoading model...', end=' ')

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

# 多行的开头这样输入：
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
安柏：这里离蒙德不远，就由身为骑士的我来护送你们一程。
派蒙：欸？你出城没有什么别的任务吗？
安柏：当然有，不过放心，任务路上也会保证你们的安全。
安柏：而且……我也不能放着可疑人士不管！
"""

##############################################################################

context = context.strip().split('\n')
for c in range(len(context)):
    context[c] = context[c].strip()
context = '\n' + '\n'.join(context)

with open(WORD_NAME + '.json', "r", encoding="utf-16") as result_file:
    word_table = json.load(result_file)   

vocab_size = len(word_table)

train_dataset = lambda: None
train_dataset.stoi = {v: int(k) for k, v in word_table.items()}
train_dataset.itos = {int(k): v for k, v in word_table.items()}
UNKNOWN_CHAR = train_dataset.stoi['\ue083']

nLayers = 12
nHead = 12
nEmb = 768
block_size = 256

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
            char = src.utils.sample_logits(out, pos, temperature=1.0, top_p=0.995)
        else:
            char = src.utils.sample_logits(out, pos, temperature=1.0, min_p_pow=2.0, min_p_ratio=min_p_ratio)
#             char = src.utils.sample_logits(out, pos, temperature=1.0, top_k=100, top_p=0.99)
#             char = src.utils.sample_logits(out, pos, temperature=1.0, top_p=0.95)
    
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
