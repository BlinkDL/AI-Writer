import numpy as np
import math, json
import torch
import torch.nn as nn
from torch.nn import functional as F

import src.utils
from src.model_v101 import GPT, GPTConfig

src.utils.set_seed(42) # 是否固定随机数（固定后每次运行的生成结果都一样）

print('\nAI-writer demo https://github.com/BlinkDL/AI-Writer')
print('\n声明：模型的训练数据全部来自网文，缺乏生活常识。生成的文字仅供娱乐。请遵守法律法规。')
print('\nLoading model...', end=' ')

MODEL_NAME = 'model/ww-101-L12-H12-C768-T256-20210723'
WORD_NAME = 'model/ww-20210723'

NUM_OF_RUNS = 9999
LENGTH_OF_EACH = 400

# 这是你的开头，建议开头用 \n 表示这是新段落
# 注意，不应出现 \r，不应出现重复空行（\n\n），段落开头和结尾不应有空格，因为清洗数据时去除了这些情况

# context = "\n我"
# context = "\n他"
# context = "\n她"
# context = "\n魔法"
context = "\n魔皇"
# context = "\n总裁"
# context = "\n都城"
# context = "\n龙傲天"
# context = "\n星际旅行"
# context = "\n三体舰队"
# context = "\n乾坤混元一气鼎！这是"

##############################################################################

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

model = GPT(GPTConfig(vocab_size, block_size,
                        n_layer=nLayers, n_head=nHead, n_embd=nEmb)).cuda()
model.load_state_dict(torch.load(MODEL_NAME + '.pth'))

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
            xxx = torch.tensor(x[-block_size:], dtype=torch.long)[None,...].to("cuda:0")
            out, _ = model(xxx)
        pos = -1 if real_len >= block_size else real_len - 1

        if train_dataset.itos[int(x[real_len-1])] == '\n':
            char = src.utils.sample_logits(out, pos, temperature=1.0, top_p=0.995)
        else:
            char = src.utils.sample_logits(out, pos, temperature=1.0, min_p_pow=2.0, min_p_ratio=0.02)
#             char = src.utils.sample_logits(out, pos, temperature=1.0, top_k=100, top_p=0.99)
#             char = src.utils.sample_logits(out, pos, temperature=1.0, top_p=0.95)
    
        if real_len < block_size:
            x[real_len] = char
        else:
            x = np.append(x, char)
        real_len += 1

        if i % 10 == 9 or i == LENGTH_OF_EACH-1 or i < 10:
            completion = ''.join([train_dataset.itos[int(i)] for i in x[print_begin:real_len]])
            print(completion.replace('\n', '\n  '), end = '')
            print_begin = real_len
    print()
