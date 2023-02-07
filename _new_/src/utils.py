# -*- coding:utf-8 -*-
import json, math
import random
import numpy as np
import torch
from torch.nn import functional as F

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class CHN_TOKENIZER():
    def __init__(self, WORD_NAME, UNKNOWN_CHAR = '\ue083'):
        with open(WORD_NAME + '.json', "r", encoding="utf-16") as result_file:
            self.word_table = json.load(result_file)

        self.vocab_size = len(self.word_table)

        self.stoi = {v: int(k) for k, v in self.word_table.items()}
        self.itos = {int(k): v for k, v in self.word_table.items()}

        self.UNKNOWN_CHAR = self.stoi[UNKNOWN_CHAR]

    def refine_context(self, context):
        context = context.strip().split('\n')
        for c in range(len(context)):
            context[c] = context[c].strip().strip('\u3000').strip('\r')
        context = list(filter(lambda c: c != '', context))
        context = '\n' + ('\n'.join(context)).strip()
        if context == '':
            context = '\n'
        return context

    def sample_logits(self, out, x, ctx_len, temperature=1.0, top_p_usual=None, top_p_newline=None, allow_p=None):
        out[self.UNKNOWN_CHAR] = -float('Inf')
        
        lastChar = int(x[-1])        
        probs = F.softmax(torch.tensor(out), dim=-1)
        if self.itos[lastChar] == '\n':
            top_p = top_p_newline
        else:
            top_p = top_p_usual
        sorted_probs, s_index = torch.sort(probs, descending=True)

        cumulative_probs = torch.cumsum(sorted_probs, dim=-1).numpy()
        cutoff = float(sorted_probs[np.argmax(cumulative_probs > top_p)])
        cutoff = min(cutoff, allow_p)

        probs[probs < cutoff] = 0

        if temperature != 1.0:
            probs = probs.pow(1.0 / temperature)
        
        try:
            x = torch.multinomial(probs, num_samples=1)[0]
        except:
            print(probs)
        return x
