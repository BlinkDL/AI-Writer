########################################################################################################
# AI人工智障写作 - https://github.com/BlinkDL/AI-Writer
########################################################################################################

import random
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

def to_float(x):
    return x.cpu().detach().numpy().flatten()[0].astype(float)

def sample_logits(logits, pos, temperature=1.0, top_p=None):
    logits = logits[0][pos, :]
    probs = F.softmax(logits, dim=-1)

    if top_p is not None:
        out = probs.clone()
        sorted_probs, _ = torch.sort(out, descending=True)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1).cpu().numpy()
        cutoff = float(sorted_probs[np.argmax(cumulative_probs > top_p)].cpu())
        probs[probs < cutoff] = 0

    if temperature != 1.0:
        probs = probs.pow(1.0 / temperature)
    ix = torch.multinomial(probs, num_samples=1)
    return ix[0].cpu()

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda:
        torch.cuda.manual_seed_all(seed)
