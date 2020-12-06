import random

import torch
import numpy as np

from torch.nn import functional as F


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def top_k_logits_(logits, k):
    v, _ = torch.topk(logits, k)
    return logits.masked_fill_(logits.lt(v[..., [-1]]), float('-inf'))


def generate(model, x, steps, temperature=1.0, sample=False, top_k=None):
    """
    take a conditioning sequence of indices in x (of shape (b,t)) and predict the next token in
    the sequence, feeding the predictions back into the model each time. Clearly the sampling
    has quadratic complexity unlike an RNN that is only linear, and has a finite context window
    of block_size, unlike an RNN that has an infinite context window.
    """
    block_size = model.get_block_size()

    # essentially a crude circular buffer for characters
    i0, i1 = 0, min(len(x), block_size)
    state = torch.empty(2 * block_size, dtype=x.dtype, device=x.device)
    state[i0:i1].copy_(x[-block_size:])  # crop context if needed

    model.eval()
    for k in range(steps):
        with torch.no_grad():
            logits, _ = model(state[i0:i1].unsqueeze(0))

        # pluck the logits at the final step and scale by temperature
        logits = logits[0, -1, :] / temperature

        # optionally crop probabilities to only the top k options
        if top_k is not None:
            logits = top_k_logits_(logits, top_k)

        # apply softmax to convert to probabilities
        probs = F.softmax(logits, dim=-1)
        # sample from the distribution or take the most likely
        if sample:
            ix = torch.multinomial(probs, num_samples=1)
        else:
            _, ix = torch.topk(probs, k=1, dim=-1)

        # append to the sequence and continue
        if i1 >= len(state):
            state[:i1 - i0].copy_(state[i0:i1])
            i0, i1 = 0, i1 - i0
        ix = state[i1] = int(ix)

        yield ix, probs.cpu()

        # shift the context in the buffer
        i1 += 1
        if i1 - i0 >= block_size:  # can only be == in fact!
            i0 += 1


@torch.no_grad()
def sample(model, x, steps, temperature=1.0, sample=False, top_k=None):
    result = list(generate(model, x, steps, temperature=temperature,
                           sample=sample, top_k=top_k))

    return torch.cat([x, torch.tensor(result).to(x)])
