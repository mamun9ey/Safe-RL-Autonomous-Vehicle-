import torch, random, numpy as np

def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def discount_cumsum(x, gamma):
    # x: [T], returns y[t] = sum_{k>=t} gamma^{k-t} x[k]
    y = np.zeros_like(x, dtype=np.float32)
    acc = 0.0
    for t in reversed(range(len(x))):
        acc = x[t] + gamma * acc
        y[t] = acc
    return y

def gae(rews, vals, dones, gamma=0.99, lam=0.95):
    T = len(rews)
    adv = np.zeros(T, dtype=np.float32)
    lastgaelam = 0.0
    for t in reversed(range(T)):
        nextnonterminal = 1.0 - float(dones[t])
        nextv = 0.0 if t == T-1 else vals[t+1]
        delta = rews[t] + gamma * nextv * nextnonterminal - vals[t]
        lastgaelam = delta + gamma * lam * nextnonterminal * lastgaelam
        adv[t] = lastgaelam
    ret = adv + vals
    return adv, ret
