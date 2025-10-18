import torch, torch.nn as nn
import torch.nn.functional as F

def mlp(sizes, act=nn.Tanh, out_act=nn.Identity):
    layers = []
    for i in range(len(sizes)-1):
        layers += [nn.Linear(sizes[i], sizes[i+1])]
        layers += [out_act() if i==len(sizes)-2 else act()]
    return nn.Sequential(*layers)

class ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, continuous=True):
        super().__init__()
        hid = 128
        self.continuous = continuous
        self.pi = mlp([obs_dim, hid, hid, act_dim], act=nn.Tanh, out_act=nn.Identity)
        self.log_std = nn.Parameter(torch.zeros(act_dim)) if continuous else None
        self.v = mlp([obs_dim, hid, hid, 1], act=nn.Tanh, out_act=nn.Identity)
        self.vc = mlp([obs_dim, hid, hid, 1], act=nn.Tanh, out_act=nn.Identity)  # cost value

    def pi_dist(self, x):
        if self.continuous:
            mu = self.pi(x)
            std = torch.exp(self.log_std)
            return torch.distributions.Normal(mu, std)
        else:
            logits = self.pi(x)
            return torch.distributions.Categorical(logits=logits)

    def act(self, x):
        dist = self.pi_dist(x)
        a = dist.sample()
        logp = dist.log_prob(a).sum(-1) if self.continuous else dist.log_prob(a)
        return a, logp

    def value(self, x): return self.v(x).squeeze(-1)
    def cost_value(self, x): return self.vc(x).squeeze(-1)
