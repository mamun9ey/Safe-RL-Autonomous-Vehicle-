import numpy as np, torch, torch.nn as nn, torch.optim as optim
from utils import set_seed

class QNet(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim,128), nn.ReLU(),
            nn.Linear(128,128), nn.ReLU(),
            nn.Linear(128, act_dim)
        )
    def forward(self, x): return self.net(x)

class SafeDQN:
    def __init__(self, obs_space, act_space, gamma=0.99, lr=1e-4, tau=0.005,
                 batch_size=64, replay_size=100_000, cost_weight=5.0, device="cpu"):
        obs_dim = int(np.prod(obs_space.shape)); act_dim = act_space.n
        self.q = QNet(obs_dim, act_dim).to(device)
        self.q_tgt = QNet(obs_dim, act_dim).to(device)
        self.q_tgt.load_state_dict(self.q.state_dict())
        self.opt = optim.Adam(self.q.parameters(), lr=lr)
        self.gamma, self.tau = gamma, tau
        self.bs, self.replay_size = batch_size, replay_size
        self.device = device
        self.cost_weight = cost_weight
        self.buf = []  # naive replay

    def push(self, s,a,r,c,ns,d):
        if len(self.buf)>=self.replay_size: self.buf.pop(0)
        self.buf.append((s,a,r,c,ns,d))

    def act(self, obs, eps, act_space):
        if np.random.rand() < eps: return act_space.sample()
        with torch.no_grad():
            q = self.q(torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0))
            return int(q.argmax(dim=-1).item())

    def update(self):
        if len(self.buf) < self.bs: return {}
        batch = np.random.choice(len(self.buf), self.bs, replace=False)
        s,a,r,c,ns,d = zip(*[self.buf[i] for i in batch])
        s = torch.tensor(np.array(s), dtype=torch.float32, device=self.device)
        ns = torch.tensor(np.array(ns), dtype=torch.float32, device=self.device)
        a = torch.tensor(a, dtype=torch.long, device=self.device)
        r = torch.tensor(r, dtype=torch.float32, device=self.device)
        c = torch.tensor(c, dtype=torch.float32, device=self.device)
        d = torch.tensor(d, dtype=torch.float32, device=self.device)

        # Cost-aware shaped reward
        rs = r - self.cost_weight * c

        with torch.no_grad():
            q_next = self.q_tgt(ns).max(dim=-1).values
            y = rs + self.gamma*(1.0 - d)*q_next

        q_pred = self.q(s).gather(1, a.unsqueeze(1)).squeeze(1)
        loss = ((q_pred - y)**2).mean()

        self.opt.zero_grad(); loss.backward(); self.opt.step()

        # soft update
        for p, tp in zip(self.q.parameters(), self.q_tgt.parameters()):
            tp.data.copy_(self.tau*p.data + (1-self.tau)*tp.data)
        return {"loss": float(loss.item())}
