import numpy as np, torch, torch.nn as nn, torch.optim as optim
from nets import ActorCritic
from utils import gae, set_seed

class SafePPO:
    def __init__(self, obs_space, act_space, continuous=True, clip=0.2, lr=3e-4,
                 gamma=0.99, lam=0.95, cost_gamma=0.99, epochs=10, batch_size=4096,
                 minibatch_size=256, vf_coef=0.5, ent_coef=0.0, max_grad_norm=0.5,
                 cost_penalty=10.0, device="cpu"):
        obs_dim = int(np.prod(obs_space.shape))
        act_dim = act_space.shape[0] if continuous else act_space.n
        self.ac = ActorCritic(obs_dim, act_dim, continuous).to(device)
        self.opt = optim.Adam(self.ac.parameters(), lr=lr)
        self.clip, self.gamma, self.lam = clip, gamma, lam
        self.cost_gamma, self.epochs = cost_gamma, epochs
        self.batch_size, self.minibatch_size = batch_size, minibatch_size
        self.vf_coef, self.ent_coef = vf_coef, ent_coef
        self.max_grad_norm = max_grad_norm
        self.cost_penalty = cost_penalty
        self.device = device
        self.continuous = continuous

    def rollout(self, env, seed):
        set_seed(seed)
        obs_buf, act_buf, logp_buf, rew_buf, cost_buf, val_buf, cval_buf, done_buf = [],[],[],[],[],[],[],[]
        obs, _ = env.reset(seed=seed)
        for _ in range(self.batch_size):
            x = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            with torch.no_grad():
                a, logp = self.ac.act(x)
                v = self.ac.value(x); vc = self.ac.cost_value(x)
            a_np = a.cpu().numpy().squeeze()
            next_obs, r, c, done, _ = env.step(a_np)
            obs_buf.append(obs); act_buf.append(a_np); logp_buf.append(logp.item())
            rew_buf.append(r); cost_buf.append(c); val_buf.append(v.item()); cval_buf.append(vc.item())
            done_buf.append(done)
            obs = next_obs if not done else env.reset(seed=seed)[0]
        # Advantages & returns
        adv, ret = gae(rew_buf, val_buf, done_buf, self.gamma, self.lam)
        cadv, cret = gae(cost_buf, cval_buf, done_buf, self.cost_gamma, self.lam)
        data = {
            "obs": torch.tensor(np.array(obs_buf), dtype=torch.float32, device=self.device),
            "act": torch.tensor(np.array(act_buf), dtype=torch.float32 if self.continuous else torch.long, device=self.device),
            "logp_old": torch.tensor(np.array(logp_buf), dtype=torch.float32, device=self.device),
            "adv": torch.tensor((adv - adv.mean())/(adv.std()+1e-8), dtype=torch.float32, device=self.device),
            "ret": torch.tensor(ret, dtype=torch.float32, device=self.device),
            "cadv": torch.tensor((cadv - cadv.mean())/(cadv.std()+1e-8), dtype=torch.float32, device=self.device),
            "cret": torch.tensor(cret, dtype=torch.float32, device=self.device),
        }
        return data

    def update(self, data):
        inds = np.arange(self.batch_size)
        for _ in range(self.epochs):
            np.random.shuffle(inds)
            for start in range(0, self.batch_size, self.minibatch_size):
                mb = inds[start:start+self.minibatch_size]
                obs, act = data["obs"][mb], data["act"][mb]
                logp_old = data["logp_old"][mb]
                adv, ret = data["adv"][mb], data["ret"][mb]
                cadv, cret = data["cadv"][mb], data["cret"][mb]

                dist = self.ac.pi_dist(obs)
                if self.ac.continuous:
                    logp = dist.log_prob(act).sum(-1)
                    ent = dist.entropy().sum(-1).mean()
                else:
                    logp = dist.log_prob(act)
                    ent = dist.entropy().mean()
                ratio = torch.exp(logp - logp_old)
                surr1 = ratio * adv
                surr2 = torch.clamp(ratio, 1.0-self.clip, 1.0+self.clip) * adv
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value losses
                v = self.ac.value(obs); vc = self.ac.cost_value(obs)
                v_loss = ((v - ret)**2).mean()
                vc_loss = ((vc - cret)**2).mean()

                # Cost penalty on policy (encourage decreasing cost-advantage)
                cost_loss = (ratio * cadv).mean()

                loss = policy_loss + self.vf_coef*(v_loss + vc_loss) - self.ent_coef*ent + self.cost_penalty*cost_loss

                self.opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(self.ac.parameters(), self.max_grad_norm)
                self.opt.step()
