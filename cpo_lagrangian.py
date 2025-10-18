import numpy as np, torch, torch.nn as nn, torch.optim as optim
from nets import ActorCritic
from utils import gae, set_seed

class LagrangeMultiplier(nn.Module):
    def __init__(self, init_lambda=0.5, lr=1e-3):
        super().__init__()
        self.log_lambda = nn.Parameter(torch.log(torch.tensor(init_lambda)))
        self.opt = optim.Adam([self.log_lambda], lr=lr)

    @property
    def lam(self): return torch.clamp(self.log_lambda.exp(), 0.0, 1e6)

    def step(self, Jc, d):
        # maximize w.r.t. lambda: L(θ,λ) = J_r(θ) - λ (J_c(θ) - d)
        loss = -(self.lam * (Jc - d))
        self.opt.zero_grad(); loss.backward(); self.opt.step()
        return float(self.lam.item())

class CPO_Lag:
    def __init__(self, obs_space, act_space, d=0.05, continuous=True, lr=3e-4,
                 gamma=0.99, lam=0.95, cost_gamma=0.99, epochs=10, batch_size=4096,
                 minibatch_size=256, vf_coef=0.5, ent_coef=0.0, max_grad_norm=0.5,
                 device="cpu"):
        obs_dim = int(np.prod(obs_space.shape))
        act_dim = act_space.shape[0] if continuous else act_space.n
        self.ac = ActorCritic(obs_dim, act_dim, continuous).to(device)
        self.opt = optim.Adam(self.ac.parameters(), lr=lr)
        self.gamma, self.lam = gamma, lam
        self.cost_gamma = cost_gamma
        self.epochs = epochs
        self.batch_size = batch_size
        self.minibatch_size = minibatch_size
        self.vf_coef, self.ent_coef = vf_coef, ent_coef
        self.max_grad_norm = max_grad_norm
        self.device = device
        self.d = torch.tensor(d, dtype=torch.float32, device=device)
        self.lagr = LagrangeMultiplier(init_lambda=1.0, lr=1e-3)
        self.continuous = continuous

    def rollout(self, env, seed):
        # identical to PPO rollout but no clipping
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

        from utils import gae
        adv, ret = gae(rew_buf, val_buf, done_buf, self.gamma, self.lam)
        cadv, cret = gae(cost_buf, cval_buf, done_buf, self.cost_gamma, self.lam)
        data = {
            "obs": torch.tensor(np.array(obs_buf), dtype=torch.float32, device=self.device),
            "act": torch.tensor(np.array(act_buf), dtype=torch.float32 if self.continuous else torch.long, device=self.device),
            "logp_old": torch.tensor(np.array(logp_buf), dtype=torch.float32, device=self.device),
            "adv": torch.tensor((adv - np.mean(adv))/(np.std(adv)+1e-8), dtype=torch.float32, device=self.device),
            "ret": torch.tensor(ret, dtype=torch.float32, device=self.device),
            "cadv": torch.tensor((cadv - np.mean(cadv))/(np.std(cadv)+1e-8), dtype=torch.float32, device=self.device),
            "cret": torch.tensor(cret, dtype=torch.float32, device=self.device),
            "Jc": float(np.mean(cost_buf))
        }
        return data

    def update(self, data):
        # Update lambda first (dual ascent) using batch Jc
        lam_val = self.lagr.step(torch.tensor(data["Jc"]), self.d.cpu())
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
                    logp = dist.log_prob(act); ent = dist.entropy().mean()
                ratio = torch.exp(logp - logp_old)

                # Unclipped surrogate with Lagrangian penalty on cost-advantage
                policy_loss = -(ratio * (adv - self.lagr.lam.detach()*cadv)).mean()

                v = self.ac.value(obs); vc = self.ac.cost_value(obs)
                v_loss = ((v - ret)**2).mean()
                vc_loss = ((vc - cret)**2).mean()

                loss = policy_loss + self.vf_coef*(v_loss + vc_loss) - self.ent_coef*ent

                self.opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(self.ac.parameters(), self.max_grad_norm)
                self.opt.step()
        return {"lambda": lam_val}
