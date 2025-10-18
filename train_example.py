import gym
import torch
from env_wrapper import SafeEnv
from ppo import SafePPO
from dqn import SafeDQN
from cpo_lagrangian import CPO_Lag
from utils import set_seed

def make_env(env_id="Pendulum-v1", continuous=True):
    base = gym.make(env_id)
    return SafeEnv(base), continuous

def main(algo="ppo", seeds=(0,1,2,3,4), total_iters=200):
    env, continuous = make_env("Pendulum-v1", continuous=True)  # replace with CARLA wrapper
    device="cuda" if torch.cuda.is_available() else "cpu"

    if algo=="ppo":
        agent = SafePPO(env.observation_space, env.action_space, continuous=continuous, device=device)
    elif algo=="dqn":
        assert not continuous, "Use a discrete-action env for DQN"
        agent = SafeDQN(env.observation_space, env.action_space, device=device)
    else:
        agent = CPO_Lag(env.observation_space, env.action_space, continuous=continuous, device=device, d=0.05)

    for s in seeds:
        set_seed(s)
        if algo in ("ppo","cpo"):
            for it in range(total_iters):
                data = agent.rollout(env, seed=s+it)
                logs = agent.update(data)
                if it % 10 == 0: print(algo.upper(), "iter", it, logs if logs else "")
        else:  # DQN loop (off-policy)
            obs, _ = env.reset(seed=s)
            eps, eps_min, eps_decay = 1.0, 0.05, 0.995
            for it in range(total_iters*1000):
                a = agent.act(obs, eps, env.action_space)
                nobs, r, c, done, _ = env.step(a)
                agent.push(obs, a, r, c, nobs, done)
                obs = nobs if not done else env.reset(seed=s)[0]
                logs = agent.update()
                eps = max(eps_min, eps*eps_decay)
                if it % 1000 == 0: print("DQN step", it, logs if logs else "")

if __name__ == "__main__":
    # Options: "ppo", "dqn", "cpo"
    main(algo="ppo")
