# Minimal Safe-RL env adapter (works with Gym/Gymnasium-like envs)
import numpy as np

class SafeEnv:
    """
    Expected env API:
      obs, info = env.reset(seed=...)
      obs, reward, terminated, truncated, info = env.step(action)
    info may include {"cost": float}. If not, set cost=0.
    """
    def __init__(self, base_env, cost_key="cost"):
        self.env = base_env
        self.cost_key = cost_key

    def reset(self, seed=None):
        try:
            obs, info = self.env.reset(seed=seed)
        except TypeError:  # if old gym
            obs = self.env.reset()
            info = {}
        return obs, info

    def step(self, action):
        out = self.env.step(action)
        if len(out) == 5:
            obs, reward, terminated, truncated, info = out
            done = terminated or truncated
        else:
            obs, reward, done, info = out
        cost = float(info.get(self.cost_key, 0.0))
        return obs, reward, cost, done, info

    @property
    def action_space(self): return self.env.action_space
    @property
    def observation_space(self): return self.env.observation_space
