"""PPO Rollout Buffer。收集 n_steps 的轨迹数据用于策略更新。"""

from dataclasses import dataclass, field
from typing import List

import numpy as np
import torch


@dataclass
class RolloutBuffer:
    """存储单次 rollout 的所有经验（含两阶段带牌数据）。"""

    obs: List[np.ndarray] = field(default_factory=list)
    actions: List[int] = field(default_factory=list)
    log_probs: List[float] = field(default_factory=list)
    values: List[float] = field(default_factory=list)
    rewards: List[float] = field(default_factory=list)
    dones: List[bool] = field(default_factory=list)
    masks: List[np.ndarray] = field(default_factory=list)
    # 两阶段带牌数据
    kicker_faces: List[np.ndarray] = field(default_factory=list)   # shape (3,) int32, -1=无带牌
    kicker_avails: List[np.ndarray] = field(default_factory=list)  # shape (13,) int32, 可用张数

    def add(self, obs, action, log_prob, value, reward, done, mask,
            kicker_faces=None, kicker_avail=None):
        self.obs.append(obs)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.rewards.append(reward)
        self.dones.append(done)
        self.masks.append(mask)
        self.kicker_faces.append(
            kicker_faces if kicker_faces is not None else np.full(8, -1, dtype=np.int32)
        )
        self.kicker_avails.append(
            kicker_avail if kicker_avail is not None else np.zeros(13, dtype=np.int32)
        )

    def __len__(self):
        return len(self.obs)

    def compute_returns_and_advantages(
        self,
        last_value: float,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        计算 GAE advantages 和 discounted returns。

        last_value: V(s_{T+1})，下一状态的价值估计（若为终止则为 0）
        """
        n = len(self.rewards)
        advantages = np.zeros(n, dtype=np.float32)
        last_gae = 0.0

        for t in reversed(range(n)):
            next_value = last_value if t == n - 1 else self.values[t + 1]
            next_done = 1.0 if (t == n - 1 and self.dones[-1]) else float(self.dones[t])

            delta = self.rewards[t] + gamma * next_value * (1.0 - next_done) - self.values[t]
            last_gae = delta + gamma * gae_lambda * (1.0 - next_done) * last_gae
            advantages[t] = last_gae

        returns = advantages + np.array(self.values, dtype=np.float32)
        return returns, advantages

    def to_tensors(self, device: str = 'cpu'):
        obs_t = torch.FloatTensor(np.array(self.obs)).to(device)
        actions_t = torch.LongTensor(self.actions).to(device)
        log_probs_t = torch.FloatTensor(self.log_probs).to(device)
        masks_t = torch.BoolTensor(np.array(self.masks)).to(device)
        kicker_faces_t = torch.LongTensor(np.array(self.kicker_faces)).to(device)    # (N, 3)
        kicker_avails_t = torch.FloatTensor(np.array(self.kicker_avails)).to(device) # (N, 13)
        return obs_t, actions_t, log_probs_t, masks_t, kicker_faces_t, kicker_avails_t
