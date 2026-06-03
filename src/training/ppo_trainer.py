"""PPO 更新器。管理策略和价值函数的梯度更新。"""

import numpy as np
import torch
import torch.nn.functional as F

from .actor_critic import ActorCritic
from .rollout_buffer import RolloutBuffer


class PPOTrainer:
    """执行 PPO 梯度更新（clipped surrogate + value loss + entropy bonus）。"""

    def __init__(
        self,
        model: ActorCritic,
        optimizer: torch.optim.Optimizer,
        n_epochs: int,
        batch_size: int,
        clip_eps: float = 0.2,
        value_coeff: float = 0.5,
        entropy_coeff: float = 0.01,
        max_grad_norm: float = 0.5,
        normalize_advantages: bool = True,
        device: str = 'cpu',
    ):
        self.model = model
        self.optimizer = optimizer
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.clip_eps = clip_eps
        self.value_coeff = value_coeff
        self.entropy_coeff = entropy_coeff
        self.max_grad_norm = max_grad_norm
        self.normalize_advantages = normalize_advantages
        self.device = device

    def update(
        self,
        buffer: RolloutBuffer,
        returns: np.ndarray,
        advantages: np.ndarray,
    ) -> dict:
        """执行 n_epochs 轮 PPO 更新，返回平均损失统计。"""
        obs_t, actions_t, old_log_probs_t, masks_t, kicker_faces_t, kicker_avails_t = \
            buffer.to_tensors(self.device)
        returns_t = torch.FloatTensor(returns).to(self.device)
        advantages_t = torch.FloatTensor(advantages).to(self.device)

        n = len(buffer)
        indices = np.arange(n)

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        total_updates = 0

        for _ in range(self.n_epochs):
            np.random.shuffle(indices)
            for start in range(0, n, self.batch_size):
                mb = indices[start:start + self.batch_size]

                mb_advantages = advantages_t[mb]
                if self.normalize_advantages:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                        mb_advantages.std() + 1e-8
                    )

                _, new_log_probs, entropy, new_values, _ = self.model.get_full_action_and_value(
                    obs_t[mb], masks_t[mb],
                    kicker_faces_t=kicker_faces_t[mb],
                    kicker_avail_t=kicker_avails_t[mb],
                    action=actions_t[mb],
                )

                # Clipped surrogate policy objective
                ratio = torch.exp(new_log_probs - old_log_probs_t[mb])
                policy_loss = -torch.min(
                    ratio * mb_advantages,
                    torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * mb_advantages,
                ).mean()

                # Value function MSE loss
                value_loss = F.mse_loss(new_values, returns_t[mb])

                # Entropy bonus（最大化探索）
                entropy_loss = -entropy.mean()

                loss = (
                    policy_loss
                    + self.value_coeff * value_loss
                    + self.entropy_coeff * entropy_loss
                )

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.mean().item()
                total_updates += 1

        count = max(total_updates, 1)
        return {
            'policy_loss': total_policy_loss / count,
            'value_loss': total_value_loss / count,
            'entropy': total_entropy / count,
        }
