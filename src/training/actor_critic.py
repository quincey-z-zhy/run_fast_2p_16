"""Actor-Critic 网络（共享主干 MLP）+ 带牌头（kicker head）。

两阶段动作设计：
  阶段 1：主头（294 维）— 选择牌型 / 起始面值 / 带牌数量
  阶段 2：带牌头（13 维）— 学习选取哪些具体带牌（面值 3-15，共 13 个）
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


class ActorCritic(nn.Module):
    """
    共享主干的 Actor-Critic MLP，附带 13 维带牌头。

    主头：obs_dim → hidden → hidden → 294（主动作分布）
    带牌头：hidden → 13（带牌面值偏好分，face 3-15 → idx 0-12）
    价值头：hidden → 1
    """

    def __init__(self, obs_dim: int = 80, act_dim: int = 294, hidden: int = 256, kicker_dim: int = 13):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.actor_head = nn.Linear(hidden, act_dim)
        self.critic_head = nn.Linear(hidden, 1)
        self.kicker_head = nn.Linear(hidden, kicker_dim)

        # 正交初始化：主头使用小 gain 以稳定早期训练
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.01 if m.out_features == act_dim else 1.0)
                nn.init.zeros_(m.bias)

    def forward(self, obs: torch.Tensor):
        h = self.shared(obs)
        logits = self.actor_head(h)
        value = self.critic_head(h)
        return logits, value

    def get_action_and_value(
        self,
        obs: torch.Tensor,
        action_mask: torch.Tensor,
        action: torch.Tensor | None = None,
    ):
        """
        获取主动作、log_prob、entropy、value（不含带牌阶段，向后兼容）。

        obs:         (B, obs_dim) float32
        action_mask: (B, 294) bool — True 表示合法
        action:      (B,) int64 或 None（None 时采样）
        """
        logits, value = self.forward(obs)
        logits = logits.masked_fill(~action_mask, float('-1e9'))

        dist = Categorical(logits=logits)
        if action is None:
            action = dist.sample()

        return action, dist.log_prob(action), dist.entropy(), value.squeeze(-1)

    def get_full_action_and_value(
        self,
        obs: torch.Tensor,
        action_mask: torch.Tensor,
        kicker_faces_t: torch.Tensor | None = None,
        kicker_avail_t: torch.Tensor | None = None,
        action: torch.Tensor | None = None,
    ):
        """
        单次前向传播，返回主动作 + 带牌 log_prob 之和 + entropy + value + kicker_logits。

        kicker_faces_t: (B, 3) int64，带牌面值索引（0-12），-1 表示该槽无带牌
        kicker_avail_t: (B, 13) float32，带牌选取前各 face 的可用张数
        当 kicker_faces_t/kicker_avail_t 为 None 时，带牌 log_prob 为 0（用于采样/推理）。

        :return: (action, log_prob, entropy, value, kicker_logits)
                 log_prob = log_prob_main + log_prob_kicker
        """
        h = self.shared(obs)
        logits = self.actor_head(h).masked_fill(~action_mask, float('-1e9'))
        value = self.critic_head(h).squeeze(-1)
        kicker_logits = self.kicker_head(h)  # (B, 13)

        dist = Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        log_prob_main = dist.log_prob(action)

        if kicker_faces_t is not None and kicker_avail_t is not None:
            log_prob_kicker = self._compute_kicker_log_probs_batch(
                kicker_logits, kicker_faces_t, kicker_avail_t
            )
        else:
            log_prob_kicker = torch.zeros_like(log_prob_main)

        return action, log_prob_main + log_prob_kicker, dist.entropy(), value, kicker_logits

    @torch.no_grad()
    def sample_kickers(
        self,
        kicker_logits_1d: torch.Tensor,
        avail_counts_13d: np.ndarray,
        kicker_count: int,
        deterministic: bool = False,
    ) -> tuple[list[int], float]:
        """
        从带牌头输出中顺序选取 kicker_count 张带牌。

        kicker_logits_1d: (13,) tensor，来自带牌头（已在目标设备上）
        avail_counts_13d: (13,) numpy int32，各 face 的可用张数（主牌面值已排除）
        kicker_count:     需要选取的带牌数
        deterministic:    True → argmax 选取（推理/评估用，确定性）
                          False → softmax 采样（训练 rollout 用，保留探索）

        :return: (kicker_faces, log_prob_sum)
                 kicker_faces: 升序排列的带牌面值列表（face 值 3-15）
                 log_prob_sum: 各步选取 log_prob 之和（float）
                              argmax 模式下仍按当前 softmax 概率计算，仅用于日志
        """
        if kicker_count == 0:
            return [], 0.0

        if isinstance(avail_counts_13d, np.ndarray):
            avail = torch.from_numpy(avail_counts_13d.astype(np.float32)).to(kicker_logits_1d.device)
        else:
            avail = avail_counts_13d.float().clone().to(kicker_logits_1d.device)
        kicker_face_indices: list[int] = []
        total_log_prob = 0.0

        for _ in range(kicker_count):
            avail_mask = avail > 0  # (13,)
            if not avail_mask.any():
                break  # 安全保险，理论上合法动作不应触发
            masked_logits = kicker_logits_1d.masked_fill(~avail_mask, float('-1e9'))
            d = Categorical(logits=masked_logits)
            if deterministic:
                idx = masked_logits.argmax()
            else:
                idx = d.sample()
            total_log_prob += d.log_prob(idx).item()
            kicker_face_indices.append(idx.item())
            avail[idx] -= 1

        # 还原为 face 值（索引 0 → face 3，索引 12 → face 15）
        kicker_faces = sorted(i + 3 for i in kicker_face_indices)
        return kicker_faces, total_log_prob

    def _compute_kicker_log_probs_batch(
        self,
        kicker_logits: torch.Tensor,
        kicker_faces_t: torch.Tensor,
        kicker_avail_t: torch.Tensor,
    ) -> torch.Tensor:
        """
        批量重算带牌选择的 log_prob（用于 PPO 更新阶段的策略比值计算）。

        kicker_logits:  (B, 13) float32，带牌头输出
        kicker_faces_t: (B, 3) int64，存储的带牌面值索引（0-12），-1 表示该槽未使用
        kicker_avail_t: (B, 13) float32，带牌选取前各 face 的可用张数

        :return: (B,) float32 — 每个样本的带牌选择 log_prob 之和
        """
        log_prob = torch.zeros(kicker_logits.shape[0], device=kicker_logits.device)
        avail = kicker_avail_t.float().clone()  # (B, 13)

        for slot in range(8):  # 最多 8 个带牌槽（s_trips length=4 × mode=2）
            faces = kicker_faces_t[:, slot]           # (B,) 值为 0-12 或 -1
            has_kicker = faces >= 0                    # (B,) bool

            if not has_kicker.any():
                break  # 本槽及后续槽均无带牌

            avail_mask = avail > 0                     # (B, 13)
            masked_logits = kicker_logits.masked_fill(~avail_mask, float('-1e9'))
            log_softmax = F.log_softmax(masked_logits, dim=-1)  # (B, 13)

            face_idx = faces.clamp(min=0)              # (B,) 防止 -1 越界 gather
            slot_log_prob = log_softmax.gather(1, face_idx.unsqueeze(1)).squeeze(1)  # (B,)

            # 仅当该槽有带牌时才累加 log_prob
            log_prob = log_prob + slot_log_prob * has_kicker.float()

            # 更新可用张数：选中的 face 减 1
            face_one_hot = F.one_hot(face_idx, num_classes=13).float()       # (B, 13)
            avail = avail - face_one_hot * has_kicker.float().unsqueeze(1)

        return log_prob
