"""
基线对手实现。

RandomLegalAgent: 从合法动作中均匀随机采样
GreedyLowAgent:   优先出面值最低的合法牌型，尽量不 pass
"""

import random
import sys
import os

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np

from src.engine.action_generator import decode_action


class RandomLegalAgent:
    """从合法动作中均匀随机采样。"""

    def act(self, obs: np.ndarray, action_mask: np.ndarray, game=None) -> int:
        legal = np.where(action_mask)[0].tolist()
        return random.choice(legal)


class GreedyLowAgent:
    """优先出面值最低的合法牌型（尽量不 pass）。

    决策优先级：
    1. 若有非-pass 合法动作 → 选面值最小的（按动作索引升序，索引小对应更小面值）
    2. 仅有 pass 合法 → 选 pass（idx=0）
    """

    def act(self, obs: np.ndarray, action_mask: np.ndarray, game=None) -> int:
        legal = np.where(action_mask)[0].tolist()
        # 索引 0 = pass；非 pass 动作索引 1-293 按面值从小到大粗略排列
        non_pass = [a for a in legal if a != 0]
        if non_pass:
            return min(non_pass)  # 最小索引对应最低面值牌型
        return 0  # 只能 pass
