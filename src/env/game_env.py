"""
两人跑得快 Gymnasium 环境。

观测空间：80 维 float32 向量（从当前玩家视角）
动作空间：Discrete(294)
合法动作掩码通过 info["action_mask"] 传出（294 维 bool 数组）
"""

import sys
import os

# 将项目根目录加入 sys.path，确保 game 包可被导入
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import random as _random

import numpy as np
import gymnasium

from game.cards import shuffle
from game.game import Game
from game.player import Player
from game.config import CARD_TO_CHAR

from src.engine.action_generator import compute_action_mask, decode_action
from src.engine.obs_encoder import encode_obs


class RunFast2pEnv(gymnasium.Env):
    """两人跑得快纸牌游戏的 Gymnasium 环境。

    每一步由当前玩家选取一个动作（动作索引 0-293）。
    游戏结束时返回当前玩家的最终得分作为 reward。
    中间步 reward 为 0.0。
    """

    metadata = {'render_modes': ['human', 'ansi']}

    def __init__(self, render_mode: str | None = None):
        super().__init__()

        # 观测空间：80 维 float32，理论范围 [0, 1]
        self.observation_space = gymnasium.spaces.Box(
            low=0.0, high=1.0, shape=(80,), dtype=np.float32
        )

        # 动作空间：离散 294 个动作（16-hand 变种）
        self.action_space = gymnasium.spaces.Discrete(294)

        self.render_mode: str | None = render_mode

        # 游戏对象（reset 后初始化）
        self._game: Game | None = None
        self._players: list[Player] | None = None

    # ── 公开接口 ─────────────────────────────────────────────────────────────

    def reset(
            self,
            seed: int | None = None,
            options: dict | None = None,
    ) -> tuple[np.ndarray, dict]:
        """初始化新一局游戏。

        :param seed:    随机种子（传给 super().reset 以设置 np_random）
        :param options: 可含 first_player (0/1/None)，控制先手玩家
        :return: (obs, info)
                 info 含：
                   action_mask      (294, bool)   合法动作掩码
                   current_player   int           当前玩家索引
                   hand_size_self   int           当前玩家手牌数
                   hand_size_opponent int         对手手牌数
        """
        super().reset(seed=seed)

        # 同步 Python random 模块，确保 cards.shuffle() 可复现
        if seed is not None:
            _random.seed(seed)

        # 解析 options
        first_player: int | None = None
        if options is not None:
            first_player = options.get('first_player', None)

        # 洗牌发牌
        p0_cards, p1_cards = shuffle()
        self._players = [Player(p0_cards), Player(p1_cards)]

        # 创建并启动游戏
        self._game = Game()
        self._game.start(self._players, first_player=first_player)

        obs, info = self._get_obs_and_info()

        if self.render_mode == 'human':
            self.render()

        return obs, info

    def step(self, action: int, kicker_faces: list[int] | None = None) -> tuple[np.ndarray, float, bool, bool, dict]:
        """执行一个动作，推进游戏状态。

        :param action:       动作索引 [0, 293]
        :param kicker_faces: 由带牌头指定的带牌面值列表（可选）；
                             为 None 时退回 greedy_lowest 策略
        :return: (obs, reward, terminated, truncated, info)
                 - reward:     终局时为当前玩家最终得分；非法动作返回 -1；其余步为 0.0
                 - terminated: 游戏是否结束
                 - truncated:  始终 False
                 - info 含：
                     action_mask      (294, bool)
                     current_player   int
                     illegal_action   bool   该步是否为非法动作
        """
        assert self._game is not None, "请先调用 reset() 初始化环境"

        # 计算当前合法掩码
        mask = compute_action_mask(self._game)

        # 非法动作处理：不推进游戏，返回惩罚 reward
        if not mask[action]:
            obs, info = self._get_obs_and_info()
            info['illegal_action'] = True
            return obs, -1.0, False, False, info

        # 解码动作（带牌面值由外部指定时覆盖 greedy_lowest）
        act_obj = decode_action(action, self._game, kicker_faces=kicker_faces)
        if act_obj is None:
            # 理论上不应到达此分支（掩码已过滤），保守处理
            obs, info = self._get_obs_and_info()
            info['illegal_action'] = True
            return obs, -1.0, False, False, info

        # 记录执行前的当前玩家（得分归属）
        acting_player_idx: int = self._game.current_player_index

        # 执行动作
        self._game.act(act_obj)

        terminated: bool = self._game.is_end
        reward: float = 0.0

        if terminated:
            # 返回执行动作的玩家得分
            reward = float(self._game.players[acting_player_idx].score)

        obs, info = self._get_obs_and_info()
        info['illegal_action'] = False
        if terminated:
            # 暴露双方最终得分，供训练代码获取对手视角 reward
            info['final_rewards'] = [float(p.score) for p in self._game.players]

        if self.render_mode == 'human':
            self.render()

        return obs, reward, terminated, False, info

    def render(self) -> str | None:
        """渲染当前游戏状态。

        human 模式：打印到 stdout
        ansi  模式：返回字符串
        """
        if self._game is None:
            return None

        lines: list[str] = []
        lines.append("=" * 50)
        lines.append(f"当前玩家：P{self._game.current_player_index}")

        for i, player in enumerate(self._game.players):
            hand_str = ''.join(CARD_TO_CHAR[c] for c in player.cards)
            lines.append(f"  P{i} 手牌({len(player.cards)}张)：{hand_str}")

        # 最近一次出牌
        if self._game.actions:
            last_act = self._game.actions[-1]
            from game.config import FORMATION_NAMES
            fname = FORMATION_NAMES.get(last_act.formation, str(last_act.formation))
            try:
                cards_str = str(last_act) if last_act.formation != 0 else "（不出）"
            except Exception:
                cards_str = "（无法显示）"
            lines.append(f"  上一动作：{fname} {cards_str}")

        # 剩余牌摘要（仅显示还剩多张的面值）
        remain_summary = {
            CARD_TO_CHAR[f]: cnt
            for f, cnt in self._game.remain_cards.items()
            if cnt > 0
        }
        lines.append(f"  剩余牌摘要：{remain_summary}")
        lines.append(f"  is_new_round={self._game.is_new_round}")
        lines.append("=" * 50)

        output = "\n".join(lines)
        if self.render_mode == 'human':
            print(output)
            return None
        return output

    # ── 内部方法 ─────────────────────────────────────────────────────────────

    def _get_obs_and_info(self) -> tuple[np.ndarray, dict]:
        """返回当前游戏状态的观测向量和 info 字典。"""
        obs = encode_obs(self._game)
        mask = compute_action_mask(self._game)

        current_idx = self._game.current_player_index
        opponent_idx = 1 - current_idx

        info: dict = {
            'action_mask':         mask,
            'current_player':      current_idx,
            'hand_size_self':      len(self._game.players[current_idx].cards),
            'hand_size_opponent':  len(self._game.players[opponent_idx].cards),
        }
        return obs, info
