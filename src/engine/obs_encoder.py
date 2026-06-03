"""
观测编码器：将游戏状态编码为 80 维 float32 观测向量，
并提供 Action → 动作索引 的反向编码功能（用于测试/回放）。

观测维度布局：
  dim  0-12 : own_hand[face] / DECK_COUNTS[face]，face 3..15
  dim 13-25 : remain_cards[face] / DECK_COUNTS[face]，face 3..15
  dim 26-34 : last_action formation one-hot（9种牌型按 FORMATIONS 值升序）
  dim 35-47 : last_action start_card one-hot，face 3..15
  dim 48-60 : last_action end_card one-hot，face 3..15（非顺牌型/新轮时全零）
  dim 61-64 : last_action num_kicker one-hot，0/1/2/3
  dim    65 : is_new_round（0.0 或 1.0）
  dim    66 : current_player_index（0.0 或 1.0）
  dim 67-79 : opp_pass_upper_bound[face] / DECK_COUNTS[face]，face 3..15
              对手历次 Pass 推断出的各面值张数上界（归一化）。
              初始均为 1.0；对手每次 Pass 后收紧：
                - 对手 Pass 单张 X → face > X 均降为 0.0
                - 对手 Pass 对子 X → face ∈ (X, 13] 降至 ≤ 1/DECK_COUNTS
                - 对手 Pass 三张 X → face ∈ (X, 13] 降至 ≤ 2/DECK_COUNTS
                - 对手 Pass 炸弹/四带 X → face ∈ (X, 12] 降至 ≤ 3/DECK_COUNTS
"""

import sys
import os

# 确保可以导入 game 包
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
from game.config import FORMATIONS, DECK_COUNTS, QUAD_FACES, POCKET_TRIPLE_FACES
from game.action import Action
from src.engine.action_generator import (
    _S_POCKET_MAP, _S_TRIPS_MAP, _STRAIGHT_MAP,
    _IDX_PASS, _IDX_SINGLE_START, _IDX_POCKET_START,
    _IDX_S_POCKET_START, _IDX_TRIPS_START, _IDX_S_TRIPS_START,
    _IDX_QUADS_START, _IDX_STRAIGHT_START, _IDX_BOMB_START,
)

# FORMATIONS 值升序排列（用于 one-hot 索引）
_FORMATION_VALUES_SORTED: list[int] = sorted(FORMATIONS.values())
# {formation_value: one-hot 索引}
_FORMATION_TO_IDX: dict[int, int] = {v: i for i, v in enumerate(_FORMATION_VALUES_SORTED)}

# face 3..15 共 13 个
_FACES_LIST: list[int] = list(range(3, 16))
# {face: 在 _FACES_LIST 中的索引}
_FACE_TO_IDX: dict[int, int] = {f: i for i, f in enumerate(_FACES_LIST)}

# 新增维度偏移量
_OPP_PASS_UB_OFFSET: int = 67  # dim 67-79：对手 Pass 推断上界

# obs 总维度
OBS_DIM: int = 80


def _compute_opp_pass_upper_bound(game) -> np.ndarray:
    """扫描历史动作，计算对手 Pass 推断的各面值持有数上界（归一化）。

    当对手 Pass 时，意味着他手中没有能压过桌面牌的牌，可据此收紧对对手
    手牌的估计。

    推断规则（"出得过就要出"）：
      - 对手 Pass 单张 X → 对手没有 face > X 的任何牌（上界降为 0）
      - 对手 Pass 对子 X → 对手每个 face ∈ (X, 13] 至多 1 张
      - 对手 Pass 三张 X → 对手每个 face ∈ (X, 13] 至多 2 张
      - 对手 Pass 炸弹/四带 X → 对手每个 face ∈ (X, 12] 至多 3 张
      - 连对/飞机/顺子：涉及多面联合约束，本版本暂不处理

    历史约束单调有效：对手 Pass 后只能打出牌（不会增牌），约束持续成立。

    :param game: 当前 Game 对象（is_end=False）
    :return: shape=(13,) float32，opp_pass_ub[i] = 上界 / DECK_COUNTS[face i+3]
             初始值 1.0（无约束），Pass 后逐步收紧。
    """
    ub = np.ones(13, dtype=np.float32)

    if not game.actions:
        return ub

    n = len(game.actions)
    current_idx: int = game.current_player_index
    # 先手玩家推导：经过 n 次交替后，当前玩家 = first_player XOR (n%2)
    # 故 first_player = current_idx XOR (n%2)
    first_player: int = current_idx ^ (n % 2)
    opp_idx: int = 1 - current_idx

    prev_played = None  # 最近一次非 Pass 动作（对手须超越的桌面牌）
    for i, action in enumerate(game.actions):
        actor = first_player if i % 2 == 0 else 1 - first_player
        if action.formation == FORMATIONS['none']:  # Pass
            if actor == opp_idx and prev_played is not None:
                _apply_pass_bound(ub, prev_played)
        else:
            prev_played = action

    return ub


def _apply_pass_bound(ub: np.ndarray, card) -> None:
    """根据对手 Pass 时的桌面牌，原地收紧上界数组。

    :param ub:   shape=(13,) float32，索引 i 对应 face i+3
    :param card: 对手须超越但未能超越的 Action 对象
    """
    f = card.formation
    x = card.start_card
    if x is None:
        return

    if f == FORMATIONS['single']:
        # 对手没有任何 face > x 的牌
        for face in range(x + 1, 16):
            ub[face - 3] = 0.0

    elif f == FORMATIONS['pocket']:
        # 对手每个 face ∈ (x, 14] 至多 1 张（无法成对）
        for face in range(x + 1, 15):  # 对子最高 A(14)（16-hand 变种）
            idx = face - 3
            bound = 1.0 / DECK_COUNTS[face]
            if bound < ub[idx]:
                ub[idx] = bound

    elif f == FORMATIONS['trips']:
        # 对手每个 face ∈ (x, 14] 至多 2 张（无法成三条）
        for face in range(x + 1, 15):
            idx = face - 3
            bound = 2.0 / DECK_COUNTS[face]
            if bound < ub[idx]:
                ub[idx] = bound

    elif f == FORMATIONS['quads'] or f == FORMATIONS['bomb']:
        # 对手每个 face ∈ (x, 13] 至多 3 张（无法成四张/炸弹）
        for face in range(x + 1, 14):  # 炸弹/四带最高 K(13)（16-hand 变种）
            idx = face - 3
            bound = 3.0 / DECK_COUNTS[face]
            if bound < ub[idx]:
                ub[idx] = bound

    # s_pocket / s_trips / straight 涉及多面联合约束，暂不处理


def encode_obs(game) -> np.ndarray:
    """从当前玩家视角编码 80 维 float32 观测向量。

    :param game: 游戏对象（Game 实例）
    :return: shape=(80,), dtype=float32 的观测向量
    """
    obs = np.zeros(OBS_DIM, dtype=np.float32)

    current_idx: int = game.current_player_index
    current_player = game.players[current_idx]

    # ── dim 0-12: 当前玩家手牌比例 ────────────────────────────────────────────
    hand_counts: dict[int, int] = {}
    for card in current_player.cards:
        hand_counts[card] = hand_counts.get(card, 0) + 1

    for i, face in enumerate(_FACES_LIST):
        deck_count = DECK_COUNTS[face]
        obs[i] = hand_counts.get(face, 0) / deck_count

    # ── dim 13-25: 剩余牌比例 ─────────────────────────────────────────────────
    for i, face in enumerate(_FACES_LIST):
        deck_count = DECK_COUNTS[face]
        obs[13 + i] = game.remain_cards.get(face, 0) / deck_count

    # ── is_new_round 标志 ─────────────────────────────────────────────────────
    is_new_round: bool = game.is_new_round
    obs[65] = 1.0 if is_new_round else 0.0

    # ── current_player_index ──────────────────────────────────────────────────
    obs[66] = float(current_idx)

    # ── dim 26-64: last_action 特征（is_new_round 时全零） ──────────────────────
    if not is_new_round:
        prev: Action = game.previous_action
        if prev is not None and prev.formation != FORMATIONS['none']:
            # dim 26-34: formation one-hot
            f_idx = _FORMATION_TO_IDX.get(prev.formation)
            if f_idx is not None:
                obs[26 + f_idx] = 1.0

            # dim 35-47: start_card one-hot
            if prev.start_card is not None:
                sc_idx = _FACE_TO_IDX.get(prev.start_card)
                if sc_idx is not None:
                    obs[35 + sc_idx] = 1.0

            # dim 48-60: end_card one-hot（仅顺牌型）
            if prev.end_card is not None:
                ec_idx = _FACE_TO_IDX.get(prev.end_card)
                if ec_idx is not None:
                    obs[48 + ec_idx] = 1.0

            # dim 61-64: num_kicker one-hot（0/1/2/3）
            nk = min(prev.num_kicker, 3)
            obs[61 + nk] = 1.0

    # ── dim 67-79: 对手 Pass 推断上界 ─────────────────────────────────────────
    opp_ub = _compute_opp_pass_upper_bound(game)
    obs[_OPP_PASS_UB_OFFSET: _OPP_PASS_UB_OFFSET + 13] = opp_ub

    return obs


def encode_action(action: Action, game) -> int:
    """将 Action 对象编码为动作索引。

    对于带牌动作，只匹配 kicker_count，不匹配具体带牌内容。

    :param action: Action 对象
    :param game:   当前游戏状态（保留参数，便于扩展；当前未使用）
    :return: 动作索引 [0, 293]；若动作不在 294 动作空间内则返回 -1
    """
    try:
        if not action.is_legal:
            return -1
    except Exception:
        return -1

    f = action.formation

    # Pass
    if f == FORMATIONS['none']:
        return _IDX_PASS

    # Single: idx 1-13, face 3..15
    if f == FORMATIONS['single']:
        sc = action.start_card
        if sc is None or sc < 3 or sc > 15:
            return -1
        return _IDX_SINGLE_START + (sc - 3)

    # Pocket: idx 14-25, face 3..14
    if f == FORMATIONS['pocket']:
        sc = action.start_card
        if sc is None or sc < 3 or sc > 14:
            return -1
        return _IDX_POCKET_START + (sc - 3)

    # S_pocket: idx 26-76
    if f == FORMATIONS['s_pocket']:
        sc, ec = action.start_card, action.end_card
        if sc is None or ec is None:
            return -1
        length = ec - sc + 1
        idx = _S_POCKET_MAP.get((length, sc))
        return idx if idx is not None else -1

    # Trips: idx 77-112, kicker_count ∈ {0,1,2}
    if f == FORMATIONS['trips']:
        sc = action.start_card
        if sc is None or sc < 3 or sc > 14:
            return -1
        kicker_count = action.num_kicker
        if kicker_count > 2:
            return -1
        return _IDX_TRIPS_START + kicker_count * 12 + (sc - 3)

    # S_trips: idx 113-202
    if f == FORMATIONS['s_trips']:
        sc, ec = action.start_card, action.end_card
        if sc is None or ec is None:
            return -1
        length = ec - sc + 1
        kicker_count = action.num_kicker
        # 判断 kicker_mode
        if kicker_count == 0:
            kicker_mode = 0
        elif kicker_count == length:
            kicker_mode = 1
        elif kicker_count == 2 * length:
            kicker_mode = 2
        else:
            return -1
        idx = _S_TRIPS_MAP.get((length, kicker_mode, sc))
        return idx if idx is not None else -1

    # Quads: idx 203-246, face 3..13, kicker_count 0..3
    if f == FORMATIONS['quads']:
        sc = action.start_card
        if sc is None or sc < 3 or sc > 13:
            return -1
        kicker_count = action.num_kicker
        if kicker_count > 3:
            return -1
        return _IDX_QUADS_START + (sc - 3) * 4 + kicker_count

    # Straight: idx 247-282
    if f == FORMATIONS['straight']:
        sc, ec = action.start_card, action.end_card
        if sc is None or ec is None:
            return -1
        length = ec - sc + 1
        idx = _STRAIGHT_MAP.get((length, sc))
        return idx if idx is not None else -1

    # Bomb: idx 283-293, face 3..13
    if f == FORMATIONS['bomb']:
        sc = action.start_card
        if sc is None or sc < 3 or sc > 13:
            return -1
        return _IDX_BOMB_START + (sc - 3)

    return -1
