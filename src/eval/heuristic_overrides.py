"""
启发式覆盖层 — 独立于神经网络策略的对局优化规则。

规则 — last_card_rule（对手剩 1 张牌）：
    优先打出张数最多的牌型；若多个合法动作张数相同，则取主牌面值最大的。
    等效语义：先出多牌（对手无法应对），再出单牌且从大到小，
    以最快速度清空手牌并防止对手用唯一的牌将我方的单牌压掉。
"""

import sys
import os
from typing import Optional

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.engine.action_generator import (
    _S_POCKET_INV,   # dict[int, (length, start_face)]
    _S_TRIPS_INV,    # dict[int, (length, kicker_mode, start_face)]
    _STRAIGHT_INV,   # dict[int, (length, start_face)]
    _IDX_PASS,
    _IDX_SINGLE_START,
    _IDX_POCKET_START,
    _IDX_S_POCKET_START,
    _IDX_TRIPS_START,
    _IDX_S_TRIPS_START,
    _IDX_QUADS_START,
    _IDX_STRAIGHT_START,
    _IDX_BOMB_START,
)

# ── 开关配置 ──────────────────────────────────────────────────────────────────

HEURISTIC_FLAGS: dict[str, bool] = {
    'last_card_rule': True,  # 规则：对手剩 1 张牌时的出牌策略
}


# ── 主入口 ────────────────────────────────────────────────────────────────────

def apply_heuristics(
    nn_action: int,
    nn_kickers: list[int],
    mask: np.ndarray,
    game,
) -> tuple[int, list[int]]:
    """对 NN 输出应用启发式覆盖，返回 (最终动作索引, 最终带牌面值列表)。

    若覆盖规则触发，返回的 kicker_faces 为空列表 []，调用方应将其视为
    "使用 greedy_lowest 自动选取带牌"（即传 kicker_faces=None 给 decode_action）。
    """
    opp_idx = 1 - game.current_player_index
    opp_hand_size = len(game.players[opp_idx].cards)

    # ── 规则：对手仅剩 1 张牌 ────────────────────────────────────────────────
    if HEURISTIC_FLAGS.get('last_card_rule', False) and opp_hand_size == 1:
        override = _last_card_rule(mask)
        if override is not None:
            return override, []

    return nn_action, nn_kickers


# ── 规则实现 ──────────────────────────────────────────────────────────────────

def _last_card_rule(mask: np.ndarray) -> Optional[int]:
    """选出合法动作中出牌张数最多的，同等张数取主牌面值最大的。

    在新一轮场景中，这等价于：先出多牌型，再出大到小的单牌。
    在跟牌场景中，同类型下选面值最大的（尽量让对手无法回应）。
    """
    return _max_cards_action(mask)


# ── 通用工具 ──────────────────────────────────────────────────────────────────

def _max_cards_action(mask: np.ndarray) -> Optional[int]:
    """在所有合法非 Pass 动作中，选出打出张数最多的；
    同等张数时取主牌面值最大的（tie-break）。
    """
    best_idx: Optional[int] = None
    best_count = 0
    best_face = -1

    for idx in range(1, 294):  # 跳过 Pass (0)
        if not mask[idx]:
            continue
        count = _action_card_count(idx)
        face = _action_primary_face(idx)
        if count > best_count or (count == best_count and face > best_face):
            best_count = count
            best_face = face
            best_idx = idx

    return best_idx


def _action_card_count(action_idx: int) -> int:
    """返回该动作打出的总牌张数（主牌 + 带牌数量，与 greedy_lowest 选取结果一致）。"""
    if action_idx == _IDX_PASS:
        return 0
    if _IDX_SINGLE_START <= action_idx <= 13:
        return 1
    if _IDX_POCKET_START <= action_idx <= 25:
        return 2
    if _IDX_S_POCKET_START <= action_idx <= 76:
        if action_idx in _S_POCKET_INV:
            length, _ = _S_POCKET_INV[action_idx]
            return length * 2
        return 0
    if _IDX_TRIPS_START <= action_idx <= 112:
        offset = action_idx - _IDX_TRIPS_START
        kicker_count = offset // 12
        return 3 + kicker_count
    if _IDX_S_TRIPS_START <= action_idx <= 202:
        if action_idx in _S_TRIPS_INV:
            length, kicker_mode, _ = _S_TRIPS_INV[action_idx]
            return length * 3 + kicker_mode * length
        return 0
    if _IDX_QUADS_START <= action_idx <= 246:
        offset = action_idx - _IDX_QUADS_START
        kicker_count = offset % 4
        return 4 + kicker_count
    if _IDX_STRAIGHT_START <= action_idx <= 282:
        if action_idx in _STRAIGHT_INV:
            length, _ = _STRAIGHT_INV[action_idx]
            return length
        return 0
    if _IDX_BOMB_START <= action_idx <= 293:
        return 4
    return 0


def _action_primary_face(action_idx: int) -> int:
    """返回该动作的主牌（起始）面值，用于同等张数的 tie-break。"""
    if action_idx == _IDX_PASS:
        return -1
    if _IDX_SINGLE_START <= action_idx <= 13:
        return action_idx - _IDX_SINGLE_START + 3
    if _IDX_POCKET_START <= action_idx <= 25:
        return action_idx - _IDX_POCKET_START + 3
    if _IDX_S_POCKET_START <= action_idx <= 76:
        if action_idx in _S_POCKET_INV:
            _, start_face = _S_POCKET_INV[action_idx]
            return start_face
        return -1
    if _IDX_TRIPS_START <= action_idx <= 112:
        offset = action_idx - _IDX_TRIPS_START
        return offset % 12 + 3
    if _IDX_S_TRIPS_START <= action_idx <= 202:
        if action_idx in _S_TRIPS_INV:
            _, _, start_face = _S_TRIPS_INV[action_idx]
            return start_face
        return -1
    if _IDX_QUADS_START <= action_idx <= 246:
        offset = action_idx - _IDX_QUADS_START
        return offset // 4 + 3
    if _IDX_STRAIGHT_START <= action_idx <= 282:
        if action_idx in _STRAIGHT_INV:
            _, start_face = _STRAIGHT_INV[action_idx]
            return start_face
        return -1
    if _IDX_BOMB_START <= action_idx <= 293:
        return action_idx - _IDX_BOMB_START + 3
    return -1
