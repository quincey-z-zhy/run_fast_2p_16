"""
动作生成器：负责动作空间的编码/解码与合法性掩码计算。

动作空间共 294 个动作，索引映射见项目规格文档。
带牌（kicker）由 greedy_lowest 规则自动选取：从剩余手牌中选面值最低的若干张。
"""

import sys
import os

# 确保可以导入 game 包
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
from game.config import (
    FORMATIONS, DECK_COUNTS,
    POCKET_TRIPLE_FACES, QUAD_FACES, STRAIGHT_FACES,
    MIN_S_POCKET_LEN, MAX_S_POCKET_LEN,
    MIN_S_TRIPS_LEN, MAX_S_TRIPS_LEN,
    MIN_STRAIGHT_LEN, MAX_STRAIGHT_LEN,
)
from game.action import Action

# ── 动作空间各分段起始索引（与规格严格对应） ─────────────────────────────────

_IDX_PASS = 0                  # idx 0: Pass
_IDX_SINGLE_START = 1          # idx 1-13:   Single
_IDX_POCKET_START = 14         # idx 14-25:  Pocket (12 faces: 3..A)
_IDX_S_POCKET_START = 26       # idx 26-76:  S_pocket (51)
_IDX_TRIPS_START = 77          # idx 77-112: Trips  (3 modes × 12 faces)
_IDX_S_TRIPS_START = 113       # idx 113-202: S_trips (90)
_IDX_QUADS_START = 203         # idx 203-246: Quads (11 faces × 4 kicker counts)
_IDX_STRAIGHT_START = 247      # idx 247-282: Straight (36)
_IDX_BOMB_START = 283          # idx 283-293: Bomb (11 faces: 3..K)

_ACTION_SIZE = 294             # 动作空间总大小


# ── S_pocket 索引构建（预计算 length×start_face → idx 映射） ──────────────────

def _build_s_pocket_map() -> dict[tuple[int, int], int]:
    """构建 (length, start_face) → 动作索引 的查找表（s_pocket 分段）。"""
    mapping: dict[tuple[int, int], int] = {}
    idx = _IDX_S_POCKET_START
    # length 2..7
    for length in range(2, 8):
        # start_face 3..(15-length)，确保 end_face=start_face+length-1 ≤ 14(A)
        # A 在 16-hand 变种中有 3 张，可以入连对
        for start_face in range(3, 16 - length):
            mapping[(length, start_face)] = idx
            idx += 1
    return mapping


_S_POCKET_MAP: dict[tuple[int, int], int] = _build_s_pocket_map()
# 反向：idx → (length, start_face)
_S_POCKET_INV: dict[int, tuple[int, int]] = {v: k for k, v in _S_POCKET_MAP.items()}


# ── S_trips 索引构建 ─────────────────────────────────────────────────────────

def _build_s_trips_map() -> dict[tuple[int, int, int], int]:
    """构建 (length, kicker_mode, start_face) → 动作索引 的查找表（s_trips 分段）。

    kicker_mode: 0=带0根, 1=带N根, 2=带2N根（N为length）
    """
    mapping: dict[tuple[int, int, int], int] = {}
    idx = _IDX_S_TRIPS_START
    # length 2,3,4
    for length in range(2, 5):
        # start_face 3..(14-length+1)，确保 end_face=start_face+length-1 ≤ 14(A)
        # A 在 16-hand 变种中有 3 张，可以入飞机
        # len=2: 11个(3..13), len=3: 10个(3..12), len=4: 9个(3..11)
        num_starts = 13 - length  # len=2→11, len=3→10, len=4→9
        for kicker_mode in range(3):
            for start_face in range(3, 3 + num_starts):
                mapping[(length, kicker_mode, start_face)] = idx
                idx += 1
    return mapping


_S_TRIPS_MAP: dict[tuple[int, int, int], int] = _build_s_trips_map()
# 反向：idx → (length, kicker_mode, start_face)
_S_TRIPS_INV: dict[int, tuple[int, int, int]] = {v: k for k, v in _S_TRIPS_MAP.items()}


# ── Straight 索引构建 ────────────────────────────────────────────────────────

def _build_straight_map() -> dict[tuple[int, int], int]:
    """构建 (length, start_face) → 动作索引 的查找表（straight 分段）。"""
    mapping: dict[tuple[int, int], int] = {}
    idx = _IDX_STRAIGHT_START
    # length 5..12
    for length in range(5, 13):
        # start_face 3..(15-length)，end_face=start_face+length-1 ≤ 14(A)
        for start_face in range(3, 16 - length):
            mapping[(length, start_face)] = idx
            idx += 1
    return mapping


_STRAIGHT_MAP: dict[tuple[int, int], int] = _build_straight_map()
# 反向：idx → (length, start_face)
_STRAIGHT_INV: dict[int, tuple[int, int]] = {v: k for k, v in _STRAIGHT_MAP.items()}


# ── 公开函数 ─────────────────────────────────────────────────────────────────

def get_hand_counts(cards: list[int]) -> dict[int, int]:
    """返回手牌中每种 face 的数量。"""
    counts: dict[int, int] = {}
    for card in cards:
        counts[card] = counts.get(card, 0) + 1
    return counts


def select_kickers(
        hand_counts: dict[int, int],
        main_faces: set[int],
        kicker_count: int,
) -> list[int] | None:
    """greedy_lowest 策略：从手牌中排除主牌面值后，按面值从低到高选取 kicker_count 张。

    :param hand_counts: 手牌各 face 的数量（不会被修改）
    :param main_faces:  主牌所占用的 face 集合（这些 face 不能作为带牌）
    :param kicker_count: 需要选取的带牌张数
    :return: 升序排列的带牌列表；若手牌不足则返回 None
    """
    if kicker_count == 0:
        return []

    # 可用带牌：排除主牌 face，按 face 升序枚举；每种 face 最多取 3 张（防止炸弹进入带牌）
    available: list[int] = []
    for face in sorted(hand_counts.keys()):
        if face in main_faces:
            continue
        available.extend([face] * min(hand_counts[face], 3))

    if len(available) < kicker_count:
        return None

    return sorted(available[:kicker_count])


def get_kicker_info(
        action_idx: int,
        hand_counts: dict[int, int],
) -> tuple[int, set[int], np.ndarray]:
    """提取动作所需的带牌信息，供带牌头（kicker head）使用。

    :param action_idx:  动作索引 [0, 293]
    :param hand_counts: 手牌各 face 的数量
    :return: (kicker_count, main_faces, avail_counts_13d)
             kicker_count:     需要选取的带牌张数（0 表示无带牌）
             main_faces:       主牌面值集合（带牌不能包含这些面值）
             avail_counts_13d: shape=(13,) int32，可用带牌张数；
                               索引 i 对应 face=i+3（face 3..15 → idx 0..12）
    """
    avail_counts = np.zeros(13, dtype=np.int32)
    kicker_count = 0
    main_faces: set[int] = set()

    # trips: idx 77-112，kicker_count = 0/1/2
    if _IDX_TRIPS_START <= action_idx <= 112:
        offset = action_idx - _IDX_TRIPS_START
        kicker_count = offset // 12
        face = offset % 12 + 3
        main_faces = {face}

    # s_trips: idx 113-202，kicker_count = kicker_mode * length
    elif _IDX_S_TRIPS_START <= action_idx <= 202:
        if action_idx in _S_TRIPS_INV:
            length, kicker_mode, start_face = _S_TRIPS_INV[action_idx]
            kicker_count = kicker_mode * length
            main_faces = set(range(start_face, start_face + length))

    # quads: idx 203-246，kicker_count = 0/1/2/3
    elif _IDX_QUADS_START <= action_idx <= 246:
        offset = action_idx - _IDX_QUADS_START
        face = offset // 4 + 3
        kicker_count = offset % 4
        main_faces = {face}

    # 所有其他牌型无带牌（单牌/对子/连对/顺子/炸弹）
    if kicker_count > 0:
        for i, f in enumerate(range(3, 16)):  # face 3..15 → idx 0..12
            if f not in main_faces:
                avail_counts[i] = hand_counts.get(f, 0)

    return kicker_count, main_faces, avail_counts


def compute_action_mask(game) -> np.ndarray:
    """计算 294 维合法动作掩码。

    规则：
    - is_new_round=True：Pass 非法，其余需要手牌可用且动作格式合法
    - is_new_round=False（跟牌）：
        * Pass 仅当无任何合法跟牌（非炸弹）时合法（必须跟牌规则）
        * 炸弹：只要有该 face 4 张即合法（可压制任何非炸弹）
        * 非炸弹：formation/length/num_kicker 须与 previous_action 完全一致，
          且 start_card 须大于 previous_action.start_card，且带牌可选出
    :return: shape=(294,), dtype=bool 的掩码数组
    """
    mask = np.zeros(_ACTION_SIZE, dtype=bool)

    current_player = game.players[game.current_player_index]
    hand_counts = get_hand_counts(current_player.cards)
    is_new_round: bool = game.is_new_round
    prev: Action | None = game.previous_action

    # ── idx 0: Pass（在所有动作计算完毕后，仅无合法跟牌时允许）────────────────
    # 占位，最后统一处理

    # ── idx 1-13: Single ─────────────────────────────────────────────────────
    for face in range(3, 16):
        if hand_counts.get(face, 0) >= 1:
            action = Action(FORMATIONS['single'], start_card=face)
            if _check_beats_prev(action, prev, is_new_round):
                mask[_IDX_SINGLE_START + (face - 3)] = True

    # ── idx 14-25: Pocket ────────────────────────────────────────────────────
    for face in POCKET_TRIPLE_FACES:  # 3..14
        if hand_counts.get(face, 0) >= 2:
            action = Action(FORMATIONS['pocket'], start_card=face)
            if _check_beats_prev(action, prev, is_new_round):
                mask[_IDX_POCKET_START + (face - 3)] = True

    # ── idx 26-76: S_pocket ──────────────────────────────────────────────────
    for (length, start_face), idx in _S_POCKET_MAP.items():
        end_face = start_face + length - 1
        # 检查每个 face 是否至少有 2 张
        if all(hand_counts.get(f, 0) >= 2 for f in range(start_face, end_face + 1)):
            action = Action(FORMATIONS['s_pocket'], start_card=start_face, end_card=end_face)
            if _check_beats_prev(action, prev, is_new_round):
                mask[idx] = True

    # ── idx 77-112: Trips (kicker 0/1/2) ─────────────────────────────────────
    for face in POCKET_TRIPLE_FACES:  # 3..14
        if hand_counts.get(face, 0) >= 3:
            main_faces = {face}
            for kicker_count in range(3):  # 0, 1, 2
                base_idx = _IDX_TRIPS_START + kicker_count * 12 + (face - 3)
                kickers = select_kickers(hand_counts, main_faces, kicker_count)
                if kickers is not None:
                    action = Action(FORMATIONS['trips'], start_card=face, tail_cards=kickers)
                    if _check_beats_prev(action, prev, is_new_round):
                        mask[base_idx] = True

    # ── idx 113-202: S_trips ─────────────────────────────────────────────────
    for (length, kicker_mode, start_face), idx in _S_TRIPS_MAP.items():
        end_face = start_face + length - 1
        # 检查主牌：每个 face 至少 3 张
        if not all(hand_counts.get(f, 0) >= 3 for f in range(start_face, end_face + 1)):
            continue
        main_faces = set(range(start_face, end_face + 1))
        # kicker_mode: 0→带0张, 1→带N张, 2→带2N张
        kicker_count = kicker_mode * length  # 0, N, 2N
        kickers = select_kickers(hand_counts, main_faces, kicker_count)
        if kickers is not None:
            action = Action(
                FORMATIONS['s_trips'],
                start_card=start_face,
                end_card=end_face,
                tail_cards=kickers,
            )
            if _check_beats_prev(action, prev, is_new_round):
                mask[idx] = True

    # ── idx 203-246: Quads ───────────────────────────────────────────────────
    for face in QUAD_FACES:  # 3..13
        if hand_counts.get(face, 0) >= 4:
            main_faces = {face}
            for kicker_count in range(4):  # 0, 1, 2, 3
                base_idx = _IDX_QUADS_START + (face - 3) * 4 + kicker_count
                kickers = select_kickers(hand_counts, main_faces, kicker_count)
                if kickers is not None:
                    action = Action(FORMATIONS['quads'], start_card=face, tail_cards=kickers)
                    if _check_beats_prev(action, prev, is_new_round):
                        mask[base_idx] = True

    # ── idx 247-282: Straight ────────────────────────────────────────────────
    for (length, start_face), idx in _STRAIGHT_MAP.items():
        end_face = start_face + length - 1
        # 顺子要求 end_face ≤ 14（A），且每个 face 至少 1 张
        if end_face > 14:
            continue
        if all(hand_counts.get(f, 0) >= 1 for f in range(start_face, end_face + 1)):
            action = Action(FORMATIONS['straight'], start_card=start_face, end_card=end_face)
            if _check_beats_prev(action, prev, is_new_round):
                mask[idx] = True

    # ── idx 283-293: Bomb ────────────────────────────────────────────────────
    for face in QUAD_FACES:  # 3..13
        if hand_counts.get(face, 0) >= 4:
            action = Action(FORMATIONS['bomb'], start_card=face)
            if _check_beats_prev(action, prev, is_new_round):
                idx = _IDX_BOMB_START + (face - 3)
                mask[idx] = True

    # ── 必须跟牌规则：仅当无任何合法非 Pass 动作时，才允许 Pass ─────────────
    if not is_new_round and not mask[1:].any():
        mask[0] = True

    return mask


def _resolve_kickers(
        hand_counts: dict[int, int],
        main_faces: set[int],
        kicker_count: int,
        kicker_faces_override: list[int] | None,
) -> list[int] | None:
    """统一带牌选取入口：优先使用外部指定的带牌，否则退回 greedy_lowest。"""
    if kicker_count == 0:
        return []
    if kicker_faces_override is not None:
        kickers = sorted(kicker_faces_override)
        # 验证带牌合法性：同一面值不超过3张，否则退回 greedy_lowest
        face_counts: dict[int, int] = {}
        valid = True
        for f in kickers:
            face_counts[f] = face_counts.get(f, 0) + 1
            if face_counts[f] > 3:
                valid = False
                break
        if valid:
            return kickers
    return select_kickers(hand_counts, main_faces, kicker_count)


def decode_action(action_idx: int, game, kicker_faces: list[int] | None = None) -> 'Action | None':
    """将动作索引解码为 Action 对象。

    :param action_idx:   动作索引 [0, 293]
    :param game:         当前游戏状态（用于获取手牌）
    :param kicker_faces: 外部指定的带牌面值列表（来自带牌头采样）；
                         为 None 时退回 greedy_lowest 策略
    :return: 对应的 Action 对象；若索引越界或手牌不足则返回 None
    """
    if action_idx < 0 or action_idx >= _ACTION_SIZE:
        return None

    current_player = game.players[game.current_player_index]
    hand_counts = get_hand_counts(current_player.cards)

    try:
        # idx 0: Pass
        if action_idx == _IDX_PASS:
            return Action(FORMATIONS['none'])

        # idx 1-13: Single
        if _IDX_SINGLE_START <= action_idx <= 13:
            face = action_idx - _IDX_SINGLE_START + 3
            if hand_counts.get(face, 0) >= 1:
                return Action(FORMATIONS['single'], start_card=face)
            return None

        # idx 14-25: Pocket
        if _IDX_POCKET_START <= action_idx <= 25:
            face = action_idx - _IDX_POCKET_START + 3
            if hand_counts.get(face, 0) >= 2:
                return Action(FORMATIONS['pocket'], start_card=face)
            return None

        # idx 26-76: S_pocket
        if _IDX_S_POCKET_START <= action_idx <= 76:
            if action_idx not in _S_POCKET_INV:
                return None
            length, start_face = _S_POCKET_INV[action_idx]
            end_face = start_face + length - 1
            if all(hand_counts.get(f, 0) >= 2 for f in range(start_face, end_face + 1)):
                return Action(FORMATIONS['s_pocket'], start_card=start_face, end_card=end_face)
            return None

        # idx 77-112: Trips
        if _IDX_TRIPS_START <= action_idx <= 112:
            offset = action_idx - _IDX_TRIPS_START
            kicker_count = offset // 12
            face = offset % 12 + 3
            if hand_counts.get(face, 0) >= 3:
                kickers = _resolve_kickers(hand_counts, {face}, kicker_count, kicker_faces)
                if kickers is not None:
                    return Action(FORMATIONS['trips'], start_card=face, tail_cards=kickers)
            return None

        # idx 113-202: S_trips
        if _IDX_S_TRIPS_START <= action_idx <= 202:
            if action_idx not in _S_TRIPS_INV:
                return None
            length, kicker_mode, start_face = _S_TRIPS_INV[action_idx]
            end_face = start_face + length - 1
            kicker_count = kicker_mode * length
            if all(hand_counts.get(f, 0) >= 3 for f in range(start_face, end_face + 1)):
                main_faces = set(range(start_face, end_face + 1))
                kickers = _resolve_kickers(hand_counts, main_faces, kicker_count, kicker_faces)
                if kickers is not None:
                    return Action(
                        FORMATIONS['s_trips'],
                        start_card=start_face,
                        end_card=end_face,
                        tail_cards=kickers,
                    )
            return None

        # idx 203-246: Quads
        if _IDX_QUADS_START <= action_idx <= 246:
            offset = action_idx - _IDX_QUADS_START
            face = offset // 4 + 3
            kicker_count = offset % 4
            if hand_counts.get(face, 0) >= 4:
                kickers = _resolve_kickers(hand_counts, {face}, kicker_count, kicker_faces)
                if kickers is not None:
                    return Action(FORMATIONS['quads'], start_card=face, tail_cards=kickers)
            return None

        # idx 247-282: Straight
        if _IDX_STRAIGHT_START <= action_idx <= 282:
            if action_idx not in _STRAIGHT_INV:
                return None
            length, start_face = _STRAIGHT_INV[action_idx]
            end_face = start_face + length - 1
            if end_face <= 14 and all(hand_counts.get(f, 0) >= 1 for f in range(start_face, end_face + 1)):
                return Action(FORMATIONS['straight'], start_card=start_face, end_card=end_face)
            return None

        # idx 283-293: Bomb
        if _IDX_BOMB_START <= action_idx <= 293:
            face = action_idx - _IDX_BOMB_START + 3
            if hand_counts.get(face, 0) >= 4:
                return Action(FORMATIONS['bomb'], start_card=face)
            return None

    except Exception:
        return None

    return None


def get_all_legal_actions(game) -> list:
    """返回所有合法动作的 Action 对象列表（用于测试/评估）。

    通过遍历掩码并调用 decode_action 实现，保证与掩码逻辑一致。
    """
    mask = compute_action_mask(game)
    legal_actions: list = []
    for idx in range(_ACTION_SIZE):
        if mask[idx]:
            action = decode_action(idx, game)
            if action is not None:
                legal_actions.append(action)
    return legal_actions


# ── 内部辅助函数 ─────────────────────────────────────────────────────────────

def _check_beats_prev(action: Action, prev: Action | None, is_new_round: bool) -> bool:
    """检查动作是否满足出牌约束。

    - is_new_round=True：动作必须合法（不能是 Pass）
    - is_new_round=False：炸弹始终可出；非炸弹须大于 prev_action
    """
    try:
        if not action.is_legal:
            return False
    except Exception:
        return False

    if is_new_round:
        # 新一轮不能 Pass（Pass 在外部单独处理）
        return action.formation != FORMATIONS['none']

    # 跟牌场景
    if prev is None:
        # prev 为 None 等价于新一轮，理论上不应走到这里
        return action.formation != FORMATIONS['none']

    # 炸弹：压制所有非炸弹；炸弹间须 face 更大
    if action.formation == FORMATIONS['bomb']:
        if prev.formation == FORMATIONS['bomb']:
            # 炸弹对炸弹：须面值更大
            return action.start_card > prev.start_card
        # 炸弹压制所有非炸弹
        return True

    # 非炸弹：必须与 prev 的 formation 相同（炸弹压制非炸弹已在上面处理）
    if prev.formation == FORMATIONS['bomb']:
        # 对方出了炸弹，只有炸弹才能压制
        return False

    # 检查是否能大于 prev
    try:
        return action > prev
    except RuntimeError:
        # formation/length/kicker_count 不匹配时 __gt__ 会 raise RuntimeError
        return False
    except Exception:
        return False
