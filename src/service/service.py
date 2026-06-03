"""
两人跑得快 AI 推理服务。

用法：
    python src/service/service.py --model checkpoints/194M.pt --port 7788

API:
    GET  /health_check          → {"result": "OK"}
    POST /suggestion            → AI 决策
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
from flask import Flask, jsonify, request

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))

from game.action import Action
from game.cards import cards_to_int, cards_to_str
from game.config import DECK_COUNTS, FORMATIONS
from game.game import Game
from game.player import Player
from src.engine.action_generator import (
    compute_action_mask,
    decode_action,
    get_hand_counts,
    get_kicker_info,
)
from src.engine.obs_encoder import encode_obs
from src.eval.heuristic_overrides import apply_heuristics
from src.training.actor_critic import ActorCritic

# ── 全局模型 ──────────────────────────────────────────────────────────────────

app = Flask(__name__)
_model: ActorCritic | None = None
_device: str = 'cpu'


def _load_model(model_path: str, device: str = 'cpu') -> ActorCritic:
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    model = ActorCritic(obs_dim=80, act_dim=294, hidden=256).to(device)
    model.load_state_dict(ckpt['model_state_dict'], strict=False)
    model.eval()
    print(f"[service] 模型加载成功：{model_path}", flush=True)
    return model


# ── 游戏状态重建 ───────────────────────────────────────────────────────────────

def _action_str_to_idx(action_str: str, game: Game) -> tuple[int, list[int]]:
    """将动作字符串转换为 (动作索引, 带牌面值列表)。

    支持：
    - "" 或 "PASS" → (0, [])（不出牌）
    - 牌面字符串（如 "33355"）→ (对应的动作索引, 带牌面值列表)

    同时返回带牌面值，确保重建游戏时 decode_action 使用与原始动作完全相同的带牌，
    避免 greedy 带牌选择与实际出牌不一致导致手牌状态错误。
    """
    if not action_str or action_str.upper() == 'PASS':
        return 0, []

    target_cards = sorted(cards_to_int(action_str))
    mask = compute_action_mask(game)
    hand_counts = get_hand_counts(game.players[game.current_player_index].cards)

    for idx in range(1, 294):
        if not mask[idx]:
            continue

        # 先用 greedy 解码尝试快速匹配
        decoded = decode_action(idx, game)
        if decoded is None:
            continue
        if sorted(decoded.deserialise()) == target_cards:
            return idx, []  # greedy 带牌恰好吻合，无需指定

        # greedy 带牌选择与目标不同时，用目标中推断出的带牌重新解码
        kicker_count, _, _ = get_kicker_info(idx, hand_counts)
        if kicker_count == 0:
            continue

        # 从 decoded 动作中提取主牌列表，其余即为带牌
        main_face  = decoded.start_card
        end_face   = decoded.end_card if decoded.end_card is not None else main_face
        multiple   = {
            FORMATIONS['trips']:   3,
            FORMATIONS['s_trips']: 3,
            FORMATIONS['quads']:   4,
        }.get(decoded.formation)
        if multiple is None:
            continue

        main_cards = sorted(
            f for f in range(main_face, end_face + 1) for _ in range(multiple)
        )
        remaining = list(target_cards)
        try:
            for c in main_cards:
                remaining.remove(c)
        except ValueError:
            continue

        if len(remaining) != kicker_count:
            continue

        # 带牌不能与主牌面相同（否则非法），跳过此索引
        main_faces_set = set(range(main_face, end_face + 1))
        if any(f in main_faces_set for f in remaining):
            continue

        decoded2 = decode_action(idx, game, kicker_faces=remaining)
        if decoded2 is None:
            continue
        try:
            if sorted(decoded2.deserialise()) == target_cards:
                return idx, remaining  # 返回实际带牌面值
        except RuntimeError:
            continue

    raise ValueError(
        f"无法从字符串 '{action_str}' 找到对应的动作（游戏步骤 {len(game.actions)}）"
    )


def _current_to_starting_hand(
    current_str: str,
    player_idx: int,
    actions: list[str | int],
    first_player: int,
) -> str:
    """从当前手牌和历史动作反推初始手牌。

    初始手牌 = 当前手牌 + 该玩家历史中打出的所有牌（Pass 不计）。
    """
    cards = list(cards_to_int(current_str))
    for i, action in enumerate(actions):
        actor = first_player if i % 2 == 0 else 1 - first_player
        if actor != player_idx:
            continue
        if isinstance(action, str) and action:
            cards.extend(cards_to_int(action))
    return cards_to_str(sorted(cards))


def _infer_opponent_cards(my_cards: list[int]) -> list[int]:
    """当只知道己方手牌时，推断对手起始手牌（含死牌）。

    对手起始手牌 = 48 张完整牌池 − 己方起始手牌（16-hand 变种）。
    结果包含对手实际持有的 16 张 + 16 张死牌，多余的死牌不影响游戏逻辑，
    因为 remain_cards 只依赖已出牌动作，不依赖初始手牌分配。
    """
    full_deck: list[int] = []
    for face, count in DECK_COUNTS.items():
        full_deck.extend([face] * count)
    remaining = full_deck[:]
    for card in my_cards:
        remaining.remove(card)
    return sorted(remaining)


def _reconstruct_game(
    hand_cards: list[str],
    actions: list[str | int],
    first_player: int,
    player_index: int = 0,
) -> tuple[Game, bool]:
    """从初始手牌和动作历史重建游戏状态。

    :param hand_cards:   两位玩家的初始手牌字符串列表，如 ["33445566...", "7788..."]；
                         也可只传 1 项（当前行动玩家的手牌），对手手牌自动从 48 张牌池推断。
    :param actions:      历史动作列表；每项可为整数索引（0-293）或牌面字符串（如 "33355"）
    :param first_player: 先手玩家索引（0 或 1）
    :param player_index: 仅在 hand_cards 只有 1 项时生效，标识该手牌属于哪位玩家（0 或 1）
    :return: (game, is_end)
    """
    if len(hand_cards) == 1:
        my_cards = cards_to_int(hand_cards[0])
        opp_cards = _infer_opponent_cards(my_cards)
        if player_index == 0:
            cards0, cards1 = my_cards, opp_cards
        else:
            cards0, cards1 = opp_cards, my_cards
    else:
        cards0 = cards_to_int(hand_cards[0])
        cards1 = cards_to_int(hand_cards[1])

    p0 = Player(cards=cards0)
    p1 = Player(cards=cards1)

    game = Game()
    game.start(players=[p0, p1], first_player=first_player)

    for raw_action in actions:
        if game.is_end:
            break
        if isinstance(raw_action, str):
            action_idx, kicker_faces = _action_str_to_idx(raw_action, game)
        else:
            action_idx, kicker_faces = int(raw_action), []
        action = decode_action(action_idx, game, kicker_faces=kicker_faces if kicker_faces else None)
        if action is None:
            raise ValueError(f"非法动作 {raw_action!r}（游戏步骤 {len(game.actions)}）")
        game.act(action)

    return game, game.is_end


# ── 推理 ───────────────────────────────────────────────────────────────────────

def _inference(game: Game) -> tuple[int, list[int], str]:
    """对当前游戏状态运行一步推理。

    :return: (action_idx, kicker_faces, action_str)
    """
    obs = encode_obs(game)
    mask = compute_action_mask(game)

    obs_t = torch.FloatTensor(obs).unsqueeze(0).to(_device)
    mask_t = torch.BoolTensor(mask).unsqueeze(0).to(_device)

    with torch.no_grad():
        h = _model.shared(obs_t)
        logits = _model.actor_head(h).masked_fill(~mask_t, float('-1e9'))
        kicker_logits_t = _model.kicker_head(h)
        action_t = logits.argmax(dim=-1)  # 贪心推理，确保确定性

    nn_action_idx = action_t.item()

    # 启发式覆盖（可通过 heuristic_overrides.HEURISTIC_FLAGS 单独开关）
    action_idx, _ = apply_heuristics(nn_action_idx, [], mask, game)

    # 带牌头采样
    hand_counts = get_hand_counts(game.players[game.current_player_index].cards)
    kicker_count, _, avail_13d = get_kicker_info(action_idx, hand_counts)

    kicker_faces: list[int] = []
    if kicker_count > 0:
        # 始终使用 NN 带牌头采样
        kicker_faces, _ = _model.sample_kickers(kicker_logits_t[0], avail_13d, kicker_count, deterministic=True)

    decoded = decode_action(action_idx, game, kicker_faces=kicker_faces if kicker_faces else None)
    action_str = str(decoded) if decoded is not None else ''

    return action_idx, kicker_faces, action_str


# ── 路由 ───────────────────────────────────────────────────────────────────────

@app.route('/health_check', methods=['GET', 'POST'])
def health_check():
    return jsonify({'result': 'OK'}), 200


@app.route('/suggestion', methods=['POST'])
def suggestion():
    """获取 AI 决策。

    Request JSON:
    {
        "game_id":      str,          # 对局唯一标识（日志追踪用）
        "player_id":    str,          # 玩家标识（日志追踪用）
        "player_index": int,          # 请求方的玩家索引（0 或 1）
        "hand_cards":   [str, str],   # 两位玩家的初始手牌字符串
        "actions":      [int, ...],   # 历史动作索引列表（0-293）
        "first_player": int           # 先手玩家索引（0 或 1，默认 0）
    }

    Response JSON:
    {
        "action_index":  int,         # AI 选择的动作索引（0-293）
        "action_str":    str,         # 动作的牌面字符串（如 "555666"，Pass 为 ""）
        "kicker_faces":  [int, ...],  # 带牌面值列表（无带牌时为 []）
        "is_end":        bool,        # 执行该动作后游戏是否结束
        "is_success":    bool,        # 请求是否成功
        "error_message": str          # 错误描述（成功时为 ""）
    }
    """
    response = {
        'action_str': '',
        'kicker_faces': [],
        'is_end': False,
        'is_success': False,
        'error_message': '',
    }

    try:
        data = request.get_json(force=True)
        if data is None:
            response['error_message'] = 'Request body must be JSON'
            return jsonify(response), 400

        # ── 字段解析 ──────────────────────────────────────────────────────────
        game_id      = data.get('game_id', '')
        player_id    = data.get('player_id', '')
        player_index = int(data.get('player_index', 0))
        hand_cards   = data.get('hand_cards', [])
        actions      = data.get('actions', [])
        first_player = int(data.get('first_player', 0))

        # ── 基本校验 ──────────────────────────────────────────────────────────
        if len(hand_cards) not in (1, 2):
            response['error_message'] = 'hand_cards must contain 1 or 2 strings'
            return jsonify(response), 400

        if player_index not in (0, 1):
            response['error_message'] = 'player_index must be 0 or 1'
            return jsonify(response), 400

        if first_player not in (0, 1):
            response['error_message'] = 'first_player must be 0 or 1'
            return jsonify(response), 400

        # ── 当前手牌 → 初始手牌 ──────────────────────────────────────────────
        if len(hand_cards) == 1:
            starting_hands = [
                _current_to_starting_hand(hand_cards[0], player_index, actions, first_player)
            ]
        else:
            starting_hands = [
                _current_to_starting_hand(hand_cards[i], i, actions, first_player)
                for i in range(2)
            ]

        # ── 重建游戏状态 ──────────────────────────────────────────────────────
        game, is_end = _reconstruct_game(starting_hands, actions, first_player, player_index)

        if is_end:
            response['is_end'] = True
            response['is_success'] = True
            response['error_message'] = 'Game is already over'
            return jsonify(response), 200

        # ── 校验当前行动玩家 ──────────────────────────────────────────────────
        if game.current_player_index != player_index:
            response['error_message'] = (
                f'Wrong player turn: expected player {game.current_player_index}, '
                f'got player_index={player_index}'
            )
            return jsonify(response), 400

        # ── 推理 ──────────────────────────────────────────────────────────────
        action_idx, kicker_faces, action_str = _inference(game)

        # ── 执行动作，判断是否终局 ────────────────────────────────────────────
        decoded = decode_action(
            action_idx, game,
            kicker_faces=kicker_faces if kicker_faces else None,
        )
        if decoded is not None:
            game.act(decoded)

        response['action_str']    = action_str
        response['kicker_faces']  = kicker_faces
        response['is_end']        = game.is_end
        response['is_success']    = True

        print(
            f"[service] game={game_id} player={player_id}({player_index}) "
            f"action={action_idx}({action_str}) is_end={game.is_end}",
            flush=True,
        )

    except ValueError as e:
        response['error_message'] = str(e)
        return jsonify(response), 400
    except Exception as e:
        response['error_message'] = f'Internal error: {e}'
        return jsonify(response), 500

    return jsonify(response), 200


# ── 入口 ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='两人跑得快 AI 推理服务')
    parser.add_argument('--model', default='checkpoints/194M.pt', help='模型路径')
    parser.add_argument('--port', type=int, default=7788, help='监听端口')
    parser.add_argument('--host', default='0.0.0.0', help='监听地址')
    parser.add_argument('--device', default='cpu', help='推理设备（cpu / cuda）')
    args = parser.parse_args()

    global _model, _device
    _device = args.device
    _model = _load_model(args.model, _device)

    print(f"[service] 启动服务：http://{args.host}:{args.port}", flush=True)
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == '__main__':
    main()
