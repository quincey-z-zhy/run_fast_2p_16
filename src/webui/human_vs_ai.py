"""人类 vs AI 对战 Web 服务 (run_fast_2p_v1)

用法：
    cd /path/to/run_fast_2p_v1
    PYTHONPATH=. python src/webui/human_vs_ai.py [--model checkpoints/best.pt] [--port 7788]

然后在浏览器打开 http://localhost:7788
"""
import sys
import os
import json
import uuid
import itertools
import argparse

# 确保项目根目录在 sys.path
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from pathlib import Path
from flask import Flask, render_template, request, jsonify, make_response

import torch

from src.training.actor_critic import ActorCritic
from src.env.game_env import RunFast2pEnv
from src.engine.action_generator import compute_action_mask, decode_action, get_hand_counts, get_kicker_info
from src.engine.obs_encoder import encode_obs, encode_action
from src.eval.heuristic_overrides import apply_heuristics
from game.config import (
    FORMATION_NAMES, FORMATIONS, FACES,
    POCKET_TRIPLE_FACES, QUAD_FACES,
    MIN_S_POCKET_LEN, MAX_S_POCKET_LEN,
    MIN_S_TRIPS_LEN, MAX_S_TRIPS_LEN,
    MIN_STRAIGHT_LEN, MAX_STRAIGHT_LEN,
)
from game.action import Action
from game.cards import cards_to_str

_parser = argparse.ArgumentParser(description='RunFast 2P WebUI')
_parser.add_argument('--model', default='checkpoints/best.pt', help='模型 checkpoint 路径（相对于项目根目录或绝对路径）')
_parser.add_argument('--port', type=int, default=7788, help='监听端口')
_args, _ = _parser.parse_known_args()

app = Flask(__name__, template_folder='templates')
app.secret_key = os.urandom(24)

BASE_DIR = Path(_PROJECT_ROOT)

# ── 加载模型 ───────────────────────────────────────────────────────────────
_model_arg = Path(_args.model)
MODEL_PATH = _model_arg if _model_arg.is_absolute() else BASE_DIR / _model_arg
_model = ActorCritic(obs_dim=80, act_dim=294, hidden=256)
_ckpt = torch.load(str(MODEL_PATH), map_location='cpu', weights_only=True)
_missing, _ = _model.load_state_dict(_ckpt['model_state_dict'], strict=False)
if _missing:
    print(f"[WebUI] 带牌头权重未找到（旧版 checkpoint），从随机初始化开始：{_missing}")
_model.eval()
print(f"[WebUI] 模型加载成功：{MODEL_PATH}")

# ── 多用户会话状态（session_id → game state dict）────────────────────────
_sessions: dict[str, dict] = {}

_SID_COOKIE = 'rfp_sid'


def _get_gs() -> tuple[str, dict]:
    """获取当前请求的 session id 和对应的 game state dict。

    session id 从请求 Cookie 中读取；若不存在则创建新的 UUID。
    """
    sid = request.cookies.get(_SID_COOKIE)
    if not sid or sid not in _sessions:
        sid = str(uuid.uuid4())
        _sessions[sid] = {}
    return sid, _sessions[sid]


def _set_sid_cookie(response, sid: str):
    """将 session id 写入响应 Cookie（仅在新建时调用）。"""
    response.set_cookie(_SID_COOKIE, sid, httponly=True, samesite='Lax')
    return response


# ── 工具函数 ──────────────────────────────────────────────────────────────

def _kicker_set_prob(kicker_faces: list[int], kicker_logits: torch.Tensor,
                     avail_init: 'np.ndarray') -> float:
    """NN 带牌头下选中 kicker_faces 这一无序多重集的联合概率（求和所有排列）。"""
    import numpy as np
    from itertools import permutations as _perms

    if not kicker_faces:
        return 1.0

    indices = tuple(f - 3 for f in kicker_faces)
    avail_init = avail_init.astype(np.float32)
    total = 0.0
    seen = set()

    for perm in _perms(indices):
        if perm in seen:
            continue
        seen.add(perm)
        avail = avail_init.copy()
        prob = 1.0
        for idx in perm:
            if avail[idx] <= 0:
                prob = 0.0
                break
            mask_t = torch.from_numpy(avail > 0)
            step_probs = torch.softmax(
                kicker_logits.masked_fill(~mask_t, float('-inf')), dim=-1
            )
            prob *= step_probs[idx].item()
            avail[idx] -= 1
        total += prob

    return total


def _capture_snapshot(gs: dict, actor: str, action_obj) -> None:
    """在动作执行之后捕获游戏状态快照（供下载使用）。"""
    env = gs['env']
    game = env._game
    is_pass = action_obj is None or action_obj.formation == FORMATIONS['none']

    def _record_desc(act):
        d = _action_desc(act)
        return '' if d == '不出' else d

    def _cards_only(act):
        """仅返回出牌的牌面字符串；Pass / None → ''。"""
        if act is None or act.formation == FORMATIONS['none']:
            return ''
        cards = act.deserialise()
        return cards_to_str(cards) if cards else ''

    snap = {
        'step': len(gs.setdefault('snapshots', [])),
        'actor': actor,  # 'ai' / 'human'
        'action_desc': _record_desc(action_obj),
        'action_cards': [] if is_pass else action_obj.deserialise(),
        'after_state': {
            'current_player': game.current_player_index,
            'is_new_round':   game.is_new_round,
            'hands':          [list(p.cards) for p in game.players],
            'num_cards':      [len(p.cards) for p in game.players],
            'remain_cards':   dict(game.remain_cards),
            'action_history': [_cards_only(a) for a in game.actions],
            'is_end':         game.is_end,
            'num_bomb':       game.num_bomb,
        },
    }
    if game.is_end:
        snap['after_state']['winner_index'] = next(
            (i for i, p in enumerate(game.players) if p.is_winner), -1)
        snap['after_state']['scores'] = [p.score for p in game.players]
        snap['after_state']['is_spring'] = game.is_spring
    gs['snapshots'].append(snap)


def _action_desc(action_obj) -> str:
    """将 Action 对象转为可读字符串，pass → '不出'"""
    if action_obj is None:
        return '不出'
    cards = action_obj.deserialise()
    if not cards:
        return '不出'
    fname = FORMATION_NAMES.get(action_obj.formation, str(action_obj.formation))
    return f"{fname}：{cards_to_str(cards)}"


def _enumerate_human_actions(game) -> list[dict]:
    """枚举当前人类玩家的所有合法动作，包括每种带牌组合。

    与 294-动作空间不同，这里穷举每一种具体的带牌选择，
    使玩家可以自由决定使用哪些牌作为带牌。
    """
    player = game.players[game.current_player_index]
    hand_counts = get_hand_counts(player.cards)
    is_new_round = game.is_new_round
    prev = game.previous_action

    results: list[dict] = []

    # ── 合法性检查 ────────────────────────────────────────────────────────
    def is_playable(action_obj: Action) -> bool:
        try:
            if not action_obj.is_legal:
                return False
        except Exception:
            return False
        if action_obj.formation == FORMATIONS['none']:
            return False  # pass 不通过此路径加入，见函数末尾
        if is_new_round:
            return True  # 新一轮可出任何非 pass 牌型
        if action_obj.formation == FORMATIONS['bomb']:
            if prev is None:
                return True
            if prev.formation != FORMATIONS['bomb']:
                return True  # 炸弹压制一切非炸弹
            return action_obj.start_card > prev.start_card
        if prev is None:
            return True
        if prev.formation == FORMATIONS['bomb']:
            return False
        try:
            return action_obj > prev
        except Exception:
            return False

    def add(action_obj: Action, main_c: list, tail_c: list):
        if not is_playable(action_obj):
            return
        all_cards = sorted(main_c + list(tail_c))
        fname = FORMATION_NAMES.get(action_obj.formation, str(action_obj.formation))
        desc = f"{fname}：{cards_to_str(all_cards)}" if all_cards else '不出'
        results.append({
            'id': len(results),
            'action': action_obj,
            'desc': desc,
            'main_cards': sorted(main_c),
            'tail_cards': sorted(tail_c),
        })

    def unique_combos(pool: list, count: int):
        """从 pool 中取 count 张，枚举所有不重复的组合（pool 含重复元素）。"""
        if count == 0:
            yield []
            return
        seen: set = set()
        for combo in itertools.combinations(pool, count):
            if combo not in seen:
                seen.add(combo)
                yield list(combo)

    def kicker_pool(exclude: set) -> list:
        """构建带牌候选池（排除主牌面值，保留重复牌）。"""
        pool = []
        for f in sorted(hand_counts):
            if f not in exclude:
                pool.extend([f] * hand_counts[f])
        return pool

    # ── 单牌 ────────────────────────────────────────────────────────────────
    for face in FACES:
        if hand_counts.get(face, 0) >= 1:
            add(Action(FORMATIONS['single'], start_card=face), [face], [])

    # ── 对子 ────────────────────────────────────────────────────────────────
    for face in POCKET_TRIPLE_FACES:
        if hand_counts.get(face, 0) >= 2:
            add(Action(FORMATIONS['pocket'], start_card=face), [face, face], [])

    # ── 连对（16-hand 变种 A 可入对，end ≤ 14） ─────────────────────────────
    for length in range(MIN_S_POCKET_LEN, MAX_S_POCKET_LEN + 1):
        for start in range(3, 16 - length):
            end = start + length - 1
            if all(hand_counts.get(f, 0) >= 2 for f in range(start, end + 1)):
                main_c = [f for f in range(start, end + 1) for _ in range(2)]
                add(Action(FORMATIONS['s_pocket'], start_card=start, end_card=end), main_c, [])

    # ── 三条（带 0/1/2 张，枚举所有带牌组合）─────────────────────────────
    for face in POCKET_TRIPLE_FACES:
        if hand_counts.get(face, 0) >= 3:
            main_c = [face] * 3
            pool = kicker_pool({face})
            for k in range(3):  # 0, 1, 2 带牌
                for tc in unique_combos(pool, k):
                    add(Action(FORMATIONS['trips'], start_card=face, tail_cards=tc),
                        main_c, tc)

    # ── 飞机（连续三条，带 0/N/2N 张；16-hand 变种 A 可入飞机，end ≤ 14）──
    for length in range(MIN_S_TRIPS_LEN, MAX_S_TRIPS_LEN + 1):
        for start in range(3, 15 - length + 1):  # end ≤ 14(A)
            end = start + length - 1
            if not all(hand_counts.get(f, 0) >= 3 for f in range(start, end + 1)):
                continue
            main_faces = set(range(start, end + 1))
            main_c = [f for f in range(start, end + 1) for _ in range(3)]
            pool = kicker_pool(main_faces)
            for km in range(3):  # kicker_mode: 带 0/N/2N 张
                k_count = km * length
                if k_count > len(pool):
                    continue
                for tc in unique_combos(pool, k_count):
                    add(Action(FORMATIONS['s_trips'], start_card=start, end_card=end,
                               tail_cards=tc), main_c, tc)

    # ── 四带（带 0/1/2/3 张，枚举所有带牌组合）──────────────────────────────
    for face in QUAD_FACES:
        if hand_counts.get(face, 0) >= 4:
            main_c = [face] * 4
            pool = kicker_pool({face})
            for k in range(4):  # 0, 1, 2, 3 带牌
                if k > len(pool):
                    continue
                for tc in unique_combos(pool, k):
                    add(Action(FORMATIONS['quads'], start_card=face, tail_cards=tc),
                        main_c, tc)

    # ── 顺子 ────────────────────────────────────────────────────────────────
    for length in range(MIN_STRAIGHT_LEN, MAX_STRAIGHT_LEN + 1):
        for start in range(3, 16 - length):
            end = start + length - 1
            if end > 14:
                continue
            if all(hand_counts.get(f, 0) >= 1 for f in range(start, end + 1)):
                main_c = list(range(start, end + 1))
                add(Action(FORMATIONS['straight'], start_card=start, end_card=end),
                    main_c, [])

    # ── 炸弹 ────────────────────────────────────────────────────────────────
    for face in QUAD_FACES:
        if hand_counts.get(face, 0) >= 4:
            add(Action(FORMATIONS['bomb'], start_card=face), [face] * 4, [])

    # ── Pass（仅当跟牌阶段且无任何合法出牌时） ──────────────────────────────
    # 规则：能压上家必须出牌，只有真的没有任何合法打法才允许不出。
    if not is_new_round and not results:
        results.append({
            'id': 0,
            'action': Action(FORMATIONS['none']),
            'desc': '不出',
            'main_cards': [],
            'tail_cards': [],
        })

    return results


def _build_response(gs: dict) -> dict:
    """构建发送给前端的完整状态字典。"""
    env = gs['env']
    game = env._game
    h = gs['human_index']  # 人类玩家索引（固定 0）

    current = game.current_player_index
    is_human_turn = (current == h) and not game.is_end

    prev = game.previous_action
    if prev is None:
        previous_play = '新一轮 — 可任意出牌'
        previous_cards = []
    else:
        previous_play = _action_desc(prev)
        previous_cards = prev.deserialise()

    if is_human_turn:
        actions_list = _enumerate_human_actions(game)
        # 存储 id → Action 映射供 /api/action 查找
        gs['human_actions_map'] = {a['id']: a['action'] for a in actions_list}

        # NN 推理：actor_head 给出 P(action_idx)，kicker_head 给出 13 维带牌偏好
        obs = encode_obs(game)
        mask = compute_action_mask(game)
        hand_counts = get_hand_counts(game.players[h].cards)
        obs_t = torch.FloatTensor(obs).unsqueeze(0)
        mask_t = torch.BoolTensor(mask).unsqueeze(0)
        with torch.no_grad():
            hidden = _model.shared(obs_t)
            logits = _model.actor_head(hidden).masked_fill(~mask_t, float('-1e9'))
            probs = torch.softmax(logits, dim=-1)[0].tolist()
            kicker_logits = _model.kicker_head(hidden)[0]  # (13,) tensor

        valid_actions = []
        for a in actions_list:
            action_idx = encode_action(a['action'], game)
            if action_idx < 0:
                p = 0.0
            else:
                p_main = probs[action_idx]
                kicker_count, _, avail_13d = get_kicker_info(action_idx, hand_counts)
                if kicker_count == 0 or not a['tail_cards']:
                    p = p_main
                else:
                    p = p_main * _kicker_set_prob(a['tail_cards'], kicker_logits, avail_13d)
            valid_actions.append({
                'id':          a['id'],
                'desc':        a['desc'],
                'main_cards':  a['main_cards'],
                'tail_cards':  a['tail_cards'],
                'type':        'main',
                'probability': p,
            })
    else:
        gs['human_actions_map'] = {}
        valid_actions = []

    return {
        # 双方手牌（前端用）
        'players': [
            {'cards': list(game.players[0].cards), 'num_cards': len(game.players[0].cards)},
            {'cards': list(game.players[1].cards), 'num_cards': len(game.players[1].cards)},
        ],
        'human_index': h,
        'hand_cards': list(game.players[h].cards),
        'ai_hand_cards': list(game.players[1 - h].cards),
        # 游戏信息
        'num_bomb': game.num_bomb,
        'current_player': current,
        'is_human_turn': is_human_turn,
        'deciding_tail': False,
        'is_new_round': game.is_new_round,
        'previous_play': previous_play,
        'previous_cards': previous_cards,
        # 结局
        'game_over': game.is_end,
        'winner': gs.get('winner'),
        'is_spring': game.is_spring,
        'scores': [game.players[0].score, game.players[1].score] if game.is_end else None,
        # 操作
        'action_log': gs.get('action_log', []),
        'valid_actions': valid_actions,
    }


def _ai_play_loop(gs: dict):
    """AI 连续出牌，直到轮到人类或游戏结束，然后返回 JSON 响应。"""
    env = gs['env']
    game = env._game
    h = gs['human_index']

    while not game.is_end and game.current_player_index != h:
        obs = gs['obs']
        info = gs['info']

        obs_t = torch.FloatTensor(obs).unsqueeze(0)
        mask_t = torch.BoolTensor(info['action_mask']).unsqueeze(0)

        with torch.no_grad():
            # argmax 推理（与 service.py 一致，确保同 obs → 同动作）
            hidden = _model.shared(obs_t)
            logits = _model.actor_head(hidden).masked_fill(~mask_t, float('-1e9'))
            kicker_logits_t = _model.kicker_head(hidden)
            action_t = logits.argmax(dim=-1)

        nn_action_idx = action_t.item()

        # 启发式覆盖（可通过 heuristic_overrides.HEURISTIC_FLAGS 单独开关）
        action_idx, _ = apply_heuristics(nn_action_idx, [], info['action_mask'], game)

        # 带牌头采样
        hand_counts = get_hand_counts(game.players[game.current_player_index].cards)
        kicker_count, _, avail_13d = get_kicker_info(action_idx, hand_counts)
        kicker_faces_list: list[int] = []
        if kicker_count > 0:
            # 始终使用 NN 带牌头采样
            kicker_faces_list, _ = _model.sample_kickers(kicker_logits_t[0], avail_13d, kicker_count, deterministic=True)

        action_obj = decode_action(action_idx, game, kicker_faces=kicker_faces_list if kicker_faces_list else None)
        desc = _action_desc(action_obj)
        gs['action_log'].append(f'AI: {desc}')

        obs, _reward, terminated, _truncated, info = env.step(
            action_idx,
            kicker_faces=kicker_faces_list if kicker_faces_list else None,
        )
        gs['obs'] = obs
        gs['info'] = info
        _capture_snapshot(gs, 'ai', action_obj)

        if terminated:
            gs['winner'] = '你' if game.players[h].is_winner else 'AI'
            print('[WIN] ai_loop: is_winner[{}]={} score[{}]={} winner={}'.format(
                h, game.players[h].is_winner, h, game.players[h].score, gs['winner']))
            break

    return jsonify(_build_response(gs))


# ── 路由 ─────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    resp = make_response(render_template('index.html'))
    resp.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate'
    resp.headers['Pragma'] = 'no-cache'
    return resp


@app.route('/api/new_game', methods=['POST'])
def new_game():
    incoming_sid = request.cookies.get(_SID_COOKIE)
    sid, gs = _get_gs()

    env = RunFast2pEnv()
    obs, info = env.reset()

    gs.clear()
    gs['env'] = env
    gs['human_index'] = 0
    gs['obs'] = obs
    gs['info'] = info
    gs['action_log'] = []
    gs['winner'] = None
    gs['human_actions_map'] = {}
    gs['snapshots'] = []
    gs['initial_hands'] = [list(env._game.players[i].cards) for i in range(2)]

    # 若 AI 先手，直接让 AI 出牌
    if info['current_player'] != gs['human_index']:
        resp = _ai_play_loop(gs)
    else:
        resp = jsonify(_build_response(gs))

    # 写入 session cookie：首次访问，或 sid 已更新（旧 cookie 过期/服务重启后）
    if incoming_sid != sid:
        _set_sid_cookie(resp, sid)
    return resp


@app.route('/api/action', methods=['POST'])
def take_action():
    sid, gs = _get_gs()
    if not gs:
        return jsonify({'error': '请先开始新游戏'}), 400

    data = request.get_json(silent=True) or {}
    action_id = data.get('human_action_id')
    if action_id is None:
        return jsonify({'error': '缺少 human_action_id 参数'}), 400

    try:
        action_id = int(action_id)
    except (TypeError, ValueError):
        return jsonify({'error': 'human_action_id 必须为整数'}), 400

    env = gs['env']
    game = env._game
    h = gs['human_index']

    if game.is_end:
        return jsonify({'error': '游戏已结束，请开始新游戏'}), 400
    if game.current_player_index != h:
        return jsonify({'error': '非你的回合'}), 400

    action_map = gs.get('human_actions_map', {})
    if action_id not in action_map:
        return jsonify({'error': '无效动作 ID，请刷新后重试'}), 400

    action_obj = action_map[action_id]
    desc = _action_desc(action_obj)
    gs['action_log'].append(f'你: {desc}')

    # 直接通过 game.act() 执行（绕过 294-动作索引限制）
    game.act(action_obj)
    obs, info = env._get_obs_and_info()
    gs['obs'] = obs
    gs['info'] = info
    _capture_snapshot(gs, 'human', action_obj)

    if game.is_end:
        gs['winner'] = '你' if game.players[h].is_winner else 'AI'
        print('[WIN] human_turn: is_winner[{}]={} score[{}]={} winner={}'.format(
            h, game.players[h].is_winner, h, game.players[h].score, gs['winner']))
        return jsonify(_build_response(gs))

    # 轮到 AI
    return _ai_play_loop(gs)


# ── 下载牌谱 ─────────────────────────────────────────────────────────────

@app.route('/api/download_record', methods=['GET'])
def download_record():
    """下载本局完整对局记录（JSONL：第一行 header，后续每行一个 step）。"""
    sid, gs = _get_gs()
    if not gs or 'snapshots' not in gs:
        return jsonify({'error': '无对局记录'}), 400

    header = {
        'type':          'header',
        'human_index':   gs.get('human_index'),
        'initial_hands': gs.get('initial_hands'),
        'total_steps':   len(gs['snapshots']),
    }
    lines = [json.dumps(header, ensure_ascii=False)]
    for snap in gs['snapshots']:
        lines.append(json.dumps({'type': 'step', **snap}, ensure_ascii=False))

    body = '\n'.join(lines) + '\n'
    resp = make_response(body)
    resp.headers['Content-Type'] = 'application/x-ndjson; charset=utf-8'
    resp.headers['Content-Disposition'] = 'attachment; filename="game_record.jsonl"'
    return resp


@app.route('/api/download_step/<int:step>', methods=['GET'])
def download_step(step: int):
    """下载某一步的游戏状态快照（单行 JSONL）。"""
    sid, gs = _get_gs()
    if not gs or 'snapshots' not in gs:
        return jsonify({'error': '无对局记录'}), 400

    snaps = gs['snapshots']
    if step < 0 or step >= len(snaps):
        return jsonify({'error': f'步数 {step} 越界 (0..{len(snaps)-1})'}), 400

    body = json.dumps({'type': 'step', **snaps[step]}, ensure_ascii=False) + '\n'
    resp = make_response(body)
    resp.headers['Content-Type'] = 'application/x-ndjson; charset=utf-8'
    resp.headers['Content-Disposition'] = f'attachment; filename="step_{step:03d}.jsonl"'
    return resp


if __name__ == '__main__':
    print(f"[WebUI] 启动服务：http://localhost:{_args.port}")
    app.run(host='0.0.0.0', port=_args.port, debug=False, threaded=True)
