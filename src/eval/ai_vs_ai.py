"""
AI vs AI 对战脚本：两个对手互相对战，统计胜率、得分、春天、炸弹。

用法：
    # 两个 checkpoint 互打
    python src/eval/ai_vs_ai.py \\
        --model-a checkpoints/194M.pt \\
        --model-b checkpoints/baseline.pt \\
        --games 1000 --seed 42

    # 模型 vs 基线
    python src/eval/ai_vs_ai.py \\
        --model-a checkpoints/194M.pt \\
        --model-b random_legal \\
        --games 1000

特点：
- 每副牌打 2 次（双向轮换先后手）以消除发牌优势
- 同副牌用同一个 seed，A vs B 对位与 B vs A 对位完全一致
- model-b 也可以是基线名 (random_legal / greedy_low)
- 春天用 env._game.is_spring 直接判定（不依赖 SPRING_SCORE 阈值）
- 按先后手分组统计（A 先手胜率 / A 后手胜率）
"""

import argparse
import random
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))

import numpy as np
import torch

from src.engine.action_generator import get_hand_counts, get_kicker_info
from src.env.game_env import RunFast2pEnv
from src.eval.baselines import GreedyLowAgent, RandomLegalAgent
from src.eval.heuristic_overrides import apply_heuristics
from src.training.actor_critic import ActorCritic


BASELINES = {
    'random_legal': RandomLegalAgent,
    'greedy_low':   GreedyLowAgent,
}


# ── 模型加载 ─────────────────────────────────────────────────────────────────

class ModelAgent:
    """ActorCritic checkpoint 包装：选动作 + 选带牌（argmax 确定性）+ 启发式覆盖。"""

    def __init__(self, ckpt_path: str, device: str = 'cpu'):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        self.model = ActorCritic().to(device)
        self.model.load_state_dict(ckpt['model_state_dict'], strict=False)
        self.model.eval()
        self.device = device
        self.step = ckpt.get('step', -1)

    def act(self, obs: np.ndarray, mask: np.ndarray, game) -> tuple[int, list[int]]:
        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        mask_t = torch.BoolTensor(mask).unsqueeze(0).to(self.device)
        with torch.no_grad():
            hidden = self.model.shared(obs_t)
            logits = self.model.actor_head(hidden).masked_fill(~mask_t, float('-1e9'))
            kicker_logits_t = self.model.kicker_head(hidden)
            nn_action = logits.argmax(dim=-1).item()

        action, _ = apply_heuristics(nn_action, [], mask, game)

        hand_counts = get_hand_counts(game.players[game.current_player_index].cards)
        kicker_count, _, avail_13d = get_kicker_info(action, hand_counts)
        kicker_faces: list[int] = []
        if kicker_count > 0:
            kicker_faces, _ = self.model.sample_kickers(
                kicker_logits_t[0], avail_13d, kicker_count, deterministic=True
            )
        return action, kicker_faces


class BaselineAgent:
    """基线对手包装：random_legal / greedy_low，无带牌头（用 greedy_lowest 自动带）。"""

    def __init__(self, name: str):
        if name not in BASELINES:
            raise ValueError(f"未知基线：{name}（可用：{list(BASELINES)}）")
        self.agent = BASELINES[name]()
        self.step = -1
        self.name = name

    def act(self, obs: np.ndarray, mask: np.ndarray, game) -> tuple[int, list[int]]:
        action = self.agent.act(obs, mask, game)
        return action, []  # 基线不指定带牌，由 env 端 greedy_lowest 选


def load_agent(spec: str, device: str = 'cpu'):
    """spec 可以是 checkpoint 路径或基线名。返回 agent + 用于报表的 label。"""
    if spec in BASELINES:
        return BaselineAgent(spec), spec
    p = Path(spec)
    if not p.exists():
        raise FileNotFoundError(f"checkpoint 不存在：{spec}")
    return ModelAgent(spec, device), p.stem


# ── 单局对战 ─────────────────────────────────────────────────────────────────

def play_one(env: RunFast2pEnv, agent_a, agent_b, a_player: int, seed: int) -> dict:
    """跑一局，A 占 a_player 位置。返回单局统计 dict。"""
    obs, info = env.reset(seed=seed)
    n_pass = [0, 0]
    n_bomb = [0, 0]
    steps = 0

    while True:
        cur = info['current_player']
        mask = info['action_mask']
        agent = agent_a if cur == a_player else agent_b
        action, kickers = agent.act(obs, mask, env._game)
        if action == 0:
            n_pass[cur] += 1
        else:
            # 炸弹 idx 由 action_generator._IDX_BOMB_START 起算（283 in 16-hand）
            from src.engine.action_generator import _IDX_BOMB_START, _ACTION_SIZE
            if _IDX_BOMB_START <= action < _ACTION_SIZE:
                n_bomb[cur] += 1
        obs, _r, term, _trunc, info = env.step(action, kicker_faces=kickers if kickers else None)
        steps += 1
        if term:
            break

    game = env._game
    b_player = 1 - a_player
    return {
        'a_won':     bool(game.players[a_player].is_winner),
        'score_a':   info['final_rewards'][a_player],
        'score_b':   info['final_rewards'][b_player],
        'is_spring': game.is_spring,
        'spring_winner': a_player if (game.is_spring and game.players[a_player].is_winner) else (
            b_player if game.is_spring else -1),
        'n_bomb_a':  n_bomb[a_player],
        'n_bomb_b':  n_bomb[b_player],
        'n_pass_a':  n_pass[a_player],
        'n_pass_b':  n_pass[b_player],
        'steps':     steps,
        'a_first':   a_player == 0,
    }


# ── 主入口 ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='AI vs AI 对战评估')
    parser.add_argument('--model-a', required=True,
                        help='模型 A：checkpoint 路径或基线名 (random_legal / greedy_low)')
    parser.add_argument('--model-b', required=True,
                        help='模型 B：checkpoint 路径或基线名')
    parser.add_argument('--games', type=int, default=1000, help='总对局数（向上取偶）')
    parser.add_argument('--seed',  type=int, default=42, help='board 采样随机种子（保证可复现）')
    parser.add_argument('--device', default='cpu', help='cpu / cuda')
    parser.add_argument('--verbose', action='store_true', help='每 100 局打印中间进度')
    args = parser.parse_args()

    if args.games % 2 != 0:
        args.games += 1
    n_boards = args.games // 2

    # 设种子 → board_seeds 与 NN（启发式之外的 stochastic）都可复现
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    board_seeds = [random.randint(0, 2**31 - 1) for _ in range(n_boards)]

    print(f"[load] A = {args.model_a}")
    agent_a, label_a = load_agent(args.model_a, args.device)
    print(f"[load] B = {args.model_b}")
    agent_b, label_b = load_agent(args.model_b, args.device)
    print(f"\n[match] {label_a} vs {label_b}  共 {args.games} 局（{n_boards} 副牌 × 双向）"
          f"  seed={args.seed}\n")

    env = RunFast2pEnv()
    stats = []

    for ep in range(args.games):
        a_player = ep % 2  # 偶数局 A=P0, 奇数局 A=P1
        seed = board_seeds[ep // 2]
        stats.append(play_one(env, agent_a, agent_b, a_player, seed))

        if args.verbose and (ep + 1) % 100 == 0:
            wins = sum(s['a_won'] for s in stats)
            print(f"  [{ep+1:>5}/{args.games}]  {label_a} 累计胜率 {wins/(ep+1):.1%}")

    # ── 汇总 ──────────────────────────────────────────────────────────────
    n = len(stats)
    wins_a = sum(s['a_won'] for s in stats)
    wins_b = n - wins_a

    wins_a_first = sum(s['a_won'] for s in stats if s['a_first'])
    wins_a_second = sum(s['a_won'] for s in stats if not s['a_first'])
    n_first = sum(1 for s in stats if s['a_first'])
    n_second = n - n_first

    avg_score_a = sum(s['score_a'] for s in stats) / n
    avg_score_b = sum(s['score_b'] for s in stats) / n

    n_spring = sum(1 for s in stats if s['is_spring'])
    spring_a = sum(1 for s in stats if s['spring_winner'] == (0 if s['a_first'] else 1))
    spring_b = n_spring - spring_a

    avg_bomb_a = sum(s['n_bomb_a'] for s in stats) / n
    avg_bomb_b = sum(s['n_bomb_b'] for s in stats) / n
    avg_pass_a = sum(s['n_pass_a'] for s in stats) / n
    avg_pass_b = sum(s['n_pass_b'] for s in stats) / n
    avg_steps = sum(s['steps'] for s in stats) / n

    print(f"╔══════════════════════════════════════════════════════════════╗")
    print(f"║                AI vs AI  对战结果                              ║")
    print(f"╠══════════════════════════════════════════════════════════════╣")
    print(f"║  对局数：{n}    seed={args.seed}                                  ║")
    print(f"╠═════════════════════════╦═══════════════════╦══════════════════╣")
    print(f"║  指标                   ║   {label_a[:15]:<15}   ║   {label_b[:15]:<15}   ║")
    print(f"╠═════════════════════════╬═══════════════════╬══════════════════╣")
    print(f"║  总胜率                 ║      {wins_a/n:6.1%}        ║      {wins_b/n:6.1%}        ║")
    print(f"║    其中先手             ║   {wins_a_first}/{n_first} ({wins_a_first/n_first:.1%})   ║   {n_second-wins_a_second}/{n_second} ({(n_second-wins_a_second)/n_second:.1%})  ║")
    print(f"║    其中后手             ║   {wins_a_second}/{n_second} ({wins_a_second/n_second:.1%})   ║   {n_first-wins_a_first}/{n_first} ({(n_first-wins_a_first)/n_first:.1%})  ║")
    print(f"║  平均得分               ║     {avg_score_a:+6.2f}        ║     {avg_score_b:+6.2f}        ║")
    print(f"║  春天获胜局数           ║      {spring_a}            ║      {spring_b}            ║")
    print(f"║  平均炸弹/局            ║      {avg_bomb_a:5.2f}        ║      {avg_bomb_b:5.2f}        ║")
    print(f"║  平均 Pass/局           ║      {avg_pass_a:5.2f}        ║      {avg_pass_b:5.2f}        ║")
    print(f"╠═════════════════════════╩═══════════════════╩══════════════════╣")
    print(f"║  平均步数/局：{avg_steps:5.1f}                                          ║")
    print(f"╚══════════════════════════════════════════════════════════════╝")


if __name__ == '__main__':
    main()
