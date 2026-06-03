"""
评估脚本：将训练好的 PPO 模型对战基线对手。

用法：
    python src/eval/evaluate.py \\
        --model checkpoints/baseline.pt \\
        --opponent random_legal \\
        --games 1000 \\
        --seed 42
"""

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))

import numpy as np
import torch

from src.engine.action_generator import get_hand_counts, get_kicker_info
from src.env.game_env import RunFast2pEnv
from src.eval.baselines import RandomLegalAgent, GreedyLowAgent
from src.training.actor_critic import ActorCritic


def load_model(checkpoint_path: str, device: str = 'cpu') -> ActorCritic:
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = ActorCritic()
    missing, _ = model.load_state_dict(ckpt['model_state_dict'], strict=False)
    if missing:
        print(f"[load_model] 带牌头权重未找到（旧版 checkpoint），从随机初始化开始：{missing}")
    model.eval()
    return model.to(device)


def select_action_with_kickers(
    model: ActorCritic,
    obs: np.ndarray,
    mask: np.ndarray,
    game,
    device: str,
) -> tuple[int, list[int]]:
    """两阶段动作选择：主动作（greedy argmax）+ 带牌头采样。"""
    obs_t = torch.FloatTensor(obs).unsqueeze(0).to(device)
    mask_t = torch.BoolTensor(mask).unsqueeze(0).to(device)
    with torch.no_grad():
        action_t, _, _, _, kicker_logits_t = model.get_full_action_and_value(obs_t, mask_t)
    action = action_t.item()

    hand_counts = get_hand_counts(game.players[game.current_player_index].cards)
    kicker_count, _, avail_13d = get_kicker_info(action, hand_counts)
    kicker_faces: list[int] = []
    if kicker_count > 0:
        kicker_faces, _ = model.sample_kickers(kicker_logits_t[0], avail_13d, kicker_count, deterministic=True)

    return action, kicker_faces


class ModelAgent:
    """将 ActorCritic checkpoint 封装为与 baseline agent 相同接口的对手。"""

    def __init__(self, checkpoint_path: str, device: str = 'cpu'):
        self.model = load_model(checkpoint_path, device)
        self.device = device
        self._env_ref = None  # 由调用方在每局开始时注入

    def act(self, obs: np.ndarray, mask: np.ndarray) -> int:
        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        mask_t = torch.BoolTensor(mask).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action_t, _, _, _, _ = self.model.get_full_action_and_value(obs_t, mask_t)
        return action_t.item()


def run_evaluation(
    model_path: str,
    opponent_name: str,
    n_games: int,
    seed: int,
    device: str = 'cpu',
    verbose: bool = True,
    opponent_model_path: str | None = None,
) -> dict:
    """
    对战评估。返回评估结果 dict。

    model_path:           模型 checkpoint 路径
    opponent_name:        'random_legal'、'greedy_low' 或 'model'
    opponent_model_path:  当 opponent_name='model' 时，对手 checkpoint 路径
    n_games:              总对局数（前半作为先手，后半作为后手）
    seed:                 评估随机种子（用于环境初始化）
    """
    rng = random.Random(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    model = load_model(model_path, device)
    env = RunFast2pEnv()

    if opponent_name == 'random_legal':
        opponent = RandomLegalAgent()
    elif opponent_name == 'greedy_low':
        opponent = GreedyLowAgent()
    elif opponent_name == 'model':
        if not opponent_model_path:
            raise ValueError("opponent_name='model' 时必须提供 --opponent-model 路径")
        opponent = ModelAgent(opponent_model_path, device)
    else:
        raise ValueError(f"未知对手: {opponent_name}")

    wins = 0
    total_score = 0.0
    opponent_remaining = []
    spring_count = 0
    bomb_counts = []
    step_counts = []
    scores = []

    for ep in range(n_games):
        # 前半局 model 作 player0，后半局 model 作 player1
        model_player = 0 if ep < n_games // 2 else 1
        ep_seed = seed * 10000 + ep

        obs, info = env.reset(seed=ep_seed)
        done = False
        steps = 0

        while not done:
            cur = info['current_player']
            mask = info['action_mask']

            if cur == model_player:
                action, kicker_faces = select_action_with_kickers(model, obs, mask, env._game, device)
                obs, rew, terminated, truncated, info = env.step(
                    action, kicker_faces=kicker_faces if kicker_faces else None
                )
            else:
                action = opponent.act(obs, mask)
                obs, rew, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            steps += 1

        # 解析最终结果
        model_score = info['final_rewards'][model_player]
        opp_score = info['final_rewards'][1 - model_player]
        scores.append(model_score)
        step_counts.append(steps)

        if env._game.players[model_player].is_winner:
            wins += 1
            # 对手剩余手牌数（近似用对手得分 / base计算比较复杂，直接记0）
            opponent_remaining.append(info.get('hand_size_opponent', 0))

        total_score += model_score

        # 春天检测（得分 ≥ 30 分基础 + 30 春天奖励 = 较高，简单阈值检测）
        if model_score >= 31:
            spring_count += 1

    win_rate = wins / n_games
    avg_score = total_score / n_games
    avg_steps = sum(step_counts) / n_games

    result = {
        'model': model_path,
        'opponent': opponent_name,
        'n_games': n_games,
        'eval_seed': seed,
        'win_rate': round(win_rate, 4),
        'wins': wins,
        'avg_score': round(avg_score, 4),
        'avg_steps': round(avg_steps, 1),
        'spring_rate': round(spring_count / n_games, 4),
        'score_min': round(min(scores), 1),
        'score_max': round(max(scores), 1),
    }

    opp_label = (os.path.basename(opponent_model_path)
                 if opponent_name == 'model' and opponent_model_path
                 else opponent_name)
    if verbose:
        print(f"\n=== 评估结果：{os.path.basename(model_path)} vs {opp_label} ===")
        print(f"对局数:      {n_games}")
        print(f"胜率:        {win_rate:.1%}  ({wins}/{n_games})")
        print(f"平均得分:    {avg_score:.2f}")
        print(f"平均步数:    {avg_steps:.1f}")
        print(f"春天率:      {spring_count/n_games:.1%}")

    result['opponent_label'] = opp_label
    return result


def main():
    parser = argparse.ArgumentParser(description='PPO 模型评估')
    parser.add_argument('--model', required=True, help='模型 checkpoint 路径')
    parser.add_argument('--opponent', default='random_legal',
                        choices=['random_legal', 'greedy_low', 'model'],
                        help='对手类型')
    parser.add_argument('--opponent-model', default=None,
                        help='对手模型 checkpoint 路径（opponent=model 时使用）')
    parser.add_argument('--games', type=int, default=1000, help='对局数')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--device', default='cpu', help='推理设备')
    parser.add_argument('--output', default=None, help='结果 JSON 输出路径')
    args = parser.parse_args()

    result = run_evaluation(
        model_path=args.model,
        opponent_name=args.opponent,
        n_games=args.games,
        seed=args.seed,
        device=args.device,
        verbose=True,
        opponent_model_path=args.opponent_model,
    )

    if args.output:
        with open(args.output, 'w') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"\n结果已写入: {args.output}")
    else:
        print(f"\n结果 JSON:\n{json.dumps(result, indent=2, ensure_ascii=False)}")


if __name__ == '__main__':
    main()
