"""
PPO 基线训练脚本。

用法：
    python src/training/train.py --config src/training/config.yaml --stage baseline
    python src/training/train.py --config src/training/config.yaml --stage baseline \\
        --resume checkpoints/step_100000.pt
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import yaml

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))

from src.engine.action_generator import get_hand_counts, get_kicker_info
from src.env.game_env import RunFast2pEnv
from src.eval.baselines import RandomLegalAgent, GreedyLowAgent
from src.eval.evaluate import run_evaluation
from src.training.actor_critic import ActorCritic
from src.training.league import League
from src.training.ppo_trainer import PPOTrainer
from src.training.rollout_buffer import RolloutBuffer


class LeagueOpponent:
    """
    联赛对手：持有一个历史快照模型，定期通过 PFSP 切换对手。

    - 每局结束时以概率 switch_prob 重采样对手
    - 重采样时以概率 sp_prob 切换为当前学习模型（自博弈），否则 PFSP 选历史快照
    - 调用 on_episode_end(result) 更新 League 的胜率追踪
    """

    def __init__(
        self,
        league: League,
        learner_model: 'ActorCritic',
        obs_dim: int,
        act_dim: int,
        hidden: int,
        device: str,
        sp_prob: float = 0.2,
        switch_prob: float = 0.01,
    ):
        self.league = league
        self.learner_model = learner_model
        self.device = device
        self.sp_prob = sp_prob
        self.switch_prob = switch_prob

        self._opp_model = ActorCritic(obs_dim=obs_dim, act_dim=act_dim, hidden=hidden).to(device)
        self._opp_model.eval()
        self.current_pid: int | None = None
        self.is_selfplay: bool = False

    def initialize(self):
        """训练开始时调用，初始选一个对手。"""
        if not self.league.initialized():
            return
        self._resample()

    def act(self, obs: np.ndarray, mask: np.ndarray, game=None) -> int:
        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        mask_t = torch.BoolTensor(mask).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action_t, _, _, _ = self._opp_model.get_action_and_value(obs_t, mask_t)
        return action_t.item()

    def on_episode_end(self, result: float):
        """每局结束时调用：更新胜率追踪，以概率 switch_prob 切换对手。"""
        self.league.update_result(self.current_pid, result, selfplay=self.is_selfplay)
        if np.random.random() < self.switch_prob:
            self._resample()

    def _resample(self):
        if np.random.random() < self.sp_prob:
            self._opp_model.load_state_dict(self.learner_model.state_dict())
            self._opp_model.eval()
            self.is_selfplay = True
            self.current_pid = None
        else:
            pid, weights = self.league.select_opponent()
            self._opp_model.load_state_dict(weights, strict=False)
            self._opp_model.eval()
            self.current_pid = pid
            self.is_selfplay = False


class FrozenModelOpponent:
    """冻结的 ActorCritic 模型作为对手（自博弈用）。权重不参与梯度更新。"""

    def __init__(self, model: ActorCritic, device: str):
        self.model = model
        self.device = device
        self.model.eval()

    def act(self, obs: np.ndarray, action_mask: np.ndarray, game=None) -> int:
        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        mask_t = torch.BoolTensor(action_mask).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action_t, _, _, _ = self.model.get_action_and_value(obs_t, mask_t)
        return action_t.item()

    def sync_from(self, learner: ActorCritic):
        """将学习模型的权重同步到冻结对手。"""
        self.model.load_state_dict(learner.state_dict())
        self.model.eval()


# ── 工具函数 ─────────────────────────────────────────────────────────────────

def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def save_checkpoint(model, optimizer, step: int, config: dict, path: Path, extra: dict = None):
    ckpt = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'step': step,
        'config': config,
    }
    if extra:
        ckpt.update(extra)
    torch.save(ckpt, path)


def load_checkpoint(path: str, model, optimizer, device: str) -> int:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    missing, _ = model.load_state_dict(ckpt['model_state_dict'], strict=False)
    if missing:
        print(f"[resume] 带牌头权重未找到（旧版 checkpoint），从随机初始化继续：{missing}")
    optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    return ckpt['step']


def _cleanup_checkpoints(ckpt_dir: Path, manifest: dict, keep: int = 5):
    """保留最新 keep 个 step_*.pt，删除更早的。"""
    step_ckpts = sorted(
        [k for k in manifest['checkpoints'] if k.startswith('step_')],
        key=lambda k: manifest['checkpoints'][k]['step'],
    )
    to_delete = step_ckpts[:-keep] if len(step_ckpts) > keep else []
    for name in to_delete:
        p = ckpt_dir / name
        if p.exists():
            p.unlink()
        manifest['checkpoints'].pop(name, None)


# ── Rollout 收集 ──────────────────────────────────────────────────────────────

def collect_rollout(
    env: RunFast2pEnv,
    model: ActorCritic,
    opponent: RandomLegalAgent,
    n_steps: int,
    device: str,
    model_player_start: int = 0,
) -> tuple[RolloutBuffer, float, int]:
    """
    收集 n_steps 步的 model rollout（仅统计 model 的步数）。

    两人交替游戏处理：
    - model 回合：收集 (obs, action, log_prob, value, reward, done, mask)
    - 对手回合：跳过，但若对手走出最后一步，回填 model 最近一步的 reward/done

    返回：(buffer, last_value, next_model_player_start)
    """
    buffer = RolloutBuffer()
    model_player = model_player_start
    obs, info = env.reset()
    model_acted_this_ep = False  # 当前局 model 是否已行动过
    ep_final_reward: float = 0.0   # 当局终局时 model 的奖励（用于联赛胜率更新）

    while len(buffer) < n_steps:
        cur = info['current_player']
        mask = info['action_mask']

        if cur == model_player:
            # ── model 行动（两阶段：主动作 + 带牌头）──────────────────────
            obs_t = torch.FloatTensor(obs).unsqueeze(0).to(device)
            mask_t = torch.BoolTensor(mask).unsqueeze(0).to(device)

            with torch.no_grad():
                action_t, log_prob_main_t, _, value_t, kicker_logits_t = \
                    model.get_full_action_and_value(obs_t, mask_t)

            action = action_t.item()
            value = value_t.item()

            # 带牌头采样
            hand_counts = get_hand_counts(env._game.players[env._game.current_player_index].cards)
            kicker_count, _, avail_13d = get_kicker_info(action, hand_counts)

            kicker_faces_arr = np.full(8, -1, dtype=np.int32)  # 最多 8 张带牌（s_trips length=4 × mode=2）
            kicker_faces_list: list[int] = []
            log_prob_kicker = 0.0

            if kicker_count > 0:
                kicker_faces_list, log_prob_kicker = model.sample_kickers(
                    kicker_logits_t[0], avail_13d, kicker_count
                )
                for i, f in enumerate(kicker_faces_list):
                    kicker_faces_arr[i] = f - 3  # 存储为 0-indexed 面值索引

            total_log_prob = log_prob_main_t.item() + log_prob_kicker

            next_obs, _, terminated, truncated, next_info = env.step(
                action,
                kicker_faces=kicker_faces_list if kicker_count > 0 else None,
            )
            done = terminated or truncated

            # 稀疏奖励：仅终局时有奖励
            reward = next_info['final_rewards'][model_player] if terminated else 0.0
            if terminated:
                ep_final_reward = reward
            buffer.add(obs, action, total_log_prob, value, reward, done, mask,
                       kicker_faces_arr, avail_13d)
            model_acted_this_ep = True

        else:
            # ── 对手行动 ────────────────────────────────────────────────────
            opp_action = opponent.act(obs, mask)
            next_obs, _, terminated, truncated, next_info = env.step(opp_action)
            done = terminated or truncated

            # 对手走出最后一步 → 回填 model 最近一步的 reward 和 done 标志
            if terminated and model_acted_this_ep:
                ep_final_reward = next_info['final_rewards'][model_player]
                buffer.rewards[-1] = ep_final_reward
                buffer.dones[-1] = True

        obs = next_obs
        info = next_info

        if done:
            # 联赛胜率更新（LeagueOpponent 专用）
            if terminated and hasattr(opponent, 'on_episode_end'):
                opponent.on_episode_end(ep_final_reward)
            # 新一局交替身份
            ep_final_reward = 0.0
            model_player = 1 - model_player
            model_acted_this_ep = False
            obs, info = env.reset()

    # ── 计算 GAE bootstrap 所需的末尾价值 ───────────────────────────────────
    if buffer.dones[-1]:
        last_value = 0.0
    elif info['current_player'] == model_player:
        # 轮到 model 行动，直接评估
        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(device)
        with torch.no_grad():
            _, v = model(obs_t)
        last_value = v.squeeze().item()
    else:
        # 对手回合，以 0 近似（truncation 偏差在多局均值下可忽略）
        last_value = 0.0

    return buffer, last_value, model_player


# ── 向量化 Rollout 收集（多进程版）──────────────────────────────────────────────

def merge_buffers(buffers: list[RolloutBuffer]) -> RolloutBuffer:
    """将 N 个 per-env RolloutBuffer 合并为一个，用于 PPO 更新。"""
    merged = RolloutBuffer()
    for buf in buffers:
        merged.obs.extend(buf.obs)
        merged.actions.extend(buf.actions)
        merged.log_probs.extend(buf.log_probs)
        merged.values.extend(buf.values)
        merged.rewards.extend(buf.rewards)
        merged.dones.extend(buf.dones)
        merged.masks.extend(buf.masks)
        merged.kicker_faces.extend(buf.kicker_faces)
        merged.kicker_avails.extend(buf.kicker_avails)
    return merged


def collect_rollout_vec_mp(
    vec,          # SubprocVecGame
    model: ActorCritic,
    opponent,
    n_steps: int,
    device: str,
) -> tuple[list[RolloutBuffer], list[float]]:
    """
    向量化并行 rollout 收集（N 个子进程）。

    设计要点：
    - vec.pending_obs[j] 存放 worker j 等待推理的最新 obs
    - 每步：先回填上步 reward/done → 批量 GPU 推理 → 发 action → 更新 pending_obs
    - 最后一步结果在循环结束后回填
    - 返回 N 个独立 per-env buffer（保证 GAE 在 episode 边界处正确）

    reward 回填规则：
      worker 发来 ('obs', ep_r, is_done, ...) 时，ep_r/is_done 是上一个 action 的结果。
      主进程在下一步开始时用 ep_r/is_done 回填上一步的 buffer 条目。
    """
    num_envs = vec.num_envs
    steps_per_env = n_steps // num_envs

    buffers = [RolloutBuffer() for _ in range(num_envs)]
    last_values = [0.0] * num_envs

    for step_idx in range(steps_per_env):
        # ── 回填上步 reward/done（从 step_idx=1 开始）────────────────────
        if step_idx > 0:
            for j in range(num_envs):
                ep_r, is_done, _, _, _ = vec.pending_obs[j]
                buffers[j].rewards[-1] = ep_r
                buffers[j].dones[-1]   = is_done
                if is_done and hasattr(opponent, 'on_episode_end'):
                    opponent.on_episode_end(1.0 if ep_r > 0 else 0.0)

        # ── 批量 GPU 推理 ─────────────────────────────────────────────────
        obs_arr  = np.stack([vec.pending_obs[j][2] for j in range(num_envs)])
        mask_arr = np.stack([vec.pending_obs[j][3] for j in range(num_envs)])
        hc_list  = [vec.pending_obs[j][4] for j in range(num_envs)]

        obs_t  = torch.FloatTensor(obs_arr).to(device)
        mask_t = torch.BoolTensor(mask_arr).to(device)

        with torch.no_grad():
            action_t, log_prob_t, _, value_t, kicker_logits_t = \
                model.get_full_action_and_value(obs_t, mask_t)

        actions_out:  list[int]       = []
        kickers_out:  list[list[int]] = []

        for j in range(num_envs):
            _, _, obs, mask, hc = vec.pending_obs[j]
            action = action_t[j].item()
            value  = value_t[j].item()
            lp     = log_prob_t[j].item()

            kicker_count, _, avail_13d = get_kicker_info(action, hc)
            kicker_arr  = np.full(8, -1, dtype=np.int32)
            kicker_list: list[int] = []
            lp_kicker   = 0.0

            if kicker_count > 0:
                kicker_list, lp_kicker = model.sample_kickers(
                    kicker_logits_t[j], avail_13d, kicker_count
                )
                for k, f in enumerate(kicker_list):
                    kicker_arr[k] = f - 3

            # 用占位 reward=0/done=False 写入 buffer；下一步回填真实值
            buffers[j].add(obs, action, lp + lp_kicker, value,
                           0.0, False, mask, kicker_arr, avail_13d)
            actions_out.append(action)
            kickers_out.append(kicker_list)

        # ── 推进所有 worker，接收新 pending_obs ──────────────────────────
        results = vec.step(actions_out, kickers_out)
        for j, result in enumerate(results):
            vec.pending_obs[j] = result

    # ── 回填最后一步结果 + 计算 bootstrap value ──────────────────────────
    for j in range(num_envs):
        ep_r, is_done, obs, _, _ = vec.pending_obs[j]
        buffers[j].rewards[-1] = ep_r
        buffers[j].dones[-1]   = is_done
        if is_done and hasattr(opponent, 'on_episode_end'):
            opponent.on_episode_end(1.0 if ep_r > 0 else 0.0)

        if is_done:
            last_values[j] = 0.0
        else:
            obs_t = torch.FloatTensor(obs).unsqueeze(0).to(device)
            with torch.no_grad():
                _, v = model(obs_t)
            last_values[j] = v.squeeze().item()

    return buffers, last_values


# ── 主训练循环 ─────────────────────────────────────────────────────────────────

def train(args):
    cfg = load_config(args.config)
    t_cfg = cfg['training']
    ppo_cfg = cfg['ppo']
    net_cfg = cfg['network']

    device = args.device
    stage = args.stage

    # ── 日志和 checkpoint 目录 ────────────────────────────────────────────────
    run_name = f"{stage}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    log_dir = Path(cfg['paths']['log_dir']) / run_name
    log_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = Path(cfg['paths']['checkpoint_dir'])
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # 保存本次运行超参快照
    with open(log_dir / 'config.yaml', 'w') as f:
        yaml.dump(cfg, f, allow_unicode=True)

    # ── 初始化模型和优化器 ────────────────────────────────────────────────────
    num_envs = t_cfg.get('num_envs', 1)
    # 单环境保留作 fallback（eval 用）；多环境由 SubprocVecGame 管理
    env = RunFast2pEnv()

    model = ActorCritic(
        obs_dim=net_cfg['obs_dim'],
        act_dim=net_cfg['act_dim'],
        hidden=net_cfg['hidden_dim'],
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=ppo_cfg['learning_rate'])

    trainer = PPOTrainer(
        model=model,
        optimizer=optimizer,
        n_epochs=t_cfg['n_epochs'],
        batch_size=t_cfg['batch_size'],
        clip_eps=ppo_cfg['clip_eps'],
        value_coeff=ppo_cfg['value_coeff'],
        entropy_coeff=ppo_cfg['entropy_coeff'],
        max_grad_norm=ppo_cfg['max_grad_norm'],
        normalize_advantages=ppo_cfg['normalize_advantages'],
        device=device,
    )

    # ── 对手初始化 ────────────────────────────────────────────────────────────
    league_cfg = cfg.get('league', {})
    league_enabled = league_cfg.get('enabled', False)
    sp_cfg = cfg.get('self_play', {})
    self_play_enabled = sp_cfg.get('enabled', False) and not league_enabled

    league: League | None = None
    ppo_update_count = 0

    if league_enabled:
        league = League(
            output_dir=str(ckpt_dir),
            tracker_n=league_cfg.get('tracker_n', 200),
            last_num=league_cfg.get('last_num', 30),
        )
        # 断点续训：恢复联赛快照元数据（权重仍在磁盘上，延迟加载）
        if args.resume:
            league.load_manifest()
        # LeagueOpponent 在此创建，但首个快照和 initialize() 延迟到 checkpoint 加载之后
        opponent = LeagueOpponent(
            league=league,
            learner_model=model,
            obs_dim=net_cfg['obs_dim'],
            act_dim=net_cfg['act_dim'],
            hidden=net_cfg['hidden_dim'],
            device=device,
            sp_prob=league_cfg.get('sp_prob', 0.2),
            switch_prob=league_cfg.get('switch_prob', 0.01),
        )
        sp_update_freq = None
        next_sp_update_at = float('inf')
    elif self_play_enabled:
        opp_model = ActorCritic(
            obs_dim=net_cfg['obs_dim'],
            act_dim=net_cfg['act_dim'],
            hidden=net_cfg['hidden_dim'],
        ).to(device)
        opp_init = sp_cfg.get('opponent_init', None)
        if opp_init and Path(opp_init).exists():
            ckpt = torch.load(opp_init, map_location=device, weights_only=False)
            opp_model.load_state_dict(ckpt['model_state_dict'])
        else:
            opp_model.load_state_dict(model.state_dict())
        opponent = FrozenModelOpponent(opp_model, device)
        sp_update_freq = sp_cfg.get('update_freq', 200_000)
        next_sp_update_at = sp_update_freq
    else:
        opponent = RandomLegalAgent()
        sp_update_freq = None
        next_sp_update_at = float('inf')

    # ── 断点续训 ──────────────────────────────────────────────────────────────
    global_step = 0
    if args.resume:
        global_step = load_checkpoint(args.resume, model, optimizer, device)
        if getattr(args, 'reset_steps', False):
            print(f"[resume] 加载权重自 step {global_step}，步数重置为 0（全新训练周期）")
            global_step = 0
        else:
            print(f"[resume] 从 step {global_step} 继续训练")
        if self_play_enabled:
            next_sp_update_at = ((global_step // sp_update_freq) + 1) * sp_update_freq

    # ── League：在 checkpoint 加载完成后才取首个快照 ──────────────────────────
    # 确保快照权重来自已加载的 checkpoint（而非随机初始化），修正 resume 路径的顺序问题
    if league_enabled and league is not None:
        if not league.initialized():
            pid0 = league.add_weight(model.state_dict())
        opponent.initialize()

    # ── 并行向量化环境（num_envs > 1）────────────────────────────────────────
    vec = None
    if num_envs > 1:
        from src.training.vecenv import SubprocVecGame
        if league_enabled or self_play_enabled:
            opp_type = 'model'
        elif isinstance(opponent, GreedyLowAgent):
            opp_type = 'greedy'
        else:
            opp_type = 'random'

        # 初始对手权重（仅 model 类型需要）
        if opp_type == 'model':
            if hasattr(opponent, '_opp_model'):
                init_opp_sd = opponent._opp_model.state_dict()
            elif hasattr(opponent, 'model'):
                init_opp_sd = opponent.model.state_dict()
            else:
                init_opp_sd = None
        else:
            init_opp_sd = None

        model_players_init = [i % 2 for i in range(num_envs)]
        vec = SubprocVecGame(
            num_envs=num_envs,
            obs_dim=net_cfg['obs_dim'],
            act_dim=net_cfg['act_dim'],
            hidden=net_cfg['hidden_dim'],
            opp_type=opp_type,
        )
        vec.initialize(model_players_init, init_opp_sd)

    # 计算下一个 eval / checkpoint 触发点
    eval_freq = t_cfg['eval_freq']
    ckpt_freq = t_cfg['checkpoint_freq']
    next_eval_at = ((global_step // eval_freq) + 1) * eval_freq
    next_ckpt_at = ((global_step // ckpt_freq) + 1) * ckpt_freq

    # ── Manifest（checkpoint 索引）─────────────────────────────────────────────
    manifest_path = ckpt_dir / 'manifest.json'
    if manifest_path.exists():
        with open(manifest_path) as f:
            manifest = json.load(f)
    else:
        manifest = {
            'checkpoints': {},
            'retention_policy': 'keep_latest_5_plus_best_plus_baseline',
        }

    # ── 日志文件 ──────────────────────────────────────────────────────────────
    log_file = open(log_dir / 'stdout.log', 'w', buffering=1)

    def log(msg: str):
        print(msg, flush=True)
        log_file.write(msg + '\n')

    if league_enabled:
        opp_label = f'league(n={league.n_agents()})'
        log(f"[league] 联赛初始化 | n_agents={league.n_agents()} | pid={opponent.current_pid} | selfplay={opponent.is_selfplay}")
    else:
        opp_label = 'self_play' if self_play_enabled else 'random_legal'
    log(f"[train] 开始训练 | stage={stage} | device={device} | opponent={opp_label}")
    log(f"[train] 总步数={t_cfg['total_steps']} | n_steps={t_cfg['n_steps']} | batch={t_cfg['batch_size']}")
    log(f"[train] lr={ppo_cfg['learning_rate']} | gamma={ppo_cfg['gamma']} | entropy_coeff={ppo_cfg['entropy_coeff']}")

    best_win_rate = 0.0
    model_player_start = 0
    t_start = time.time()
    step_start = global_step  # 用于 sps 计算，避免 resume 时 global_step 偏大导致 sps 虚高
    last_log_step = global_step
    log_freq = t_cfg['log_freq']

    if num_envs > 1:
        log(f"[train] 并行环境 num_envs={num_envs} | steps_per_env={t_cfg['n_steps'] // num_envs} | opp_type={opp_type}")

    # ── 主循环 ─────────────────────────────────────────────────────────────────
    while global_step < t_cfg['total_steps']:
        model.train()

        if vec is not None:
            # ── 向量化收集（并行多进程）─────────────────────────────────────
            # 每轮开始前同步最新对手权重到所有 worker
            if opp_type == 'model':
                if hasattr(opponent, '_opp_model'):
                    vec.sync_opp(opponent._opp_model.state_dict())
                elif hasattr(opponent, 'model'):
                    vec.sync_opp(opponent.model.state_dict())

            per_env_bufs, per_env_last_vals = collect_rollout_vec_mp(
                vec, model, opponent,
                n_steps=t_cfg['n_steps'],
                device=device,
            )

            # 各环境独立计算 GAE，再合并
            all_returns:    list[np.ndarray] = []
            all_advantages: list[np.ndarray] = []
            for buf, lv in zip(per_env_bufs, per_env_last_vals):
                r, a = buf.compute_returns_and_advantages(
                    last_value=lv,
                    gamma=ppo_cfg['gamma'],
                    gae_lambda=ppo_cfg['gae_lambda'],
                )
                all_returns.append(r)
                all_advantages.append(a)

            merged_buf = merge_buffers(per_env_bufs)
            merged_returns    = np.concatenate(all_returns)
            merged_advantages = np.concatenate(all_advantages)

            losses = trainer.update(merged_buf, merged_returns, merged_advantages)
            global_step += len(merged_buf)

        else:
            # ── 原始单环境收集 ───────────────────────────────────────────────
            buffer, last_value, model_player_start = collect_rollout(
                env, model, opponent,
                n_steps=t_cfg['n_steps'],
                device=device,
                model_player_start=model_player_start,
            )

            returns, advantages = buffer.compute_returns_and_advantages(
                last_value=last_value,
                gamma=ppo_cfg['gamma'],
                gae_lambda=ppo_cfg['gae_lambda'],
            )

            losses = trainer.update(buffer, returns, advantages)
            global_step += len(buffer)

        # ── 联赛快照（League 模式）────────────────────────────────────────────
        if league_enabled and league is not None:
            ppo_update_count += 1
            snapshot_gap = league_cfg.get('snapshot_gap', 200)
            snapshot_upscore = league_cfg.get('snapshot_upscore', 1.5)
            should_snapshot = (
                league.score_all_above(snapshot_upscore)
                or (ppo_update_count % snapshot_gap == 0)
            )
            if should_snapshot:
                pid = league.add_weight(model.state_dict())
                ppo_update_count = 0
                stats = league.get_stats()
                log(
                    f"[league {global_step:>9d}] 新快照 pid={pid} | "
                    f"n_agents={stats['n_agents']} | "
                    f"sp_score={stats['selfplay_score']:.2f} | "
                    f"recent_scores={stats['recent_scores']}"
                )

        # ── 自博弈对手更新（非 League 模式）─────────────────────────────────
        if self_play_enabled and not league_enabled and global_step >= next_sp_update_at:
            next_sp_update_at += sp_update_freq
            opponent.sync_from(model)
            log(f"[sp    {global_step:>9d}] 对手模型已同步（self-play update）")

        # ── 日志 ────────────────────────────────────────────────────────────
        if global_step - last_log_step >= log_freq:
            elapsed = time.time() - t_start
            sps = (global_step - step_start) / elapsed if elapsed > 0 else 0
            log(
                f"[step {global_step:>9d}] "
                f"policy={losses['policy_loss']:+.4f} "
                f"value={losses['value_loss']:.4f} "
                f"entropy={losses['entropy']:.4f} "
                f"sps={sps:.0f}"
            )
            last_log_step = global_step
            # 联赛 manifest 周期性刷盘（确保 league table 实时可见）
            if league_enabled and league is not None:
                league._save_manifest()

        # ── 评估 ────────────────────────────────────────────────────────────
        if global_step >= next_eval_at:
            next_eval_at += eval_freq
            model.eval()

            eval_ckpt = ckpt_dir / '_eval_tmp.pt'
            save_checkpoint(model, optimizer, global_step, cfg, eval_ckpt)

            eval_opponents_list = cfg.get('eval_opponents', ['random_legal'])
            eval_results = {}
            for opp_name in eval_opponents_list:
                r = run_evaluation(
                    model_path=str(eval_ckpt),
                    opponent_name=opp_name,
                    n_games=t_cfg['eval_games'],
                    seed=42,
                    device=device,
                    verbose=False,
                )
                eval_results[opp_name] = r
                threshold_key = 'stage_7_heuristic' if opp_name == 'greedy_low' else 'stage_7_main'
                threshold = cfg['eval_thresholds'].get(threshold_key, 0.80)
                log(
                    f"[eval  {global_step:>9d}] vs {opp_name}: "
                    f"win_rate={r['win_rate']:.3f} (目标≥{threshold:.2f}) "
                    f"avg_score={r['avg_score']:.2f}"
                )

            # 以 greedy_low 为主排名指标（若有），否则用 random_legal
            primary_opp = 'greedy_low' if 'greedy_low' in eval_results else 'random_legal'
            win_rate = eval_results[primary_opp]['win_rate']

            if win_rate > best_win_rate:
                best_win_rate = win_rate
                best_path = ckpt_dir / 'best.pt'
                save_checkpoint(model, optimizer, global_step, cfg, best_path,
                                extra={
                                    'win_rate': win_rate,
                                    'eval': {k: v['win_rate'] for k, v in eval_results.items()},
                                })
                manifest['checkpoints']['best.pt'] = {
                    'step': global_step,
                    'eval': {f'win_rate_vs_{k}': v['win_rate'] for k, v in eval_results.items()},
                    'created_at': datetime.now(timezone.utc).isoformat(),
                }
                log(f"[best  {global_step:>9d}] 新最优 vs {primary_opp}={win_rate:.3f} → {best_path}")

            eval_ckpt.unlink(missing_ok=True)
            model.train()

        # ── Checkpoint ──────────────────────────────────────────────────────
        if global_step >= next_ckpt_at:
            next_ckpt_at += ckpt_freq
            ckpt_name = f'step_{global_step}.pt'
            ckpt_path = ckpt_dir / ckpt_name
            save_checkpoint(model, optimizer, global_step, cfg, ckpt_path)
            manifest['checkpoints'][ckpt_name] = {
                'step': global_step,
                'eval': {},
                'created_at': datetime.now(timezone.utc).isoformat(),
            }
            with open(manifest_path, 'w') as f:
                json.dump(manifest, f, indent=2, ensure_ascii=False)
            _cleanup_checkpoints(ckpt_dir, manifest, keep=5)
            log(f"[ckpt  {global_step:>9d}] 保存 {ckpt_path}")

    # ── 训练结束：保存 baseline checkpoint ──────────────────────────────────
    elapsed = time.time() - t_start
    log(f"[done ] 训练完成 | step={global_step} | best_win_rate={best_win_rate:.3f} | elapsed={elapsed:.0f}s")

    baseline_path = ckpt_dir / 'baseline.pt'
    # 保存当前模型（如果 best_win_rate 没超过当前，保存 best.pt 的副本）
    best_path = ckpt_dir / 'best.pt'
    if best_path.exists():
        import shutil
        shutil.copy2(best_path, baseline_path)
    else:
        save_checkpoint(model, optimizer, global_step, cfg, baseline_path,
                        extra={'win_rate': best_win_rate})

    manifest['checkpoints']['baseline.pt'] = {
        'step': global_step,
        'eval': {'win_rate_vs_random': best_win_rate},
        'created_at': datetime.now(timezone.utc).isoformat(),
    }
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    log(f"[done ] baseline.pt 已保存至 {baseline_path}")
    log_file.close()

    # 关闭并行环境子进程
    if vec is not None:
        vec.close()


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='PPO 训练（两人跑得快）')
    parser.add_argument('--config', default='src/training/config.yaml',
                        help='超参数配置文件路径')
    parser.add_argument('--stage', default='baseline',
                        choices=['baseline', 'formal', 'league'],
                        help='训练阶段标识（用于日志命名）')
    parser.add_argument('--resume', default=None,
                        help='断点续训的 checkpoint 路径')
    parser.add_argument('--reset_steps', action='store_true',
                        help='加载 checkpoint 权重但将步数重置为 0（全新训练周期）')
    parser.add_argument('--device',
                        default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='训练设备（cuda / cpu）')
    args = parser.parse_args()
    train(args)


if __name__ == '__main__':
    main()
