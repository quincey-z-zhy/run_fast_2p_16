"""
并行向量化游戏环境 (SubprocVecGame)。

N 个子进程各维护一个游戏环境 + 本地 CPU 对手，主进程统一做 GPU 批量推理。

设计要点：
- 游戏步骤在 N 个 CPU 核上真正并行，不再是主进程串行瓶颈
- 每轮 rollout 只需 steps_per_env 次 GPU 批量推理（batch=N），而非 N×steps_per_env 次单步推理
- 对手轮由 worker 本地处理，无需 IPC；仅 model 轮需要发送 obs/action

消息协议（主进程 → worker）：
  ('init', model_player: int, opp_state_dict: dict | None)
  ('action', action: int, kicker_faces: list[int] | None)
  ('sync_opp', state_dict: dict)   -- 更新 worker 本地对手权重，在下一个 action 之前插入
  ('close',)

消息协议（worker → 主进程）：
  ('obs', ep_r: float, is_done: bool, obs: ndarray, mask: ndarray, hc: ndarray)

  ep_r / is_done：上一个 action 的执行结果
    - ep_r=0.0, is_done=False：局继续
    - ep_r≠0, is_done=True：局已结束，ep_r = final_rewards[model_player]
  obs / mask / hc：当前 model 轮的状态（已处理完所有对手轮，可能已跨局重置）
"""

import multiprocessing as mp
from multiprocessing.connection import Connection
import sys
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parent.parent.parent


# ── Worker 子进程 ──────────────────────────────────────────────────────────────

def _env_worker(conn: Connection, obs_dim: int, act_dim: int, hidden: int,
                opp_type: str = 'model'):
    """
    子进程主循环。

    opp_type:
      'model'     — 使用本地 ActorCritic（权重由主进程 sync_opp 维护）
      'random'    — RandomLegalAgent
      'greedy'    — GreedyLowAgent
    """
    root = str(_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)

    from src.env.game_env import RunFast2pEnv
    from src.engine.action_generator import get_hand_counts

    env = RunFast2pEnv()
    model_player: int = 0

    # ── 本地对手 ──────────────────────────────────────────────────────────
    if opp_type == 'random':
        from src.eval.baselines import RandomLegalAgent
        _opp_agent = RandomLegalAgent()

        def opp_act(obs, mask):
            return _opp_agent.act(obs, mask)

        def sync_opp(sd):
            pass  # 无模型，无需同步

    elif opp_type == 'greedy':
        from src.eval.baselines import GreedyLowAgent
        _opp_agent = GreedyLowAgent()

        def opp_act(obs, mask):
            return _opp_agent.act(obs, mask)

        def sync_opp(sd):
            pass

    else:  # 'model'
        from src.training.actor_critic import ActorCritic
        _opp_model = ActorCritic(obs_dim=obs_dim, act_dim=act_dim, hidden=hidden)
        _opp_model.eval()

        def opp_act(obs, mask):
            with torch.no_grad():
                a, *_ = _opp_model.get_action_and_value(
                    torch.FloatTensor(obs).unsqueeze(0),
                    torch.BoolTensor(mask).unsqueeze(0),
                )
            return a.item()

        def sync_opp(sd):
            _opp_model.load_state_dict(sd)
            _opp_model.eval()

    # ── 辅助函数 ──────────────────────────────────────────────────────────

    def skip_opp_turns(obs, info):
        """推进对手轮，直到 model 轮或本局结束。
        返回 (obs, info, ep_r_or_None)。"""
        while info['current_player'] != model_player:
            a = opp_act(obs, info['action_mask'])
            obs, _, done, _, info = env.step(a)
            if done:
                return obs, info, float(info['final_rewards'][model_player])
        return obs, info, None

    def reset_to_model_turn():
        """重置并推进到 model 轮（交替身份）。对手连赢则最多重置 3 次。"""
        nonlocal model_player
        for _ in range(3):
            obs, info = env.reset()
            model_player = 1 - model_player
            obs, info, ep_r = skip_opp_turns(obs, info)
            if ep_r is None:
                return obs, info
            # 极少见：对手在 model 行动前赢了，再重置
        return obs, info  # fallback

    # ── 初始化 ────────────────────────────────────────────────────────────
    cmd, *args = conn.recv()
    assert cmd == 'init', f"期待 init，收到 {cmd}"
    model_player = args[0]
    if opp_type == 'model' and args[1] is not None:
        sync_opp(args[1])

    obs, info = env.reset()
    obs, info, ep_r0 = skip_opp_turns(obs, info)
    if ep_r0 is not None:
        obs, info = reset_to_model_turn()

    hc = get_hand_counts(env._game.players[env._game.current_player_index].cards)
    conn.send(('obs', 0.0, False, obs, info['action_mask'].copy(), hc.copy()))

    # ── 主循环 ────────────────────────────────────────────────────────────
    pending_ep_r: float = 0.0
    pending_is_done: bool = False

    while True:
        # 接收命令（sync_opp 可在 action 之前多次插入）
        while True:
            msg = conn.recv()
            if msg[0] == 'sync_opp':
                sync_opp(msg[1])
            elif msg[0] == 'close':
                env.close()
                conn.close()
                return
            else:  # 'action'
                action, kicker_faces = msg[1], msg[2]
                break

        # 执行 model 动作
        obs, _, done, _, info = env.step(
            action, kicker_faces=kicker_faces if kicker_faces else None
        )

        if done:
            pending_ep_r = float(info['final_rewards'][model_player])
            pending_is_done = True
            obs, info = reset_to_model_turn()
        else:
            obs, info, opp_ep_r = skip_opp_turns(obs, info)
            if opp_ep_r is not None:
                pending_ep_r = opp_ep_r
                pending_is_done = True
                obs, info = reset_to_model_turn()
            else:
                pending_ep_r = 0.0
                pending_is_done = False

        hc = get_hand_counts(env._game.players[env._game.current_player_index].cards)
        conn.send(('obs', pending_ep_r, pending_is_done,
                   obs, info['action_mask'].copy(), hc.copy()))
        pending_ep_r = 0.0
        pending_is_done = False


# ── SubprocVecGame ────────────────────────────────────────────────────────────

class SubprocVecGame:
    """
    N 个子进程并行游戏环境管理器。

    pending_obs[i] 存放 worker i 最新发来的 obs（等待主进程推理）：
      (ep_r: float, is_done: bool, obs: ndarray, mask: ndarray, hc: ndarray)

    主进程在每次 rollout 开始时读取 pending_obs → 批量 GPU 推理 → 调用 step() → 更新 pending_obs。
    """

    def __init__(self, num_envs: int, obs_dim: int, act_dim: int, hidden: int,
                 opp_type: str = 'model'):
        self.num_envs = num_envs
        self.obs_dim = obs_dim
        self.opp_type = opp_type

        ctx = mp.get_context('spawn')
        self._conns: list[Connection] = []
        self._procs: list[mp.Process] = []

        for _ in range(num_envs):
            parent_conn, child_conn = ctx.Pipe(duplex=True)
            proc = ctx.Process(
                target=_env_worker,
                args=(child_conn, obs_dim, act_dim, hidden, opp_type),
                daemon=True,
            )
            proc.start()
            child_conn.close()
            self._conns.append(parent_conn)
            self._procs.append(proc)

        # (ep_r, is_done, obs, mask, hc) per worker
        self.pending_obs: list[tuple] = [None] * num_envs  # type: ignore

    def initialize(self, model_players: list[int], opp_state_dict):
        """发送 init 命令，接收所有 worker 的首个 obs。"""
        for i, conn in enumerate(self._conns):
            conn.send(('init', model_players[i], opp_state_dict))
        for i, conn in enumerate(self._conns):
            msg = conn.recv()
            self.pending_obs[i] = msg[1:]  # (ep_r, is_done, obs, mask, hc)

    def step(self, actions: list[int], kicker_faces_list: list) -> list[tuple]:
        """
        向所有 worker 发送 action（同步），阻塞等待所有 worker 回包。
        返回 N 个 (ep_r, is_done, obs, mask, hc)。
        """
        for i, conn in enumerate(self._conns):
            conn.send(('action', actions[i], kicker_faces_list[i]))
        results = []
        for conn in self._conns:
            msg = conn.recv()
            results.append(msg[1:])  # strip 'obs' tag
        return results

    def sync_opp(self, state_dict: dict):
        """向所有 worker 广播新对手权重（在下一个 action 之前生效）。"""
        for conn in self._conns:
            conn.send(('sync_opp', state_dict))

    def close(self):
        for conn in self._conns:
            try:
                conn.send(('close',))
            except Exception:
                pass
        for proc in self._procs:
            proc.join(timeout=3)
            if proc.is_alive():
                proc.kill()
