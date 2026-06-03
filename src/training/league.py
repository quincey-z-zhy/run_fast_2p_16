"""
League training utilities — adapted from fast_run_ai.

核心组件：
- pfsp:             Prioritized Fictitious Self-Play 采样权重
- ScoreTracker:     每个历史对手的平均得分指数滑动平均（原 WinrateTracker）
- League:           历史快照池（无 Ray 依赖，纯本地磁盘 + manifest）

快照触发逻辑：对所有近期对手的平均得分均超过 snapshot_upscore 时触发。
"""

import json
import os

import numpy as np
import torch


# ── PFSP ──────────────────────────────────────────────────────────────────────

# 游戏得分归一化范围（用于 PFSP 权重计算）
_SCORE_MIN = -30.0
_SCORE_MAX =  40.0


def pfsp(win_rates: list[float], weighting: str = "squared") -> np.ndarray:
    """Prioritized Fictitious Self-Play 采样概率。

    胜率越低的对手（越难赢）→ 被选中概率越高。
    win_rates 被 clip 到 0.95 防止权重坍塌。
    """
    win_rates = [min(w, 0.95) for w in win_rates]
    weightings = {
        "variance":      lambda x: x * (1 - x),
        "linear":        lambda x: 1 - x,
        "linear_capped": lambda x: np.minimum(0.5, 1 - x),
        "squared":       lambda x: (1 - x) ** 2,
    }
    fn = weightings[weighting]
    probs = fn(np.asarray(win_rates, dtype=np.float32))
    norm = probs.sum()
    if norm < 1e-10:
        return np.ones(len(win_rates), dtype=np.float32) / len(win_rates)
    return probs / norm


def pfsp_from_scores(scores: list[float], weighting: str = "squared") -> np.ndarray:
    """基于平均得分的 PFSP 采样概率。

    得分越低（越难赢）→ 被选中概率越高。
    先将得分线性归一化到 [0, 1]，再套用标准 pfsp 公式。
    """
    span = _SCORE_MAX - _SCORE_MIN
    normalized = [float(np.clip((s - _SCORE_MIN) / span, 0.0, 1.0)) for s in scores]
    return pfsp(normalized, weighting)


# ── WinrateTracker ────────────────────────────────────────────────────────────

class ScoreTracker:
    """每个历史对手的平均得分指数滑动平均追踪器。

    更新公式：v = v * (n-1)/n + result/n，其中 n = clip(self.n, nmin, nmax)。
    nmin/nmax 控制历史权重的衰减速率；nmin=nmax=200 时约等于 200 局滑动窗口。
    result 为每局结束时 model 的实际得分（可负）。
    """

    def __init__(self, nmin: int = 200, nmax: int = 200):
        self.n: int = 0
        self.v: float = 0.0      # 初始平均得分中性值（0 = 不赢不输）
        self.nmin = nmin
        self.nmax = nmax

    def update(self, result: float):
        self.n += 1
        clp_n = int(np.clip(self.n, self.nmin, self.nmax))
        self.v = self.v * (clp_n - 1) / clp_n + result / clp_n


# 向后兼容别名
WinrateTracker = ScoreTracker


# ── League ────────────────────────────────────────────────────────────────────

class League:
    """
    历史快照池。无 Ray 依赖，纯本地实现。

    快照权重存储为 output_dir/league/league_{pid}.pt，
    manifest.json 追踪 pid、胜率历史，支持断点续训。

    使用示例：
        league = League(output_dir='checkpoints', tracker_n=200, last_num=30)
        pid = league.add_weight(model.state_dict())   # 初始快照
        pid, sd = league.select_opponent()            # PFSP 采样
        league.update_result(pid, 1.0)                # 更新胜率
    """

    def __init__(self, output_dir: str, tracker_n: int = 200, last_num: int = 30):
        self.output_dir = output_dir
        self.tracker_n = tracker_n
        self.last_num = last_num

        self.pids: list[int] = []
        self.winrates: list[ScoreTracker] = []
        self.selfplay_winrate = ScoreTracker(tracker_n, tracker_n)
        self.current_pid: int = -1

        self._league_dir = os.path.join(output_dir, 'league')
        os.makedirs(self._league_dir, exist_ok=True)

    # ── 状态查询 ──────────────────────────────────────────────────────────────

    def initialized(self) -> bool:
        return len(self.pids) > 0

    def n_agents(self) -> int:
        return len(self.pids)

    # ── 快照管理 ──────────────────────────────────────────────────────────────

    def _ckpt_path(self, pid: int) -> str:
        return os.path.join(self._league_dir, f'league_{pid}.pt')

    def add_weight(self, state_dict: dict) -> int:
        """将当前模型快照加入联赛，返回新 pid。

        同时保留旧追踪器的历史胜率数据（仅扩展，不清零）。
        """
        self.current_pid += 1
        pid = self.current_pid
        self.pids.append(pid)

        torch.save({'model_state_dict': state_dict}, self._ckpt_path(pid))

        # 扩展得分追踪器，保留旧数据
        old = self.winrates
        self.winrates = [ScoreTracker(self.tracker_n, self.tracker_n) for _ in self.pids]
        for i in range(min(len(self.winrates) - 1, len(old))):
            self.winrates[i].v = old[i].v
            self.winrates[i].n = old[i].n

        # 重置自博弈追踪器（新快照意味着新的自博弈基准）
        self.selfplay_winrate = ScoreTracker(self.tracker_n, self.tracker_n)

        self._save_manifest()
        return pid

    def get_weight(self, pid: int) -> dict:
        """从磁盘加载 pid 对应的模型权重（state dict）。"""
        ckpt = torch.load(self._ckpt_path(pid), map_location='cpu', weights_only=False)
        return ckpt['model_state_dict']

    # ── 对手采样 ──────────────────────────────────────────────────────────────

    def select_opponent(self) -> tuple[int, dict]:
        """PFSP 从最近 last_num 个联赛成员中采样对手，返回 (pid, state_dict)。"""
        recent_pids = self.pids[-self.last_num:]
        recent_idx = list(range(len(self.pids)))[-self.last_num:]
        recent_scores = [self.winrates[i].v for i in recent_idx]
        probs = pfsp_from_scores(recent_scores)
        pid = int(np.random.choice(recent_pids, p=probs))
        return pid, self.get_weight(pid)

    # ── 胜率更新 ──────────────────────────────────────────────────────────────

    def update_result(self, pid: int | None, result: float, selfplay: bool = False):
        """更新对 pid 对手（或自博弈）的胜率追踪。

        result: 1.0 = 胜，0.0 = 负
        """
        if selfplay:
            self.selfplay_winrate.update(result)
        else:
            if pid not in self.pids:
                return
            idx = self.pids.index(pid)
            self.winrates[idx].update(result)

    def score_all_above(self, threshold: float) -> bool:
        """最近 last_num 个对手的平均得分全部超过 threshold。"""
        recent = self.winrates[-self.last_num:]
        return len(recent) > 0 and all(tr.v > threshold for tr in recent)

    # 向后兼容别名
    def win_rate_all_above(self, threshold: float) -> bool:
        return self.score_all_above(threshold)

    def get_stats(self) -> dict:
        recent_pids = self.pids[-self.last_num:]
        recent_idx = list(range(len(self.pids)))[-self.last_num:]
        recent_scores = {pid: round(self.winrates[i].v, 3)
                         for pid, i in zip(recent_pids, recent_idx)}
        return {
            'n_agents': len(self.pids),
            'selfplay_score': round(self.selfplay_winrate.v, 3),
            'recent_scores': recent_scores,
        }

    # ── Manifest（断点续训）──────────────────────────────────────────────────

    def _save_manifest(self):
        manifest = {
            'metric': 'score',
            'current_pid': self.current_pid,
            'pids': self.pids,
            'winrates': [{'v': tr.v, 'n': tr.n} for tr in self.winrates],
            'selfplay': {'v': self.selfplay_winrate.v, 'n': self.selfplay_winrate.n},
        }
        with open(os.path.join(self._league_dir, 'manifest.json'), 'w') as f:
            json.dump(manifest, f, indent=2)

    def load_manifest(self) -> bool:
        """从 manifest.json 恢复联赛状态，返回是否成功。

        兼容旧格式（metric='winrate'）：旧胜率值在 [0,1] 之间，
        切换到得分追踪后重置为 0.0，避免 PFSP 混乱。
        """
        path = os.path.join(self._league_dir, 'manifest.json')
        if not os.path.exists(path):
            return False
        with open(path) as f:
            manifest = json.load(f)
        self.current_pid = manifest['current_pid']
        self.pids = manifest['pids']
        metric = manifest.get('metric', 'winrate')
        self.winrates = []
        for d in manifest['winrates']:
            tr = ScoreTracker(self.tracker_n, self.tracker_n)
            if metric == 'score':
                tr.v = d['v']
                tr.n = d['n']
            else:
                # 旧格式为胜率，重置为中性得分 0.0
                tr.v = 0.0
                tr.n = 0
            self.winrates.append(tr)
        sp = manifest.get('selfplay', {})
        if metric == 'score':
            self.selfplay_winrate.v = sp.get('v', 0.0)
            self.selfplay_winrate.n = sp.get('n', 0)
        else:
            self.selfplay_winrate.v = 0.0
            self.selfplay_winrate.n = 0
        return True
