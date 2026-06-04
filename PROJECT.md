# run_fast_2p_16_v0 — 项目文档

> 两人跑得快 **16-hand 变种** AI 项目
> fork 自 `run_fast_2p_15_v0`，沿用相同的模型架构与训练管线，仅调整牌池/手牌/动作空间
> 仓库：[`quincey-z-zhy/run_fast_2p_16`](https://github.com/quincey-z-zhy/run_fast_2p_16)

---

## 一、项目简介

16-hand 变种把基线版 45 张 / 15 手扩展到 **48 张 / 16 手**——K 由 3 张升为 4 张、A 由 1 张升为 3 张，2 仍 1 张。规则和流程与基线完全一致，只是动作空间扩展（对/三/四带/炸弹的合法 face 范围相应放宽）。

技术上沿用 15_v0 的全套：

- Gymnasium 环境 + 合法动作 mask + greedy_lowest 默认带牌
- Actor-Critic MLP + 带牌头（两阶段动作）
- 多进程向量化并行 rollout
- League/PFSP 历史快照池
- Flask 推理服务 + 人机对战 WebUI
- AI vs AI 对战评估

---

## 二、与 15-hand 基线的差异

| 项 | 15-hand 基线（`run_fast_2p_15_v0`） | **16-hand 变种（本项目）** |
|---|---:|---:|
| 牌池总数 | 45 | **48** |
| 牌池组成 | 3-Q ×4，K ×3，A ×1，2 ×1 | **3-K ×4，A ×3，2 ×1** |
| 每人初始手牌 | 15 | **16** |
| 死牌数 | 15 | **16** |
| 可成对 / 三条 face | 3-K（K 三张可三条不可对... 实为 3-K 各 ≥3）| **3-A**（A 三张可对/三条）|
| 可成炸弹 / 四带 face | 3-Q（K 不足 4 张） | **3-K** |
| 动作空间 | 270 维 | **294 维** |
| obs 维度 | 80 | 80（结构不变）|
| 春天分 SPRING_SCORE | 30 | **32**（= 2 × HAND_SIZE）|
| 炸弹分 BOMB_BONUS | ±10 zero-sum | ±10 zero-sum（同步） |

**规则不变**：流程、跟牌约束（must-beat）、计分公式、春天触发条件、炸弹即时分、信息边界——与基线 100% 一致。

---

## 三、游戏规则核心点

完整规则见 [`docs/rules_confirmed.md`](docs/rules_confirmed.md)。关键：

- **9 种牌型**：Pass / 单 / 对 / 连对 / 三条(带 0/1/2) / 飞机(带 0/N/2N) / 四带 / 顺子 / 炸弹
- **顺子**：长度 5-12，face ∈ [3, A]，2 不入顺
- **must-beat 跟牌**：`is_new_round=False` 时若有同型更大牌或炸弹，**禁止** Pass
- **春天**：败者全程未出牌 → 基础分置为 32（替代，非加法），不含炸弹分
- **炸弹分（zero-sum）**：出牌方 +10，对方 −10；与最终胜负无关；输家可凭炸弹分翻为正
- **终局**：任一方手牌出空，立即结束

---

## 四、系统架构

### 4.1 观测空间（80 维 float32）

与 15-hand 完全相同的布局：

| dim | 段 | 含义 |
|---|---|---|
| 0–12 | own_hand | 己方手牌归一化计数（face 3–15） |
| 13–25 | remain_cards | 牌池剩余归一化计数 |
| 26–34 | last_formation | 上家牌型 one-hot |
| 35–47 | last_start_face | 上家起始面值 one-hot |
| 48–60 | last_end_face | 上家结束面值 one-hot |
| 61–64 | last_num_kicker | 上家带牌数 one-hot |
| 65 | is_new_round | 是否新一轮 |
| 66 | current_player_index | 当前玩家 |
| 67–79 | opp_pass_ub | 对手 Pass 历史推断的各面值上界 |

归一化的 `max_counts` 适配新牌池：`[4,4,4,4,4,4,4,4,4,4,4,3,1]`（3-K=4, A=3, 2=1）。

### 4.2 动作空间（294 维 + mask）

```
Pass        idx 0                            1
Single      idx 1-13       face 3-15        13
Pocket      idx 14-25      face 3-14        12     ← +1 (A 可入对)
S_pocket    idx 26-76      (len 2-7)        51     ← +6
Trips×3     idx 77-112     带 0/1/2         36     ← +3 (A 可成三条)
S_trips     idx 113-202    (len, mode)      90     ← +9
Quads       idx 203-246    face 3-13        44     ← +4 (K 可成四带)
Straight    idx 247-282    (len 5-12)       36
Bomb        idx 283-293    face 3-13        11     ← +1 (K 可成炸弹)
                                           ----
                                           294
```

### 4.3 网络

```
   obs (80,)
      │
   Linear(80→256) + ReLU
   Linear(256→256) + ReLU
      │
   ┌──┼──────────────────────────┐
   ▼  ▼                          ▼
 actor_head     critic_head    kicker_head
 (256→294)     (256→1)         (256→13)
```

- 与 15_v0 完全相同，只是 actor_head 输出从 270 改为 **294**
- 推理（service / webui / eval）：`argmax`（确定性）
- 训练（rollout）：`Categorical.sample()`（保留探索）
- 切换通过 `sample_kickers(deterministic=True/False)` 控制

### 4.4 训练算法

PPO + GAE(λ=0.95, γ=0.99) + League/PFSP，与 15_v0 同。本项目实战中将 League 参数调到：

```yaml
league:
  snapshot_upscore: 1.0    # ← 从 2.0 调低，让快照池真正流动
  snapshot_gap:     2000   # ← 每 2000 个 PPO update 强制落，~100 个 snapshot / 200M 步
```

效果：上轮 400M 步训练中 League 池累计到 **74 个** 历史快照，PFSP 在低胜率对手上自动加权。

---

## 五、仓库结构

```
run_fast_2p_16_v0/
├── README.md                    # 简介
├── PROJECT.md                   # 本文档
├── Dockerfile + start.sh        # 推理服务容器化（zhcloud/run_fast_2p:c16_v1.0）
├── game/                        # 游戏核心
│   └── config.py                # DECK_COUNTS / HAND_SIZE=16 / 牌型常量
├── src/
│   ├── engine/
│   │   ├── action_generator.py  # 294 维动作空间编码/解码/mask
│   │   └── obs_encoder.py       # 80 维 obs 编码（pass_bound 上界扩展）
│   ├── env/
│   │   └── game_env.py          # Gymnasium env (Discrete(294))
│   ├── training/
│   │   ├── actor_critic.py      # actor 输出 294；sample_kickers 支持 deterministic
│   │   ├── ppo_trainer.py
│   │   ├── rollout_buffer.py
│   │   ├── league.py
│   │   ├── vecenv.py
│   │   ├── train.py
│   │   ├── config.yaml          # 长训练 config
│   │   ├── config_gpu_400m.yaml # 400M 步 GPU 训练（League 调参版）
│   │   ├── config_cpu.yaml      # 短 CPU 烟测 config
│   │   └── config_smoke.yaml    # 3K 步秒级烟测
│   ├── eval/
│   │   ├── baselines.py
│   │   ├── evaluate.py
│   │   ├── ai_vs_ai.py          # 重写：支持基线对手、先后手分组、可复现 seed
│   │   └── heuristic_overrides.py
│   ├── service/
│   │   ├── service.py           # Flask 推理 API（默认 7788，模型 194M.pt）
│   │   ├── API.md
│   │   ├── test_service.py
│   │   └── requirements.txt
│   └── webui/
│       ├── human_vs_ai.py       # 人机对战 WebUI（多 session，argmax 确定性）
│       └── templates/index.html
├── docs/
│   ├── rules_confirmed.md       # 16-hand 规则
│   ├── obs_design.md
│   ├── obs_review.md
│   ├── eval_method.md
│   ├── strategy_analysis.md
│   └── unknown_rules.md
├── project_spec/
│   ├── game_spec.yaml           # game_id=run_fast_2p_16_v0
│   ├── production_profile.yaml  # 48 张 / 16 手 / spring_score 32
│   ├── obs_action_contract.yaml # 294 维契约
│   └── info_boundary.yaml
├── project_state/
├── reports/
├── tests/
│   └── test_engine.py           # 52 个单元测试（含 16-hand 适配）
└── checkpoints/                 # git 忽略
    ├── best.pt                  # = 110M.pt（symlink-ish 复制，便于 docker/webui 默认引用）
    ├── 110M.pt                  # vs greedy_low 86.2% (zero-sum 修复后训出)
    ├── 194M.pt                  # vs greedy_low 86.0% (旧奖励训出，已归档)
    ├── 252M.pt                  # vs greedy_low 82.2% (训练后期一个 step ckpt)
    └── baseline.pt              # 训练结束时的 final snapshot
```

---

## 六、快速开始

### 6.1 环境依赖

```bash
pip install torch numpy gymnasium pyyaml flask requests
```

### 6.2 单元测试

```bash
python -m pytest tests/ -v          # 期望 52 passed
```

### 6.3 训练

```bash
# 从零起 GPU 训练（用预设的 400M 步 + League 调参版）
PYTHONPATH=. python src/training/train.py \
    --config src/training/config_gpu_400m.yaml \
    --stage league --device cuda

# 从某个 ckpt 续训（step 计数重置为 0）
PYTHONPATH=. python src/training/train.py \
    --config src/training/config_gpu_400m.yaml \
    --stage league --device cuda \
    --resume checkpoints/best.pt --reset_steps

# CPU 短训练（验证管线，10 min 出 500K 步模型）
PYTHONPATH=. python src/training/train.py \
    --config src/training/config_cpu.yaml \
    --stage baseline --device cpu
```

### 6.4 评估

```bash
# 单模型 vs 基线（1000 局）
PYTHONPATH=. python src/eval/evaluate.py \
    --model checkpoints/best.pt --opponent greedy_low --games 1000

# 两模型互打（支持基线对手）
PYTHONPATH=. python src/eval/ai_vs_ai.py \
    --model-a checkpoints/best.pt \
    --model-b checkpoints/194M.pt \
    --games 1000 --seed 42

# 模型 vs 基线（同样走 ai_vs_ai 接口）
PYTHONPATH=. python src/eval/ai_vs_ai.py \
    --model-a checkpoints/best.pt \
    --model-b random_legal --games 1000
```

### 6.5 推理服务

```bash
# 直接跑
PYTHONPATH=. python src/service/service.py \
    --model checkpoints/best.pt --port 7788

# Docker（推荐生产环境）
./start.sh                                # 默认 7788 + best.pt
PORT=8000 MODEL=checkpoints/194M.pt ./start.sh

# 或直接拉镜像
docker pull zhcloud/run_fast_2p:c16_v1.0
docker run -d -p 7788:7788 \
    -v $(pwd)/checkpoints:/app/checkpoints:ro \
    zhcloud/run_fast_2p:c16_v1.0
```

API：

```
GET  /health_check     → {"result": "OK"}
POST /suggestion       → 输入完整 hand_cards + actions 历史，返回 AI 决策
```

完整 schema 见 [`src/service/API.md`](src/service/API.md)。

### 6.6 WebUI

```bash
PYTHONPATH=. python src/webui/human_vs_ai.py \
    --model checkpoints/best.pt --port 9001
```

浏览器打开 `http://localhost:9001`。多 session 通过 cookie 隔离，支持并发玩家。

---

## 七、模型现状

| 模型 | step | vs random_legal | vs greedy_low | 训练环境 | 备注 |
|---|---:|---:|---:|---|---|
| **`best.pt` (= 110M.pt)** | 110M | 88.6% | **86.2%** | zero-sum 修复后 | **当前推荐**：在正确奖励下训出 |
| 194M.pt | 194M | 91.4% | 86.0% | 旧（非零和）奖励 | 已归档；评估值更高但 reward 失真 |
| 252M.pt | 252M | 89.2% | 82.2% | 旧奖励 | 训练后期 ckpt（best 后未刷新）|

### 训练轨迹（GPU 第二轮：从零开始 → 400M 步 + League）

| 里程碑 | 步数 | vs greedy_low |
|---|---:|---:|
| 起步 | 1M | 71% |
| 首次过 Stage 7 目标 | 11M | **80.0%** |
| 中期峰值 | 83M | 85.0% |
| 全局最优 | **194M** | **86.0%**（旧奖励）|

### 训练轨迹（GPU 第三轮：从 194M.pt resume，zero-sum 修复后）

| 步数 | vs greedy_low | 备注 |
|---:|---:|---|
| 1M | 83.0% | resume 起点已是强模型 |
| 5M | 85.6% | |
| **110M** | **86.2%** | 当前 best.pt |
| 240M | ~82-84% | 已稳态，主动停训 |

> 86%+ 区间趋于饱和。进一步提升需要更广泛的 League 多样性（如加入启发式对手）或加深网络。

---

## 八、Bug 修复记录

### 2026-06-02 — 炸弹奖励改为 zero-sum

原 `game/game.py` 在出炸弹时只给出牌方 +10，对方无变化，违反双人零和。修复后：

```python
if action.formation == FORMATIONS['bomb']:
    opp = 1 - self._current_player_index
    self._bomb_bonuses[self._current_player_index] += GAME.BOMB_BONUS
    self._bomb_bonuses[opp] -= GAME.BOMB_BONUS     # ← 对方扣同等分
    self.num_bomb += 1
```

随机对局 5/5 验证两方最终得分和 = 0。同时也修了同样的 bug 在 15_v0。

### 2026-06-02 — SPRING_SCORE 改为 32

16-hand 应为 2 × HAND_SIZE = 32，原值 30 是从 15_v0 继承的写死值。

### 2026-06-02 — 推理一致性（argmax + deterministic kicker）

`actor_critic.sample_kickers` 加 `deterministic` 参数（默认 False，训练用；service / webui / eval 传 True）；webui 主动作改为 manual argmax（顺便修了 `h` 变量名冲突导致的运行时 RuntimeError）。

---

## 九、已知限制

1. **vs greedy_low 86% 区间饱和**：194M / 110M 都在这个量级抖动，再延长训练收益有限。可能需要：
   - 把 `GreedyLowAgent` 直接放进 League 池（让 AI 见过启发式分布）
   - 加深网络（256-hidden → 512）
   - 调高 entropy_coeff 防止策略过早收敛
2. **春天得分文档/实现不一致**：`docs/rules_confirmed.md` 已对齐**替换**口径，但旧 15_v0 文档曾写"加法"。`test_spring_bonus` 已按 code 写法对齐。
3. **多局赛制 / 超时机制**：未定义（继承自 15_v0 的 `docs/unknown_rules.md`）。

---

## 十、交付物索引

| 位置 | 内容 |
|---|---|
| GitHub [`quincey-z-zhy/run_fast_2p_16`](https://github.com/quincey-z-zhy/run_fast_2p_16) | 源代码（main 分支） |
| DockerHub [`zhcloud/run_fast_2p:c16_v1.0`](https://hub.docker.com/r/zhcloud/run_fast_2p) | 推理服务镜像（1.32 GB CPU-only） |
| `checkpoints/best.pt` | 当前最优模型（step 110M，zero-sum 环境训出） |
| `src/service/API.md` | HTTP API schema |
| `docs/rules_confirmed.md` | 16-hand 规则权威定义 |

---

## 十一、姐妹项目

- `run_fast_2p_15_v0`（基线 15-hand 版本）— 主要差异是牌池/手牌/动作空间；架构与训练管线完全一致。详见该项目的 `PROJECT.md`。
