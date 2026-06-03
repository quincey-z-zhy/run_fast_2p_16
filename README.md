# run_fast_2p_16_v0

两人跑得快 **16-hand 变种** AI 项目，fork 自 `run_fast_2p_15_v0`，沿用相同的模型与训练管线。

## 与 15-hand 版本的差异

| 项 | 15-hand（基线）| 16-hand（本项目）|
|---|---|---|
| 每人初始手牌 | 15 | **16** |
| 牌池总数 | 45 | **48** |
| 牌池组成 | 3-Q 各 4 张；K=3；A=1；2=1 | **3-K 各 4 张；A=3；2=1** |
| 死牌数 | 15 | **16** |
| 可成对/三条面值 | 3-K | **3-A** |
| 可成炸弹/四带面值 | 3-Q | **3-K** |
| 动作空间维度 | 270 | **294** |
| obs 维度 | 80 | 80（结构不变） |

游戏规则（流程、跟牌约束、计分、春天、炸弹奖励等）与 15-hand 版本完全一致。

## 当前状态

- 阶段：Stage 6（基线训练，待启动）
- 模式：full-auto
- GPU：待配置
- 代码：通过全部 52 个单元测试

## 目录结构

- `game/` — 游戏核心（牌池、动作、玩家、对局）
- `project_spec/` — 稳定事实与契约
- `project_state/` — 推进状态
- `docs/` — 设计文档
- `reports/` — 执行报告
- `logs/` — 原始日志
- `src/` — 源代码（engine / env / training / eval / inference / webui）
- `tests/` — 单元测试
- `checkpoints/` — 模型 checkpoint
- `artifacts/` — 最终交付物

## 策略方向

PPO 自博弈（Deep RL）+ Actor-Critic + League/PFSP，与 15-hand 版本同结构。

## 如何继续

```bash
cd /home/quincey_zhang_china/run_fast_2p_16_v0
python -m pytest tests/        # 验证环境
python src/training/train.py   # Stage 6 基线训练
```
