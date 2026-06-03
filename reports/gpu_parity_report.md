# GPU 环境 Parity 验证报告

生成时间：2026-04-15
阶段：Stage 5（GPU 环境验证）
项目：run_fast_2p_16_v0

---

## 验证结论

**结论：Parity 验证通过。** 远程 GPU 机器与本地机器运行相同环境代码，产出一致且符合规范的结果。

---

## 验证场景

本项目游戏环境（`RunFast2pEnv`）为纯 CPU Python 环境，GPU 仅用于 PPO 神经网络训练。因此 parity 验证目标为：

| 验证项 | 描述 |
|--------|------|
| 确定性验证 | 相同 seed 在远程机器上多次运行产出相同结果 |
| obs 规范性 | obs shape / dtype / 值域符合契约 |
| 单元测试 | 52 个测试在远程机器全部通过 |
| 统计一致性 | 500 局随机 rollout 统计指标与本地接近 |

---

## 修复记录

**发现并修复 1 个 Bug（Seeding 不一致）：**

- **问题**：`env.reset(seed=42)` 调用 `super().reset(seed=42)` 设置了 gymnasium 的 `np_random`，但 `game/cards.py::shuffle()` 使用 Python 全局 `random.shuffle()`，未被 seed 控制，导致相同 seed 产出不同发牌
- **根因**：`shuffle()` 使用 `import random; random.shuffle(deck)`（Python 全局 RNG），不受 gymnasium seed 影响
- **修复**：在 `src/env/game_env.py::reset()` 中添加 `if seed is not None: _random.seed(seed)` 同步 Python random 模块
- **验证**：修复后 seed=42 两次调用产出相同初始 obs 和相同 trajectory（PASS）

---

## Parity 测试结果

| 测试项 | 结果 | 说明 |
|--------|------|------|
| 单元测试（52 个）| ✅ 52/52 PASS | 与本地机器一致 |
| seed=42 确定性 | ✅ PASS | 两次 reset(42) 产出相同 obs 和 trajectory |
| obs shape | ✅ (67,) float32 | 符合 obs_action_contract.yaml |
| action_mask shape | ✅ (270,) bool | 符合契约 |
| obs 值域 | ✅ [0.000, 1.000] | 全部在 [0,1] 内 |
| 500 局 rollout 均值 | ✅ 4.95 | 与本地 4.74 差异 < 5%（正常统计波动）|
| 500 局最大 reward | 45 | 有效（含炸弹奖励）|
| 500 局最小 reward | 1 | 有效（最小基础得分）|

---

## 环境配置

| 项目 | 值 |
|------|-----|
| 机器 | root@34.96.143.142 (port 10022) |
| GPU | NVIDIA RTX5880-Ada-48Q (48 GB) |
| Python | 3.10.12 |
| gymnasium | 1.2.2 |
| torch | 2.11.0+cu128 |
| CUDA | Available (driver 570.172.18) |

---

## 注意事项

环境为 CPU 仿真，GPU 用于神经网络推理和训练：
- 确定性测试中 action 选择使用固定 seed 的 Python Random 实例
- 非确定性 rollout（不传 seed）表现正常，与本地统计指标一致
