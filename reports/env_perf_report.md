# 环境性能报告

生成时间：2026-04-15
项目：run_fast_2p_16_v0
阶段：Stage 4

## 随机 rollout 压测结果

| 指标 | 数值 |
|------|------|
| 压测局数 | 1000 局 |
| 非法动作错误数 | 0 |
| 平均步数/局 | 27.1 步 |
| 最短局 | 3 步 |
| 最长局 | 46 步 |
| 平均 reward | 4.74 |
| 总步数 | 27,063 步 |
| 总耗时 | 5.15 秒 |
| 吞吐量（CPU） | **5,252 steps/sec** |

## 动作掩码验证

- 所有局均使用合法动作列表采样，未出现非法动作触发 RuntimeError
- 初始状态合法动作数均 ≥ 1
- is_new_round 时 pass（idx=0）正确被掩码为 False

## Bug 修复记录

压测初次运行时发现 1 个 Bug：
- **描述**：炸弹跟炸弹场景下，action mask 未验证新炸弹 face > 旧炸弹 face，导致小炸弹错误获得合法标记
- **位置**：`src/engine/action_generator.py`，`_check_beats_prev()` 函数
- **修复**：在炸弹对炸弹分支补充 `action.start_card > prev.start_card` 判断
- **修复后验证**：52 个单元测试全部通过，1000 局 rollout 零错误

## 回放重演验证

本项目无回放数据，跳过回放重演验证。

## WebUI 验收

full-auto 模式，跳过 WebUI 验收，依赖随机 rollout + 单元测试覆盖。
