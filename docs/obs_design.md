# Obs/Action 空间设计文档

版本：v1.0
日期：2026-04-15
状态：待审核

---

## 一、观测空间设计

### 1.1 总览

| 段 | 维度 | 数据含义 | 信息类型 |
|----|------|----------|---------|
| own_hand | 13 | 己方手牌各面值计数（归一化） | 私有 |
| remain_cards | 13 | 全牌池各面值未打出计数（归一化） | 公开 |
| last_action_formation | 9 | 上家出牌牌型 one-hot | 公开 |
| last_action_start_face | 13 | 上家出牌起始面值 one-hot | 公开 |
| last_action_end_face | 13 | 上家出牌结束面值 one-hot | 公开 |
| last_action_num_kicker | 4 | 上家带牌数量 one-hot | 公开 |
| game_status | 2 | is_new_round + current_player | 公开 |
| **合计** | **67** | float32 向量 | — |

动作掩码（270 维）通过 `info["action_mask"]` 传出，不包含在 obs 向量中。

### 1.2 各段详细说明

#### own_hand（维度 0–12）

13 维浮点向量，索引 i 对应 face = i+3（即 face 3 → 索引 0，face 15 → 索引 12）。

值的含义：`hand_count[face] / DECK_COUNTS[face]`，归一化到 [0, 1]。

例：手中有 3 张 7（face=7, DECK_COUNTS=4）→ 值为 0.75。

**信息边界合规性：** 仅包含自身手牌，未包含对手手牌或死牌，符合 `info_boundary.yaml` 的 `production_visible_fields`。

#### remain_cards（维度 13–25）

与 own_hand 格式相同，但表示整个牌池中各 face 剩余未打出的张数。

初始值为 DECK_COUNTS（归一化后全部为 1.0）；每次有人打出该 face 的牌，对应值递减。

**可推导信息：** `remain_cards[face] - own_hand_count[face]` = 对手 + 死牌中该 face 的数量，属于合法推断（不是直接泄露对手手牌）。

#### last_action（维度 26–64）

编码桌面上当前玩家须超越的"前序有效动作"（`previous_action`）。

- `is_new_round=True` 时：全部为零向量（无前序有效动作）
- `is_new_round=False` 时：编码 previous_action 的 formation、start_card、end_card、num_kicker

formation 编码：按 FORMATIONS 整数值从小到大排列（none=0, single=1, pocket=2, s_pocket=3, trips=8, s_trips=9, quads=12, straight=13, bomb=14），共 9 个类别。

#### game_status（维度 65–66）

- dim 65：`is_new_round`（0.0 或 1.0）
- dim 66：`current_player_index`（0.0 或 1.0）

`current_player_index` 用于多智能体场景中的视角归一化（始终以当前行动玩家为"自己"）。

---

## 二、动作空间设计

### 2.1 设计原则

动作空间采用**固定大小（270 维）+ 动作掩码**的策略：
- 每个动作 index 对应一种"动作模板"（formation + 主牌范围 + 带牌数量）
- 带牌的具体面值由系统按**贪心最低**规则自动选取（不纳入动作决策）
- 掩码标记当前手牌下实际可执行的动作

### 2.2 带牌选取规则（greedy_lowest）

对于需要带牌的动作（trips/s_trips/quads），系统按以下规则自动选取带牌：
1. 从当前手牌中剔除主牌所需的牌
2. 在剩余牌中，按面值从低到高排列
3. 选取前 kicker_count 张作为带牌
4. 若剩余牌数 < kicker_count，该动作被掩码为非法

### 2.3 动作模板编码（共 270 个）

| 段 | 索引范围 | 数量 | 说明 |
|----|---------|------|------|
| Pass | 0 | 1 | 不出牌 |
| Single | 1–13 | 13 | 单牌，face 3..15 |
| Pocket | 14–24 | 11 | 对子，face 3..13 |
| S_pocket | 25–69 | 45 | 连对，(len, start) 组合 |
| Trips, kicker=0 | 70–80 | 11 | 三条不带 |
| Trips, kicker=1 | 81–91 | 11 | 三条带1张 |
| Trips, kicker=2 | 92–102 | 11 | 三条带2张 |
| S_trips | 103–183 | 81 | 飞机，(len, kicker_mode, start) |
| Quads | 184–223 | 40 | 四带，(face, kicker_count) |
| Straight | 224–259 | 36 | 顺子，(len, start) |
| Bomb | 260–269 | 10 | 炸弹，face 3..12 |

S_pocket 编码顺序：先按 length 从小到大（2..7），再按 start_face 从小到大（3..13-len+1）。

S_trips 编码顺序：先按 length（2,3,4），再按 kicker_mode（0, N, 2N），再按 start_face。

### 2.4 动作掩码逻辑

掩码设置为 0（非法）的情况：
1. Pass（idx=0）且 `is_new_round=True`
2. 主牌面值在手牌中数量不足
3. 带牌所需数量超出剩余可用牌数
4. `is_new_round=False` 时，出牌动作未超越 previous_action
5. 非炸弹动作且 formation 与 previous_action 不匹配（formation/长度/带牌数不一致）

炸弹（bomb）在 `is_new_round=False` 时，只要手中有对应面值 4 张，始终合法（无视当前桌面牌型）。

---

## 三、信息可见性分析

| 信息类型 | 来源 | obs 中对应段 | 生产环境可见？ |
|---------|------|------------|-------------|
| 己方手牌 | Player.cards | own_hand | ✅ 是 |
| 牌池剩余 | Game.remain_cards | remain_cards | ✅ 是 |
| 动作历史 | Game.actions | last_action（仅最近一步有效动作） | ✅ 是 |
| 对手手牌 | 禁止使用 | 未包含 | ❌ 否 |
| 死牌 | 禁止使用 | 未包含 | ❌ 否 |

**信息泄露风险评估：**

`remain_cards` 中隐含了对手手牌的信息（通过排除法可推断对手 + 死牌的合计数量），这是**合法的不确定性推断**，不构成信息泄露。AI 无法区分 remain_cards 中哪些属于对手、哪些是死牌，这与实战完全一致。

---

## 四、动作掩码设计

### 4.1 掩码计算时机

每次 `env.step()` 执行后，更新 `info["action_mask"]`，供下一步决策使用。`env.reset()` 时也返回初始状态的掩码。

### 4.2 掩码计算流程

```
1. 从 game.current_player_index 获取当前玩家手牌
2. 对所有 270 个动作模板逐一检查：
   a. 解码 (formation, start, end, kicker_count)
   b. 尝试从手牌中选取主牌 + 带牌（greedy_lowest）
   c. 构造 Action 对象，调用 action.is_legal 验证
   d. 若 is_new_round=False 且 formation≠bomb：验证 action > game.previous_action
3. 返回 270 维 bool 数组
```

### 4.3 合法性保证

所有最终传入 `game.act()` 的动作均已经过 `Action.is_legal` + `Action > previous_action` 双重验证，不会触发 RuntimeError。

---

## 五、设计决策

| 决策 | 选择 | 理由 |
|------|------|------|
| 手牌编码方式 | 面值计数向量（13维） | 简洁直观；比 one-hot 手牌列表节省维度；与 remain_cards 格式统一 |
| 动作空间大小 | 270（抽象带牌数量） | 完整枚举（含具体带牌选择）超过 10000，270 足够表达关键决策且可行；带牌选择对最终胜负影响较小 |
| 带牌选取策略 | greedy_lowest | 实现简单，且在大多数情况下符合人类直觉（留高牌打低牌）；后续可升级为可选择的带牌策略 |
| 动作掩码位置 | info dict（非 obs） | 符合 MaskablePPO 等标准库惯例；避免 obs 因掩码而含有规则信息（不同于状态信息） |
| last_action 编码 | 仅前序有效动作 | 保持 Markov 性；RL 只需知道"当前须超越的牌"，完整历史不必要（历史压缩） |
| 是否加入对手历史统计 | 否（v1 不加） | 增加复杂度，v1 先验证基础架构正确性；后续版本可添加 |
