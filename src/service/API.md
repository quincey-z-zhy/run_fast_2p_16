# 两人跑得快 AI 推理服务 API 文档

## 概述

基于 Flask 的无状态 HTTP 服务，为两人跑得快游戏提供 AI 决策。
客户端每次请求时传入完整的初始手牌和历史动作，服务端重建局面并返回 AI 的下一步动作。

**默认地址：** `http://localhost:7788`

---

## 牌面表示

| 字符 | 含义 |
|------|------|
| `3`–`9` | 数字牌 3–9 |
| `T` | 10 |
| `J` | J（11） |
| `Q` | Q（12） |
| `K` | K（13） |
| `A` | A（14） |
| `2` | 2（15，最大单牌） |

**牌池组成（45 张）：** 3–Q 各 4 张，K 3 张，A 1 张，2 1 张。
每局每位玩家各发 15 张，剩余 15 张为死牌。

手牌字符串按牌面从小到大排序，如 `"3456789TJQKKA2"`。

---

## 端点

### 1. 健康检查

```
GET /health_check
```

**响应示例：**

```json
{
    "result": "OK"
}
```

| 字段 | 类型 | 说明 |
|------|------|------|
| `result` | string | 固定值 `"OK"` |

---

### 2. AI 决策

```
POST /suggestion
Content-Type: application/json
```

#### 请求体

```json
{
    "game_id":      "game_001",
    "player_id":    "player_A",
    "player_index": 0,
    "hand_cards":   ["3456789TJQKA2", "3456789TJQKK"],
    "actions":      ["3344", "", "56789"],
    "first_player": 0
}
```

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `game_id` | string | 否 | 对局唯一标识，用于日志追踪 |
| `player_id` | string | 否 | 玩家标识，用于日志追踪 |
| `player_index` | int | 是 | 请求方的玩家索引，`0` 或 `1` |
| `hand_cards` | [string] 或 [string, string] | 是 | 玩家**当前**手牌字符串（即本步请求时实际持有的牌），按牌面从小到大排序。服务端会结合 `actions` 历史自动反推初始手牌。**双人模式**：传 2 项，索引 0 对应玩家 0，索引 1 对应玩家 1。**单人模式**：只传 1 项（当前 `player_index` 玩家的手牌），服务端自动推断对手手牌。 |
| `actions` | [string, ...] | 是 | 从开局至今的**完整动作历史**；每项为已打出的牌面字符串，Pass（不出牌）用空字符串 `""` 表示；开局时传空数组 `[]` |
| `first_player` | int | 否 | 先手玩家索引，`0` 或 `1`，默认 `0` |

**`actions` 约定：**

- 历史中奇偶位置交替归属两位玩家：偶数下标（0、2、4…）为先手玩家的动作，奇数下标（1、3、5…）为后手玩家的动作。
- 服务端会依次重建完整局面，再为当前玩家生成动作。
- 调用方须确保 `player_index` 与当前轮次一致，否则返回错误。

---

#### 响应体

```json
{
    "action_str":    "33355",
    "kicker_faces":  [5, 5],
    "is_end":        false,
    "is_success":    true,
    "error_message": ""
}
```

| 字段 | 类型 | 说明 |
|------|------|------|
| `action_str` | string | 本次打出的完整牌面字符串（含带牌）；Pass 时为空字符串 `""` |
| `kicker_faces` | [int, ...] | 带牌的面值列表（整数，3–15）；无带牌时为 `[]` |
| `is_end` | bool | 执行本次动作后，游戏是否结束（当前玩家手牌已出完） |
| `is_success` | bool | 请求是否成功处理 |
| `error_message` | string | 错误描述；成功时为 `""` |

---

#### 错误情况

请求校验失败时，`is_success` 为 `false`，HTTP 状态码为 `400`（或 `500`）。

| 场景 | HTTP 状态码 | `error_message` 示例 |
|------|-------------|----------------------|
| `player_index` 不是 0 或 1 | 400 | `"player_index must be 0 or 1"` |
| `first_player` 不是 0 或 1 | 400 | `"first_player must be 0 or 1"` |
| `hand_cards` 元素数量不为 1 或 2 | 400 | `"hand_cards must contain 1 or 2 strings"` |
| 非当前玩家请求 | 400 | `"Wrong player turn: expected player 1, got player_index=0"` |
| `actions` 中含非法动作 | 400 | `"无法从字符串 'XYZ' 找到对应的动作（游戏步骤 3）"` |
| 游戏已结束 | 200 | `"Game is already over"`（`is_end=true`，`is_success=true`） |
| 请求体非 JSON | 400 | `"Request body must be JSON"` |
| 服务内部异常 | 500 | `"Internal error: ..."` |

---

## 完整示例

### 对局开局（第 0 步，玩家 0 先手）

**请求：**

```json
{
    "game_id":      "demo_game",
    "player_id":    "bot_0",
    "player_index": 0,
    "hand_cards":   ["3345679TJJQKK", "456788TJQAA2"],
    "actions":      [],
    "first_player": 0
}
```

**响应：**

```json
{
    "action_str":    "3344",
    "kicker_faces":  [],
    "is_end":        false,
    "is_success":    true,
    "error_message": ""
}
```

---

### 对局第 1 步（玩家 1 跟牌）

玩家 0 已打出 `"3344"`，轮到玩家 1：

**请求：**

```json
{
    "game_id":      "demo_game",
    "player_id":    "bot_1",
    "player_index": 1,
    "hand_cards":   ["3345679TJJQKK", "456788TJQAA2"],
    "actions":      ["3344"],
    "first_player": 0
}
```

**响应（Pass）：**

```json
{
    "action_str":    "",
    "kicker_faces":  [],
    "is_end":        false,
    "is_success":    true,
    "error_message": ""
}
```

---

### 带牌动作示例（三带二）

**响应：**

```json
{
    "action_str":    "33355",
    "kicker_faces":  [5, 5],
    "is_end":        false,
    "is_success":    true,
    "error_message": ""
}
```

`action_str` 为 `"33355"`，表示打出三张 3 并带一对 5 作为带牌；`kicker_faces` 返回带牌的面值列表。

---

### 单人模式示例（只传当前玩家手牌）

当客户端只持有己方初始手牌时，可使用单人模式：

**请求：**

```json
{
    "game_id":      "demo_game",
    "player_id":    "bot_0",
    "player_index": 0,
    "hand_cards":   ["3345679TJJQKK"],
    "actions":      [],
    "first_player": 0
}
```

服务端自动将对手手牌设为 45 张牌池中剩余的 30 张（含 15 张死牌），由此重建完整局面并完成推理。响应格式与双人模式完全相同。

---

## 注意事项

1. **服务无状态**：每次请求必须传入完整的 `hand_cards` 和 `actions` 历史；服务不保存任何局面状态。
2. **手牌为当前手牌**：`hand_cards` 传入的是本步请求时玩家实际持有的牌，会随出牌减少。服务端内部通过 `actions` 历史反推初始手牌后再重建局面。
3. **动作历史完整性**：`actions` 必须包含从第 0 步到上一步的所有动作（含双方的 Pass），不可跳步。
4. **推理策略**：AI 使用贪心（argmax）策略，相同局面下每次返回相同动作，保证确定性。
5. **带牌（kicker）**：`action_str` 已包含带牌，`kicker_faces` 单独返回带牌面值，供需要精确卡牌信息的客户端使用。
6. **单人模式推断**：`hand_cards` 只传 1 项时，服务端以「45 张牌池 − 己方手牌」作为对手起始手牌（含 15 张死牌）。对手多余的死牌不影响 obs 计算，因为 `remain_cards` 仅依赖已出牌动作。
