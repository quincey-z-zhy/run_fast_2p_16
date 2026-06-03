# AI 运行流程

当你接手这个项目时，请严格按以下顺序执行：

1. 读取 `README.md` 了解项目概况
2. 读取 `project_state/current_stage.json` 确认当前阶段
3. 读取 `project_state/pipeline_control.json` 确认运行模式和约束
4. 读取 `project_spec/data_inventory.yaml` 了解数据来源
5. 如果 `human_rough_idea` 不为 null，读取用户的开发思路
6. 根据当前阶段，执行对应的域 skill

## 阶段 → 域 Skill 映射

| current_stage | 域 skill |
|--------------|----------|
| stage_1, stage_2 | `gaming_ai_meta_skills:game-rules-confirmation` |
| stage_3, stage_4 | `gaming_ai_meta_skills:env-development` |
| stage_5 | `gaming_ai_meta_skills:ai-strategy` |
| stage_6 | `gaming_ai_meta_skills:model-training`（基线） |
| stage_7 | `gaming_ai_meta_skills:model-training`（正式） |
| stage_8 | `gaming_ai_meta_skills:model-delivery` |

## 关键原则

- 只完成当前阶段要求的产出物，不要跳阶段
- 遵守 `pipeline_control.json` 中的 `global_constraints` 和 `stage_constraints`
- 所有文档产出必须中文，不允许占位符
- `docs/` 放设计意图，`reports/` 放执行结果，不要混放
- 修改 `project_spec/` 下的文件需要经过审核

## 阶段推进

完成当前阶段后：
1. 确认所有 required_files 和 required_reports 存在且非空
2. 确认所有 artifacts 状态为 `verified`（或 full-auto 模式下 `auto_approved`）
3. 更新 `current_stage.json` 到下一阶段
4. full-auto 模式自动继续，stage 模式等待用户确认

## 项目特有信息

- 游戏核心：`game/`（已验证无 Bug，直接使用）
- 无回放数据，obs 设计与策略完全自主设计
- 用户指定策略方向：PPO 自我博弈（Deep RL）
- GPU 资源：待配置（进入 ai-strategy 阶段前须完成 GPU 配置）
