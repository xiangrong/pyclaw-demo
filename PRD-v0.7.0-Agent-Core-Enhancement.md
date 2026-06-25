# PRD: PyClaw Agent 核心架构增强 (v0.7.0)

## Problem Statement

PyClaw 当前的 Agent 核心层（`agent.py`）采用"单次调用即返回"的简单模式。随着功能逐步完善，以下问题日益突出：

1. **无自愈能力**：工具调用失败后直接返回错误，LLM 无法看到错误信息并自我修正，用户体验差
2. **消息历史膨胀**：长会话中 Token 消耗快速增加，没有截断/压缩策略
3. **RAG 缺乏分层**：语义记忆检索结果没有质量分层，噪音信息干扰 Agent 判断
4. **技能暴露过于宽泛**：所有已安装技能的工具全部暴露给 LLM，模型的推理负担大
5. **安全机制不完善**：`TerminalTool` 等高风险工具仅有简单关键词拦截，缺少操作沙箱
6. **记忆写入被动**：语义记忆只读不写，Agent 无法从交互中积累经验

需要通过 v0.7.0 版本系统性解决上述问题，使 PyClaw 从"原型阶段"进化到"生产可用"阶段。

---

## Solution

对 Agent 核心层做系统性增强，围绕 **自愈循环（Self-Correction Loop）** 为核心，配套实现消息历史压缩、分层 RAG 召回、技能渐进暴露、操作沙箱化和记忆自动写入六大功能。

---

## User Stories

1. 作为 PyClaw 用户，我希望 Agent 在工具调用失败后能自动分析错误原因并重试，这样我不需要手动重发指令
2. 作为 PyClaw 用户，我希望长会话不会消耗过多 Token，这样我的使用成本可控
3. 作为 PyClaw 用户，我希望 Agent 能记住之前对话中的关键信息并主动引用，这样体验更连贯
4. 作为 PyClaw 用户，我希望 Agent 只暴露当前任务需要的工具，这样 Agent 的决策更精准
5. 作为 PyClaw 用户，我希望高风险操作（如删除文件、执行危险命令）需要我确认，这样系统更安全
6. 作为 PyClaw 用户，我希望 Agent 能从每次交互中学习经验，这样它越来越"懂"我
7. 作为 PyClaw 开发者，我希望核心功能有测试覆盖，这样重构时更有信心
8. 作为 PyClaw 开发者，我希望 Agent 的自愈循环有最大轮次限制，这样不会陷入无限循环
9. 作为 PyClaw 开发者，我希望消息历史压缩策略是可配置的，这样不同场景可以灵活调整
10. 作为 PyClaw 开发者，我希望每种工具可以声明自己的风险等级，这样安全策略可扩展

---

## Implementation Decisions

### 1. Agent 自愈循环（Self-Correction Loop）

`agent.py` 的 `process_message()` 方法从单次调用改为 **While 循环**：

- 向 LLM 发送消息并获取响应
- 如果响应包含工具调用 → 执行工具 → 将结果追加回消息历史 → 继续循环
- 如果响应不包含工具调用 → 返回最终回复
- **最大迭代 5 轮**，连续 3 次工具调用失败则放弃并给出失败摘要
- 错误信息格式化为 `<error_context>` 标签，帮助 LLM 理解失败原因

**接口变更**：`process_message()` 签名不变，内部实现从单次调用改为循环。

### 2. 消息历史混合压缩策略

`Session.get_history(limit)` 增加混合模式：

- **最近 10 轮**：完整保留（原始消息）
- **第 11-30 轮**：用 LLM 自动生成的摘要替代（每个 session 的 metadata 中缓存 `summary` 字段）
- **30 轮之前**：丢弃

**触发时机**：`get_history()` 调用时自动检查，如果消息超过阈值则自动压缩。

**接口变更**：`Session` 模型新增 `summary: Optional[str]` 字段和 `summarize_and_compress()` 方法。

### 3. RAG 分层召回

`SemanticMemory.search()` 增加 L2 距离阈值过滤：

- **score < 0.5**：高精度匹配，直接召回
- **0.5 ≤ score < 1.0**：中等匹配，仅在精确匹配不足时召回
- **score ≥ 1.0**：低质量匹配，丢弃

同时增加**近因层合并**：对语义相似的近期交互做去重（按 session_id + timestamp 排序保留最新的一条）。

**接口变更**：无需变更，延续现有 `search(query, limit)` 签名。

### 4. 技能渐进暴露

`ToolRegistry.get_all_specs()` 修改为两种模式：

- **Active 技能**：当前会话已激活的技能（`activate_skill` 调用过），完整暴露所有工具的 function calling schema
- **Available 技能**：已安装但未激活的技能，仅在系统提示词中以文本列表形式提及，不暴露工具 schema

Agent 可在工具调用中调用 `activate_skill` 激活新技能，激活后工具 schema 立即生效。

**接口变更**：无接口签名变更，`get_all_specs()` 内部实现修改。

### 5. 操作沙箱化

`TerminalTool` 增加三级安全策略，扩展 `ToolResult` 返回更多元数据：

- **级别 1（安全）**：`ls`, `cat`, `pwd`, `echo` 等只读命令 → 直接执行
- **级别 2（需确认）**：`mkdir`, `touch`, `cp`, `mv`, `pip install` 等有副作用命令 → 需用户确认
- **级别 3（高风险）**：`rm -rf`, `dd`, `shutdown`, `mkfs` → 默认拒绝，除非用户显式 `approved=True`

同时引入**工作目录限定**：所有文件操作强制在 `work_dir` 范围内执行。

**接口变更**：`TerminalTool` 新增 `_classify_command()` 和 `_validate_path()` 方法。

### 6. 分层记忆写入

`Agent` 的 `process_message()` 中集成记忆写入：

- **短期记忆自动写入**：每次交互 (`user_msg + assistant_msg`) 自动调用 `memory.add_session_interaction()`，标记 `type: "interaction"`
- **长期记忆主动写入**：Agent 通过新的 `save_memory(text, importance)` 工具调用主动写入经验，标记 `type: "experience"`

**接口变更**：`SemanticMemory` 新增 `add_session_interaction()` 方法（已有原型，需完善）。

---

## Testing Decisions

### 测试原则

- 只测试**外部行为**，不测试实现细节
- 对 LLM 调用层进行 mock，避免真实模型调用
- 优先使用**最高接缝（集成测试）**，降低测试维护成本

### 测试模块和接缝

| 模块 | 接缝位置 | 层级 | 测试内容 | 先例 |
|:----|:---------|:----|:---------|:-----|
| `agent.py` | `process_message()` | 集成测试 | 自愈循环：单次成功、失败后重试、连续失败放弃 | 新建 `tests/test_agent.py` |
| `session.py` | `get_history()` | 单元测试 | 混合压缩：阈值触发、摘要注入、边界条件 | 新建 |
| `memory.py` | `search()` | 单元测试 | 分层召回：阈值过滤、近因层去重 | 新建 |
| `registry.py` | `get_all_specs()` | 单元测试 | 渐进暴露：Active vs Available 模式切换 | 新建 |
| `tools/terminal.py` | `execute()` | 单元测试 | 安全沙箱：命令分类、路径校验、HITL | 新建 |

### 测试方式

- 使用 `pytest` 作为测试框架
- 工具调用使用 mock（模拟 LLM 返回 function_call）
- 数据库操作使用内存 SQLite 或临时文件
- 集成测试通过 `gateway.py` 的 `_on_message` 为入口，模拟完整的 Message 流转

---

## Out of Scope

- **多 Agent 协作**：v0.7.0 聚焦单 Agent 核心能力增强，多 Agent 编排在 v0.8.0
- **Telegram 通道 HTML 渲染修复**：此为独立问题，不在本 PRD 范围内
- **模型 Provider 扩展**：当前仅支持 OpenAI 和本地模型，其他 Provider 在后续版本
- **监控和可观测性**：日志系统、Token 用量仪表盘不在本版本范围
- **Cron 任务调度增强**：当前 `wait-for-next` 策略可接受，待后续优化

---

## Further Notes

- 所有新功能均以**向后兼容**为原则，不破坏现有 API 签名
- v0.7.0 版本号对应 `pyproject.toml` 中的版本字段
- 优先级排序：自愈循环 > 操作沙箱 > 消息压缩 > 分层记忆 > 技能暴露 > RAG 分层
- 本 PRD 对应的 11 条架构决策来源于 `grill-me` 深度讨论，已达成完全一致