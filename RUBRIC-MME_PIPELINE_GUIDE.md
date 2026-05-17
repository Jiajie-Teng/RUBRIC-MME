# RUBRIC-MME 五阶段评测管线说明

## 1. 文档目的

这份文档用于系统性说明当前 `RUBRIC-MME` 五阶段评测管线的设计、代码结构、运行方式、输入输出、并行与修复机制，以及当前已实现的能力与后续建议。

本文档的目标不是只介绍“怎么跑”，而是尽量把下面这些问题一次讲清楚：

- 这个 benchmark 整体在做什么
- 五个阶段分别负责什么
- 每个阶段涉及哪些代码文件，它们的职责是什么
- 每个阶段会生成哪些结果文件，这些文件分别代表什么
- 每个阶段如何运行，常用命令是什么
- 哪些参数最重要，分别控制什么
- 并行、`resume`、`repair` 是怎么工作的
- 当前系统已经达到什么程度，后面还建议补什么

这份文档可以视为当前 `tools/RUBRIC-MME` 目录下最完整的总说明。后续如果增加统一总控脚本，建议在本文档基础上继续补充总控章节，而不是另起一份完全独立的新文档。

---

## 2. 项目整体目标

`RUBRIC-MME` 的目标是构建一个面向真实用户场景的、多模态、多轮交互 benchmark。当前实现重点覆盖以下四类任务：

- `omnibench_image_multi_text`
  - 多图、多轮、文本提问
- `omnibench_image_multi_tts`
  - 多图、多轮、语音提问
- `omnibench_video_stream_text`
  - 视频流、多轮、文本提问
- `omnibench_video_stream_tts`
  - 视频流、多轮、语音提问

整个评测链路目前分为五个阶段：

1. `Phase 1`：被测模型完成多轮推理，生成原始回答
2. `Phase 2`：将 Phase 1 输出标准化为统一中间格式
3. `Phase 3`：裁判模型进行 turn-level / session-level 评分
4. `Phase 4`：基于评分进行低分筛选、错误归因和统计分析
5. `Phase 5`：基于 Phase 4 的结构化结果自动生成最终分析报告

这五个阶段已经形成完整主链，当前代码可支持从被测模型推理到最终报告生成的全流程。

---

## 3. 目录与代码结构概览

### 3.1 主要目录

- `lmms_eval/tasks/omnibench`
  - RUBRIC-MME 任务在 `lmms-eval` 任务系统中的定义入口
- `tools/RUBRIC-MME`
  - RUBRIC-MME 五阶段的核心实现目录
- `logs* / logs_parallel / logs2`
  - 不同阶段运行结果目录

### 3.2 `lmms_eval/tasks/omnibench` 主要文件

- `omnibench.yaml`
  - RUBRIC-MME 总任务入口
- `omnibench_image_multi_text.yaml`
  - 多图文本任务定义
- `omnibench_image_multi_tts.yaml`
  - 多图语音任务定义
- `omnibench_video_stream_text.yaml`
  - 视频文本任务定义
- `omnibench_video_stream_tts.yaml`
  - 视频语音任务定义
- `utils.py`
  - 任务系统集成所需的共享工具
- `README.md`
  - `lmms-eval` 任务侧的简要说明

### 3.3 `tools/RUBRIC-MME` 主要文件分类

#### Phase 1

- `phase1_common.py`
- `run_gemini_phase1_internal.py`
- `run_gemini_phase1.py`

#### Phase 2

- `phase2_results.py`
- `normalize_phase1_outputs.py`

#### Phase 3

- `judge_media.py`
- `judge_parsing.py`
- `judge_prompts.py`
- `judge_runner.py`
- `judge_pipeline.py`
- `aggregation.py`
- `run_judge_phase3_internal.py`
- `run_judge_phase3.py`
- `prompt_templates/turn_core_cn.txt`
- `prompt_templates/session_core_cn.txt`

#### Phase 4

- `low_score_selector.py`
- `error_taxonomy.py`
- `attribution_prompts.py`
- `attribution_parsing.py`
- `attribution_runner.py`
- `attribution_aggregation.py`
- `phase4_pipeline.py`
- `run_phase4_internal.py`
- `run_phase4.py`

#### Phase 5

- `report_prompts.py`
- `report_parsing.py`
- `report_render.py`
- `phase5_pipeline.py`
- `run_phase5_internal.py`
- `run_phase5.py`
- `prompt_templates/report_phase5_step1_cn.txt`
- `prompt_templates/report_phase5_step2_scope_cn.txt`
- `prompt_templates/report_phase5_step3_causes_cn.txt`
- `prompt_templates/report_phase5_step4_recommendations_cn.txt`

#### 其他说明

`tools/RUBRIC-MME` 目录下还存在一些非主链文件，例如历史临时文件、外部文档、诱饵文件说明、`__pycache__` 等。它们不属于五阶段管线核心组件，后续可以按需清理，但当前不影响主链运行。

---

## 4. 五阶段总体数据流

整体数据流如下：

1. `Phase 1`
   - 读取原始 RUBRIC-MME 数据
   - 调用被测模型完成多轮交互推理
   - 输出原始样本结果

2. `Phase 2`
   - 读取 Phase 1 样本
   - 统一格式
   - 产出标准化 `dialogues` / `rounds`

3. `Phase 3`
   - 读取 Phase 2 标准化结果
   - 调用裁判模型进行评分
   - 输出 turn-level / session-level judgement

4. `Phase 4`
   - 读取 Phase 3 judgement
   - 筛选低分样本
   - 调用归因模型进行错误类型打标
   - 汇总任务、模态、能力、错误交叉统计

5. `Phase 5`
   - 读取 Phase 4 结构化结果
   - 通过多步分析生成结构化报告分析结果
   - 渲染 Markdown / HTML 最终报告

---

## 5. Phase 1：被测模型多轮推理

### 5.1 阶段目标

Phase 1 的目标是：

- 读取四类 RUBRIC-MME 任务数据
- 严格按任务对应模态组织输入
- 调用被测模型完成多轮对话式推理
- 记录每一轮的模型回答、请求上下文、错误信息和运行统计

Phase 1 是整个 benchmark 的起点。后面所有阶段都建立在 Phase 1 成功产出的样本之上。

### 5.2 主要代码文件与职责

#### `phase1_common.py`

这是 Phase 1 的公共层，负责与具体模型后端无关的逻辑，包括：

- 四类任务的统一定义
- `TaskSpec` 与任务解析
- 原始数据加载
- 将原始数据构造成统一的 dialogue/session 工作项
- 公共 JSON/JSONL 写盘
- 已完成 dialogue 的识别
- repair 选择逻辑
- 可选接续 Phase 2 / Phase 3

它的意义是：
- 把“任务与数据处理逻辑”从具体模型调用中剥离出来
- 便于未来支持更多被测模型，而不是每种模型都复制一份大脚本

#### `run_gemini_phase1_internal.py`

公司内部接口版 Phase 1 入口，主要负责：

- 组织内部多模态请求
- 处理图片、视频、音频输入格式
- 调用内部 Gemini / MatrixLLM 接口
- 处理重试、退避、限流等待
- 支持并行、repair、实时写入

#### `run_gemini_phase1.py`

官方 Gemini API 版 Phase 1 入口，主要负责：

- 组织官方 Gemini 输入
- 调用官方生成接口
- 支持并行、repair、实时写入

### 5.3 输入模态与任务严格对应关系

当前 Phase 1 已经保证：

- `omnibench_image_multi_text`
  - 输入图片
  - 输入文本问题
- `omnibench_image_multi_tts`
  - 输入图片
  - 输入音频问题
- `omnibench_video_stream_text`
  - 输入视频
  - 输入文本问题
- `omnibench_video_stream_tts`
  - 输入视频
  - 输入音频问题

正常主跑与 repair 都遵循这一点。  
即使 repair 进入轻量模式，也只会“去掉历史上下文”，不会改掉当前任务本身的模态输入类型。

### 5.4 运行输出文件

每个 task 目录下通常会产出：

- `*_samples.jsonl`
  - 每个 dialogue 一条记录，包含该 dialogue 的全部轮次信息
- `*_summary.json`
  - 当前 task 的汇总信息，例如总轮次、失败轮次、成功对话数等

顶层通常会产出：

- `*_run_summary.json`
  - 本次命令级别的整体运行摘要

### 5.5 样本记录大致包含什么

`samples.jsonl` 中的每个样本通常包含：

- `dialogue_id`
- `task_name`
- `media_mode`
- `question_mode`
- `rounds`
  - 每轮问题
  - 参考答案
  - 模型回答
  - 请求上下文
  - 错误信息
  - 修复信息

### 5.6 并行机制

Phase 1 当前支持 `session/dialogue` 级并行：

- 并行单位是一个完整 dialogue
- 同一个 dialogue 内部轮次仍按顺序执行
- 每个 worker 负责一整组 session

当前已支持：

- `--max-workers`

并且 Phase 1 现在已经支持实时写入：

- 某个 dialogue 完成后，结果会尽快反映到 `samples.jsonl`
- 不再必须等整批任务全部完成后才落盘

### 5.7 repair 机制

Phase 1 当前有两类 repair 模式：

#### `resume_from_failure`

- 找到某个 dialogue 最早失败的轮次
- 从这个轮次开始继续跑
- 后续轮次一并重跑
- 适合尽量保留多轮上下文一致性

#### `current_turn_only`

- 只修失败或缺失的当前轮
- 不再带历史轮上下文
- 更适合应对“请求过重、视频太大、上下文太长、连接不稳”的失败场景

### 5.8 resume 与 repair 的区别

- `--resume`
  - 倾向于跳过已存在的成功 dialogue
  - 适合断点续跑

- `--repair-failed`
  - 只针对失败/不完整的 dialogue 做补齐
  - 适合主跑后修复残留失败项

### 5.9 失败重试机制

Phase 1 内部保留单次请求级重试，通常包括：

- `--max-retries`
- `--retry-sleep`
- 对 `429` 和连接错误的额外退避
- 轮次间冷却

### 5.10 常用命令示例

#### 内部接口，四任务各跑 10 个 session，并行 10

```powershell
$env:MATRIXLLM_API_KEY="你的token"
python D:\lmms-eval\tools\RUBRIC-MME\run_gemini_phase1_internal.py `
  --tasks rubric-mme `
  --model gemini-2.5-pro `
  --limit 10 `
  --max-workers 10 `
  --output-dir D:\lmms-eval\logs_parallel\omnibench_phase1_internal_gemini25pro
```

#### 内部接口，repair 失败项

```powershell
$env:MATRIXLLM_API_KEY="你的token"
python D:\lmms-eval\tools\RUBRIC-MME\run_gemini_phase1_internal.py `
  --tasks rubric-mme `
  --model gemini-2.5-pro `
  --repair-failed `
  --repair-mode current_turn_only `
  --max-workers 4 `
  --output-dir D:\lmms-eval\logs_parallel\omnibench_phase1_internal_gemini25pro
```

#### 官方 Gemini 版

```powershell
$env:GOOGLE_API_KEY="你的token"
python D:\lmms-eval\tools\RUBRIC-MME\run_gemini_phase1.py `
  --tasks rubric-mme `
  --model gemini-2.5-pro `
  --limit 10 `
  --max-workers 10 `
  --output-dir D:\lmms-eval\logs_parallel\omnibench_phase1_official_gemini25pro
```

### 5.11 主要参数说明

#### `run_gemini_phase1_internal.py`

重要参数包括：

- `--tasks`
- `--model`
- `--data-root`
- `--media-root`
- `--output-dir`
- `--limit`
- `--dialogue-id`
- `--resume`
- `--repair-failed`
- `--max-workers`
- `--repair-mode`
  - `resume_from_failure`
  - `current_turn_only`
- `--api-url`
- `--api-key-env`
- `--timeout`
- `--max-retries`
- `--retry-sleep`
- `--rate-limit-retry-sleep`
- `--rate-limit-max-sleep`
- `--rate-limit-round-cooldown`
- `--inter-round-sleep`
- `--temperature`
- `--top-p`
- `--max-output-tokens`
- `--audio-content-mode`
- `--video-content-mode`
- `--max-inline-video-bytes`
- `--save-request-blueprint`

可选联动参数：

- `--phase2-output-dir`
- `--phase2-keep-existing`
- `--phase3-output-dir`
- `--phase3-judge-model`
- `--phase3-judge-api-key-env`
- `--phase3-keep-existing`
- `--phase3-save-prompt-text`
- `--phase3-allow-incomplete-dialogues`

#### `run_gemini_phase1.py`

重要参数包括：

- `--tasks`
- `--model`
- `--data-root`
- `--output-dir`
- `--limit`
- `--dialogue-id`
- `--resume`
- `--repair-failed`
- `--max-workers`
- `--repair-mode`
- `--max-retries`
- `--retry-sleep`
- `--poll-interval`
- `--timeout-seconds`
- `--api-key-env`
- `--temperature`
- `--max-output-tokens`
- `--save-raw-response`

可选联动参数同样支持接 Phase 2 / Phase 3。

### 5.12 当前建议

Phase 1 已经可以稳定运行，但后续仍建议继续提升：

- 按 task 类型设置不同默认并发
- 增加自动多轮 repair pass
- 输出更清晰的失败队列与完整性审计
- 对不同错误类型采取更细粒度的 repair 策略

---

## 6. Phase 2：结果标准化

### 6.1 阶段目标

Phase 2 的目标是将 Phase 1 产生的不同任务、不同运行方式下的结果统一成标准中间格式，便于 Phase 3/4/5 稳定消费。

### 6.2 主要代码文件与职责

#### `phase2_results.py`

负责：

- 定义 Phase 2 的统一结果 schema
- 将 Phase 1 原始样本转换为 dialogue-level 与 round-level 结构化记录
- 生成各类 summary 和 validation 结果

#### `normalize_phase1_outputs.py`

Phase 2 的 CLI 入口，负责：

- 读取一个或多个 Phase 1 输出目录或文件
- 调用 `phase2_results.py`
- 写出标准化结果

### 6.3 输出文件

Phase 2 典型输出包括：

- `dialogues.jsonl`
  - 每个 dialogue 一条记录
- `rounds.jsonl`
  - 每个 turn / round 一条记录
- `errors.jsonl`
  - 标准化过程中发现的问题
- `task_summary.json`
  - 按 task 汇总
- `category_summary.json`
  - 按能力类别汇总
- `validation_summary.json`
  - 覆盖率与完整性检查
- `manifest.json`
  - 本次 Phase 2 运行元信息

### 6.4 命令示例

```powershell
python D:\lmms-eval\tools\RUBRIC-MME\normalize_phase1_outputs.py `
  --input D:\lmms-eval\logs_parallel\omnibench_phase1_internal_gemini25pro `
  --output-dir D:\lmms-eval\logs_parallel\omnibench_phase2_gemini25pro
```

### 6.5 主要参数

- `--input`
- `--output-dir`
- `--glob`
- `--strict`
- `--keep-existing`

### 6.6 当前状态

Phase 2 当前比较稳定，后续更多是作为中间层继续沿用，不是当前优先级最高的改造对象。

---

## 7. Phase 3：多模态裁判评分

### 7.1 阶段目标

Phase 3 的目标是：

- 使用裁判模型对被测模型回答进行结构化评分
- 同时支持 turn-level 与 session-level 两个维度
- 当前已经是“多模态 judge”，而不是纯文本 judge

### 7.2 当前评分维度

#### Turn-level 指标

当前 turn-level 评分覆盖：

- `accuracy`
- `completeness`
- `relevance`
- `conciseness`
- `naturalness`
- `proactiveness_helpfulness`
- `intent_understanding_depth`
- `user_state_adaptation`

#### Session-level 指标

当前 session-level 评分覆盖：

- `session_consistency`
- `intent_fulfillment`
- `persona_adaptation`
- `overall_helpfulness_trustworthiness`

### 7.3 输入逻辑

当前 Phase 3 已经接入视觉信息：

#### Turn-level

裁判模型会看到：

- 当前轮视觉证据
- 历史轮视觉证据
- 当前轮问题
- 当前轮参考答案
- 当前轮模型回答
- 历史轮对话文本

#### Session-level

裁判模型会看到：

- 整组 transcript
- `interaction_goal`
- `user_persona`
- 整组视觉证据

### 7.4 主要代码文件与职责

#### `judge_media.py`

负责：

- 组织图片与视频证据
- turn / session repair 轻量模式下的媒体裁剪逻辑

#### `judge_parsing.py`

负责：

- 解析裁判模型输出
- turn/session judgement schema 校验

#### `judge_prompts.py`

负责：

- 组装 turn/session prompt

#### `judge_runner.py`

负责：

- 封装官方 / 内部裁判模型调用后端

#### `judge_pipeline.py`

负责：

- 读取 Phase 2 数据
- 调度并行 judge
- repair 失败项
- 实时写入 judgement 文件
- 最终汇总与收敛结果

#### `aggregation.py`

负责：

- 生成 Phase 3 的基本汇总统计

#### Prompt 模板

- `prompt_templates/turn_core_cn.txt`
- `prompt_templates/session_core_cn.txt`

### 7.5 输出文件

- `turn_judgements.jsonl`
- `session_judgements.jsonl`
- `judge_errors.jsonl`
- `turn_summary.json`
- `session_summary.json`
- `task_summary.json`
- `category_summary.json`
- `validation_summary.json`
- `manifest.json`

### 7.6 并行机制

Phase 3 当前支持 `dialogue/session` 级并行：

- 一个 worker 负责一个完整 dialogue
- 在该 dialogue 内部顺序完成：
  - 所有 turn judgement
  - session judgement

当前支持：

- `--max-workers`

并且已经支持实时写入：

- turn judgement 产生后尽快写入 `turn_judgements.jsonl`
- session judgement 产生后尽快写入 `session_judgements.jsonl`

### 7.7 repair 机制

Phase 3 repair 只会处理：

- 当前输出目录里已经存在的失败项
- 不会误扫全量未跑数据

并且当前已经支持 turn/session 两类 repair 模式。

#### turn-level repair 模式

- `full_context`
  - 保留完整历史文本与历史视觉
- `current_turn_only`
  - 只保留当前轮视觉与当前轮文本
  - 去掉历史轮上下文

#### session-level repair 模式

- `full_context`
  - 保留整组完整信息
- `light_context`
  - 保留整段 transcript
  - 压缩视觉证据到代表性轮次

### 7.8 结果收敛方式

当前 Phase 3 repair 已改为“收敛最新有效记录”，即：

- 最终文件中保留的是最新有效版本
- 不再长期堆积旧失败记录
- 更适合人工阅读和后续阶段消费

### 7.9 命令示例

#### 内部接口，正常并行跑

```powershell
$env:MATRIXLLM_API_KEY="你的token"
python D:\lmms-eval\tools\RUBRIC-MME\run_judge_phase3_internal.py `
  --phase2-dir D:\lmms-eval\logs_parallel\omnibench_phase2_gemini25pro `
  --output-dir D:\lmms-eval\logs_parallel\omnibench_phase3_gemini25pro `
  --model gemini-2.5-pro `
  --max-workers 4 `
  --save-prompt-text
```

#### 内部接口，repair 失败项

```powershell
$env:MATRIXLLM_API_KEY="你的token"
python D:\lmms-eval\tools\RUBRIC-MME\run_judge_phase3_internal.py `
  --phase2-dir D:\lmms-eval\logs_parallel\omnibench_phase2_gemini25pro `
  --output-dir D:\lmms-eval\logs_parallel\omnibench_phase3_gemini25pro `
  --model gemini-2.5-pro `
  --repair-failed `
  --turn-repair-mode current_turn_only `
  --session-repair-mode light_context `
  --max-workers 2 `
  --save-prompt-text
```

#### 官方 Gemini 版

```powershell
$env:GOOGLE_API_KEY="你的token"
python D:\lmms-eval\tools\RUBRIC-MME\run_judge_phase3.py `
  --phase2-dir D:\lmms-eval\logs_parallel\omnibench_phase2_gemini25pro `
  --output-dir D:\lmms-eval\logs_parallel\omnibench_phase3_gemini25pro_official `
  --model gemini-2.5-pro `
  --max-workers 4 `
  --save-prompt-text
```

### 7.10 主要参数

#### `run_judge_phase3_internal.py`

- `--phase2-dir`
- `--output-dir`
- `--model`
- `--api-url`
- `--api-key-env`
- `--timeout`
- `--max-retries`
- `--retry-sleep`
- `--rate-limit-retry-sleep`
- `--rate-limit-max-sleep`
- `--inter-request-sleep`
- `--max-workers`
- `--repair-passes`
- `--repair-pass-cooldown`
- `--turn-repair-mode`
- `--session-repair-mode`
- `--temperature`
- `--top-p`
- `--max-output-tokens`
- `--dialogue-id`
- `--limit-dialogues`
- `--resume`
- `--repair-failed`
- `--keep-existing`
- `--save-prompt-text`
- `--allow-incomplete-dialogues`

#### `run_judge_phase3.py`

与 internal 版基本一致，主要差异是：

- 使用官方 Gemini 接口
- 支持 `--disable-response-schema`

### 7.11 当前建议

Phase 3 主逻辑已经稳定。后续建议重点是：

- 根据任务类型设置更细粒度并发
- 自动多轮 repair pass
- 更清晰的 failure audit

---

## 8. Phase 4：低分归因与统计分析

### 8.1 阶段目标

Phase 4 的目标不是重新评分，而是在评分结果之上做结构化分析：

- 识别低分样本
- 为低分样本打错误原因标签
- 统计任务、模态、能力、错误类型之间的关系
- 为最终 Phase 5 报告提供分析底座

### 8.2 当前输入来源

主要输入包括：

- Phase 3:
  - `turn_judgements.jsonl`
  - `session_judgements.jsonl`
- Phase 2:
  - `dialogues.jsonl`
  - 必要时辅助补充 session 对话上下文

### 8.3 什么算低分

当前低分判定规则主要由 `low_score_selector.py` 负责，核心规则为：

- 平均分低于阈值
  - 默认 `avg_score < 4.0`
- 任一关键指标低于阈值
  - 默认 `metric <= 3`
- 或 judgement 本身失败
  - `status != success`

严重度进一步区分为：

- `critical`
  - 例如：
    - `status != success`
    - 或更严重的单项低分
- `warning`
  - 触发低分条件，但未达到 critical

### 8.4 主要代码文件与职责

#### `low_score_selector.py`

负责：

- 根据阈值从 Phase 3 judgement 中筛选低分 turn / session 候选

#### `error_taxonomy.py`

负责：

- 定义 turn-level 与 session-level 低分原因体系
- turn-level taxonomy 已吸收原 `vqa_config.py` 中的 `VQA_Error_Categories_CN`

#### `attribution_prompts.py`

负责：

- 归因 prompt 组装
- 当前是纯文本归因，不再输入视觉数据

#### `attribution_parsing.py`

负责：

- 解析归因模型输出

#### `attribution_runner.py`

负责：

- 调用归因模型

#### `attribution_aggregation.py`

负责：

- 低分原因分布
- 任务层、模态层、能力层、metric 交叉统计
- 高分统计
- 代表性样本池构建

#### `phase4_pipeline.py`

负责：

- 串联 selector、runner、aggregation
- 并行 attribution
- repair 失败 candidate
- 最终收敛输出

### 8.5 并行机制

Phase 4 的并行粒度不是 dialogue，而是 `candidate`：

- 一个低分 turn candidate 是一个 work item
- 一个低分 session candidate 也是一个 work item

这么设计的原因是：

- Phase 4 attribution 本身已经没有多轮依赖
- turn/session 候选天然相互独立
- 归因调用是 Phase 4 的主要耗时来源

### 8.6 repair 机制

Phase 4 repair 是 candidate 级：

- 只重跑失败归因 candidate
- 不重跑已成功 candidate

当前不需要像 Phase 1 / 3 那样分 strict / light repair：

- 因为 Phase 4 本身是 text-only
- 失败更多是 API 或解析错误，而不是输入过重

### 8.7 输出文件

#### 低分候选与归因

- `low_score_turns.jsonl`
- `low_score_sessions.jsonl`
- `turn_error_attributions.jsonl`
- `session_error_attributions.jsonl`
- `phase4_errors.jsonl`

#### 低分归因汇总

- `turn_attribution_summary.json`
- `session_attribution_summary.json`
- `error_category_summary.json`
- `task_error_summary.json`
- `metric_failure_summary.json`

#### 全量评分统计

- `score_summary.json`
- `ability_score_summary.json`

#### 增强后的交叉统计

- `error_reason_by_task_summary.json`
- `error_reason_by_mode_summary.json`
- `error_reason_by_ability_summary.json`
- `metric_error_cross_summary.json`
- `high_score_summary.json`
- `representative_cases.json`

#### 运行元信息

- `validation_summary.json`
- `manifest.json`

### 8.8 这些统计具体回答什么问题

#### `score_summary.json`

回答：

- 各 task / mode / media / question_mode 的总体分数表现
- turn/session 不同硬指标的平均分、分布、低分率

#### `ability_score_summary.json`

回答：

- 不同 `primary_category / secondary_category` 能力层面的分数分布

#### `error_category_summary.json`

回答：

- turn/session 最常见低分原因是什么

#### `task_error_summary.json`

回答：

- 各 task 下最常见的低分原因是什么

#### `metric_failure_summary.json`

回答：

- 哪些评分维度最容易低分
- 这些低分通常和哪些错误原因共现

#### `error_reason_by_task_summary.json`

回答：

- `错误原因 × task` 的分布

#### `error_reason_by_mode_summary.json`

回答：

- `错误原因 × task_mode/question_mode/media_mode` 的分布

#### `error_reason_by_ability_summary.json`

回答：

- `错误原因 × ability` 的分布

#### `metric_error_cross_summary.json`

回答：

- `metric × error reason` 的交叉关系

#### `high_score_summary.json`

回答：

- 高分样本集中在哪些 task / mode / ability
- 作为低分分析的对照组

#### `representative_cases.json`

回答：

- 报告中最值得展示的高分 / 低分代表性案例有哪些

### 8.9 命令示例

#### 内部接口，正常并行跑

```powershell
$env:MATRIXLLM_API_KEY="你的token"
python D:\lmms-eval\tools\RUBRIC-MME\run_phase4_internal.py `
  --phase3-dir D:\lmms-eval\logs_parallel\omnibench_phase3_gemini25pro `
  --phase2-dir D:\lmms-eval\logs_parallel\omnibench_phase2_gemini25pro `
  --output-dir D:\lmms-eval\logs_parallel\omnibench_phase4_gemini25pro `
  --model gemini-2.5-pro `
  --max-workers 4 `
  --save-prompt-text
```

#### 内部接口，resume + repair pass

```powershell
$env:MATRIXLLM_API_KEY="你的token"
python D:\lmms-eval\tools\RUBRIC-MME\run_phase4_internal.py `
  --phase3-dir D:\lmms-eval\logs_parallel\omnibench_phase3_gemini25pro `
  --phase2-dir D:\lmms-eval\logs_parallel\omnibench_phase2_gemini25pro `
  --output-dir D:\lmms-eval\logs_parallel\omnibench_phase4_gemini25pro `
  --model gemini-2.5-pro `
  --max-workers 4 `
  --resume `
  --repair-passes 1 `
  --save-prompt-text
```

#### 官方 Gemini 版

```powershell
$env:GOOGLE_API_KEY="你的token"
python D:\lmms-eval\tools\RUBRIC-MME\run_phase4.py `
  --phase3-dir D:\lmms-eval\logs_parallel\omnibench_phase3_gemini25pro `
  --phase2-dir D:\lmms-eval\logs_parallel\omnibench_phase2_gemini25pro `
  --output-dir D:\lmms-eval\logs_parallel\omnibench_phase4_gemini25pro_official `
  --model gemini-2.5-pro `
  --max-workers 4 `
  --save-prompt-text
```

### 8.10 主要参数

#### `run_phase4_internal.py`

- `--phase3-dir`
- `--phase2-dir`
- `--output-dir`
- `--model`
- `--api-url`
- `--api-key-env`
- `--timeout`
- `--max-retries`
- `--retry-sleep`
- `--rate-limit-retry-sleep`
- `--rate-limit-max-sleep`
- `--inter-request-sleep`
- `--max-workers`
- `--repair-passes`
- `--repair-pass-cooldown`
- `--temperature`
- `--top-p`
- `--max-output-tokens`
- `--dialogue-id`
- `--limit-dialogues`
- `--resume`
- `--keep-existing`
- `--save-prompt-text`
- `--avg-threshold`
- `--metric-threshold`
- `--critical-threshold`

#### `run_phase4.py`

与 internal 版大体一致，主要差异是：

- 使用官方 Gemini 接口
- 支持 `--use-response-schema`

### 8.11 当前状态

Phase 4 当前已经具备：

- 低分归因
- 全量分数统计
- 能力层统计
- 任务 / 模态 / 能力 / metric 与错误原因交叉统计
- 高分对照统计
- 代表性案例池

这使得 Phase 4 已经成为 Phase 5 报告生成的有效数据底座。

---

## 9. Phase 5：自动生成分析报告

### 9.1 阶段目标

Phase 5 的目标是：

- 利用 Phase 4 的结构化结果自动生成一份完整 benchmark 分析报告
- 回答三个核心问题：
  - 模型哪里表现好 / 不好
  - 为什么不好
  - 建议怎么改

### 9.2 当前实现思路

当前 Phase 5 已经从“单个超长 prompt 一步生成”重构为“三层式 + 多步分析”的实现。

### 9.3 三层结构

#### 第一层：全量证据保留 + digest 压缩

首先构造：

- `report_payload.json`
  - 保存全量分析材料

然后基于全量 `payload` 生成：

- `report_digest.json`
  - 给模型使用的高信息密度摘要

注意：

- 全量证据并没有被丢弃
- 只是模型不再直接吃超长原始统计
- 这样做是为了解决单次 prompt 过长、API 稳定性差的问题

#### 第二层：分步调用分析模型

当前 refined 版已经是四步分析：

1. `step1_overview`
   - 总体表现与执行摘要
2. `step2_scope_findings`
   - task / mode / ability 范围分析
3. `step3_root_causes_cases`
   - 低分原因与代表性案例分析
4. `step4_recommendations`
   - 改进建议与路线图

每一步会：

- 使用不同的 prompt 模板
- 使用 `digest` 中不同的证据切片
- 解析为结构化结果

#### 第三层：本地合成与渲染

四步结果完成后：

- 合并为统一 `report_analysis.json`
- 渲染成：
  - `benchmark_report.md`
  - `benchmark_report.html`

### 9.4 主要代码文件与职责

#### `report_prompts.py`

负责：

- 组装每个步骤的分析 prompt
- 读取中文 prompt 模板

#### `report_parsing.py`

负责：

- 解析四步分析结果
- 容错 JSON 代码块或轻微脏字符

#### `report_render.py`

负责：

- 根据结构化分析结果渲染 Markdown / HTML 报告

#### `phase5_pipeline.py`

负责：

- 读取 Phase 4 结果
- 构建 `payload` 与 `digest`
- 顺序调用四个分析 step
- 支持 resume / step repair
- 合并输出

#### 主要模板

- `prompt_templates/report_phase5_step1_cn.txt`
- `prompt_templates/report_phase5_step2_scope_cn.txt`
- `prompt_templates/report_phase5_step3_causes_cn.txt`
- `prompt_templates/report_phase5_step4_recommendations_cn.txt`

### 9.5 Phase 5 输出文件

#### 输入快照层

- `report_payload.json`
  - 全量分析材料
- `report_digest.json`
  - 给分析模型使用的摘要材料

#### 分析过程层

- `report_prompt.txt`
  - 当前运行各 step prompt 的调试拼接文件
- `report_analysis_raw.txt`
  - 四个步骤的原始模型返回
- `report_step_results.json`
  - 各 step 的状态、解析情况、错误原因
- `report_analysis.json`
  - 最终合并后的结构化报告分析结果
- `manifest.json`
  - 本次 Phase 5 运行状态、步骤状态、最终 `analysis_status`

#### 最终展示层

- `benchmark_report.md`
- `benchmark_report.html`

### 9.6 Phase 5 repair 机制

Phase 5 当前支持 step 级 repair：

- 如果某一步失败，不需要重跑全部步骤
- 可以只 repair 失败 step

当前支持：

- `--resume`
- `--repair-failed`
- `--repair-steps`

### 9.7 命令示例

#### 内部接口，正常运行

```powershell
$env:MATRIXLLM_API_KEY="你的token"
python D:\lmms-eval\tools\RUBRIC-MME\run_phase5_internal.py `
  --phase4-dir D:\lmms-eval\logs_parallel\omnibench_phase4_gemini25pro `
  --output-dir D:\lmms-eval\logs_parallel\omnibench_phase5_gemini25pro `
  --model gemini-2.5-pro `
  --save-prompt-text
```

#### 内部接口，repair 失败 step

```powershell
$env:MATRIXLLM_API_KEY="你的token"
python D:\lmms-eval\tools\RUBRIC-MME\run_phase5_internal.py `
  --phase4-dir D:\lmms-eval\logs_parallel\omnibench_phase4_gemini25pro `
  --output-dir D:\lmms-eval\logs_parallel\omnibench_phase5_gemini25pro `
  --model gemini-2.5-pro `
  --resume `
  --repair-failed `
  --repair-steps step2_scope_findings,step4_recommendations `
  --save-prompt-text
```

#### 官方 Gemini 版

```powershell
$env:GOOGLE_API_KEY="你的token"
python D:\lmms-eval\tools\RUBRIC-MME\run_phase5.py `
  --phase4-dir D:\lmms-eval\logs_parallel\omnibench_phase4_gemini25pro `
  --output-dir D:\lmms-eval\logs_parallel\omnibench_phase5_gemini25pro_official `
  --model gemini-2.5-pro `
  --save-prompt-text
```

### 9.8 主要参数

#### `run_phase5_internal.py`

- `--phase4-dir`
- `--output-dir`
- `--model`
- `--api-url`
- `--api-key-env`
- `--timeout`
- `--max-retries`
- `--retry-sleep`
- `--rate-limit-retry-sleep`
- `--rate-limit-max-sleep`
- `--temperature`
- `--top-p`
- `--max-output-tokens`
- `--keep-existing`
- `--resume`
- `--repair-failed`
- `--repair-steps`
- `--save-prompt-text`

#### `run_phase5.py`

与 internal 版大体一致，主要差异是：

- 使用官方 Gemini 接口
- 支持 `--use-response-schema`
- 使用官方接口的超时与轮询参数

### 9.9 当前状态与设计要点

Phase 5 当前已经解决了早期“单个超长 prompt 经常失败”的问题，主要依赖两点：

1. 保留全量证据，但构造给模型的 `digest`
2. 将报告分析改造成多步调用

当前 Phase 5 的关键设计原则是：

- 不让模型一次性吃所有原始统计
- 但又不完全丢弃关键信息
- 通过阶段化分析提高成功率与稳定性

### 9.10 当前仍建议继续提升的方向

- 进一步优化 step2 的输入体量
- 加强各 step 的 parser 容错
- 升级 HTML 展示层
- 后续如果后端支持，可探索文件输入方式

---

## 10. 当前推荐的分阶段运行顺序

### 10.1 开发/调试小样本

推荐顺序：

1. Phase 1 小样本主跑
2. Phase 1 repair
3. Phase 2 标准化
4. Phase 3 主跑
5. Phase 3 repair
6. Phase 4 主跑
7. Phase 4 repair / resume
8. Phase 5 主跑
9. Phase 5 step repair

### 10.2 正式评测建议策略

#### Phase 1

- 主跑
- `resume_from_failure`
- `current_turn_only`

#### Phase 3

- 主跑
- `turn=full_context + session=full_context`
- `turn=current_turn_only + session=light_context`

#### Phase 4

- 主跑
- candidate repair

#### Phase 5

- 主跑
- step repair

### 10.3 为什么建议这样跑

这样做的原因是：

- 主跑优先保留完整上下文
- repair 先严格，再轻量
- 尽量在保证质量的前提下，兼顾运行效率与成功率

---

## 11. 当前并行与 repair 的总体经验

### 11.1 并行不等于无限加速

当前经验很明确：

- 图片任务更适合较高并发
- 视频任务更重，应该保守并发
- 过高并发不一定更快，反而更容易触发：
  - `429`
  - 连接中断
  - 长尾请求

### 11.2 repair 不是越重越好

经验上：

- 严格 repair 适合优先保留质量
- 轻量 repair 适合解决“过重请求导致一直修不回来”的问题

### 11.3 各阶段的 repair 粒度不同

- Phase 1：dialogue/session 级
- Phase 3：turn judgement / session judgement 级
- Phase 4：candidate 级
- Phase 5：step 级

---

## 12. 当前系统能力总结

截至当前版本，系统已经具备：

- 四类任务的多模态多轮推理
- 统一标准化中间层
- 多模态 turn/session 裁判评分
- 低分归因
- 多维统计分析
- 高分对照分析
- 代表性案例池
- 多步自动报告生成
- 并行运行
- 实时写入
- `resume`
- 多阶段 repair

这意味着当前 RUBRIC-MME 五阶段主链已经完整可用。

---

## 13. 当前已知不足与后续建议

虽然五阶段主链已经可用，但仍有一些很值得后续继续提升的点：

### 13.1 Phase 1 / Phase 3 更智能的运行策略

建议后续继续补：

- 按 task 类型自适应并发
- 自动多轮 repair pass
- 更清晰的 failure manifest
- 更强的 completion audit

### 13.2 统一总控脚本

当前五阶段已经具备较完整能力，下一步非常值得做的是：

- 写一个 Python 总控脚本
- 统一调度 Phase 1 -> Phase 5
- 自动读取各阶段 manifest / validation summary
- 自动触发各阶段 repair
- 形成整条链路的运行日志与总 manifest

### 13.3 文档持续更新

这份文档建议后续继续维护，并在总控脚本完成后补充：

- 总控脚本的参数说明
- 一键运行示例
- 全链路运行策略

---

## 14. 推荐的最小可运行示例

下面给出一个较常见的“内部接口 + gemini-2.5-pro + logs_parallel”风格的手动分阶段命令链。

### Phase 1

```powershell
$env:MATRIXLLM_API_KEY="你的token"
python D:\lmms-eval\tools\RUBRIC-MME\run_gemini_phase1_internal.py `
  --tasks rubric-mme `
  --model gemini-2.5-pro `
  --limit 10 `
  --max-workers 10 `
  --output-dir D:\lmms-eval\logs_parallel\omnibench_phase1_internal_gemini25pro
```

### Phase 1 repair

```powershell
$env:MATRIXLLM_API_KEY="你的token"
python D:\lmms-eval\tools\RUBRIC-MME\run_gemini_phase1_internal.py `
  --tasks rubric-mme `
  --model gemini-2.5-pro `
  --repair-failed `
  --repair-mode current_turn_only `
  --max-workers 4 `
  --output-dir D:\lmms-eval\logs_parallel\omnibench_phase1_internal_gemini25pro
```

### Phase 2

```powershell
python D:\lmms-eval\tools\RUBRIC-MME\normalize_phase1_outputs.py `
  --input D:\lmms-eval\logs_parallel\omnibench_phase1_internal_gemini25pro `
  --output-dir D:\lmms-eval\logs_parallel\omnibench_phase2_gemini25pro
```

### Phase 3

```powershell
$env:MATRIXLLM_API_KEY="你的token"
python D:\lmms-eval\tools\RUBRIC-MME\run_judge_phase3_internal.py `
  --phase2-dir D:\lmms-eval\logs_parallel\omnibench_phase2_gemini25pro `
  --output-dir D:\lmms-eval\logs_parallel\omnibench_phase3_gemini25pro `
  --model gemini-2.5-pro `
  --max-workers 4 `
  --save-prompt-text
```

### Phase 3 repair

```powershell
$env:MATRIXLLM_API_KEY="你的token"
python D:\lmms-eval\tools\RUBRIC-MME\run_judge_phase3_internal.py `
  --phase2-dir D:\lmms-eval\logs_parallel\omnibench_phase2_gemini25pro `
  --output-dir D:\lmms-eval\logs_parallel\omnibench_phase3_gemini25pro `
  --model gemini-2.5-pro `
  --repair-failed `
  --turn-repair-mode current_turn_only `
  --session-repair-mode light_context `
  --max-workers 2 `
  --save-prompt-text
```

### Phase 4

```powershell
$env:MATRIXLLM_API_KEY="你的token"
python D:\lmms-eval\tools\RUBRIC-MME\run_phase4_internal.py `
  --phase3-dir D:\lmms-eval\logs_parallel\omnibench_phase3_gemini25pro `
  --phase2-dir D:\lmms-eval\logs_parallel\omnibench_phase2_gemini25pro `
  --output-dir D:\lmms-eval\logs_parallel\omnibench_phase4_gemini25pro `
  --model gemini-2.5-pro `
  --max-workers 4 `
  --save-prompt-text
```

### Phase 5

```powershell
$env:MATRIXLLM_API_KEY="你的token"
python D:\lmms-eval\tools\RUBRIC-MME\run_phase5_internal.py `
  --phase4-dir D:\lmms-eval\logs_parallel\omnibench_phase4_gemini25pro `
  --output-dir D:\lmms-eval\logs_parallel\omnibench_phase5_gemini25pro `
  --model gemini-2.5-pro `
  --save-prompt-text
```

### Phase 5 step repair

```powershell
$env:MATRIXLLM_API_KEY="你的token"
python D:\lmms-eval\tools\RUBRIC-MME\run_phase5_internal.py `
  --phase4-dir D:\lmms-eval\logs_parallel\omnibench_phase4_gemini25pro `
  --output-dir D:\lmms-eval\logs_parallel\omnibench_phase5_gemini25pro `
  --model gemini-2.5-pro `
  --resume `
  --repair-failed `
  --repair-steps step2_scope_findings,step4_recommendations `
  --save-prompt-text
```

---

## 15. 建议的后续文档演进方向

这份文档当前已经覆盖了五阶段主链说明。  
后续最适合继续补充的内容是：

1. 统一总控脚本使用说明
2. 一键运行的推荐策略
3. 不同模型类型的接入说明
   - 官方 Gemini
   - 内部接口模型
   - 其他 API 模型
   - 本地加载模型
4. 常见故障排查章节

---

## 16. 结语

当前 RUBRIC-MME 五阶段评测体系已经从最初的单点脚本，演进为一条较完整的多阶段管线：

- 前端输入覆盖图片、视频、文本、语音
- 中间层格式统一
- 裁判评分支持多模态 turn/session 评估
- 低分归因与多维统计已具备研究价值
- 报告生成已具备自动化能力

后续最重要的工程工作将是：

- 统一总控调度
- 更智能的 repair / audit
- 更完善的对外使用说明

但从当前阶段看，这套系统已经具备了稳定推进与持续迭代的基础。

---

## 17. 当前代码状态补充：RUBRIC-MME 重命名后的实际运行约定

本节补充旧版手册没有覆盖的最新工程状态，尤其是后续新增的 GPT、Claude、ERNIE、Qwen、GLM、Doubao 等 Phase 1 被测模型路线。

### 17.1 命名与兼容边界

当前 benchmark 展示名统一为 `RUBRIC-MME`。新生成结果中的 `benchmark_name` 应写为 `RUBRIC-MME`。

以下历史名称仍被刻意保留，不应在当前阶段强行替换：

| 名称 | 是否保留 | 原因 |
| --- | --- | --- |
| `omnibench_image_multi_text` | 保留 | Phase 1 到 Phase 5 的任务 ID、输出子目录和聚合 key 依赖它 |
| `omnibench_image_multi_tts` | 保留 | 同上 |
| `omnibench_video_stream_text` | 保留 | 同上 |
| `omnibench_video_stream_tts` | 保留 | 同上 |
| `omnibench_dataset` | 保留 | 数据集目录名，当前不修改数据集文件和相对媒体路径 |
| `--tasks omnibench` | 保留为兼容别名 | 旧命令仍可运行 |
| `--tasks rubric-mme` | 推荐新别名 | 新命令建议使用 |
| `run_omnibench_pipeline.py` | 暂时保留 | 总控脚本名属于历史入口，改名会影响大量已记录命令 |

当前代码目录已经改为：

```text
tools/RUBRIC-MME
```

总控脚本内部已经不再硬编码旧工具目录名，而是使用脚本所在目录查找同级阶段脚本。

### 17.2 总控脚本当前职责

入口：

```powershell
python D:\lmms-eval\tools\RUBRIC-MME\run_omnibench_pipeline.py --help
```

`run_omnibench_pipeline.py` 当前负责：

| 职责 | 说明 |
| --- | --- |
| 阶段调度 | 根据 `--start-stage`、`--end-stage` 执行 Phase 1 到 Phase 5 的任意连续区间 |
| backend 选择 | 根据 `--phase1-backend`、`--phase3-backend`、`--phase4-backend`、`--phase5-backend` 选择具体脚本 |
| 任务裁剪 | GPT、Claude、Qwen、GLM 等只支持 text 任务的 Phase 1 backend 会自动过滤 TTS 任务 |
| resume | 通过 `--resume` 复用已有阶段结果 |
| repair | Phase 1、Phase 3、Phase 4、Phase 5 都有阶段级修复循环 |
| audit | 每个阶段结束后检查 manifest、覆盖率、错误数和剩余失败数 |
| 运行日志 | 写出 `pipeline_events.jsonl`、`pipeline_manifest.json`、`pipeline_stage_status.json`、`pipeline_summary.md` |

总控输出：

| 文件 | 位置 | 说明 |
| --- | --- | --- |
| `pipeline_events.jsonl` | `<output-root>/pipeline_events.jsonl` | 每个阶段启动、结束、audit、repair 的事件流 |
| `pipeline_manifest.json` | `<output-root>/pipeline_manifest.json` | 总控 manifest，包含 `benchmark_name: RUBRIC-MME` |
| `pipeline_stage_status.json` | `<output-root>/pipeline_stage_status.json` | 阶段状态快照 |
| `pipeline_summary.md` | `<output-root>/pipeline_summary.md` | 人类可读摘要 |

### 17.3 Phase 1 backend 到脚本的映射

| `--phase1-backend` | 实际脚本 | 当前支持任务 | 典型模型/接口 | 视频策略 |
| --- | --- | --- | --- | --- |
| `internal` | `run_gemini_phase1_internal.py` | 四任务 | MatrixLLM Gemini | 原生图片/视频/音频 |
| `official` | `run_gemini_phase1.py` | 四任务 | 官方 Gemini API | 官方文件上传/媒体输入 |
| `openai_compatible` | `run_openai_compatible_phase1.py` | 取决于模型，Doubao 常用 text 两任务 | Doubao 和通用多模态接口 | 原视频或预压缩视频 |
| `gpt_openai_compatible` | `run_gpt_openai_compatible_phase1.py` | 图片 text、视频 text | GPT-4o/GPT-5 系列 | 视频转帧图片 |
| `claude_vision_openai_compatible` | `run_claude_vision_openai_compatible_phase1.py` | 图片 text、视频 text | Claude 旧兼容接口 | 视频转帧图片 |
| `claude_openai_sdk` | `run_claude_openai_sdk_phase1.py` | 图片 text、视频 text | Claude OpenAI SDK 路线 | 视频转帧图片 |
| `qwen_openai_compatible` | `run_qwen25_vl_openai_compatible_phase1.py` | 图片 text、视频 text | MatrixLLM Qwen2.5-VL | video 或 frames |
| `qwen_antchat_openai_sdk` | `run_qwen_antchat_openai_sdk_phase1.py` | 图片 text、视频 text | AntChat Qwen2.5/Qwen3/Qwen3.5 | video 或 frames |
| `glm_antchat_openai_sdk` | `run_glm_antchat_openai_sdk_phase1.py` | 图片 text、视频 text | GLM-4.5V/GLM-4.6V | 原视频，可带有限历史视频 |

### 17.4 Phase 1 新增模型路线详解

#### 17.4.1 Doubao 与通用 OpenAI-compatible 路线

入口脚本：

```text
run_openai_compatible_phase1.py
```

适用模型：支持图片、文本、视频输入，但不支持或不稳定支持音频输入的 OpenAI-compatible 多模态模型。Doubao 当前按这个路线处理。

任务建议：

| 任务 | 是否建议运行 | 原因 |
| --- | --- | --- |
| `omnibench_image_multi_text` | 是 | 图片和文本正常支持 |
| `omnibench_video_stream_text` | 是 | 视频可通过预压缩文件输入 |
| `omnibench_image_multi_tts` | 否 | Doubao 不支持音频输入 |
| `omnibench_video_stream_tts` | 否 | Doubao 不支持音频输入 |

视频输入顺序：

1. 如果数据中存在 `compressed_clip_path` 且文件存在，优先使用预压缩视频。
2. 如果没有预压缩视频，根据 `--video-compress-mode` 决定是否运行时压缩。
3. 如果视频仍超过大小限制，记录失败，由 repair 或前置压缩脚本解决。
4. 历史文本默认完整保留；历史视频是否输入取决于 `--video-history-mode` 和模型上限。

常用参数：

```powershell
--phase1-backend openai_compatible `
--phase1-api-url https://matrixllm.alipay.com/v1/chat/completions `
--phase1-video-precompressed-mode prefer `
--phase1-video-precompressed-field compressed_clip_path `
--phase1-video-compress-mode auto `
--phase1-video-max-inline-bytes 5500000
```

前置压缩：

```powershell
python D:\lmms-eval\tools\RUBRIC-MME\prepare_doubao_videos.py `
  --dataset-json D:\lmms-eval\omnibench_dataset\video_final_with_vqa_category.json `
  --media-root D:\lmms-eval\omnibench_dataset `
  --max-inline-bytes 5500000 `
  --max-workers 4
```

#### 17.4.2 GPT 路线

入口脚本：

```text
run_gpt_openai_compatible_phase1.py
```

GPT 路线只运行图片 text 和视频 text。视频任务不能直接输入视频文件，而是使用抽帧图片模拟视频输入。

视频帧输入逻辑：

1. 优先读取数据集中已经准备好的预抽帧字段。
2. 如果没有预抽帧，运行时 fallback 到 `_frame_cache`。
3. 当前轮视频帧优先级最高。
4. 历史文本默认完整保留。
5. 历史视觉帧可通过 `--phase1-video-history-mode frames` 打开，但受图片数量和字节预算限制。

推荐保守参数：

```powershell
--phase1-video-prepared-frame-mode prefer `
--phase1-video-history-mode text_only `
--phase1-video-frame-sampling-strategy hybrid_tail `
--phase1-video-frame-count 10 `
--phase1-video-frame-max-side 768 `
--phase1-video-frame-jpeg-quality 8 `
--phase1-video-frame-max-inline-bytes 3000000
```

如果要测试历史视觉帧：

```powershell
--phase1-video-history-mode frames `
--phase1-video-max-images-per-request 50 `
--phase1-video-history-max-frames-per-round 3
```

注意事项：

- GPT 系列可能出现 API 成功但 `prediction` 为空的情况。
- 常见原因是输出 token 不足或 reasoning 消耗过多。
- 排查时优先看 `samples.jsonl` 中的 `raw_response`、`finish_reason`、`usage` 和 `error` 字段。

#### 17.4.3 Claude 路线

当前 Claude 有三条入口：

| 脚本 | 用途 |
| --- | --- |
| `run_claude_openai_compatible_phase1.py` | Claude 通用 OpenAI-compatible 路线 |
| `run_claude_vision_openai_compatible_phase1.py` | Claude 旧视觉兼容接口路线 |
| `run_claude_openai_sdk_phase1.py` | OpenAI SDK 调用方式，保持 Claude 数据处理逻辑不变 |

Claude 只运行：

| 任务 | 说明 |
| --- | --- |
| `omnibench_image_multi_text` | 当前轮图片和历史上下文正常输入 |
| `omnibench_video_stream_text` | 视频转帧图片输入 |

Claude 实测图片数量上限高于 GPT，因此可以使用更积极的历史视觉参数：

```powershell
--phase1-video-history-mode frames `
--phase1-video-max-images-per-request 100 `
--phase1-video-history-max-frames-per-round 8
```

如果发现模型回答“看不到图片/画面”，优先排查：

1. 当前运行走的是哪个 Claude backend。
2. 是否使用了 SDK 路线还是旧 chat/completions 路线。
3. `samples.jsonl` 中是否保存了当前轮 prompt 和媒体说明。
4. 请求 payload 是否包含当前轮 image/frame content。
5. 请求是否过大，导致服务端丢弃或裁剪视觉输入。

#### 17.4.4 ERNIE 路线

入口脚本：

| 脚本 | 用途 |
| --- | --- |
| `run_ernie_openai_compatible_phase1.py` | ERNIE 图片 text 路线 |
| `run_ernie_frame_openai_compatible_phase1.py` | ERNIE 视频转帧路线 |

当前 ERNIE 不稳定支持音频输入。视频 text 如果原生视频输入失败，优先使用 frames 路线。

#### 17.4.5 Qwen 路线

核心脚本：

```text
run_qwen_openai_compatible_phase1.py
```

包装脚本：

| 脚本 | 接口 |
| --- | --- |
| `run_qwen25_vl_openai_compatible_phase1.py` | MatrixLLM 旧接口 Qwen2.5-VL |
| `run_qwen_antchat_openai_sdk_phase1.py` | AntChat 新接口 Qwen2.5/Qwen3/Qwen3.5 |

Qwen 视频任务支持两种路线：

| 路线 | 参数 | 说明 |
| --- | --- | --- |
| 视频文件模式 | `--phase1-video-route video` | 当前轮输入视频文件，优先预压缩视频 |
| 帧图片模式 | `--phase1-video-route frames` | 当前轮和历史轮输入抽帧图片 |

视频文件模式推荐默认：

```powershell
--phase1-video-route video `
--phase1-video-history-mode text_only `
--phase1-video-precompressed-mode prefer `
--phase1-video-precompressed-field compressed_clip_path `
--phase1-video-compress-mode auto `
--phase1-video-max-inline-bytes 5500000
```

frames 模式推荐默认：

```powershell
--tasks omnibench_video_stream_text `
--phase1-video-route frames `
--phase1-video-history-mode frames `
--phase1-video-prepared-frame-mode prefer `
--phase1-video-max-frame-images-per-request 30 `
--phase1-video-history-max-frames-per-round 3
```

AntChat Qwen2.5-VL 之前实测上限较严格，所以默认 video 模式只保留历史文本。Qwen3/Qwen3.5 如果实测支持更多视觉历史，可以逐步增加历史轮数、图片数或总字节预算，但不要直接默认全量历史视觉。

#### 17.4.6 GLM 路线

入口脚本：

```text
run_glm_antchat_openai_sdk_phase1.py
```

当前用于 GLM-4.5V、GLM-4.6V。GLM 支持图片、文本、视频，不支持音频，因此只运行两个 text 任务。

推荐默认：

```powershell
--phase1-video-route video `
--phase1-video-history-mode visual `
--phase1-video-history-max-visual-rounds 1 `
--phase1-video-history-max-inline-bytes-total 12000000
```

如果 GLM 某次运行出现大量视频输入失败或请求过大，优先回退：

```powershell
--phase1-video-history-mode text_only
```

### 17.5 预处理脚本

#### `prepare_doubao_videos.py`

用途：为支持视频文件输入但有大小限制的模型准备压缩视频。

典型适用模型：

- Doubao
- Qwen video 模式
- GLM video 模式
- 其他支持视频文件但限制请求大小的模型

典型命令：

```powershell
python D:\lmms-eval\tools\RUBRIC-MME\prepare_doubao_videos.py `
  --dataset-json D:\lmms-eval\omnibench_dataset\video_final_with_vqa_category.json `
  --media-root D:\lmms-eval\omnibench_dataset `
  --max-inline-bytes 5500000 `
  --max-workers 4
```

#### `prepare_gpt_video_frames.py`

用途：为只支持图片输入的模型提前准备视频帧。

典型适用模型：

- GPT
- Claude
- Qwen frames 模式
- ERNIE frames 模式

典型命令：

```powershell
python D:\lmms-eval\tools\RUBRIC-MME\prepare_gpt_video_frames.py `
  --dataset-json D:\lmms-eval\omnibench_dataset\video_final_with_vqa_category.json `
  --media-root D:\lmms-eval\omnibench_dataset `
  --max-workers 4
```

抽帧策略重点：

| 参数 | 说明 |
| --- | --- |
| `hybrid_tail` | 均匀采样结合尾部采样，缓解问题集中在视频后段的情况 |
| `--video-frame-count` | 基础帧数 |
| `--video-frame-max-side` | 控制帧图长边 |
| `--video-frame-jpeg-quality` | 控制 JPEG 质量 |
| `--video-frame-max-inline-bytes` | 控制单轮帧图总大小 |

### 17.6 当前推荐全流程命令模板

#### Gemini 内部接口四任务

```powershell
$env:MATRIXLLM_API_KEY="<key>"
python D:\lmms-eval\tools\RUBRIC-MME\run_omnibench_pipeline.py `
  --output-root D:\lmms-eval\logs_gemini_model `
  --tasks rubric-mme `
  --tested-model gemini-2.5-pro `
  --judge-model gemini-2.5-pro `
  --attribution-model gemini-2.5-pro `
  --analysis-model gemini-2.5-pro `
  --phase1-backend internal `
  --phase3-backend internal `
  --phase4-backend internal `
  --phase5-backend internal `
  --data-root D:\lmms-eval\omnibench_dataset `
  --media-root D:\lmms-eval\omnibench_dataset `
  --phase1-workers 16 `
  --phase3-workers 32 `
  --phase4-workers 32 `
  --phase1-repair-chain resume_from_failure,current_turn_only `
  --phase1-repair-cycles 5 `
  --phase3-repair-cycles 8 `
  --phase4-repair-cycles 3 `
  --phase5-repair-cycles 5 `
  --resume `
  --save-prompt-text
```

#### GPT 系列

```powershell
$env:MATRIXLLM_API_KEY="<key>"
python D:\lmms-eval\tools\RUBRIC-MME\run_omnibench_pipeline.py `
  --output-root D:\lmms-eval\logs_gpt_model `
  --tasks rubric-mme `
  --tested-model gpt-4o-2024-11-20 `
  --judge-model gemini-2.5-pro `
  --attribution-model gemini-2.5-pro `
  --analysis-model gemini-2.5-pro `
  --phase1-backend gpt_openai_compatible `
  --phase3-backend internal `
  --phase4-backend internal `
  --phase5-backend internal `
  --data-root D:\lmms-eval\omnibench_dataset `
  --media-root D:\lmms-eval\omnibench_dataset `
  --phase1-api-url https://matrixllm.alipay.com/v1/chat/completions `
  --phase1-workers 16 `
  --phase3-workers 32 `
  --phase4-workers 32 `
  --phase1-video-prepared-frame-mode prefer `
  --phase1-video-history-mode text_only `
  --phase1-video-frame-sampling-strategy hybrid_tail `
  --phase1-video-frame-count 10 `
  --phase1-video-frame-max-side 768 `
  --phase1-video-frame-jpeg-quality 8 `
  --phase1-video-frame-max-inline-bytes 3000000 `
  --phase1-repair-chain resume_from_failure,current_turn_only `
  --phase1-repair-cycles 5 `
  --phase3-repair-cycles 8 `
  --phase4-repair-cycles 3 `
  --phase5-repair-cycles 5 `
  --resume `
  --save-prompt-text
```

#### Claude SDK 路线

```powershell
$env:MATRIXLLM_API_KEY="<key>"
python D:\lmms-eval\tools\RUBRIC-MME\run_omnibench_pipeline.py `
  --output-root D:\lmms-eval\logs_claude_model `
  --tasks rubric-mme `
  --tested-model claude-opus-4-7 `
  --judge-model gemini-2.5-pro `
  --attribution-model gemini-2.5-pro `
  --analysis-model gemini-2.5-pro `
  --phase1-backend claude_openai_sdk `
  --phase3-backend internal `
  --phase4-backend internal `
  --phase5-backend internal `
  --phase1-api-url https://matrixllm.alipay.com/v1 `
  --data-root D:\lmms-eval\omnibench_dataset `
  --media-root D:\lmms-eval\omnibench_dataset `
  --phase1-workers 16 `
  --phase3-workers 32 `
  --phase4-workers 32 `
  --phase1-video-prepared-frame-mode prefer `
  --phase1-video-history-mode frames `
  --phase1-video-max-images-per-request 100 `
  --phase1-video-history-max-frames-per-round 8 `
  --phase1-repair-chain resume_from_failure,current_turn_only `
  --phase1-repair-cycles 5 `
  --phase3-repair-cycles 8 `
  --phase4-repair-cycles 3 `
  --phase5-repair-cycles 5 `
  --resume `
  --save-prompt-text
```

#### Qwen AntChat 默认视频文件模式

```powershell
$env:MATRIXLLM_API_KEY="<phase3-5-key>"
$env:ANTCHAT_API_KEY="<phase1-antchat-key>"
python D:\lmms-eval\tools\RUBRIC-MME\run_omnibench_pipeline.py `
  --output-root D:\lmms-eval\logs_qwen_model_full_video `
  --tasks rubric-mme `
  --tested-model Qwen2.5-VL-7B-Instruct `
  --judge-model gemini-2.5-pro `
  --attribution-model gemini-2.5-pro `
  --analysis-model gemini-2.5-pro `
  --phase1-backend qwen_antchat_openai_sdk `
  --phase3-backend internal `
  --phase4-backend internal `
  --phase5-backend internal `
  --phase1-api-url https://antchat.alipay.com/v1 `
  --phase1-api-key-env ANTCHAT_API_KEY `
  --data-root D:\lmms-eval\omnibench_dataset `
  --media-root D:\lmms-eval\omnibench_dataset `
  --phase1-workers 10 `
  --phase3-workers 32 `
  --phase4-workers 32 `
  --phase1-video-route video `
  --phase1-video-history-mode text_only `
  --phase1-video-precompressed-mode prefer `
  --phase1-video-precompressed-field compressed_clip_path `
  --phase1-video-compress-mode auto `
  --phase1-video-max-inline-bytes 5500000 `
  --phase1-repair-chain resume_from_failure,current_turn_only `
  --phase1-repair-cycles 5 `
  --phase3-repair-cycles 8 `
  --phase4-repair-cycles 3 `
  --phase5-repair-cycles 5 `
  --resume `
  --save-prompt-text
```

#### Qwen AntChat 视频 frames 模式

```powershell
$env:MATRIXLLM_API_KEY="<phase3-5-key>"
$env:ANTCHAT_API_KEY="<phase1-antchat-key>"
python D:\lmms-eval\tools\RUBRIC-MME\run_omnibench_pipeline.py `
  --output-root D:\lmms-eval\logs_qwen_model_video_frames `
  --tasks omnibench_video_stream_text `
  --tested-model Qwen2.5-VL-7B-Instruct `
  --judge-model gemini-2.5-pro `
  --attribution-model gemini-2.5-pro `
  --analysis-model gemini-2.5-pro `
  --phase1-backend qwen_antchat_openai_sdk `
  --phase3-backend internal `
  --phase4-backend internal `
  --phase5-backend internal `
  --phase1-api-url https://antchat.alipay.com/v1 `
  --phase1-api-key-env ANTCHAT_API_KEY `
  --data-root D:\lmms-eval\omnibench_dataset `
  --media-root D:\lmms-eval\omnibench_dataset `
  --phase1-workers 10 `
  --phase3-workers 32 `
  --phase4-workers 32 `
  --phase1-video-route frames `
  --phase1-video-history-mode frames `
  --phase1-video-prepared-frame-mode prefer `
  --phase1-video-max-frame-images-per-request 30 `
  --phase1-video-history-max-frames-per-round 3 `
  --phase1-repair-chain resume_from_failure,current_turn_only `
  --phase1-repair-cycles 5 `
  --phase3-repair-cycles 8 `
  --phase4-repair-cycles 3 `
  --phase5-repair-cycles 5 `
  --resume `
  --save-prompt-text
```

#### GLM 默认全流程

```powershell
$env:MATRIXLLM_API_KEY="<phase3-5-key>"
$env:ANTCHAT_API_KEY="<phase1-antchat-key>"
python D:\lmms-eval\tools\RUBRIC-MME\run_omnibench_pipeline.py `
  --output-root D:\lmms-eval\logs_glm46v `
  --tasks rubric-mme `
  --tested-model GLM-4.6V `
  --judge-model gemini-2.5-pro `
  --attribution-model gemini-2.5-pro `
  --analysis-model gemini-2.5-pro `
  --phase1-backend glm_antchat_openai_sdk `
  --phase3-backend internal `
  --phase4-backend internal `
  --phase5-backend internal `
  --phase1-api-url https://antchat.alipay.com/v1 `
  --phase1-api-key-env ANTCHAT_API_KEY `
  --data-root D:\lmms-eval\omnibench_dataset `
  --media-root D:\lmms-eval\omnibench_dataset `
  --phase1-workers 10 `
  --phase3-workers 32 `
  --phase4-workers 32 `
  --phase1-video-route video `
  --phase1-video-history-mode visual `
  --phase1-video-history-max-visual-rounds 1 `
  --phase1-video-history-max-inline-bytes-total 12000000 `
  --phase1-repair-chain resume_from_failure,current_turn_only `
  --phase1-repair-cycles 5 `
  --phase3-repair-cycles 8 `
  --phase4-repair-cycles 3 `
  --phase5-repair-cycles 5 `
  --resume `
  --save-prompt-text
```

### 17.7 运行前后检查清单

运行前：

1. 确认 `tools/RUBRIC-MME` 存在。
2. 确认 `omnibench_dataset/image_final_with_mimt_category.json` 和 `omnibench_dataset/video_final_with_vqa_category.json` 存在。
3. 视频文件模式确认 `compressed_clip_path` 或原始 `clip_path` 可读。
4. frames 模式确认预抽帧字段和本地帧图片存在。
5. 确认 Phase 1 key 和 Phase 3-5 key 分别写到正确环境变量。
6. 新模型先用 `--limit 5` 或 `--limit 20` smoke test。

运行后：

1. Phase 1 先看 `<model>_run_summary.json`。
2. Phase 1 每个任务看 `<model>_summary.json`。
3. 如果 `prediction` 为空，看 `samples.jsonl` 里的 `error`、`raw_response`、`finish_reason`。
4. Phase 2 看 `manifest.json` 的 `error_round_count`。
5. Phase 3 看 `validation_summary.json` 的 turn/session coverage。
6. Phase 4 看 `phase4_errors.jsonl` 和 `validation_summary.json`。
7. Phase 5 看 `manifest.json` 的 `analysis_status`。
8. 最终看 `benchmark_report.md` 和 `benchmark_report.html`。

### 17.8 当前代码文件速查

| 文件 | 阶段 | 说明 |
| --- | --- | --- |
| `run_omnibench_pipeline.py` | 总控 | 五阶段调度、audit、resume、repair |
| `phase1_common.py` | Phase 1 | 任务定义、数据读取、samples 合并、work item 构造 |
| `run_gemini_phase1_internal.py` | Phase 1 | Gemini 内部接口 |
| `run_gemini_phase1.py` | Phase 1 | Gemini 官方接口 |
| `run_openai_compatible_phase1.py` | Phase 1 | 通用 OpenAI-compatible 多模态接口 |
| `run_gpt_openai_compatible_phase1.py` | Phase 1 | GPT 图片/帧输入路线 |
| `run_claude_openai_compatible_phase1.py` | Phase 1 | Claude 通用兼容路线 |
| `run_claude_vision_openai_compatible_phase1.py` | Phase 1 | Claude 旧视觉兼容路线 |
| `run_claude_openai_sdk_phase1.py` | Phase 1 | Claude OpenAI SDK 路线 |
| `run_ernie_openai_compatible_phase1.py` | Phase 1 | ERNIE 图片 text 路线 |
| `run_ernie_frame_openai_compatible_phase1.py` | Phase 1 | ERNIE 视频帧路线 |
| `run_qwen_openai_compatible_phase1.py` | Phase 1 | Qwen 核心 video/frames 实现 |
| `run_qwen25_vl_openai_compatible_phase1.py` | Phase 1 | Qwen2.5-VL MatrixLLM 包装 |
| `run_qwen_antchat_openai_sdk_phase1.py` | Phase 1 | Qwen AntChat 包装 |
| `run_glm_antchat_openai_sdk_phase1.py` | Phase 1 | GLM AntChat 包装 |
| `gpt_video_frame_utils.py` | Phase 1 | 抽帧、读取预抽帧、帧裁剪 |
| `prepare_doubao_videos.py` | 预处理 | 视频压缩和字段写回 |
| `prepare_gpt_video_frames.py` | 预处理 | 视频预抽帧和字段写回 |
| `normalize_phase1_outputs.py` | Phase 2 | Phase 2 CLI 入口 |
| `phase2_results.py` | Phase 2 | samples 到 dialogues/rounds 标准化 |
| `run_judge_phase3_internal.py` | Phase 3 | 内部裁判入口 |
| `run_judge_phase3.py` | Phase 3 | 官方裁判入口 |
| `judge_pipeline.py` | Phase 3 | 裁判主流程 |
| `judge_media.py` | Phase 3 | 图片/视频/音频媒体读取和组装 |
| `judge_runner.py` | Phase 3 | 裁判 API backend |
| `judge_prompts.py` | Phase 3 | turn/session prompt 组装 |
| `judge_parsing.py` | Phase 3 | 裁判结果解析 |
| `aggregation.py` | Phase 3 | 分数聚合 |
| `run_phase4_internal.py` | Phase 4 | 内部归因入口 |
| `run_phase4.py` | Phase 4 | 官方归因入口 |
| `phase4_pipeline.py` | Phase 4 | 低分选择、归因调用、repair、统计 |
| `low_score_selector.py` | Phase 4 | 低分候选选择 |
| `error_taxonomy.py` | Phase 4 | 错误分类定义 |
| `attribution_prompts.py` | Phase 4 | 归因 prompt |
| `attribution_runner.py` | Phase 4 | 归因 API backend |
| `attribution_parsing.py` | Phase 4 | 归因解析 |
| `attribution_aggregation.py` | Phase 4 | 归因统计 |
| `run_phase5_internal.py` | Phase 5 | 内部报告入口 |
| `run_phase5.py` | Phase 5 | 官方报告入口 |
| `phase5_pipeline.py` | Phase 5 | 报告主流程和 step repair |
| `report_prompts.py` | Phase 5 | 报告 prompt |
| `report_parsing.py` | Phase 5 | 报告解析 |
| `report_render.py` | Phase 5 | Markdown/HTML 渲染 |
| `aggregate_phase5_reports.py` | 汇总分析 | 多模型报告聚合 |
| `analyze_phase5_paper_insights.py` | 汇总分析 | 论文 insight 提取 |
| `render_scene_distribution_donuts.py` | 可视化 | 场景分布图 |
| `render_text_only_metric_radars.py` | 可视化 | 文本任务雷达图 |
