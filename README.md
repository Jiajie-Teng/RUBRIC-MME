# RUBRIC-MME 项目概览

`RUBRIC-MME` 是一个面向多模态、多轮对话评测的 benchmark 管线。它的核心目标不是只判断模型某一轮是否答对，而是系统性评估模型在连续上下文中的视觉理解、文本理解、时序理解、回答稳定性和错误模式。

## 核心思想

RUBRIC-MME 将评测拆成五个阶段：

| 阶段 | 作用 |
| --- | --- |
| Phase 1 | 调用被测模型，让模型完成多轮图片/视频问答 |
| Phase 2 | 将不同模型的输出统一成标准 dialogue/round 格式 |
| Phase 3 | 使用裁判模型对 turn-level 和 session-level 进行多模态评分 |
| Phase 4 | 对低分样本进行错误归因和统计分析 |
| Phase 5 | 汇总结果并生成最终 benchmark 报告 |

这个设计的重点是把“被测模型推理”和“后续裁判分析”解耦。不同模型只需要在 Phase 1 接入；Phase 2 到 Phase 5 尽量保持统一逻辑。

## 目录说明

| 路径 | 说明 |
| --- | --- |
| `tools/RUBRIC-MME` | 当前 benchmark 的主要代码目录 |
| `tools/RUBRIC-MME/run_omnibench_pipeline.py` | 五阶段总控入口，文件名暂时保留历史名称 |
| `tools/RUBRIC-MME/RUBRIC-MME_PIPELINE_GUIDE.md` | 详细管线说明文档 |
| `tools/RUBRIC-MME/prompt_templates` | 裁判、归因、报告生成提示词模板 |
| `omnibench_dataset` | 数据集和本地媒体文件目录，当前保留历史目录名 |

## 任务类型

内部任务 ID 暂时保留历史命名，用于保证各阶段协议稳定：

| 任务 ID | 含义 |
| --- | --- |
| `omnibench_image_multi_text` | 多图多轮文本问答 |
| `omnibench_image_multi_tts` | 多图多轮语音问答 |
| `omnibench_video_stream_text` | 视频流多轮文本问答 |
| `omnibench_video_stream_tts` | 视频流多轮语音问答 |

新命令推荐使用 `--tasks rubric-mme` 表示全任务运行；旧别名 `--tasks omnibench` 仍可兼容。

## 必要环境

建议环境：

| 依赖 | 用途 |
| --- | --- |
| Python 3.10+ | 运行五阶段脚本 |
| `requests` | 内部接口和 OpenAI-compatible API 调用 |
| `openai` | SDK 路线模型调用 |
| `Pillow` | 图片处理 |
| `imageio-ffmpeg` 或系统 `ffmpeg` | 视频抽帧和压缩 |
| Gemini 相关 SDK | 官方 Gemini 路线需要 |

常用环境变量：

| 环境变量 | 用途 |
| --- | --- |
| `MATRIXLLM_API_KEY` | MatrixLLM 接口，常用于 Phase 1 或 Phase 3-5 Gemini 裁判 |
| `ANTCHAT_API_KEY` | AntChat 接口，常用于 Qwen/GLM 等 Phase 1 |
| `GOOGLE_API_KEY` | 官方 Gemini API 路线 |

不要把真实 API key 写进代码或提交到仓库。

## 最小运行入口

查看总控参数：

```powershell
python D:\lmms-eval\tools\RUBRIC-MME\run_omnibench_pipeline.py --help
```

小批量 smoke test 建议加：

```powershell
--limit 5 --phase1-workers 1 --phase3-workers 1 --phase4-workers 1 --save-prompt-text
```

全流程运行一般需要指定：

```powershell
--output-root <输出目录>
--tasks rubric-mme
--tested-model <被测模型>
--judge-model gemini-2.5-pro
--attribution-model gemini-2.5-pro
--analysis-model gemini-2.5-pro
--phase1-backend <Phase 1 后端>
--phase3-backend internal
--phase4-backend internal
--phase5-backend internal
--data-root D:\lmms-eval\omnibench_dataset
--media-root D:\lmms-eval\omnibench_dataset
--resume
--save-prompt-text
```

## 重要注意事项

- `RUBRIC-MME` 是当前 benchmark 展示名，但内部任务 ID 仍是 `omnibench_*`，不要随意改动，否则会破坏阶段间对齐。
- `omnibench_dataset` 是当前数据和媒体根目录，也不要随意重命名。
- 非 Gemini 模型通常只支持图片和文本，因此一般只运行两个 text 任务。
- 视频模型如果支持原生视频，优先使用视频输入；如果接口限制较强，可以使用预压缩视频或抽帧图片。
- GPT、Claude、部分 Qwen/ERNIE 路线的视频任务使用帧图片输入，需要提前确认预抽帧文件是否存在。
- Phase 3 到 Phase 5 默认使用统一裁判和分析逻辑，不应该为每个被测模型单独修改。
- 如果运行中断，优先使用 `--resume` 和 repair 参数继续，不要手动删除结果文件，除非明确知道阶段状态。

## 推荐阅读

详细运行方式、每个脚本职责、参数说明和故障排查请阅读：

```text
tools/RUBRIC-MME/RUBRIC-MME_PIPELINE_GUIDE.md
```
