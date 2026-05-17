# RUBRIC-MME Project Overview

`RUBRIC-MME` is a multi-modal, multi-turn benchmark pipeline. Its goal is not only to check whether a model answers a single question correctly, but also to evaluate visual understanding, temporal reasoning, context tracking, answer stability, and error patterns across complete dialogue sessions.

## Core Idea

RUBRIC-MME is organized as a five-stage pipeline:

| Stage | Purpose |
| --- | --- |
| Phase 1 | Run the tested model on multi-turn image/video QA sessions |
| Phase 2 | Normalize model outputs into standard dialogue and round records |
| Phase 3 | Use a judge model to score turn-level and session-level performance |
| Phase 4 | Attribute low-score cases to structured error categories |
| Phase 5 | Generate the final benchmark report |

The key design choice is to isolate model-specific inference in Phase 1. Phases 2-5 use a unified judging, attribution, and reporting pipeline across all tested models.

## Directory Layout

| Path | Description |
| --- | --- |
| `tools/RUBRIC-MME` | Main code directory for the benchmark pipeline |
| `tools/RUBRIC-MME/run_omnibench_pipeline.py` | Five-stage controller; the filename is kept for backward compatibility |
| `tools/RUBRIC-MME/RUBRIC-MME_PIPELINE_GUIDE.md` | Detailed pipeline guide |
| `tools/RUBRIC-MME/prompt_templates` | Prompt templates for judging, attribution, and reporting |
| `omnibench_dataset` | Dataset and local media directory; the historical directory name is kept intentionally |

## Task Types

The internal task IDs are kept unchanged for compatibility between stages:

| Task ID | Meaning |
| --- | --- |
| `omnibench_image_multi_text` | Multi-image multi-turn text QA |
| `omnibench_image_multi_tts` | Multi-image multi-turn spoken-question QA |
| `omnibench_video_stream_text` | Streaming-video multi-turn text QA |
| `omnibench_video_stream_tts` | Streaming-video multi-turn spoken-question QA |

Use `--tasks rubric-mme` for all tasks in new commands. The legacy alias `--tasks omnibench` is still supported.

## Required Environment

Recommended dependencies:

| Dependency | Purpose |
| --- | --- |
| Python 3.10+ | Running the pipeline scripts |
| `requests` | Internal API and OpenAI-compatible API calls |
| `openai` | SDK-based model calls |
| `Pillow` | Image processing |
| `imageio-ffmpeg` or system `ffmpeg` | Video frame extraction and compression |
| Gemini SDK packages | Required only for the official Gemini backend |

Common environment variables:

| Environment Variable | Purpose |
| --- | --- |
| `MATRIXLLM_API_KEY` | MatrixLLM access, often used for Phase 1 or Gemini-based Phases 3-5 |
| `ANTCHAT_API_KEY` | AntChat access, often used for Qwen/GLM Phase 1 backends |
| `GOOGLE_API_KEY` | Official Gemini API access |

Do not hard-code real API keys in source files or commit them to version control.

## Minimal Entry Point

Show controller options:

```powershell
python D:\lmms-eval\tools\RUBRIC-MME\run_omnibench_pipeline.py --help
```

For a small smoke test, add:

```powershell
--limit 5 --phase1-workers 1 --phase3-workers 1 --phase4-workers 1 --save-prompt-text
```

A typical full-pipeline run specifies:

```powershell
--output-root <output-dir>
--tasks rubric-mme
--tested-model <tested-model>
--judge-model gemini-2.5-pro
--attribution-model gemini-2.5-pro
--analysis-model gemini-2.5-pro
--phase1-backend <phase1-backend>
--phase3-backend internal
--phase4-backend internal
--phase5-backend internal
--data-root D:\lmms-eval\omnibench_dataset
--media-root D:\lmms-eval\omnibench_dataset
--resume
--save-prompt-text
```

## Important Notes

- `RUBRIC-MME` is the benchmark display name, but internal task IDs still use the `omnibench_*` format. Do not rename them casually, because all stages depend on these keys.
- `omnibench_dataset` is still the data and media root. Do not rename it unless the whole pipeline is migrated.
- Most non-Gemini tested models only support image and text inputs, so they usually run only the two text tasks.
- For video-capable models, prefer native video input first. If API limits are strict, use pre-compressed videos or extracted frame images.
- GPT, Claude, and some Qwen/ERNIE routes process videos as frame images. Make sure the prepared frame files exist before large-scale runs.
- Phases 3-5 should remain model-agnostic. Avoid adding model-specific logic there unless the shared judging protocol itself changes.
- If a run is interrupted, prefer `--resume` and repair cycles instead of manually deleting output files.

## Recommended Reading

For detailed commands, file responsibilities, parameter explanations, and troubleshooting, read:

```text
tools/RUBRIC-MME/RUBRIC-MME_PIPELINE_GUIDE.md
```
