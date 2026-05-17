from __future__ import annotations

import argparse
import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import mimetypes
import os
import random
from threading import Lock
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import requests

from judge_runner import MatrixLLMJudgeBackend
from phase1_common import (
    AUDIO_EXTENSIONS,
    DEFAULT_DATA_ROOT,
    DEFAULT_MEDIA_ROOT,
    Phase1DialogueWorkItem,
    IMAGE_EXTENSIONS,
    REPO_ROOT,
    TASK_SPECS,
    VIDEO_EXTENSIONS,
    TaskSpec,
    build_phase1_work_items,
    merge_phase1_payloads,
    append_jsonl,
    dump_json,
    first_nonempty,
    load_docs_for_task,
    is_phase1_round_success,
    read_completed_dialogue_ids,
    resolve_task_names,
    run_phase1_followups,
    safe_name,
    write_jsonl,
    utc_now,
)

DEFAULT_API_URL = "https://matrixllm.alipay.com/v1/chat/completions"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "logs" / "rubric_mme_gemini_internal_phase1"
DEFAULT_MODEL = "gemini-3.1-pro-preview"
DEFAULT_TOKEN_ENV = "MATRIXLLM_API_KEY"

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行 MatrixLLM 内部接口版 RUBRIC-MME 第一阶段推理。")
    parser.add_argument("--tasks", default="rubric-mme", help="逗号分隔的任务名；rubric-mme/omnibench/all 表示四个任务全部运行。")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="被测试模型名称。")
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT if DEFAULT_DATA_ROOT.exists() else REPO_ROOT), help="RUBRIC-MME JSON 数据根目录。")
    parser.add_argument(
        "--media-root",
        default=str(DEFAULT_MEDIA_ROOT if DEFAULT_MEDIA_ROOT.exists() else REPO_ROOT),
        help="图片、视频、音频等相对路径所对应的媒体根目录。",
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="输出目录。")
    parser.add_argument("--limit", type=int, default=None, help="每个任务最多处理多少个 dialogue。")
    parser.add_argument("--dialogue-id", default=None, help="只运行指定的 dialogue_id。")
    parser.add_argument("--resume", action="store_true", help="基于已有 samples.jsonl 跳过已完成 dialogue。")
    parser.add_argument("--repair-failed", action="store_true", help="只修复已有 samples.jsonl 中失败或未完整的 dialogue。")
    parser.add_argument("--max-workers", type=int, default=1, help="session 级并行 worker 数，1 表示串行执行。")
    parser.add_argument("--repair-mode", choices=["resume_from_failure", "current_turn_only"], default="resume_from_failure", help="repair 模式：resume_from_failure 从首个失败轮继续；current_turn_only 仅修当前失败轮。")
    parser.add_argument("--api-url", default=DEFAULT_API_URL, help="内部接口地址。")
    parser.add_argument("--api-key-env", default=DEFAULT_TOKEN_ENV, help="保存访问 token 的环境变量名。")
    parser.add_argument("--timeout", type=int, default=120, help="单次请求超时时间（秒）。")
    parser.add_argument("--max-retries", type=int, default=5, help="每轮请求最大重试次数。")
    parser.add_argument("--retry-sleep", type=float, default=3.0, help="普通重试基础等待时间（秒）。")
    parser.add_argument("--rate-limit-retry-sleep", type=float, default=20.0, help="429 限流时的基础退避时间（秒）。")
    parser.add_argument("--rate-limit-max-sleep", type=float, default=120.0, help="429 限流时的最大退避时间（秒）。")
    parser.add_argument("--rate-limit-round-cooldown", type=float, default=30.0, help="某轮命中 429 后，进入下一轮前额外冷却时间。")
    parser.add_argument("--inter-round-sleep", type=float, default=2.0, help="轮与轮之间的默认等待时间（秒）。")
    parser.add_argument("--temperature", type=float, default=0.0, help="采样温度。")
    parser.add_argument("--top-p", type=float, default=0.95, help="top_p。")
    parser.add_argument("--max-output-tokens", type=int, default=512, help="单轮最大输出 token。")
    parser.add_argument(
        "--audio-content-mode",
        default="input_audio",
        choices=["audio_url", "input_audio"],
        help="TTS 问题音频的发送方式：audio_url 使用 URL，input_audio 直接内联输入。",
    )
    parser.add_argument(
        "--video-content-mode",
        default="auto",
        choices=["auto", "local", "remote"],
        help="视频输入策略：auto 自动选择，local 尽量本地输入，remote 尽量远程/云端引用。",
    )
    parser.add_argument(
        "--max-inline-video-bytes",
        type=int,
        default=5_000_000,
        help="在 auto 模式下，允许本地内联发送的视频大小上限（字节）。",
    )
    parser.add_argument("--save-request-blueprint", action="store_true", help="将请求蓝图也写入每轮记录，方便调试。")
    parser.add_argument("--phase2-output-dir", default="", help="可选：在 Phase 1 结束后继续生成 Phase 2 结果。")
    parser.add_argument("--phase2-keep-existing", action="store_true", help="保留现有 Phase 2 输出而不是先清空。")
    parser.add_argument("--phase3-output-dir", default="", help="可选：在 Phase 2 结束后继续生成 Phase 3 judge 结果。")
    parser.add_argument("--phase3-judge-model", default="gemini-3.1-pro-preview", help="Phase 3 使用的裁判模型。")
    parser.add_argument("--phase3-judge-api-key-env", default=DEFAULT_TOKEN_ENV, help="Phase 3 裁判模型 token 的环境变量名。")
    parser.add_argument("--phase3-keep-existing", action="store_true", help="保留现有 Phase 3 输出而不是先清空。")
    parser.add_argument("--phase3-save-prompt-text", action="store_true", help="把 Phase 3 judge prompt 文本保存到输出里。")
    parser.add_argument("--phase3-allow-incomplete-dialogues", action="store_true", help="当部分 round 失败时仍允许做 session-level judge。")
    return parser.parse_args()


def resolve_local_path(media_root: Path, relative_path: str) -> Optional[Path]:
    if not relative_path:
        return None
    candidate = Path(relative_path)
    if candidate.is_absolute():
        return candidate if candidate.exists() else None
    combined = (media_root / candidate).resolve()
    return combined if combined.exists() else None


def encode_file_to_data_url(file_path: Path) -> Tuple[str, str, int]:
    mime_type, _ = mimetypes.guess_type(str(file_path))
    if not mime_type:
        suffix = file_path.suffix.lower()
        if suffix in IMAGE_EXTENSIONS:
            mime_type = "image/jpeg" if suffix in {".jpg", ".jpeg"} else f"image/{suffix.lstrip('.')}"
        elif suffix in AUDIO_EXTENSIONS:
            mime_type = "audio/mpeg" if suffix == ".mp3" else f"audio/{suffix.lstrip('.')}"
        elif suffix in VIDEO_EXTENSIONS:
            mime_type = "video/mp4" if suffix == ".mp4" else f"video/{suffix.lstrip('.')}"
        else:
            mime_type = "application/octet-stream"
    raw = file_path.read_bytes()
    encoded = base64.b64encode(raw).decode("ascii")
    return f"data:{mime_type};base64,{encoded}", mime_type, len(raw)


def build_system_prompt(spec: TaskSpec) -> str:
    if spec.question_mode == "tts":
        q_desc = "问题以语音形式给出"
    else:
        q_desc = "问题以文本形式给出"
    if spec.media_mode == "video":
        m_desc = "视频片段"
    else:
        m_desc = "图像"
    return (
        "你正在执行 RUBRIC-MME 第一阶段多轮作答。"
        f"当前任务的视觉输入是{m_desc}，用户问题{q_desc}。"
        "请严格遵守以下要求："
        "1. 只能回答当前轮问题；"
        "2. 可以利用历史轮对话和历史轮视觉信息来理解上下文；"
        "3. 绝对不要假设未来轮的信息；"
        "4. 绝对不要输出你看不到的参考答案；"
        "5. 请始终使用中文直接作答，简洁、自然、信息充分。"
    )


def build_user_text(round_index: int, question_mode: str, question_text: str, *, is_history: bool) -> str:
    prefix = f"这是第{round_index + 1}轮历史对话。" if is_history else f"这是当前第{round_index + 1}轮。"
    if question_mode == "text":
        return f"{prefix}请结合提供的视觉信息理解并回答这一轮问题。\n用户问题：{question_text}\n请只围绕这一轮作答。"
    if is_history:
        return f"{prefix}这一轮用户问题以语音给出，请结合附带音频、视觉信息和随后给出的助手回答理解上下文。"
    return f"{prefix}这一轮用户问题以语音给出，请结合附带音频、视觉信息和历史对话，直接回答本轮问题。"


def build_image_content(round_data: Dict[str, Any], media_root: Path) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    content: List[Dict[str, Any]] = []
    refs: List[Dict[str, Any]] = []
    local_path = resolve_local_path(media_root, round_data.get("image_local_path", ""))
    remote_url = round_data.get("image_remote_url", "")
    if local_path is not None:
        data_url, mime_type, size = encode_file_to_data_url(local_path)
        content.append({"type": "image_url", "image_url": {"url": data_url, "detail": "auto"}})
        refs.append({
            "kind": "image",
            "transport": "data_url",
            "local_path": str(local_path),
            "mime_type": mime_type,
            "size_bytes": size,
        })
    elif remote_url:
        content.append({"type": "image_url", "image_url": {"url": remote_url, "detail": "auto"}})
        refs.append({"kind": "image", "transport": "remote_url", "remote_url": remote_url})
    else:
        raise FileNotFoundError(f"图片轮次缺少可用图像资源: {round_data}")
    return content, refs


def build_gs_uri(cloud_path: str) -> str:
    cleaned = str(cloud_path or "").strip()
    if not cleaned:
        return ""
    if cleaned.startswith("gs://"):
        return cleaned
    return f"gs://antgroup_matrix_storage/{cleaned.lstrip('/')}"


def choose_video_transport(
    round_data: Dict[str, Any],
    media_root: Path,
    *,
    is_history: bool,
    video_content_mode: str,
    max_inline_video_bytes: int,
    compact_mode: bool,
) -> Dict[str, Any]:
    local_path = resolve_local_path(media_root, round_data.get("video_local_path", ""))
    remote_url = round_data.get("video_remote_url", "")
    cloud_path = round_data.get("video_cloud_path", "")
    gs_uri = build_gs_uri(cloud_path)
    local_size = local_path.stat().st_size if local_path is not None else None

    if compact_mode:
        if gs_uri:
            return {
                "transport": "gs_uri_input_audio",
                "selection_reason": "compact_gs_uri",
                "local_path": local_path,
                "remote_url": remote_url,
                "cloud_path": cloud_path,
                "gs_uri": gs_uri,
                "size_bytes": local_size,
            }
        if is_history:
            return {
                "transport": "omitted_due_size",
                "selection_reason": "compact_history_omitted",
                "local_path": local_path,
                "remote_url": remote_url,
                "cloud_path": cloud_path,
                "gs_uri": gs_uri,
                "size_bytes": local_size,
            }
        if local_path is not None:
            return {
                "transport": "data_url",
                "selection_reason": "compact_local_fallback",
                "local_path": local_path,
                "remote_url": remote_url,
                "cloud_path": cloud_path,
                "gs_uri": gs_uri,
                "size_bytes": local_size,
            }
        raise FileNotFoundError(f"视频轮次缺少可用资源: {round_data}")

    if video_content_mode == "remote":
        if gs_uri:
            return {
                "transport": "gs_uri_input_audio",
                "selection_reason": "forced_remote_gs_uri",
                "local_path": local_path,
                "remote_url": remote_url,
                "cloud_path": cloud_path,
                "gs_uri": gs_uri,
                "size_bytes": local_size,
            }
        if remote_url:
            return {
                "transport": "remote_url",
                "selection_reason": "forced_remote_url_fallback",
                "local_path": local_path,
                "remote_url": remote_url,
                "cloud_path": cloud_path,
                "gs_uri": gs_uri,
                "size_bytes": local_size,
            }
        if local_path is not None:
            return {
                "transport": "data_url",
                "selection_reason": "forced_remote_fallback_local",
                "local_path": local_path,
                "remote_url": remote_url,
                "cloud_path": cloud_path,
                "gs_uri": gs_uri,
                "size_bytes": local_size,
            }
        raise FileNotFoundError(f"视频轮次缺少可用资源: {round_data}")

    if video_content_mode == "local":
        if local_path is not None:
            return {
                "transport": "data_url",
                "selection_reason": "forced_local_data_url",
                "local_path": local_path,
                "remote_url": remote_url,
                "cloud_path": cloud_path,
                "gs_uri": gs_uri,
                "size_bytes": local_size,
            }
        if gs_uri:
            return {
                "transport": "gs_uri_input_audio",
                "selection_reason": "forced_local_fallback_gs_uri",
                "local_path": local_path,
                "remote_url": remote_url,
                "cloud_path": cloud_path,
                "gs_uri": gs_uri,
                "size_bytes": local_size,
            }
        if remote_url:
            return {
                "transport": "remote_url",
                "selection_reason": "forced_local_fallback_remote_url",
                "local_path": local_path,
                "remote_url": remote_url,
                "cloud_path": cloud_path,
                "gs_uri": gs_uri,
                "size_bytes": local_size,
            }
        raise FileNotFoundError(f"视频轮次缺少可用资源: {round_data}")

    if gs_uri:
        return {
            "transport": "gs_uri_input_audio",
            "selection_reason": "auto_gs_uri_dest_path",
            "local_path": local_path,
            "remote_url": remote_url,
            "cloud_path": cloud_path,
            "gs_uri": gs_uri,
            "size_bytes": local_size,
        }

    if is_history:
        if local_path is not None and local_size is not None and local_size <= max_inline_video_bytes:
            return {
                "transport": "data_url",
                "selection_reason": "auto_history_local_small",
                "local_path": local_path,
                "remote_url": remote_url,
                "cloud_path": cloud_path,
                "gs_uri": gs_uri,
                "size_bytes": local_size,
            }
        return {
            "transport": "omitted_due_size",
            "selection_reason": "auto_history_omitted_due_size",
            "local_path": local_path,
            "remote_url": remote_url,
            "cloud_path": cloud_path,
            "gs_uri": gs_uri,
            "size_bytes": local_size,
        }

    if local_path is not None and local_size is not None and local_size <= max_inline_video_bytes:
        return {
            "transport": "data_url",
            "selection_reason": "auto_current_local_small",
            "local_path": local_path,
            "remote_url": remote_url,
            "cloud_path": cloud_path,
            "gs_uri": gs_uri,
            "size_bytes": local_size,
        }
    if remote_url:
        return {
            "transport": "remote_url",
            "selection_reason": "auto_current_remote_large_video",
            "local_path": local_path,
            "remote_url": remote_url,
            "cloud_path": cloud_path,
            "gs_uri": gs_uri,
            "size_bytes": local_size,
        }
    if local_path is not None:
        return {
            "transport": "data_url",
            "selection_reason": "auto_current_local_no_remote",
            "local_path": local_path,
            "remote_url": remote_url,
            "cloud_path": cloud_path,
            "gs_uri": gs_uri,
            "size_bytes": local_size,
        }
    raise FileNotFoundError(f"视频轮次缺少可用资源: {round_data}")


def build_video_history_omission_note(round_index: int) -> str:
    return (
        f"由于历史视频体积较大，本次请求未附带第 {round_index + 1} 轮的视频内容。"
        "请结合其余历史对话与可见信息理解上下文。"
    )


def build_video_content(
    round_data: Dict[str, Any],
    media_root: Path,
    *,
    is_history: bool,
    video_content_mode: str,
    max_inline_video_bytes: int,
    compact_mode: bool,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[str]]:
    content: List[Dict[str, Any]] = []
    refs: List[Dict[str, Any]] = []
    extra_texts: List[str] = []

    selected = choose_video_transport(
        round_data,
        media_root,
        is_history=is_history,
        video_content_mode=video_content_mode,
        max_inline_video_bytes=max_inline_video_bytes,
        compact_mode=compact_mode,
    )
    transport = selected["transport"]
    local_path = selected.get("local_path")
    remote_url = selected.get("remote_url", "")
    cloud_path = selected.get("cloud_path", "")
    gs_uri = selected.get("gs_uri", "")
    size_bytes = selected.get("size_bytes")
    selection_reason = selected.get("selection_reason", "")

    if transport == "data_url":
        if local_path is None:
            raise FileNotFoundError(f"视频轮次缺少本地或远程可用资源: {round_data}")
        data_url, mime_type, size = encode_file_to_data_url(local_path)
        content.append({"type": "image_url", "image_url": {"url": data_url}})
        refs.append({
            "kind": "video",
            "transport": "data_url",
            "local_path": str(local_path),
            "remote_url": remote_url,
            "cloud_path": cloud_path,
            "gs_uri": gs_uri,
            "mime_type": mime_type,
            "size_bytes": size,
            "content_type": "image_url",
            "selection_reason": selection_reason,
        })
        return content, refs, extra_texts

    if transport == "gs_uri_input_audio":
        if not gs_uri:
            raise FileNotFoundError(f"视频轮次缺少 gs:// 云端引用: {round_data}")
        content.append({"type": "input_audio", "input_audio": {"format": "video/mp4", "data": gs_uri}})
        refs.append({
            "kind": "video",
            "transport": "gs_uri_input_audio",
            "local_path": str(local_path) if local_path is not None else "",
            "remote_url": remote_url,
            "cloud_path": cloud_path,
            "gs_uri": gs_uri,
            "mime_type": "video/mp4",
            "size_bytes": size_bytes,
            "content_type": "input_audio",
            "selection_reason": selection_reason,
        })
        return content, refs, extra_texts

    if transport == "remote_url":
        if not remote_url:
            raise FileNotFoundError(f"视频轮次缺少本地或远程可用资源: {round_data}")
        content.append({"type": "image_url", "image_url": {"url": remote_url}})
        refs.append({
            "kind": "video",
            "transport": "remote_url",
            "local_path": str(local_path) if local_path is not None else "",
            "remote_url": remote_url,
            "cloud_path": cloud_path,
            "gs_uri": gs_uri,
            "mime_type": "video/mp4",
            "size_bytes": size_bytes,
            "content_type": "image_url",
            "selection_reason": selection_reason,
        })
        return content, refs, extra_texts

    if transport == "omitted_due_size":
        refs.append({
            "kind": "video",
            "transport": "omitted_due_size",
            "local_path": str(local_path) if local_path is not None else "",
            "remote_url": remote_url,
            "cloud_path": cloud_path,
            "gs_uri": gs_uri,
            "mime_type": "video/mp4" if (local_path or remote_url or gs_uri) else "",
            "size_bytes": size_bytes,
            "content_type": "omitted",
            "selection_reason": selection_reason,
        })
        if is_history:
            extra_texts.append(build_video_history_omission_note(round_data["round_index"]))
        return content, refs, extra_texts

    raise RuntimeError(f"不支持的视频传输方式: {transport}")


def build_audio_content(round_data: Dict[str, Any], media_root: Path, audio_content_mode: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    content: List[Dict[str, Any]] = []
    refs: List[Dict[str, Any]] = []
    local_path = resolve_local_path(media_root, round_data.get("question_audio_local_path", ""))
    remote_url = round_data.get("question_audio_remote_url", "")

    if local_path is not None:
        data_url, mime_type, size = encode_file_to_data_url(local_path)
        suffix = local_path.suffix.lower().lstrip(".") or "wav"
        audio_format = "mp3" if suffix == "mp3" else suffix
        if audio_content_mode == "input_audio":
            content.append({"type": "input_audio", "input_audio": {"data": data_url.split("base64,", 1)[1], "format": audio_format}})
        else:
            content.append({"type": "audio_url", "audio_url": {"url": data_url}})
        refs.append({
            "kind": "audio",
            "transport": "data_url" if audio_content_mode == "audio_url" else "input_audio",
            "local_path": str(local_path),
            "mime_type": mime_type,
            "size_bytes": size,
        })
        return content, refs

    if remote_url and audio_content_mode == "audio_url":
        content.append({"type": "audio_url", "audio_url": {"url": remote_url}})
        refs.append({"kind": "audio", "transport": "remote_url", "remote_url": remote_url})
        return content, refs

    raise FileNotFoundError(f"音频轮次缺少可用资源，或当前模式不支持远程音频: {round_data}")


def build_user_message(
    spec: TaskSpec,
    round_data: Dict[str, Any],
    media_root: Path,
    *,
    is_history: bool,
    audio_content_mode: str,
    video_content_mode: str,
    max_inline_video_bytes: int,
    video_request_strategy: str,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    content: List[Dict[str, Any]] = []
    media_refs: List[Dict[str, Any]] = []

    if spec.media_mode == "image":
        media_content, media_meta = build_image_content(round_data, media_root)
        extra_texts: List[str] = []
    else:
        media_content, media_meta, extra_texts = build_video_content(
            round_data,
            media_root,
            is_history=is_history,
            video_content_mode=video_content_mode,
            max_inline_video_bytes=max_inline_video_bytes,
            compact_mode=(video_request_strategy == "compact"),
        )
    content.extend(media_content)
    media_refs.extend(media_meta)

    if spec.question_mode == "tts":
        audio_content, audio_meta = build_audio_content(round_data, media_root, audio_content_mode)
        content.extend(audio_content)
        media_refs.extend(audio_meta)

    for note_text in extra_texts:
        content.append({"type": "text", "text": note_text})

    text = build_user_text(
        round_index=round_data["round_index"],
        question_mode=spec.question_mode,
        question_text=round_data.get("question_text", ""),
        is_history=is_history,
    )
    content.append({"type": "text", "text": text})

    blueprint = {
        "role": "user",
        "round_index": round_data["round_index"],
        "is_history": is_history,
        "text": text,
        "media_refs": media_refs,
        "extra_notes": extra_texts,
    }
    return {"role": "user", "content": content}, blueprint


def summarize_assistant_message(round_index: int, answer_text: str, *, is_history: bool) -> Dict[str, Any]:
    return {
        "role": "assistant",
        "round_index": round_index,
        "is_history": is_history,
        "text": answer_text,
    }


def build_messages(
    spec: TaskSpec,
    doc: Dict[str, Any],
    current_round_index: int,
    previous_predictions: Sequence[str],
    media_root: Path,
    audio_content_mode: str,
    video_content_mode: str,
    max_inline_video_bytes: int,
    video_request_strategy: str,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    messages: List[Dict[str, Any]] = [{"role": "system", "content": build_system_prompt(spec)}]
    blueprint: List[Dict[str, Any]] = [{"role": "system", "text": build_system_prompt(spec)}]

    for idx, previous_answer in enumerate(previous_predictions):
        round_data = doc["rounds"][idx]
        user_message, user_blueprint = build_user_message(
            spec,
            round_data,
            media_root,
            is_history=True,
            audio_content_mode=audio_content_mode,
            video_content_mode=video_content_mode,
            max_inline_video_bytes=max_inline_video_bytes,
            video_request_strategy=video_request_strategy,
        )
        messages.append(user_message)
        messages.append({"role": "assistant", "content": previous_answer})
        blueprint.append(user_blueprint)
        blueprint.append(summarize_assistant_message(idx, previous_answer, is_history=True))

    current_round = doc["rounds"][current_round_index]
    current_user_message, current_blueprint = build_user_message(
        spec,
        current_round,
        media_root,
        is_history=False,
        audio_content_mode=audio_content_mode,
        video_content_mode=video_content_mode,
        max_inline_video_bytes=max_inline_video_bytes,
        video_request_strategy=video_request_strategy,
    )
    messages.append(current_user_message)
    blueprint.append(current_blueprint)

    request_context = {
        "history_round_count": len(previous_predictions),
        "current_round_index": current_round_index,
        "question_mode": spec.question_mode,
        "media_mode": spec.media_mode,
        "video_content_mode_requested": video_content_mode if spec.media_mode == "video" else None,
        "video_request_strategy": video_request_strategy if spec.media_mode == "video" else "standard",
        "message_blueprint": blueprint,
    }
    return messages, request_context


class MatrixLLMClient:
    def __init__(
        self,
        *,
        api_url: str,
        api_key_env: str,
        model_name: str,
        timeout: int,
        max_retries: int,
        retry_sleep: float,
        rate_limit_retry_sleep: float,
        rate_limit_max_sleep: float,
        temperature: float,
        top_p: float,
        max_output_tokens: int,
    ) -> None:
        token = os.getenv(api_key_env)
        if not token:
            raise RuntimeError(f"未找到访问 token，请先设置环境变量 {api_key_env}")
        self.api_url = api_url
        self.model_name = model_name
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_sleep = retry_sleep
        self.rate_limit_retry_sleep = rate_limit_retry_sleep
        self.rate_limit_max_sleep = rate_limit_max_sleep
        self.temperature = temperature
        self.top_p = top_p
        self.max_output_tokens = max_output_tokens
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}",
        }

    @staticmethod
    def _parse_retry_after_seconds(response: Optional[requests.Response]) -> Optional[float]:
        if response is None:
            return None
        raw = response.headers.get("Retry-After")
        if not raw:
            return None
        try:
            retry_after = float(raw)
        except (TypeError, ValueError):
            return None
        return retry_after if retry_after >= 0 else None

    def generate(self, messages: Sequence[Dict[str, Any]]) -> Tuple[str, Dict[str, Any], str, int, Dict[str, Any]]:
        payload = {
            "stream": False,
            "model": self.model_name,
            "messages": list(messages),
            "generation_config": {
                "temperature": self.temperature,
                "top_p": self.top_p,
                "max_output_tokens": self.max_output_tokens,
                "response_mime_type": "text/plain",
            },
        }

        last_error = ""
        attempts_used = 0
        error_info: Dict[str, Any] = {
            "status_code": None,
            "error_type": "",
            "retriable": False,
            "retry_trace": [],
        }
        for attempt in range(1, self.max_retries + 1):
            attempts_used = attempt
            retriable = True
            status_code: Optional[int] = None
            error_type = ""
            retry_after_seconds: Optional[float] = None
            try:
                response = requests.post(
                    self.api_url,
                    headers=self.headers,
                    json=payload,
                    timeout=self.timeout,
                )
                response.raise_for_status()
                response_json = response.json()
                generated_text = ((response_json.get("choices") or [{}])[0].get("message") or {}).get("content", "")
                usage = response_json.get("usage", {}) or {}
                cleaned = str(generated_text).strip()
                if cleaned:
                    error_info.update(
                        {
                            "status_code": None,
                            "error_type": "",
                            "retriable": False,
                        }
                    )
                    return cleaned, usage, "", attempts_used, error_info
                last_error = "EmptyResponse: 响应内容为空"
                error_type = "empty_response"
            except requests.HTTPError as exc:
                body = ""
                try:
                    body = exc.response.text
                    status_code = exc.response.status_code
                    retry_after_seconds = self._parse_retry_after_seconds(exc.response)
                except Exception:
                    body = ""
                last_error = f"HTTPError: {exc}; body={body}"
                retriable = bool(status_code in {408, 409, 429} or (status_code is not None and status_code >= 500))
                if status_code == 429:
                    error_type = "rate_limit"
                elif status_code is not None and status_code >= 500:
                    error_type = "server_error"
                elif status_code in {408, 409}:
                    error_type = "transient_http"
                else:
                    error_type = "http_error"
            except (requests.Timeout, requests.ConnectionError) as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                error_type = type(exc).__name__
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                error_type = type(exc).__name__

            error_info.update(
                {
                    "status_code": status_code,
                    "error_type": error_type,
                    "retriable": retriable,
                }
            )

            if not retriable or attempt >= self.max_retries:
                break

            if status_code == 429:
                computed_sleep = min(
                    self.rate_limit_retry_sleep * (2 ** (attempt - 1)) + random.uniform(0, 1.5),
                    self.rate_limit_max_sleep,
                )
                sleep_seconds = max(retry_after_seconds or 0.0, computed_sleep)
                sleep_reason = "rate_limit_backoff"
            else:
                sleep_seconds = min(self.retry_sleep * (2 ** (attempt - 1)) + random.uniform(0, 0.5), 20.0)
                sleep_reason = "generic_backoff"
            error_info["retry_trace"].append(
                {
                    "attempt": attempt,
                    "status_code": status_code,
                    "error_type": error_type,
                    "sleep_seconds": round(float(sleep_seconds), 3),
                    "sleep_reason": sleep_reason,
                    "retry_after_seconds": retry_after_seconds,
                    "error": last_error,
                }
            )
            time.sleep(sleep_seconds)

        return "", {}, last_error, attempts_used, error_info


def is_request_size_error(error: str) -> bool:
    normalized = (error or "").lower()
    return (
        "request body length exceed" in normalized
        or "request_size_limit" in normalized
        or "request entity too large" in normalized
    )


def needs_video_compact_retry(spec: TaskSpec, error: str, request_context: Dict[str, Any]) -> bool:
    return (
        spec.media_mode == "video"
        and request_context.get("video_request_strategy") != "compact"
        and is_request_size_error(error)
    )


def is_rate_limit_error(error: str, error_info: Optional[Dict[str, Any]] = None) -> bool:
    if isinstance(error_info, dict) and error_info.get("status_code") == 429:
        return True
    normalized = (error or "").lower()
    return "rate limit exceeded" in normalized or "429 client error" in normalized or '"code":"429"' in normalized


def _generate_round_result(
    args: argparse.Namespace,
    spec: TaskSpec,
    client: MatrixLLMClient,
    media_root: Path,
    doc: Dict[str, Any],
    round_index: int,
    previous_predictions: Sequence[str],
) -> Tuple[str, Dict[str, Any], str, Optional[Dict[str, Any]]]:
    round_data = doc.get("rounds", [])[round_index]
    round_start = time.time()
    request_fallback = None
    messages, request_context = build_messages(
        spec,
        doc,
        current_round_index=round_index,
        previous_predictions=list(previous_predictions),
        media_root=media_root,
        audio_content_mode=args.audio_content_mode,
        video_content_mode=args.video_content_mode,
        max_inline_video_bytes=args.max_inline_video_bytes,
        video_request_strategy="default",
    )
    prediction, usage, error, attempt_count, error_info = client.generate(messages)
    final_messages = messages

    if needs_video_compact_retry(spec, error, request_context):
        compact_messages, compact_context = build_messages(
            spec,
            doc,
            current_round_index=round_index,
            previous_predictions=list(previous_predictions),
            media_root=media_root,
            audio_content_mode=args.audio_content_mode,
            video_content_mode=args.video_content_mode,
            max_inline_video_bytes=args.max_inline_video_bytes,
            video_request_strategy="compact",
        )
        compact_prediction, compact_usage, compact_error, compact_attempt_count, compact_error_info = client.generate(compact_messages)
        request_fallback = {
            "trigger": "request_size_limit",
            "from_strategy": request_context.get("video_request_strategy", "default"),
            "to_strategy": compact_context.get("video_request_strategy", "compact"),
            "initial_error": error,
        }
        final_messages = compact_messages
        request_context = compact_context
        prediction = compact_prediction
        usage = compact_usage
        error = compact_error
        attempt_count += compact_attempt_count
        error_info = compact_error_info

    latency_seconds = round(time.time() - round_start, 3)
    record = {
        "round_index": round_index,
        "timestamp": round_data.get("timestamp", ""),
        "category": round_data.get("category", ""),
        "question_text": round_data.get("question_text", ""),
        "question_audio_local_path": round_data.get("question_audio_local_path", ""),
        "question_audio_remote_url": round_data.get("question_audio_remote_url", ""),
        "question_audio_cloud_path": round_data.get("question_audio_cloud_path", ""),
        "reference_answer": round_data.get("reference_answer", ""),
        "prediction": prediction,
        "usage": usage,
        "attempt_count": attempt_count,
        "latency_seconds": latency_seconds,
        "request_context": request_context,
        "media_local_path": first_nonempty(round_data.get("image_local_path", ""), round_data.get("video_local_path", "")),
        "media_remote_url": first_nonempty(round_data.get("image_remote_url", ""), round_data.get("video_remote_url", "")),
        "media_cloud_path": first_nonempty(round_data.get("video_cloud_path", "")),
    }
    if error_info:
        record["error_info"] = error_info
    if request_fallback is not None:
        record["request_fallback"] = request_fallback
    if args.save_request_blueprint:
        record["request_messages"] = final_messages
    if error:
        record["error"] = error
    return prediction, record, error, error_info


def process_dialogue_work_item(
    args: argparse.Namespace,
    spec: TaskSpec,
    client: MatrixLLMClient,
    media_root: Path,
    data_root: Path,
    work_item: Phase1DialogueWorkItem,
) -> Dict[str, Any]:
    doc = work_item.doc
    dialogue_id = work_item.dialogue_id
    all_rounds = doc.get("rounds", [])
    resume_round_index = min(max(work_item.resume_round_index, 0), len(all_rounds))
    existing_payload = work_item.existing_payload if isinstance(work_item.existing_payload, dict) else None

    use_current_turn_only_repair = (
        args.repair_failed
        and args.repair_mode == "current_turn_only"
        and existing_payload is not None
        and bool(work_item.failed_round_indices)
    )

    if use_current_turn_only_repair:
        existing_rounds = list(existing_payload.get("rounds") or [])
        round_records: List[Dict[str, Any]] = list(existing_rounds)
        for round_index in work_item.failed_round_indices:
            prediction, record, error, error_info = _generate_round_result(
                args,
                spec,
                client,
                media_root,
                doc,
                round_index,
                [],
            )
            while len(round_records) <= round_index:
                round_records.append({})
            round_records[round_index] = record
            is_last_repaired_round = round_index == work_item.failed_round_indices[-1]
            if not is_last_repaired_round:
                if is_rate_limit_error(error, error_info):
                    time.sleep(max(args.rate_limit_round_cooldown, 0.0))
                elif args.inter_round_sleep > 0:
                    time.sleep(max(args.inter_round_sleep, 0.0))
    else:
        predictions: List[str] = []
        round_records = []
        if existing_payload is not None and resume_round_index > 0:
            existing_rounds = existing_payload.get("rounds") or []
            reusable_rounds = list(existing_rounds[:resume_round_index])
            round_records.extend(reusable_rounds)
            predictions.extend(str(record.get("prediction", "")) for record in reusable_rounds)

        for round_index in range(resume_round_index, len(all_rounds)):
            prediction, record, error, error_info = _generate_round_result(
                args,
                spec,
                client,
                media_root,
                doc,
                round_index,
                predictions,
            )
            predictions.append(prediction)
            round_records.append(record)

            is_last_round = round_index >= len(all_rounds) - 1
            if not is_last_round:
                if is_rate_limit_error(error, error_info):
                    time.sleep(max(args.rate_limit_round_cooldown, 0.0))
                elif args.inter_round_sleep > 0:
                    time.sleep(max(args.inter_round_sleep, 0.0))

    final_round_records: List[Dict[str, Any]] = []
    for round_index in range(len(all_rounds)):
        if round_index < len(round_records) and isinstance(round_records[round_index], dict):
            final_round_records.append(round_records[round_index])
        else:
            final_round_records.append({"round_index": round_index, "error": "MissingRoundAfterRepair", "prediction": ""})
    failed_rounds = sum(1 for record in final_round_records if not is_phase1_round_success(record))

    payload = {
        "benchmark_name": "RUBRIC-MME",
        "phase": "phase1_generation",
        "provider": "matrixllm_internal",
        "model_name": args.model,
        "task_name": spec.name,
        "task_alias": spec.task_alias,
        "dialogue_id": dialogue_id,
        "source_type": doc.get("source_type", ""),
        "environment": doc.get("environment", ""),
        "interaction_setup": doc.get("interaction_setup"),
        "conversation_meta": doc.get("conversation_meta"),
        "question_mode": spec.question_mode,
        "media_mode": spec.media_mode,
        "source_dataset_file": spec.dataset_file,
        "source_data_root": str(data_root),
        "round_count": len(final_round_records),
        "generated_at": utc_now(),
        "rounds": final_round_records,
    }
    return {
        "dialogue_id": dialogue_id,
        "doc_index": work_item.doc_index,
        "payload": payload,
        "failed_rounds": failed_rounds,
    }


def run_task(args: argparse.Namespace, spec: TaskSpec, client: MatrixLLMClient) -> Dict[str, Any]:
    data_root = Path(args.data_root).resolve()
    media_root = Path(args.media_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    task_output_dir = output_dir / spec.name
    samples_path = task_output_dir / f"{safe_name(args.model)}_samples.jsonl"
    summary_path = task_output_dir / f"{safe_name(args.model)}_summary.json"

    docs = load_docs_for_task(spec, data_root, args.dialogue_id, args.limit)
    work_items, samples_index, skipped_dialogues = build_phase1_work_items(
        docs,
        samples_path,
        resume=args.resume,
        repair_failed=args.repair_failed,
        repair_mode=args.repair_mode,
    )
    started_at = utc_now()
    attempted_dialogues = len(work_items)
    completed_dialogues = 0
    failed_rounds = 0
    max_workers = max(1, int(args.max_workers))

    results_by_index: Dict[int, Dict[str, Any]] = {}
    written_payloads_by_id: Dict[str, Dict[str, Any]] = {}
    write_lock = Lock()

    def persist_payload(payload: Dict[str, Any]) -> None:
        dialogue_id = str(payload.get("dialogue_id", "") or "")
        with write_lock:
            written_payloads_by_id[dialogue_id] = payload
            if args.repair_failed or args.resume:
                merged_payloads = merge_phase1_payloads(samples_index, written_payloads_by_id, docs)
                write_jsonl(samples_path, merged_payloads)
            else:
                current_payloads = [
                    written_payloads_by_id[str(doc.get("dialogue_id", "") or "")]
                    for doc in docs
                    if str(doc.get("dialogue_id", "") or "") in written_payloads_by_id
                ]
                write_jsonl(samples_path, current_payloads)

    if max_workers == 1 or len(work_items) <= 1:
        for work_item in work_items:
            result = process_dialogue_work_item(args, spec, client, media_root, data_root, work_item)
            results_by_index[work_item.doc_index] = result
            persist_payload(result["payload"])
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_index = {
                executor.submit(process_dialogue_work_item, args, spec, client, media_root, data_root, work_item): work_item.doc_index
                for work_item in work_items
            }
            for future in as_completed(future_to_index):
                doc_index = future_to_index[future]
                result = future.result()
                results_by_index[doc_index] = result
                persist_payload(result["payload"])

    ordered_payloads: List[Dict[str, Any]] = []
    updated_payloads_by_id: Dict[str, Dict[str, Any]] = {}
    for work_item in work_items:
        result = results_by_index[work_item.doc_index]
        payload = result["payload"]
        ordered_payloads.append(payload)
        updated_payloads_by_id[str(payload.get("dialogue_id", ""))] = payload
        completed_dialogues += 1
        failed_rounds += int(result.get("failed_rounds", 0) or 0)

    if args.repair_failed or args.resume:
        merged_payloads = merge_phase1_payloads(samples_index, updated_payloads_by_id, docs)
        write_jsonl(samples_path, merged_payloads)
    else:
        write_jsonl(samples_path, ordered_payloads)

    summary = {
        "benchmark_name": "RUBRIC-MME",
        "phase": "phase1_generation",
        "provider": "matrixllm_internal",
        "model_name": args.model,
        "task_name": spec.name,
        "task_alias": spec.task_alias,
        "started_at": started_at,
        "completed_at": utc_now(),
        "data_root": str(data_root),
        "media_root": str(media_root),
        "output_dir": str(task_output_dir),
        "attempted_dialogues": attempted_dialogues,
        "completed_dialogues": completed_dialogues,
        "skipped_dialogues": skipped_dialogues,
        "failed_rounds": failed_rounds,
        "samples_path": str(samples_path),
        "audio_content_mode": args.audio_content_mode,
        "video_content_mode": args.video_content_mode,
        "max_inline_video_bytes": args.max_inline_video_bytes,
        "rate_limit_retry_sleep": args.rate_limit_retry_sleep,
        "rate_limit_max_sleep": args.rate_limit_max_sleep,
        "rate_limit_round_cooldown": args.rate_limit_round_cooldown,
        "inter_round_sleep": args.inter_round_sleep,
        "api_url": args.api_url,
        "repair_mode": args.repair_mode if args.repair_failed else "",
    }
    dump_json(summary_path, summary)
    return summary


def main() -> None:
    args = parse_args()
    # Internal MatrixLLM path is stabilized around input_audio for TTS turns.
    if args.audio_content_mode != "input_audio":
        args.audio_content_mode = "input_audio"
    task_names = resolve_task_names(args.tasks)
    client = MatrixLLMClient(
        api_url=args.api_url,
        api_key_env=args.api_key_env,
        model_name=args.model,
        timeout=args.timeout,
        max_retries=args.max_retries,
        retry_sleep=args.retry_sleep,
        rate_limit_retry_sleep=args.rate_limit_retry_sleep,
        rate_limit_max_sleep=args.rate_limit_max_sleep,
        temperature=args.temperature,
        top_p=args.top_p,
        max_output_tokens=args.max_output_tokens,
    )

    summaries: List[Dict[str, Any]] = []
    for task_name in task_names:
        summaries.append(run_task(args, TASK_SPECS[task_name], client))

    run_summary = {
        "benchmark_name": "RUBRIC-MME",
        "phase": "phase1_generation",
        "provider": "matrixllm_internal",
        "model_name": args.model,
        "generated_at": utc_now(),
        "tasks": summaries,
    }

    run_summary = run_phase1_followups(
        summaries=summaries,
        run_summary=run_summary,
        phase2_output_dir=args.phase2_output_dir,
        phase2_keep_existing=args.phase2_keep_existing,
        phase3_output_dir=args.phase3_output_dir,
        phase3_keep_existing=args.phase3_keep_existing,
        dialogue_id=args.dialogue_id or "",
        limit=args.limit,
        phase3_save_prompt_text=args.phase3_save_prompt_text,
        phase3_allow_incomplete_dialogues=args.phase3_allow_incomplete_dialogues,
        build_phase3_backend=lambda: MatrixLLMJudgeBackend(
            api_url=args.api_url,
            api_key_env=args.phase3_judge_api_key_env,
            timeout=args.timeout,
            top_p=args.top_p,
            model_name=args.phase3_judge_model,
            max_retries=args.max_retries,
            retry_sleep=args.retry_sleep,
            temperature=args.temperature,
            max_output_tokens=max(args.max_output_tokens, 2048),
        ),
    )
    dump_json(Path(args.output_dir).resolve() / f"{safe_name(args.model)}_run_summary.json", run_summary)


if __name__ == "__main__":
    main()

