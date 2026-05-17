from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional, Sequence, Tuple

import requests

from gpt_video_frame_utils import (
    DEFAULT_VIDEO_FRAME_COUNT,
    DEFAULT_VIDEO_FRAME_JPEG_QUALITY,
    DEFAULT_VIDEO_FRAME_MAX_INLINE_BYTES,
    DEFAULT_VIDEO_FRAME_MAX_SIDE,
    DEFAULT_VIDEO_FRAME_ROOT_NAME,
    DEFAULT_VIDEO_FRAME_SAMPLING_STRATEGY,
    DEFAULT_VIDEO_PREPARED_DIR_FIELD,
    DEFAULT_VIDEO_PREPARED_DURATION_FIELD,
    DEFAULT_VIDEO_PREPARED_PATHS_FIELD,
    DEFAULT_VIDEO_PREPARED_PROFILE_FIELD,
    DEFAULT_VIDEO_PREPARED_SOURCE_SIZE_FIELD,
    DEFAULT_VIDEO_PREPARED_STRATEGY_FIELD,
    DEFAULT_VIDEO_PREPARED_TOTAL_BYTES_FIELD,
    prepare_video_frames,
    resolve_prepared_frame_paths,
    subset_prepared_frame_paths,
)
from phase1_common import (
    DEFAULT_DATA_ROOT,
    DEFAULT_MEDIA_ROOT,
    Phase1DialogueWorkItem,
    REPO_ROOT,
    TASK_SPECS,
    TaskSpec,
    build_image_docs,
    build_phase1_work_items,
    dump_json,
    is_phase1_round_success,
    load_json_list,
    merge_phase1_payloads,
    safe_name,
    utc_now,
    write_jsonl,
)
from run_openai_compatible_phase1 import (
    DEFAULT_API_KEY_ENV,
    DEFAULT_API_URL,
    DEFAULT_VIDEO_MAX_INLINE_BYTES,
    DEFAULT_VIDEO_PRECOMPRESSED_FIELD,
    OpenAICompatiblePhase1Runner,
    build_media_content,
    encode_file_to_data_url,
    extract_error_message,
    extract_text_response,
    resolve_local_path,
)


DEFAULT_MODEL = "qwen-vl-max-latest"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "logs" / "rubric_mme_qwen_phase1"
PROVIDER_NAME = "qwen_openai_compatible_api"
SUPPORTED_TASKS = [
    "omnibench_image_multi_text",
    "omnibench_image_multi_tts",
    "omnibench_video_stream_text",
    "omnibench_video_stream_tts",
]

QUESTION_MODE_TEXT = "text"
QUESTION_MODE_TRANSCRIPT = "transcript_text_fallback"
QUESTION_MODE_AUDIO_INPUT = "audio_input_audio"

VIDEO_ROUTE_VIDEO = "video"
VIDEO_ROUTE_FRAMES = "frames"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="运行 Qwen 系列 OpenAI 兼容接口版本 RUBRIC-MME Phase 1。"
    )
    parser.add_argument(
        "--tasks",
        default="rubric-mme",
        help="逗号分隔的任务名；rubric-mme/omnibench/all 表示四个任务全部运行。",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help="被测试模型名称。")
    parser.add_argument(
        "--data-root",
        default=str(DEFAULT_DATA_ROOT if DEFAULT_DATA_ROOT.exists() else REPO_ROOT),
        help="RUBRIC-MME JSON 数据根目录。",
    )
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
    parser.add_argument(
        "--repair-mode",
        choices=["resume_from_failure", "current_turn_only"],
        default="resume_from_failure",
        help="repair 模式：resume_from_failure 从首个失败轮继续；current_turn_only 仅修当前失败轮。",
    )
    parser.add_argument("--max-workers", type=int, default=1, help="session 级并行 worker 数。")
    parser.add_argument("--api-url", default=DEFAULT_API_URL, help="OpenAI 兼容 chat/completions 接口地址。")
    parser.add_argument("--api-key-env", default=DEFAULT_API_KEY_ENV, help="保存 API key 的环境变量名。")
    parser.add_argument("--timeout", type=int, default=180, help="单次请求超时时间（秒）。")
    parser.add_argument("--max-retries", type=int, default=5, help="每轮请求最大重试次数。")
    parser.add_argument("--retry-sleep", type=float, default=3.0, help="普通重试基础等待时间（秒）。")
    parser.add_argument("--rate-limit-retry-sleep", type=float, default=20.0, help="429 限流时的基础退避时间（秒）。")
    parser.add_argument("--rate-limit-max-sleep", type=float, default=120.0, help="429 限流时的最大退避时间（秒）。")
    parser.add_argument("--inter-round-sleep", type=float, default=1.0, help="轮与轮之间的默认等待时间（秒）。")
    parser.add_argument("--temperature", type=float, default=0.0, help="采样温度。")
    parser.add_argument("--top-p", type=float, default=0.95, help="top_p。")
    parser.add_argument("--max-output-tokens", type=int, default=1024, help="单轮最大输出 token。")
    parser.add_argument(
        "--stream-mode",
        choices=["auto", "off", "on"],
        default="auto",
        help="是否启用流式输出。auto 会先尝试非流式，再在需要时回退到流式。",
    )
    parser.add_argument(
        "--tts-input-mode",
        choices=["auto", "input_audio", "text_fallback"],
        default="auto",
        help="TTS 任务处理方式。auto 会先尝试 input_audio，失败后回退到文本转写。",
    )
    parser.add_argument(
        "--image-input-mode",
        choices=["local_data_url", "remote_url", "auto"],
        default="auto",
        help="图片输入策略。",
    )
    parser.add_argument(
        "--video-route",
        choices=["auto", VIDEO_ROUTE_VIDEO, VIDEO_ROUTE_FRAMES],
        default="auto",
        help="视频任务通道。video 表示原视频/压缩视频直输，frames 表示抽帧输入，auto 先试 video 再视情况回退到 frames。",
    )
    parser.add_argument(
        "--video-input-mode",
        choices=["local_data_url", "remote_url", "auto"],
        default="auto",
        help="video 直输通道时的视频输入策略。",
    )
    parser.add_argument(
        "--video-history-mode",
        choices=["text_only", "visual"],
        default="text_only",
        help="视频历史处理方式。text_only 仅保留历史文本和历史回答；visual 会在 video 通道中保留历史视频、在 frames 通道中保留历史帧图。",
    )
    parser.add_argument(
        "--video-history-max-visual-rounds",
        type=int,
        default=1,
        help="video 直输通道下，最多为多少个最近历史轮次保留历史视频。仅在 --video-history-mode visual 时生效。",
    )
    parser.add_argument(
        "--video-history-max-inline-bytes-total",
        type=int,
        default=5_500_000,
        help="video 直输通道下，当前轮与历史轮视频原始体积总预算（字节）。用于控制多段历史视频同时输入时的请求体风险。",
    )
    parser.add_argument(
        "--video-precompressed-mode",
        choices=["prefer", "off"],
        default="prefer",
        help="video 直输通道是否优先使用数据集中的预压缩视频。",
    )
    parser.add_argument(
        "--video-precompressed-field",
        default=DEFAULT_VIDEO_PRECOMPRESSED_FIELD,
        help="数据集中记录预压缩视频相对路径的字段名。",
    )
    parser.add_argument(
        "--video-compress-mode",
        choices=["auto", "off"],
        default="auto",
        help="video 直输通道中超大视频是否运行时压缩。",
    )
    parser.add_argument(
        "--video-max-inline-bytes",
        type=int,
        default=DEFAULT_VIDEO_MAX_INLINE_BYTES,
        help="video 直输通道下单个视频内联发送的原始体积上限。",
    )
    parser.add_argument(
        "--video-prepared-frame-mode",
        choices=["prefer", "off"],
        default="prefer",
        help="frames 通道中是否优先使用预抽帧结果。",
    )
    parser.add_argument(
        "--video-prepared-dir-field",
        default=DEFAULT_VIDEO_PREPARED_DIR_FIELD,
        help="数据集中记录预抽帧目录的字段名。",
    )
    parser.add_argument(
        "--video-prepared-paths-field",
        default=DEFAULT_VIDEO_PREPARED_PATHS_FIELD,
        help="数据集中记录预抽帧路径列表的字段名。",
    )
    parser.add_argument(
        "--video-prepared-profile-field",
        default=DEFAULT_VIDEO_PREPARED_PROFILE_FIELD,
        help="数据集中记录预抽帧 profile 的字段名。",
    )
    parser.add_argument(
        "--video-prepared-total-bytes-field",
        default=DEFAULT_VIDEO_PREPARED_TOTAL_BYTES_FIELD,
        help="数据集中记录预抽帧总字节数的字段名。",
    )
    parser.add_argument(
        "--video-prepared-strategy-field",
        default=DEFAULT_VIDEO_PREPARED_STRATEGY_FIELD,
        help="数据集中记录预抽帧采样策略的字段名。",
    )
    parser.add_argument(
        "--video-prepared-source-size-field",
        default=DEFAULT_VIDEO_PREPARED_SOURCE_SIZE_FIELD,
        help="数据集中记录原视频大小的字段名。",
    )
    parser.add_argument(
        "--video-prepared-duration-field",
        default=DEFAULT_VIDEO_PREPARED_DURATION_FIELD,
        help="数据集中记录视频时长的字段名。",
    )
    parser.add_argument(
        "--video-frame-root-name",
        default=DEFAULT_VIDEO_FRAME_ROOT_NAME,
        help="frames 通道运行时 fallback 抽帧缓存目录名。",
    )
    parser.add_argument("--video-frame-count", type=int, default=DEFAULT_VIDEO_FRAME_COUNT, help="当前轮目标帧数。")
    parser.add_argument("--video-frame-max-side", type=int, default=DEFAULT_VIDEO_FRAME_MAX_SIDE, help="帧图长边最大尺寸。")
    parser.add_argument(
        "--video-frame-jpeg-quality",
        type=int,
        default=DEFAULT_VIDEO_FRAME_JPEG_QUALITY,
        help="JPEG 质量参数，数值越大通常体积越小。",
    )
    parser.add_argument(
        "--video-frame-max-inline-bytes",
        type=int,
        default=DEFAULT_VIDEO_FRAME_MAX_INLINE_BYTES,
        help="frames 通道中单轮当前视频帧图总大小预算。",
    )
    parser.add_argument(
        "--video-frame-sampling-strategy",
        choices=["uniform", "hybrid_tail"],
        default=DEFAULT_VIDEO_FRAME_SAMPLING_STRATEGY,
        help="视频抽帧采样策略。",
    )
    parser.add_argument(
        "--video-max-frame-images-per-request",
        type=int,
        default=80,
        help="frames 通道下单次请求最多允许多少张帧图。保守默认值 80 适合更广的 Qwen 族模型。",
    )
    parser.add_argument(
        "--video-history-max-frames-per-round",
        type=int,
        default=4,
        help="frames 通道下每个历史轮次最多保留多少帧。",
    )
    parser.add_argument("--save-request-blueprint", action="store_true", help="将请求蓝图写入每轮记录，方便调试。")
    return parser.parse_args()


def resolve_supported_task_names(tasks_arg: str) -> List[str]:
    raw = [part.strip() for part in tasks_arg.split(",") if part.strip()]
    if not raw:
        return list(SUPPORTED_TASKS)
    lowered = {task.lower() for task in raw}
    if lowered & {"rubric-mme", "rubric_mme", "omnibench", "all"}:
        return list(SUPPORTED_TASKS)
    unknown = [task for task in raw if task not in SUPPORTED_TASKS]
    if unknown:
        raise ValueError(f"Qwen Phase 1 当前支持的任务为 {SUPPORTED_TASKS}，收到不支持任务：{unknown}")
    return raw


def audio_format_from_path(file_path: Path) -> str:
    suffix = file_path.suffix.lower().lstrip(".")
    if suffix in {"mp3", "wav", "flac", "ogg", "aac", "m4a", "webm", "opus"}:
        return suffix
    mime_type, _ = mimetypes.guess_type(str(file_path))
    if mime_type and "/" in mime_type:
        return mime_type.split("/", 1)[1]
    return "mp3"


def should_fallback_audio(error: str) -> bool:
    lowered = (error or "").lower()
    if not lowered:
        return False
    return "input_audio" in lowered or ("audio" in lowered and any(token in lowered for token in ["unsupported", "not support", "content type", "invalid", "schema"]))


def should_fallback_video_route(error: str) -> bool:
    lowered = (error or "").lower()
    if not lowered:
        return False
    hints = [
        "request body length exceed",
        "too many images",
        "video_url",
        "video",
        "content type",
        "unsupported",
        "not support",
        "invalid",
        "too large",
        "data-uri",
        "payload too large",
        "entity too large",
    ]
    return any(token in lowered for token in hints)


def should_fallback_stream(error: str) -> bool:
    lowered = (error or "").lower()
    if not lowered:
        return False
    hints = [
        "stream must be true",
        "stream should be true",
        "stream=true",
        "stream parameter",
        "must set stream",
    ]
    return "stream" in lowered and any(token in lowered for token in hints)


def should_retry_for_reasoning_exhaustion(
    prediction: str,
    response_payload: Dict[str, Any],
    max_output_tokens: int,
) -> bool:
    if prediction.strip():
        return False
    choices = response_payload.get("choices")
    if not isinstance(choices, list) or not choices:
        return False
    finish_reason = str(choices[0].get("finish_reason", "") or "").strip().lower()
    message = choices[0].get("message")
    reasoning_content = ""
    if isinstance(message, dict):
        raw_reasoning = message.get("reasoning_content")
        if isinstance(raw_reasoning, str):
            reasoning_content = raw_reasoning
        elif isinstance(raw_reasoning, list):
            parts: List[str] = []
            for item in raw_reasoning:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict):
                    text = item.get("text")
                    if isinstance(text, str):
                        parts.append(text)
            reasoning_content = "".join(parts)
    usage = response_payload.get("usage", {})
    if not isinstance(usage, dict):
        usage = {}
    completion_details = usage.get("completion_tokens_details", {})
    reasoning_tokens = 0
    if isinstance(completion_details, dict):
        reasoning_tokens = int(completion_details.get("reasoning_tokens", 0) or 0)
    completion_tokens = int(usage.get("completion_tokens", 0) or 0) if usage else 0
    if finish_reason != "length" or max_output_tokens >= 4096:
        return False
    if reasoning_tokens > 0:
        return True
    return bool(reasoning_content.strip()) and completion_tokens >= max_output_tokens


def should_retry_for_empty_completion(
    prediction: str,
    response_payload: Dict[str, Any],
    usage: Dict[str, Any],
) -> bool:
    if prediction.strip():
        return False
    if usage:
        return False
    choices = response_payload.get("choices")
    if not isinstance(choices, list) or not choices:
        return True
    first_choice = choices[0] if isinstance(choices[0], dict) else {}
    finish_reason = str(first_choice.get("finish_reason", "") or "").strip().lower()
    message = first_choice.get("message")
    content = message.get("content") if isinstance(message, dict) else None
    if isinstance(content, str) and content.strip():
        return False
    if isinstance(content, list) and any(
        isinstance(part, dict) and str(part.get("text", "") or "").strip() for part in content
    ):
        return False
    return finish_reason in {"", "stop", "length", "null"}


def build_video_docs(
    raw_items: Sequence[Dict[str, Any]],
    *,
    dialogue_id: Optional[str],
    limit: Optional[int],
    args: argparse.Namespace,
) -> List[Dict[str, Any]]:
    docs: List[Dict[str, Any]] = []
    for item in raw_items:
        current_dialogue_id = str(item.get("_ai_unique_id_", "") or "").strip()
        if dialogue_id and current_dialogue_id != dialogue_id:
            continue
        rounds: List[Dict[str, Any]] = []
        pending_user: Optional[Dict[str, Any]] = None
        for entry in (item.get("stream_conversation") or {}).get("conversations", []):
            speaker = entry.get("speaker")
            if speaker == "user":
                pending_user = entry
                continue
            if speaker != "ai":
                continue
            question_turn = pending_user or {}
            rounds.append(
                {
                    "round_index": len(rounds),
                    "timestamp": question_turn.get("timestamp") or entry.get("timestamp", ""),
                    "question_text": question_turn.get("text", ""),
                    "reference_answer": entry.get("text", ""),
                    "category": entry.get("vqa_category", ""),
                    "video_local_path": entry.get("clip_path", ""),
                    "video_remote_url": entry.get("gcs_url", ""),
                    "video_cloud_path": entry.get("destPath", ""),
                    "video_precompressed_local_path": entry.get(args.video_precompressed_field, ""),
                    "video_precompressed_profile": entry.get(f"{args.video_precompressed_field}_profile", ""),
                    "video_precompressed_size_bytes": entry.get(f"{args.video_precompressed_field}_size_bytes"),
                    "video_precompressed_original_size_bytes": entry.get(f"{args.video_precompressed_field}_original_size_bytes"),
                    "question_audio_local_path": entry.get("user_tts_path", ""),
                    "question_audio_remote_url": entry.get("tts_gcs_url", ""),
                    "question_audio_cloud_path": entry.get("tts_destPath", ""),
                    "prepared_frame_dir": entry.get(args.video_prepared_dir_field, ""),
                    "prepared_frame_paths": entry.get(args.video_prepared_paths_field, []),
                    "prepared_frame_profile": entry.get(args.video_prepared_profile_field, ""),
                    "prepared_frame_total_bytes": entry.get(args.video_prepared_total_bytes_field),
                    "prepared_frame_sampling_strategy": entry.get(args.video_prepared_strategy_field, ""),
                    "prepared_frame_source_size_bytes": entry.get(args.video_prepared_source_size_field),
                    "prepared_frame_duration_seconds": entry.get(args.video_prepared_duration_field),
                }
            )
            pending_user = None
        docs.append(
            {
                "dialogue_id": current_dialogue_id,
                "benchmark_name": "RUBRIC-MME",
                "source_type": "video_stream",
                "environment": item.get("environment", ""),
                "interaction_setup": item.get("interaction_setup"),
                "conversation_meta": item.get("conversation_meta"),
                "rounds": rounds,
            }
        )
        if limit is not None and len(docs) >= limit:
            break
    return docs


def load_docs_for_task(
    spec: TaskSpec,
    data_root: Path,
    dialogue_id: Optional[str],
    limit: Optional[int],
    args: argparse.Namespace,
) -> List[Dict[str, Any]]:
    raw_items = load_json_list((data_root / spec.dataset_file).resolve())
    if spec.source_type == "video_stream":
        return build_video_docs(raw_items, dialogue_id=dialogue_id, limit=limit, args=args)
    docs = build_image_docs(raw_items)
    filtered: List[Dict[str, Any]] = []
    for doc in docs:
        if dialogue_id and doc.get("dialogue_id") != dialogue_id:
            continue
        filtered.append(doc)
        if limit is not None and len(filtered) >= limit:
            break
    return filtered


def build_system_prompt(spec: TaskSpec, question_delivery_mode: str, video_route: str) -> str:
    if spec.media_mode == "image":
        media_desc = "单张图像"
    elif video_route == VIDEO_ROUTE_FRAMES:
        media_desc = "一组按时间顺序排列的关键帧图像，这些关键帧共同表示当前视频片段"
    else:
        media_desc = "完整视频片段"
    if spec.question_mode == "tts":
        if question_delivery_mode == QUESTION_MODE_TRANSCRIPT:
            question_desc = "用户问题原本以语音给出；当前请求中使用语音转写文本代替音频"
        else:
            question_desc = "用户问题以语音给出"
    else:
        question_desc = "用户问题以文本给出"
    return (
        "你正在执行 RUBRIC-MME 第一阶段多轮作答。"
        f"当前任务的视觉输入是{media_desc}，{question_desc}。"
        "请严格遵守以下要求："
        "1. 只能回答当前轮问题；"
        "2. 可以利用历史轮对话和历史轮视觉信息来理解上下文；"
        "3. 绝对不要假设未来轮的信息；"
        "4. 绝对不要输出你看不到的参考答案；"
        "5. 请始终使用中文直接作答，简洁、自然、信息充分。"
    )


def build_user_text(
    round_index: int,
    round_data: Dict[str, Any],
    question_delivery_mode: str,
    *,
    is_history: bool,
    media_mode: str,
    video_route: str,
) -> str:
    prefix = f"这是第{round_index + 1}轮历史对话。" if is_history else f"这是当前第{round_index + 1}轮。"
    question_text = str(round_data.get("question_text", "") or "").strip()
    if media_mode == "video":
        visual_note = "下方提供的是视频片段。" if video_route == VIDEO_ROUTE_VIDEO else "下方提供的是当前视频片段按时间顺序抽取的关键帧。"
    else:
        visual_note = "下方提供的是当前轮图像。"
    if question_delivery_mode == QUESTION_MODE_TRANSCRIPT:
        return (
            f"{prefix}{visual_note}\n这一轮用户问题原本通过语音给出；当前接口改用文本转写提供问题。"
            f"\n用户问题转写：{question_text}\n请只围绕这一轮作答。"
        )
    return f"{prefix}{visual_note}\n用户问题：{question_text}\n请只围绕这一轮作答。"


def build_audio_content(
    round_data: Dict[str, Any],
    media_root: Path,
    question_delivery_mode: str,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    if question_delivery_mode != QUESTION_MODE_AUDIO_INPUT:
        return [], []

    local_path = resolve_local_path(media_root, str(round_data.get("question_audio_local_path", "") or ""))
    remote_url = str(round_data.get("question_audio_remote_url", "") or "").strip()
    if local_path is not None:
        data_url, _mime_type, size = encode_file_to_data_url(local_path)
        audio_format = audio_format_from_path(local_path)
        return (
            [{"type": "input_audio", "input_audio": {"data": data_url, "format": audio_format}}],
            [{"kind": "audio", "transport": "input_audio_data_url", "local_path": str(local_path), "format": audio_format, "size_bytes": size}],
        )
    if remote_url:
        audio_format = audio_format_from_path(Path(remote_url))
        return (
            [{"type": "input_audio", "input_audio": {"data": remote_url, "format": audio_format}}],
            [{"kind": "audio", "transport": "input_audio_remote_url", "remote_url": remote_url, "format": audio_format}],
        )
    raise FileNotFoundError(f"当前轮缺少可用的音频资源: {round_data}")


def build_direct_visual_content(
    spec: TaskSpec,
    round_data: Dict[str, Any],
    media_root: Path,
    args: argparse.Namespace,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    return build_media_content(spec, round_data, media_root, args)


def build_frame_visual_content(
    round_data: Dict[str, Any],
    media_root: Path,
    args: argparse.Namespace,
    *,
    max_frame_count: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    prepared_paths = round_data.get("prepared_frame_paths") or []
    if args.video_prepared_frame_mode == "prefer" and isinstance(prepared_paths, list) and prepared_paths:
        selected_paths = subset_prepared_frame_paths(
            [str(path) for path in prepared_paths],
            max_frame_count,
            args.video_frame_sampling_strategy,
        )
        prepared = resolve_prepared_frame_paths(
            media_root=media_root,
            prepared_paths=selected_paths,
            max_inline_bytes=int(args.video_frame_max_inline_bytes),
        )
        if prepared is not None:
            frame_paths, frame_meta = prepared
            frame_data_urls: List[str] = []
            frame_refs: List[Dict[str, Any]] = []
            total_bytes = 0
            for frame_path in frame_paths:
                data_url, mime_type, size = encode_file_to_data_url(frame_path)
                frame_data_urls.append(data_url)
                frame_refs.append({"frame_path": str(frame_path), "mime_type": mime_type, "size_bytes": size})
                total_bytes += size
            return (
                [{"type": "video", "video": frame_data_urls}],
                [
                    {
                        "kind": "video_frames",
                        "transport": "precomputed_qwen_video_frames",
                        "local_path": str(round_data.get("video_local_path", "") or ""),
                        "prepared_local_path": frame_meta.get("prepared_local_path", ""),
                        "size_bytes": total_bytes,
                        "frame_count": len(frame_paths),
                        "frames": frame_refs,
                        "selected_source": "precomputed_video_frames",
                        **frame_meta,
                    }
                ],
            )

    local_path = resolve_local_path(media_root, str(round_data.get("video_local_path", "") or ""))
    if local_path is None:
        raise FileNotFoundError(f"frames 通道缺少可用本地视频文件：{round_data}")
    prepared_root = (Path(args.output_dir).resolve() / args.video_frame_root_name).resolve()
    frame_paths, frame_meta = prepare_video_frames(
        source_path=local_path,
        media_root=media_root,
        prepared_root=prepared_root,
        base_frame_count=int(args.video_frame_count),
        base_max_side=int(args.video_frame_max_side),
        base_jpeg_quality=int(args.video_frame_jpeg_quality),
        max_inline_bytes=int(args.video_frame_max_inline_bytes),
        sampling_strategy=args.video_frame_sampling_strategy,
    )
    if max_frame_count > 0 and len(frame_paths) > max_frame_count:
        selected_paths = subset_prepared_frame_paths(
            [str(path) for path in frame_meta.get("prepared_rel_paths", []) or []],
            max_frame_count,
            args.video_frame_sampling_strategy,
        )
        prepared = resolve_prepared_frame_paths(
            media_root=media_root,
            prepared_paths=selected_paths,
            max_inline_bytes=int(args.video_frame_max_inline_bytes),
        )
        if prepared is None:
            raise RuntimeError("Qwen frames 通道裁剪后的预抽帧结果无法重新加载。")
        frame_paths, frame_meta = prepared

    frame_data_urls: List[str] = []
    frame_refs: List[Dict[str, Any]] = []
    total_bytes = 0
    for frame_path in frame_paths:
        data_url, mime_type, size = encode_file_to_data_url(frame_path)
        frame_data_urls.append(data_url)
        frame_refs.append({"frame_path": str(frame_path), "mime_type": mime_type, "size_bytes": size})
        total_bytes += size
    return (
        [{"type": "video", "video": frame_data_urls}],
        [
            {
                "kind": "video_frames",
                "transport": "runtime_qwen_video_frames",
                "local_path": str(local_path),
                "prepared_local_path": frame_meta.get("prepared_local_path", ""),
                "size_bytes": total_bytes,
                "frame_count": len(frame_paths),
                "frames": frame_refs,
                "selected_source": "runtime_video_frames",
                **frame_meta,
            }
        ],
    )


class QwenCompatiblePhase1Runner(OpenAICompatiblePhase1Runner):
    def __init__(self, *, stream_mode: str, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.stream_mode = stream_mode
        self._tts_auto_lock = Lock()
        self._tts_resolved_mode: Optional[str] = None
        self._tts_unsupported_modes: set[str] = set()

    def get_tts_candidate_modes(self, requested_mode: str) -> List[str]:
        if requested_mode == "text_fallback":
            return [QUESTION_MODE_TRANSCRIPT]
        if requested_mode == "input_audio":
            return [QUESTION_MODE_AUDIO_INPUT]
        with self._tts_auto_lock:
            if self._tts_resolved_mode:
                return [self._tts_resolved_mode]
            ordered = [QUESTION_MODE_AUDIO_INPUT, QUESTION_MODE_TRANSCRIPT]
            return [mode for mode in ordered if mode not in self._tts_unsupported_modes] or [QUESTION_MODE_TRANSCRIPT]

    def note_tts_mode_success(self, question_delivery_mode: str) -> None:
        if question_delivery_mode != QUESTION_MODE_AUDIO_INPUT:
            return
        with self._tts_auto_lock:
            self._tts_resolved_mode = question_delivery_mode

    def note_tts_mode_unsupported(self, question_delivery_mode: str) -> None:
        if question_delivery_mode != QUESTION_MODE_AUDIO_INPUT:
            return
        with self._tts_auto_lock:
            self._tts_unsupported_modes.add(question_delivery_mode)
            self._tts_resolved_mode = QUESTION_MODE_TRANSCRIPT

    def get_stream_candidates(self) -> List[bool]:
        if self.stream_mode == "on":
            return [True]
        if self.stream_mode == "off":
            return [False]
        return [False, True]

    def _extract_stream_prediction(self, response: requests.Response) -> Tuple[str, Dict[str, Any]]:
        prediction_parts: List[str] = []
        usage: Dict[str, Any] = {}
        for raw_line in response.iter_lines(decode_unicode=True):
            if not raw_line:
                continue
            line = raw_line.strip()
            if not line.startswith("data:"):
                continue
            payload_text = line[5:].strip()
            if not payload_text or payload_text == "[DONE]":
                continue
            try:
                payload = json.loads(payload_text)
            except Exception:
                continue
            if isinstance(payload.get("usage"), dict):
                usage = payload["usage"]
            choices = payload.get("choices")
            if not isinstance(choices, list) or not choices:
                continue
            delta = choices[0].get("delta") or {}
            content = delta.get("content")
            if isinstance(content, str):
                prediction_parts.append(content)
                continue
            if isinstance(content, list):
                for item in content:
                    if not isinstance(item, dict):
                        continue
                    if isinstance(item.get("text"), str):
                        prediction_parts.append(item["text"])
                    elif item.get("type") == "text" and isinstance(item.get("content"), str):
                        prediction_parts.append(item["content"])
        return "".join(prediction_parts), usage

    def generate_round(self, messages: List[Dict[str, Any]]) -> Tuple[str, Dict[str, Any], str, Dict[str, Any]]:
        last_error = ""
        last_error_info: Dict[str, Any] = {"status_code": None, "error_type": "", "retriable": False, "retry_trace": []}
        completion_budgets = [self.max_output_tokens]
        for extra_budget in [2048, 4096]:
            if extra_budget not in completion_budgets:
                completion_budgets.append(extra_budget)

        for budget_index, completion_budget in enumerate(completion_budgets):
            parameter_profiles: List[Dict[str, Any]] = [
                {"temperature": self.temperature, "top_p": self.top_p, "max_tokens": completion_budget},
                {"temperature": self.temperature, "max_tokens": completion_budget},
                {"max_tokens": completion_budget},
                {"max_completion_tokens": completion_budget},
            ]
            deduped_profiles: List[Dict[str, Any]] = []
            seen_profiles: set[Tuple[Tuple[str, Any], ...]] = set()
            for profile in parameter_profiles:
                key = tuple(sorted(profile.items()))
                if key in seen_profiles:
                    continue
                seen_profiles.add(key)
                deduped_profiles.append(profile)

            for profile in deduped_profiles:
                for stream_enabled in self.get_stream_candidates():
                    payload: Dict[str, Any] = {"model": self.model_name, "messages": messages, **profile}
                    if stream_enabled:
                        payload["stream"] = True
                        payload["stream_options"] = {"include_usage": True}
                        if "omni" in (self.model_name or "").lower():
                            payload["modalities"] = ["text"]
                    else:
                        payload["stream"] = False

                    for attempt in range(self.max_retries):
                        try:
                            response = self.session.post(self.api_url, json=payload, timeout=self.timeout, stream=stream_enabled)
                        except requests.RequestException as exc:
                            last_error = f"{type(exc).__name__}: {exc}"
                            error_text = last_error.lower()
                            is_connection_error = "failed to establish a new connection" in error_text or "winerror 10013" in error_text
                            error_type = "connection_exception" if is_connection_error else "request_exception"
                            sleep_seconds = self.retry_sleep
                            if is_connection_error:
                                sleep_seconds = min(max(self.retry_sleep * (2 ** attempt), 10.0), self.rate_limit_max_sleep)
                                self.reset_session()
                            retry_trace = list(last_error_info.get("retry_trace", []))
                            retry_trace.append(
                                {
                                    "attempt": attempt + 1,
                                    "error_type": error_type,
                                    "sleep_seconds": round(sleep_seconds, 2),
                                    "message": last_error[-240:],
                                    "parameter_profile": profile,
                                    "completion_budget": completion_budget,
                                    "stream_enabled": stream_enabled,
                                }
                            )
                            last_error_info = {
                                "status_code": None,
                                "error_type": error_type,
                                "retriable": attempt + 1 < self.max_retries or budget_index + 1 < len(completion_budgets),
                                "retry_trace": retry_trace,
                            }
                            if attempt + 1 < self.max_retries:
                                time.sleep(sleep_seconds)
                                continue
                            if budget_index + 1 < len(completion_budgets):
                                break
                            return "", {}, last_error, last_error_info

                        if response.status_code == 429:
                            sleep_seconds = min(self.rate_limit_retry_sleep * (2 ** attempt), self.rate_limit_max_sleep)
                            retry_trace = list(last_error_info.get("retry_trace", []))
                            retry_trace.append(
                                {
                                    "attempt": attempt + 1,
                                    "status_code": 429,
                                    "error_type": "rate_limit",
                                    "sleep_seconds": round(sleep_seconds, 2),
                                    "parameter_profile": profile,
                                    "completion_budget": completion_budget,
                                    "stream_enabled": stream_enabled,
                                }
                            )
                            last_error_info = {
                                "status_code": 429,
                                "error_type": "rate_limit",
                                "retriable": attempt + 1 < self.max_retries or budget_index + 1 < len(completion_budgets),
                                "retry_trace": retry_trace,
                            }
                            last_error = f"HTTP 429: {extract_error_message(response)}"
                            if attempt + 1 < self.max_retries:
                                time.sleep(sleep_seconds)
                                continue
                            if budget_index + 1 < len(completion_budgets):
                                break
                            return "", {}, last_error, last_error_info

                        if response.status_code >= 400:
                            error_text = extract_error_message(response)
                            retriable = response.status_code >= 500
                            last_error = f"HTTP {response.status_code}: {error_text}"
                            last_error_info = {
                                "status_code": response.status_code,
                                "error_type": "http_error",
                                "retriable": retriable and (attempt + 1 < self.max_retries or budget_index + 1 < len(completion_budgets)),
                                "retry_trace": list(last_error_info.get("retry_trace", [])),
                            }
                            if retriable and attempt + 1 < self.max_retries:
                                time.sleep(self.retry_sleep)
                                continue
                            if not stream_enabled and self.stream_mode == "auto" and should_fallback_stream(last_error):
                                break
                            if budget_index + 1 < len(completion_budgets):
                                break
                            return "", {}, last_error, last_error_info

                        if stream_enabled:
                            prediction, usage = self._extract_stream_prediction(response)
                            response_payload: Dict[str, Any] = {"choices": [{"finish_reason": ""}], "usage": usage}
                        else:
                            response_payload = response.json()
                            prediction = extract_text_response(response_payload)
                            usage = response_payload.get("usage", {}) if isinstance(response_payload.get("usage"), dict) else {}
                        if should_retry_for_reasoning_exhaustion(prediction, response_payload, completion_budget) and budget_index + 1 < len(completion_budgets):
                            last_error = f"Qwen completion budget exhausted at {completion_budget} tokens without visible answer."
                            retry_trace = list(last_error_info.get("retry_trace", []))
                            retry_trace.append(
                                {
                                    "attempt": attempt + 1,
                                    "error_type": "reasoning_exhaustion",
                                    "sleep_seconds": 0.0,
                                    "message": last_error,
                                    "parameter_profile": profile,
                                    "completion_budget": completion_budget,
                                    "stream_enabled": stream_enabled,
                                }
                            )
                            last_error_info = {
                                "status_code": None,
                                "error_type": "reasoning_exhaustion",
                                "retriable": True,
                                "retry_trace": retry_trace,
                            }
                            break
                        if should_retry_for_empty_completion(prediction, response_payload, usage):
                            last_error = "Qwen returned an empty visible completion."
                            retry_trace = list(last_error_info.get("retry_trace", []))
                            retry_trace.append(
                                {
                                    "attempt": attempt + 1,
                                    "error_type": "empty_completion",
                                    "sleep_seconds": 0.0,
                                    "message": last_error,
                                    "parameter_profile": profile,
                                    "completion_budget": completion_budget,
                                    "stream_enabled": stream_enabled,
                                }
                            )
                            last_error_info = {
                                "status_code": None,
                                "error_type": "empty_completion",
                                "retriable": attempt + 1 < self.max_retries or budget_index + 1 < len(completion_budgets),
                                "retry_trace": retry_trace,
                            }
                            if attempt + 1 < self.max_retries:
                                time.sleep(self.retry_sleep)
                                continue
                            if budget_index + 1 < len(completion_budgets):
                                break
                            if not stream_enabled and self.stream_mode == "auto":
                                break
                            return "", {}, last_error, last_error_info
                        return prediction, usage, "", {"status_code": None, "error_type": "", "retriable": False, "retry_trace": []}

        return "", {}, last_error, last_error_info


def build_runner(args: argparse.Namespace) -> QwenCompatiblePhase1Runner:
    return QwenCompatiblePhase1Runner(
        model_name=args.model,
        api_url=args.api_url,
        api_key_env=args.api_key_env,
        timeout=args.timeout,
        max_retries=args.max_retries,
        retry_sleep=args.retry_sleep,
        rate_limit_retry_sleep=args.rate_limit_retry_sleep,
        rate_limit_max_sleep=args.rate_limit_max_sleep,
        temperature=args.temperature,
        top_p=args.top_p,
        max_output_tokens=args.max_output_tokens,
        stream_mode=args.stream_mode,
    )


def resolve_video_route_candidates(spec: TaskSpec, args: argparse.Namespace) -> List[str]:
    if spec.media_mode != "video":
        return [""]
    if args.video_route == "auto":
        return [VIDEO_ROUTE_VIDEO, VIDEO_ROUTE_FRAMES]
    return [args.video_route]


def build_history_frame_budgets(
    *,
    history_round_count: int,
    remaining_image_budget: int,
    per_round_max: int,
) -> Dict[int, int]:
    budgets: Dict[int, int] = {}
    minimum_sequence_images = 4
    if history_round_count <= 0 or remaining_image_budget < minimum_sequence_images or per_round_max <= 0:
        return budgets
    effective_per_round_max = max(per_round_max, minimum_sequence_images)
    for history_index in range(history_round_count - 1, -1, -1):
        if remaining_image_budget < minimum_sequence_images:
            break
        keep_count = min(effective_per_round_max, remaining_image_budget)
        if keep_count < minimum_sequence_images:
            break
        budgets[history_index] = keep_count
        remaining_image_budget -= keep_count
    return budgets


def build_history_video_budgets(
    *,
    history_round_count: int,
    max_visual_rounds: int,
) -> Dict[int, bool]:
    budgets: Dict[int, bool] = {}
    if history_round_count <= 0 or max_visual_rounds <= 0:
        return budgets
    kept = 0
    for history_index in range(history_round_count - 1, -1, -1):
        if kept >= max_visual_rounds:
            break
        budgets[history_index] = True
        kept += 1
    return budgets


def build_messages(
    runner: QwenCompatiblePhase1Runner,
    args: argparse.Namespace,
    spec: TaskSpec,
    doc: Dict[str, Any],
    round_index: int,
    media_root: Path,
    previous_predictions: Sequence[str],
    question_delivery_mode: str,
    video_route: str,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    rounds = doc.get("rounds", [])
    system_prompt = build_system_prompt(spec, question_delivery_mode, video_route)
    messages: List[Dict[str, Any]] = [{"role": "system", "content": system_prompt}]
    blueprint: List[Dict[str, Any]] = [{"role": "system", "text": system_prompt}]

    history_frame_budget: Dict[int, int] = {}
    history_video_budget: Dict[int, bool] = {}
    accumulated_history_video_bytes = 0
    current_round_visual_refs: List[Dict[str, Any]] = []
    current_round_visual_content: List[Dict[str, Any]] = []
    current_round = rounds[round_index]
    if spec.media_mode == "image":
        current_round_visual_content, current_round_visual_refs = build_direct_visual_content(spec, current_round, media_root, args)
    elif video_route == VIDEO_ROUTE_VIDEO:
        current_round_visual_content, current_round_visual_refs = build_direct_visual_content(spec, current_round, media_root, args)
        if args.video_history_mode == "visual":
            history_video_budget = build_history_video_budgets(
                history_round_count=min(len(previous_predictions), round_index),
                max_visual_rounds=max(0, int(args.video_history_max_visual_rounds)),
            )
    else:
        current_round_visual_content, current_round_visual_refs = build_frame_visual_content(
            current_round,
            media_root,
            args,
            max_frame_count=max(1, int(args.video_frame_count)),
        )
        current_frame_count = int(current_round_visual_refs[0].get("frame_count", 0) or 0) if current_round_visual_refs else 0
        remaining_budget = max(0, int(args.video_max_frame_images_per_request) - current_frame_count)
        if args.video_history_mode == "visual":
            history_frame_budget = build_history_frame_budgets(
                history_round_count=min(len(previous_predictions), round_index),
                remaining_image_budget=remaining_budget,
                per_round_max=max(0, int(args.video_history_max_frames_per_round)),
            )

    for history_index in range(min(len(previous_predictions), round_index)):
        history_round = rounds[history_index]
        history_question_mode = question_delivery_mode
        include_history_visual = True
        include_history_audio = spec.question_mode == "tts" and history_question_mode == QUESTION_MODE_AUDIO_INPUT

        if spec.media_mode == "video" and args.video_history_mode == "text_only":
            include_history_visual = False
            include_history_audio = False
            if spec.question_mode == "tts":
                history_question_mode = QUESTION_MODE_TRANSCRIPT

        user_text = build_user_text(
            history_index,
            history_round,
            history_question_mode,
            is_history=True,
            media_mode=spec.media_mode,
            video_route=video_route,
        )
        media_content: List[Dict[str, Any]] = []
        media_refs: List[Dict[str, Any]] = []
        if include_history_visual:
            if spec.media_mode == "image":
                visual_content, visual_refs = build_direct_visual_content(spec, history_round, media_root, args)
                media_content.extend(visual_content)
                media_refs.extend(visual_refs)
            elif video_route == VIDEO_ROUTE_VIDEO:
                if history_video_budget.get(history_index, False):
                    visual_content, visual_refs = build_direct_visual_content(spec, history_round, media_root, args)
                    current_size = int(current_round_visual_refs[0].get("size_bytes", 0) or 0) if current_round_visual_refs else 0
                    candidate_size = sum(
                        int(ref.get("size_bytes", 0) or 0)
                        for ref in visual_refs
                        if isinstance(ref, dict)
                    )
                    total_budget = max(0, int(args.video_history_max_inline_bytes_total))
                    if total_budget <= 0 or current_size + accumulated_history_video_bytes + candidate_size <= total_budget:
                        media_content.extend(visual_content)
                        media_refs.extend(visual_refs)
                        accumulated_history_video_bytes += candidate_size
                    else:
                        media_refs.append(
                            {
                                "kind": "video",
                                "transport": "omitted_history_video_budget",
                                "note": "该历史轮视频为控制总视频字节预算未发送视觉输入。",
                                "candidate_size_bytes": candidate_size,
                                "total_budget_bytes": total_budget,
                            }
                        )
                else:
                    media_refs.append(
                        {
                            "kind": "video",
                            "transport": "omitted_history_video_count",
                            "note": "该历史轮视频为控制历史视觉轮次数未发送视觉输入。",
                        }
                    )
            else:
                history_frame_count = history_frame_budget.get(history_index, 0)
                if history_frame_count > 0:
                    visual_content, visual_refs = build_frame_visual_content(
                        history_round,
                        media_root,
                        args,
                        max_frame_count=history_frame_count,
                    )
                    media_content.extend(visual_content)
                    media_refs.extend(visual_refs)
                else:
                    media_refs.append(
                        {
                            "kind": "video_frames",
                            "transport": "omitted_history_frames",
                            "note": "该历史轮为控制总帧图预算未发送视觉输入。",
                        }
                    )
        else:
            media_refs.append(
                {
                    "kind": spec.media_mode,
                    "transport": "omitted_history_media",
                    "note": "为控制请求体大小未重复发送历史视觉输入，仅保留历史文本与历史回答。",
                }
            )

        if include_history_audio:
            audio_content, audio_refs = build_audio_content(history_round, media_root, history_question_mode)
            media_content.extend(audio_content)
            media_refs.extend(audio_refs)
        elif spec.question_mode == "tts" and history_question_mode == QUESTION_MODE_TRANSCRIPT:
            media_refs.append(
                {
                    "kind": "audio",
                    "transport": "transcript_text_fallback",
                    "note": "历史语音问题改用文本转写提供。",
                }
            )

        user_content = [{"type": "text", "text": user_text}] + media_content
        messages.append({"role": "user", "content": user_content})
        messages.append({"role": "assistant", "content": str(previous_predictions[history_index])})
        blueprint.append({"role": "user", "round_index": history_index, "is_history": True, "text": user_text, "media_refs": media_refs})
        blueprint.append({"role": "assistant", "round_index": history_index, "is_history": True, "text": str(previous_predictions[history_index])})

    current_text = build_user_text(
        round_index,
        current_round,
        question_delivery_mode,
        is_history=False,
        media_mode=spec.media_mode,
        video_route=video_route,
    )
    current_audio_content, current_audio_refs = build_audio_content(current_round, media_root, question_delivery_mode) if spec.question_mode == "tts" else ([], [])
    current_content = [{"type": "text", "text": current_text}] + current_round_visual_content + current_audio_content
    messages.append({"role": "user", "content": current_content})
    current_refs = list(current_round_visual_refs) + list(current_audio_refs)
    blueprint.append({"role": "user", "round_index": round_index, "is_history": False, "text": current_text, "media_refs": current_refs})

    request_context: Dict[str, Any] = {
        "history_round_count": min(len(previous_predictions), round_index),
        "current_round_index": round_index,
        "question_mode": spec.question_mode,
        "media_mode": spec.media_mode,
        "question_delivery_mode": question_delivery_mode,
        "video_route_requested": args.video_route if spec.media_mode == "video" else "",
        "video_route_effective": video_route if spec.media_mode == "video" else "",
        "video_history_mode": args.video_history_mode if spec.media_mode == "video" else "",
        "video_history_max_visual_rounds": args.video_history_max_visual_rounds if spec.media_mode == "video" else 0,
        "video_history_max_inline_bytes_total": args.video_history_max_inline_bytes_total if spec.media_mode == "video" else 0,
        "tts_input_mode_requested": args.tts_input_mode if spec.question_mode == "tts" else "",
        "image_input_mode": args.image_input_mode if spec.media_mode == "image" else "",
        "video_input_mode": args.video_input_mode if spec.media_mode == "video" else "",
    }
    if current_round_visual_refs:
        request_context["current_media_ref"] = current_round_visual_refs[0]
    if current_audio_refs:
        request_context["current_audio_ref"] = current_audio_refs[0]
    if args.save_request_blueprint:
        request_context["message_blueprint"] = blueprint
    return messages, request_context


def generate_round_record(
    args: argparse.Namespace,
    spec: TaskSpec,
    runner: QwenCompatiblePhase1Runner,
    media_root: Path,
    doc: Dict[str, Any],
    round_index: int,
    previous_predictions: Sequence[str],
) -> Tuple[str, Dict[str, Any]]:
    round_data = doc["rounds"][round_index]
    request_started = time.time()
    question_modes = runner.get_tts_candidate_modes(args.tts_input_mode) if spec.question_mode == "tts" else [QUESTION_MODE_TEXT]
    video_routes = resolve_video_route_candidates(spec, args)

    prediction = ""
    usage: Dict[str, Any] = {}
    error = ""
    error_info: Dict[str, Any] = {"status_code": None, "error_type": "", "retriable": False, "retry_trace": []}
    request_context: Dict[str, Any] = {}
    selected_question_mode = QUESTION_MODE_TEXT
    selected_video_route = video_routes[0]

    completed = False
    for question_mode in question_modes:
        route_fallback_triggered = False
        for video_route in video_routes:
            selected_question_mode = question_mode
            selected_video_route = video_route
            messages, request_context = build_messages(
                runner,
                args,
                spec,
                doc,
                round_index,
                media_root,
                previous_predictions,
                question_mode,
                video_route,
            )
            prediction, usage, error, error_info = runner.generate_round(messages)
            if error and spec.question_mode == "tts" and args.tts_input_mode == "auto" and question_mode == QUESTION_MODE_AUDIO_INPUT and should_fallback_audio(error):
                runner.note_tts_mode_unsupported(question_mode)
                route_fallback_triggered = False
                break
            if error and spec.media_mode == "video" and args.video_route == "auto" and video_route == VIDEO_ROUTE_VIDEO and should_fallback_video_route(error):
                route_fallback_triggered = True
                continue
            if not error and spec.question_mode == "tts" and args.tts_input_mode == "auto" and question_mode == QUESTION_MODE_AUDIO_INPUT:
                runner.note_tts_mode_success(question_mode)
            completed = True
            break
        if completed:
            break
        if route_fallback_triggered:
            continue

    latency_seconds = round(time.time() - request_started, 3)
    media_local_path = str(round_data.get("image_local_path", "") or round_data.get("video_local_path", "") or "")
    effective_media_local_path = ""
    effective_media_source = ""
    current_media_ref = request_context.get("current_media_ref")
    if isinstance(current_media_ref, dict):
        effective_media_local_path = str(current_media_ref.get("prepared_local_path") or current_media_ref.get("local_path") or "")
        effective_media_source = str(current_media_ref.get("selected_source") or current_media_ref.get("transport") or "")
    media_remote_url = str(round_data.get("image_remote_url", "") or round_data.get("video_remote_url", "") or "")
    media_cloud_path = str(round_data.get("video_cloud_path", "") or "")
    round_record: Dict[str, Any] = {
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
        "attempt_count": 1,
        "latency_seconds": latency_seconds,
        "request_context": request_context,
        "media_local_path": media_local_path,
        "effective_media_local_path": effective_media_local_path,
        "effective_media_source": effective_media_source,
        "media_remote_url": media_remote_url,
        "media_cloud_path": media_cloud_path,
        "error_info": error_info,
        "tts_input_mode_effective": selected_question_mode if spec.question_mode == "tts" else "",
        "video_route_effective": selected_video_route if spec.media_mode == "video" else "",
    }
    if error:
        round_record["error"] = error
    return prediction, round_record


def process_dialogue_work_item(
    args: argparse.Namespace,
    spec: TaskSpec,
    media_root: Path,
    work_item: Phase1DialogueWorkItem,
    shared_runner: Optional[QwenCompatiblePhase1Runner] = None,
) -> Dict[str, Any]:
    owns_runner = shared_runner is None
    runner = shared_runner or build_runner(args)
    try:
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
                _prediction, round_record = generate_round_record(args, spec, runner, media_root, doc, round_index, [])
                while len(round_records) <= round_index:
                    round_records.append({})
                round_records[round_index] = round_record
        else:
            predictions: List[str] = []
            round_records: List[Dict[str, Any]] = []
            if existing_payload is not None and resume_round_index > 0:
                existing_rounds = existing_payload.get("rounds") or []
                reusable_rounds = list(existing_rounds[:resume_round_index])
                round_records.extend(reusable_rounds)
                predictions.extend(str(record.get("prediction", "")) for record in reusable_rounds)
            for round_index in range(resume_round_index, len(all_rounds)):
                prediction, round_record = generate_round_record(args, spec, runner, media_root, doc, round_index, predictions)
                predictions.append(prediction)
                round_records.append(round_record)
                if args.inter_round_sleep > 0 and round_index + 1 < len(all_rounds):
                    time.sleep(args.inter_round_sleep)

        final_round_records: List[Dict[str, Any]] = []
        for round_index in range(len(all_rounds)):
            if round_index < len(round_records) and isinstance(round_records[round_index], dict):
                final_round_records.append(round_records[round_index])
            else:
                final_round_records.append({"round_index": round_index, "error": "MissingRoundAfterRepair", "prediction": ""})
        failed_rounds = sum(1 for record in final_round_records if not is_phase1_round_success(record))
        effective_tts_modes = sorted(
            {
                str(record.get("tts_input_mode_effective", "")).strip()
                for record in final_round_records
                if isinstance(record, dict) and str(record.get("tts_input_mode_effective", "")).strip()
            }
        )
        effective_video_routes = sorted(
            {
                str(record.get("video_route_effective", "")).strip()
                for record in final_round_records
                if isinstance(record, dict) and str(record.get("video_route_effective", "")).strip()
            }
        )
        payload = {
            "benchmark_name": "RUBRIC-MME",
            "phase": "phase1_generation",
            "provider": PROVIDER_NAME,
            "task_name": spec.name,
            "task_alias": spec.task_alias,
            "model_name": args.model,
            "dialogue_id": dialogue_id,
            "source_type": doc.get("source_type", ""),
            "environment": doc.get("environment", ""),
            "interaction_setup": doc.get("interaction_setup"),
            "conversation_meta": doc.get("conversation_meta"),
            "round_count": len(final_round_records),
            "question_mode": spec.question_mode,
            "media_mode": spec.media_mode,
            "source_dataset_file": spec.dataset_file,
            "source_data_root": str(Path(args.data_root).resolve()),
            "generated_at": utc_now(),
            "tts_input_mode_requested": args.tts_input_mode if spec.question_mode == "tts" else "",
            "tts_input_modes_effective": effective_tts_modes if spec.question_mode == "tts" else [],
            "video_route_requested": args.video_route if spec.media_mode == "video" else "",
            "video_routes_effective": effective_video_routes if spec.media_mode == "video" else [],
            "rounds": final_round_records,
        }
        return {"dialogue_id": dialogue_id, "doc_index": work_item.doc_index, "payload": payload, "failed_rounds": failed_rounds}
    finally:
        if owns_runner:
            runner.close()


def summarize_effective_modes(payloads: Sequence[Dict[str, Any]], field_name: str) -> List[str]:
    values: set[str] = set()
    for payload in payloads:
        for value in payload.get(field_name, []) or []:
            if isinstance(value, str) and value.strip():
                values.add(value.strip())
    return sorted(values)


def run_task(args: argparse.Namespace, spec: TaskSpec) -> Dict[str, Any]:
    data_root = Path(args.data_root).resolve()
    media_root = Path(args.media_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    task_output_dir = output_dir / spec.name
    samples_path = task_output_dir / f"{safe_name(args.model)}_samples.jsonl"
    summary_path = task_output_dir / f"{safe_name(args.model)}_summary.json"
    docs = load_docs_for_task(spec, data_root, args.dialogue_id, args.limit, args)
    work_items, samples_index, skipped = build_phase1_work_items(
        docs,
        samples_path,
        resume=args.resume,
        repair_failed=args.repair_failed,
        repair_mode=args.repair_mode,
    )
    started_at = utc_now()
    attempted = len(work_items)
    completed = 0
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

    max_workers = max(1, int(args.max_workers))
    if max_workers == 1 or len(work_items) <= 1:
        active_runner: Optional[QwenCompatiblePhase1Runner] = None
        try:
            if work_items:
                active_runner = build_runner(args)
            for work_item in work_items:
                result = process_dialogue_work_item(args, spec, media_root, work_item, active_runner)
                results_by_index[work_item.doc_index] = result
                persist_payload(result["payload"])
        finally:
            if active_runner is not None:
                active_runner.close()
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_index = {
                executor.submit(process_dialogue_work_item, args, spec, media_root, work_item, None): work_item.doc_index
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
        completed += 1

    if args.repair_failed or args.resume:
        final_payloads = merge_phase1_payloads(samples_index, updated_payloads_by_id, docs)
    else:
        final_payloads = ordered_payloads
    write_jsonl(samples_path, final_payloads)

    total_failed_rounds = 0
    for payload in final_payloads:
        total_failed_rounds += sum(1 for round_record in payload.get("rounds", []) if not is_phase1_round_success(round_record))

    summary = {
        "benchmark_name": "RUBRIC-MME",
        "phase": "phase1_generation",
        "provider": PROVIDER_NAME,
        "task_name": spec.name,
        "task_alias": spec.task_alias,
        "model_name": args.model,
        "started_at": started_at,
        "completed_at": utc_now(),
        "data_root": str(data_root),
        "media_root": str(media_root),
        "output_dir": str(task_output_dir),
        "attempted_dialogues": attempted,
        "completed_dialogues": completed,
        "skipped_dialogues": skipped,
        "dialogue_count_total": len(final_payloads),
        "failed_rounds": total_failed_rounds,
        "samples_path": str(samples_path),
        "api_url": args.api_url,
        "tts_input_mode_requested": args.tts_input_mode if spec.question_mode == "tts" else "",
        "tts_input_modes_effective": summarize_effective_modes(final_payloads, "tts_input_modes_effective") if spec.question_mode == "tts" else [],
        "video_route_requested": args.video_route if spec.media_mode == "video" else "",
        "video_routes_effective": summarize_effective_modes(final_payloads, "video_routes_effective") if spec.media_mode == "video" else [],
        "video_history_mode": args.video_history_mode if spec.media_mode == "video" else "",
        "video_history_max_visual_rounds": args.video_history_max_visual_rounds if spec.media_mode == "video" else 0,
        "video_history_max_inline_bytes_total": args.video_history_max_inline_bytes_total if spec.media_mode == "video" else 0,
        "video_precompressed_mode": args.video_precompressed_mode if spec.media_mode == "video" else "",
        "video_precompressed_field": args.video_precompressed_field if spec.media_mode == "video" else "",
        "video_prepared_frame_mode": args.video_prepared_frame_mode if spec.media_mode == "video" else "",
        "video_frame_count": args.video_frame_count if spec.media_mode == "video" else 0,
        "video_frame_max_side": args.video_frame_max_side if spec.media_mode == "video" else 0,
        "video_frame_jpeg_quality": args.video_frame_jpeg_quality if spec.media_mode == "video" else 0,
        "video_frame_max_inline_bytes": args.video_frame_max_inline_bytes if spec.media_mode == "video" else 0,
        "video_frame_sampling_strategy": args.video_frame_sampling_strategy if spec.media_mode == "video" else "",
        "video_max_frame_images_per_request": args.video_max_frame_images_per_request if spec.media_mode == "video" else 0,
        "video_history_max_frames_per_round": args.video_history_max_frames_per_round if spec.media_mode == "video" else 0,
        "repair_mode": args.repair_mode if args.repair_failed else "",
    }
    dump_json(summary_path, summary)
    return summary


def main() -> None:
    args = parse_args()
    task_names = resolve_supported_task_names(args.tasks)
    summaries: List[Dict[str, Any]] = []
    for task_name in task_names:
        summaries.append(run_task(args, TASK_SPECS[task_name]))
    output_dir = Path(args.output_dir).resolve()
    run_summary = {
        "benchmark_name": "RUBRIC-MME",
        "phase": "phase1_generation",
        "provider": PROVIDER_NAME,
        "model_name": args.model,
        "generated_at": utc_now(),
        "tasks": summaries,
    }
    dump_json(output_dir / f"{safe_name(args.model)}_run_summary.json", run_summary)


if __name__ == "__main__":
    main()
