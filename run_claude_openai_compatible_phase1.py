from __future__ import annotations

import argparse
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
    subset_frame_paths,
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
    OpenAICompatiblePhase1Runner,
    extract_error_message,
    extract_text_response,
    encode_file_to_data_url,
    resolve_local_path,
)


DEFAULT_MODEL = "gpt-4o-2024-11-20"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "logs" / "rubric_mme_gpt_phase1"
SUPPORTED_TASKS = [
    "omnibench_image_multi_text",
    "omnibench_video_stream_text",
]


def is_gpt5_family_model(model_name: str) -> bool:
    normalized = (model_name or "").strip().lower()
    return normalized.startswith("gpt-5")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行 GPT 系列 OpenAI 兼容接口版本的 RUBRIC-MME Phase 1。")
    parser.add_argument("--tasks", default="rubric-mme", help="逗号分隔的任务名；rubric-mme/omnibench/all 默认只运行 GPT 当前支持的 image_text 和 video_text。")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="被测试模型名称。")
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT if DEFAULT_DATA_ROOT.exists() else REPO_ROOT), help="RUBRIC-MME JSON 数据根目录。")
    parser.add_argument("--media-root", default=str(DEFAULT_MEDIA_ROOT if DEFAULT_MEDIA_ROOT.exists() else REPO_ROOT), help="图片、视频相对路径所对应的媒体根目录。")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="输出目录。")
    parser.add_argument("--limit", type=int, default=None, help="每个任务最多处理多少个 dialogue。")
    parser.add_argument("--dialogue-id", default=None, help="只运行指定的 dialogue_id。")
    parser.add_argument("--resume", action="store_true", help="基于已有 samples.jsonl 跳过已成功 dialogue。")
    parser.add_argument("--repair-failed", action="store_true", help="只修复已有 samples.jsonl 中失败或未完整的 dialogue。")
    parser.add_argument("--repair-mode", choices=["resume_from_failure", "current_turn_only"], default="resume_from_failure", help="repair 模式。")
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
    parser.add_argument("--max-output-tokens", type=int, default=512, help="单轮最大输出 token。")
    parser.add_argument("--image-input-mode", choices=["local_data_url", "remote_url", "auto"], default="auto", help="图片输入策略。")
    parser.add_argument("--video-history-mode", choices=["text_only", "frames"], default="text_only", help="视频历史轮的处理方式。")
    parser.add_argument("--video-prepared-frame-mode", choices=["prefer", "off"], default="prefer", help="是否优先使用预抽帧结果。")
    parser.add_argument("--video-prepared-dir-field", default=DEFAULT_VIDEO_PREPARED_DIR_FIELD, help="视频 JSON 中记录预抽帧目录的字段名。")
    parser.add_argument("--video-prepared-paths-field", default=DEFAULT_VIDEO_PREPARED_PATHS_FIELD, help="视频 JSON 中记录预抽帧路径列表的字段名。")
    parser.add_argument("--video-prepared-profile-field", default=DEFAULT_VIDEO_PREPARED_PROFILE_FIELD, help="视频 JSON 中记录预抽帧 profile 的字段名。")
    parser.add_argument("--video-prepared-total-bytes-field", default=DEFAULT_VIDEO_PREPARED_TOTAL_BYTES_FIELD, help="视频 JSON 中记录预抽帧总字节数的字段名。")
    parser.add_argument("--video-prepared-strategy-field", default=DEFAULT_VIDEO_PREPARED_STRATEGY_FIELD, help="视频 JSON 中记录预抽帧采样策略的字段名。")
    parser.add_argument("--video-prepared-source-size-field", default=DEFAULT_VIDEO_PREPARED_SOURCE_SIZE_FIELD, help="视频 JSON 中记录原视频大小的字段名。")
    parser.add_argument("--video-prepared-duration-field", default=DEFAULT_VIDEO_PREPARED_DURATION_FIELD, help="视频 JSON 中记录视频时长的字段名。")
    parser.add_argument("--video-frame-root-name", default=DEFAULT_VIDEO_FRAME_ROOT_NAME, help="运行时 fallback 抽帧缓存目录名。")
    parser.add_argument("--video-frame-count", type=int, default=DEFAULT_VIDEO_FRAME_COUNT, help="基础目标帧数。")
    parser.add_argument("--video-frame-max-side", type=int, default=DEFAULT_VIDEO_FRAME_MAX_SIDE, help="帧图长边最大尺寸。")
    parser.add_argument("--video-frame-jpeg-quality", type=int, default=DEFAULT_VIDEO_FRAME_JPEG_QUALITY, help="JPEG 质量参数，数值越大体积通常越小。")
    parser.add_argument("--video-frame-max-inline-bytes", type=int, default=DEFAULT_VIDEO_FRAME_MAX_INLINE_BYTES, help="单轮所有帧图总大小预算。")
    parser.add_argument("--video-frame-sampling-strategy", choices=["uniform", "hybrid_tail"], default=DEFAULT_VIDEO_FRAME_SAMPLING_STRATEGY, help="视频抽帧采样策略。")
    parser.add_argument("--video-max-images-per-request", type=int, default=50, help="单次请求最多允许多少张图片输入。")
    parser.add_argument("--video-history-max-frames-per-round", type=int, default=4, help="当启用历史视频帧时，每个历史轮次最多保留多少帧。")
    parser.add_argument("--save-request-blueprint", action="store_true", help="将请求蓝图写入每轮记录，方便调试。")
    return parser.parse_args()


def resolve_task_names(tasks_arg: str) -> List[str]:
    raw = [part.strip() for part in tasks_arg.split(",") if part.strip()]
    if not raw:
        return list(SUPPORTED_TASKS)
    lowered = {task.lower() for task in raw}
    if lowered & {"rubric-mme", "rubric_mme", "omnibench", "all"}:
        return list(SUPPORTED_TASKS)
    unknown = [task for task in raw if task not in SUPPORTED_TASKS]
    if unknown:
        raise ValueError(f"GPT Phase 1 当前仅支持 {SUPPORTED_TASKS}，收到不支持任务：{unknown}")
    return raw


def build_system_prompt(spec: TaskSpec) -> str:
    if spec.media_mode == "video":
        media_desc = "一组按时间顺序排列的关键帧图像，这些关键帧共同表示当前视频片段"
    else:
        media_desc = "单张图像"
    return (
        "你正在执行 RUBRIC-MME 第一阶段多轮作答。"
        f"当前任务的视觉输入是{media_desc}，用户问题以文本给出。"
        "请严格遵守以下要求："
        "1. 只回答当前轮问题；"
        "2. 可以利用历史轮对话和历史轮视觉信息来理解上下文；"
        "3. 绝对不要假设未来轮的信息；"
        "4. 绝对不要输出你看不到的参考答案；"
        "5. 请始终使用中文直接作答，简洁、自然、信息充分。"
    )


def build_user_text(round_index: int, round_data: Dict[str, Any], *, is_history: bool, media_mode: str) -> str:
    prefix = f"这是第{round_index + 1}轮历史对话。" if is_history else f"这是当前第{round_index + 1}轮。"
    question_text = str(round_data.get("question_text", "") or "").strip()
    if media_mode == "video":
        visual_note = "下方提供的是当前视频片段按时间顺序抽取的关键帧图像。"
    else:
        visual_note = "下方提供的是当前轮的图像。"
    return f"{prefix}{visual_note}\n用户问题：{question_text}\n请只围绕这一轮作答。"


def build_gpt_video_docs(
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


def load_docs_for_task(spec: TaskSpec, data_root: Path, dialogue_id: Optional[str], limit: Optional[int], args: argparse.Namespace) -> List[Dict[str, Any]]:
    raw_items = load_json_list((data_root / spec.dataset_file).resolve())
    if spec.media_mode == "image":
        docs = build_image_docs(raw_items)
        filtered: List[Dict[str, Any]] = []
        for doc in docs:
            if dialogue_id and doc.get("dialogue_id") != dialogue_id:
                continue
            filtered.append(doc)
            if limit is not None and len(filtered) >= limit:
                break
        return filtered
    return build_gpt_video_docs(raw_items, dialogue_id=dialogue_id, limit=limit, args=args)


def build_image_content(round_data: Dict[str, Any], media_root: Path, args: argparse.Namespace) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    remote_url = str(round_data.get("image_remote_url", "") or "").strip()
    local_path = resolve_local_path(media_root, str(round_data.get("image_local_path", "") or ""))
    if args.image_input_mode in {"local_data_url", "auto"} and local_path is not None:
        data_url, mime_type, size = encode_file_to_data_url(local_path)
        return (
            [{"type": "image_url", "image_url": {"url": data_url}}],
            [{"kind": "image", "transport": "data_url", "local_path": str(local_path), "mime_type": mime_type, "size_bytes": size}],
        )
    if remote_url:
        return (
            [{"type": "image_url", "image_url": {"url": remote_url}}],
            [{"kind": "image", "transport": "remote_url", "local_path": str(local_path) if local_path else "", "remote_url": remote_url}],
        )
    raise FileNotFoundError(f"图片轮次缺少可用图像资源: {round_data}")


def build_video_content(
    round_data: Dict[str, Any],
    media_root: Path,
    args: argparse.Namespace,
    *,
    max_frame_count: int = 0,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    if args.video_prepared_frame_mode == "prefer":
        prepared_paths = round_data.get("prepared_frame_paths") or []
        if isinstance(prepared_paths, list) and prepared_paths:
            if max_frame_count > 0:
                prepared_paths = subset_prepared_frame_paths(prepared_paths, max_frame_count, args.video_frame_sampling_strategy)
            prepared = resolve_prepared_frame_paths(
                media_root=media_root,
                prepared_paths=[str(path) for path in prepared_paths],
                max_inline_bytes=int(args.video_frame_max_inline_bytes),
            )
            if prepared is not None:
                frame_paths, frame_meta = prepared
                content: List[Dict[str, Any]] = []
                frame_refs: List[Dict[str, Any]] = []
                total_bytes = 0
                for frame_path in frame_paths:
                    data_url, mime_type, size = encode_file_to_data_url(frame_path)
                    total_bytes += size
                    content.append({"type": "image_url", "image_url": {"url": data_url}})
                    frame_refs.append({"frame_path": str(frame_path), "mime_type": mime_type, "size_bytes": size})
                return (
                    content,
                    [
                        {
                            "kind": "video_frames",
                            "transport": "precomputed_frame_data_urls",
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
        raise FileNotFoundError(f"视频轮次缺少本地视频文件，当前 GPT 脚本需要先从本地视频抽帧: {round_data}")
    prepared_root = (Path(args.output_dir).resolve() / "_frame_cache").resolve()
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
    if max_frame_count > 0:
        frame_paths = subset_frame_paths(frame_paths, max_frame_count, args.video_frame_sampling_strategy)
        frame_meta = dict(frame_meta)
        frame_meta["prepared_size_bytes"] = sum(path.stat().st_size for path in frame_paths)
        frame_meta["frame_count"] = len(frame_paths)
    content: List[Dict[str, Any]] = []
    frame_refs: List[Dict[str, Any]] = []
    total_bytes = 0
    for frame_path in frame_paths:
        data_url, mime_type, size = encode_file_to_data_url(frame_path)
        total_bytes += size
        content.append({"type": "image_url", "image_url": {"url": data_url}})
        frame_refs.append({"frame_path": str(frame_path), "mime_type": mime_type, "size_bytes": size})
    return (
        content,
        [
            {
                "kind": "video_frames",
                "transport": "runtime_frame_data_urls",
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


def build_media_content(spec: TaskSpec, round_data: Dict[str, Any], media_root: Path, args: argparse.Namespace) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    if spec.media_mode == "image":
        return build_image_content(round_data, media_root, args)
    return build_video_content(round_data, media_root, args)


def allocate_history_image_budget(
    history_round_count: int,
    available_images: int,
    per_round_cap: int,
) -> List[int]:
    if history_round_count <= 0 or available_images <= 0 or per_round_cap <= 0:
        return [0] * max(history_round_count, 0)

    quotas = [0] * history_round_count
    remaining = available_images

    for index in range(history_round_count - 1, -1, -1):
        if remaining <= 0:
            break
        quotas[index] = 1
        remaining -= 1

    while remaining > 0:
        progressed = False
        for index in range(history_round_count - 1, -1, -1):
            if quotas[index] >= per_round_cap:
                continue
            quotas[index] += 1
            remaining -= 1
            progressed = True
            if remaining <= 0:
                break
        if not progressed:
            break

    return quotas


def build_messages(
    spec: TaskSpec,
    doc: Dict[str, Any],
    round_index: int,
    media_root: Path,
    args: argparse.Namespace,
    previous_predictions: Sequence[str],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    rounds = doc.get("rounds", [])
    system_prompt = build_system_prompt(spec)
    messages: List[Dict[str, Any]] = [{"role": "system", "content": system_prompt}]
    blueprint: List[Dict[str, Any]] = [{"role": "system", "text": system_prompt}]

    history_round_count = min(len(previous_predictions), round_index)
    current_round = rounds[round_index]
    current_text = build_user_text(round_index, current_round, is_history=False, media_mode=spec.media_mode)
    if spec.media_mode == "video":
        current_media_content, current_media_refs = build_video_content(
            current_round,
            media_root,
            args,
            max_frame_count=max(0, int(args.video_max_images_per_request)),
        )
    else:
        current_media_content, current_media_refs = build_media_content(spec, current_round, media_root, args)

    current_image_count = len(current_media_content) if spec.media_mode == "video" else 0
    history_image_budgets = [0] * history_round_count
    if spec.media_mode == "video" and args.video_history_mode == "frames":
        max_total_images = max(int(args.video_max_images_per_request), current_image_count)
        available_history_images = max(0, max_total_images - current_image_count)
        history_image_budgets = allocate_history_image_budget(
            history_round_count=history_round_count,
            available_images=available_history_images,
            per_round_cap=max(0, int(args.video_history_max_frames_per_round)),
        )

    for history_index in range(history_round_count):
        history_round = rounds[history_index]
        user_text = build_user_text(history_index, history_round, is_history=True, media_mode=spec.media_mode)
        if spec.media_mode == "video" and args.video_history_mode == "text_only":
            media_content = []
            media_refs = [{"kind": "video_frames", "transport": "omitted_history_frames", "note": "历史视频帧为控制请求体大小未重复发送，仅保留历史文本与历史回答。"}]
        elif spec.media_mode == "video" and args.video_history_mode == "frames":
            max_history_frames = history_image_budgets[history_index] if history_index < len(history_image_budgets) else 0
            if max_history_frames <= 0:
                media_content = []
                media_refs = [{"kind": "video_frames", "transport": "omitted_history_frames", "note": "历史视频帧因单次请求图片预算限制被省略，仅保留历史文本与历史回答。", "max_frame_count": 0}]
            else:
                media_content, media_refs = build_video_content(
                    history_round,
                    media_root,
                    args,
                    max_frame_count=max_history_frames,
                )
        else:
            media_content, media_refs = build_media_content(spec, history_round, media_root, args)
        user_content = [{"type": "text", "text": user_text}] + media_content
        messages.append({"role": "user", "content": user_content})
        messages.append({"role": "assistant", "content": str(previous_predictions[history_index])})
        blueprint.append({"role": "user", "round_index": history_index, "is_history": True, "text": user_text, "media_refs": media_refs})
        blueprint.append({"role": "assistant", "round_index": history_index, "is_history": True, "text": str(previous_predictions[history_index])})

    current_content = [{"type": "text", "text": current_text}] + current_media_content
    messages.append({"role": "user", "content": current_content})
    blueprint.append({"role": "user", "round_index": round_index, "is_history": False, "text": current_text, "media_refs": current_media_refs})

    request_context = {
        "history_round_count": history_round_count,
        "current_round_index": round_index,
        "question_mode": spec.question_mode,
        "media_mode": spec.media_mode,
        "question_delivery_mode": "text",
        "image_input_mode": args.image_input_mode if spec.media_mode == "image" else "",
        "video_history_mode": args.video_history_mode if spec.media_mode == "video" else "",
        "video_frame_count": args.video_frame_count if spec.media_mode == "video" else 0,
        "video_frame_max_side": args.video_frame_max_side if spec.media_mode == "video" else 0,
        "video_frame_jpeg_quality": args.video_frame_jpeg_quality if spec.media_mode == "video" else 0,
        "video_frame_max_inline_bytes": args.video_frame_max_inline_bytes if spec.media_mode == "video" else 0,
        "video_frame_sampling_strategy": args.video_frame_sampling_strategy if spec.media_mode == "video" else "",
        "video_prepared_frame_mode": args.video_prepared_frame_mode if spec.media_mode == "video" else "",
        "video_max_images_per_request": args.video_max_images_per_request if spec.media_mode == "video" else 0,
        "video_history_max_frames_per_round": args.video_history_max_frames_per_round if spec.media_mode == "video" else 0,
        "current_image_count": current_image_count if spec.media_mode == "video" else 0,
        "history_image_budget_total": sum(history_image_budgets) if spec.media_mode == "video" else 0,
        "history_image_budgets": history_image_budgets if spec.media_mode == "video" else [],
    }
    if current_media_refs:
        request_context["current_media_ref"] = current_media_refs[0]
    if args.save_request_blueprint:
        request_context["message_blueprint"] = blueprint
    return messages, request_context


def build_runner(args: argparse.Namespace) -> OpenAICompatiblePhase1Runner:
    class GPTModelAwareRunner(OpenAICompatiblePhase1Runner):
        def generate_round(self, messages: List[Dict[str, Any]]) -> Tuple[str, Dict[str, Any], str, Dict[str, Any]]:
            is_gpt5 = is_gpt5_family_model(self.model_name)
            completion_budgets = [self.max_output_tokens]
            if is_gpt5:
                completion_budgets = []
                for budget in [max(self.max_output_tokens, 2048), 4096]:
                    if budget not in completion_budgets:
                        completion_budgets.append(budget)

            last_error = ""
            last_error_info: Dict[str, Any] = {"status_code": None, "error_type": "", "retriable": False, "retry_trace": []}

            def should_retry_for_reasoning_exhaustion(usage: Dict[str, Any], budget: int, prediction: str) -> bool:
                if prediction.strip():
                    return False
                if not is_gpt5 or budget <= 0:
                    return False
                completion_tokens = int(usage.get("completion_tokens") or 0)
                if completion_tokens < budget:
                    return False
                details = usage.get("completion_tokens_details")
                if not isinstance(details, dict):
                    return False
                reasoning_tokens = int(details.get("reasoning_tokens") or 0)
                accepted_prediction_tokens = int(details.get("accepted_prediction_tokens") or 0)
                return reasoning_tokens >= completion_tokens and accepted_prediction_tokens == 0

            for budget_index, completion_budget in enumerate(completion_budgets):
                payload: Dict[str, Any] = {
                    "model": self.model_name,
                    "messages": messages,
                    "stream": False,
                }
                if is_gpt5:
                    payload["max_completion_tokens"] = completion_budget
                else:
                    payload["temperature"] = self.temperature
                    payload["top_p"] = self.top_p
                    payload["max_tokens"] = self.max_output_tokens

                for attempt in range(self.max_retries):
                    try:
                        response = self.session.post(self.api_url, json=payload, timeout=self.timeout)
                    except requests.RequestException as exc:
                        last_error = f"{type(exc).__name__}: {exc}"
                        error_text = last_error.lower()
                        is_connection_error = "failed to establish a new connection" in error_text or "winerror 10013" in error_text
                        error_type = "connection_exception" if is_connection_error else "request_exception"
                        sleep_seconds = self.retry_sleep
                        if is_connection_error:
                            sleep_seconds = min(max(self.retry_sleep * (2 ** attempt), 10.0), self.rate_limit_max_sleep)
                        retry_trace = list(last_error_info.get("retry_trace", []))
                        retry_trace.append(
                            {
                                "attempt": attempt + 1,
                                "error_type": error_type,
                                "sleep_seconds": round(sleep_seconds, 2),
                                "message": last_error[-240:],
                                "completion_budget": completion_budget if is_gpt5 else self.max_output_tokens,
                            }
                        )
                        last_error_info = {
                            "status_code": None,
                            "error_type": error_type,
                            "retriable": attempt + 1 < self.max_retries or budget_index + 1 < len(completion_budgets),
                            "retry_trace": retry_trace,
                        }
                        if attempt + 1 < self.max_retries:
                            self.reset_session()
                            time.sleep(sleep_seconds)
                            continue
                        if budget_index + 1 < len(completion_budgets):
                            break
                        return "", {}, last_error, last_error_info

                    if response.status_code == 429:
                        sleep_seconds = min(self.rate_limit_retry_sleep * (2 ** attempt), self.rate_limit_max_sleep)
                        retry_trace = list(last_error_info.get("retry_trace", []))
                        retry_trace.append({"attempt": attempt + 1, "status_code": 429, "error_type": "rate_limit", "sleep_seconds": round(sleep_seconds, 2), "completion_budget": completion_budget if is_gpt5 else self.max_output_tokens})
                        last_error_info = {"status_code": 429, "error_type": "rate_limit", "retriable": attempt + 1 < self.max_retries or budget_index + 1 < len(completion_budgets), "retry_trace": retry_trace}
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
                        if budget_index + 1 < len(completion_budgets):
                            break
                        return "", {}, last_error, last_error_info

                    response_payload = response.json()
                    prediction = extract_text_response(response_payload)
                    usage = response_payload.get("usage", {}) if isinstance(response_payload.get("usage"), dict) else {}
                    if should_retry_for_reasoning_exhaustion(usage, completion_budget, prediction) and budget_index + 1 < len(completion_budgets):
                        last_error = f"GPT-5 completion budget exhausted at {completion_budget} tokens without visible answer."
                        retry_trace = list(last_error_info.get("retry_trace", []))
                        retry_trace.append(
                            {
                                "attempt": attempt + 1,
                                "error_type": "reasoning_exhaustion",
                                "sleep_seconds": 0.0,
                                "message": last_error,
                                "completion_budget": completion_budget,
                            }
                        )
                        last_error_info = {
                            "status_code": None,
                            "error_type": "reasoning_exhaustion",
                            "retriable": True,
                            "retry_trace": retry_trace,
                        }
                        break
                    return prediction, usage, "", {"status_code": None, "error_type": "", "retriable": False, "retry_trace": []}
            return "", {}, last_error, last_error_info

    return GPTModelAwareRunner(
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
    )


def generate_round_record(
    args: argparse.Namespace,
    spec: TaskSpec,
    runner: OpenAICompatiblePhase1Runner,
    media_root: Path,
    doc: Dict[str, Any],
    round_index: int,
    previous_predictions: Sequence[str],
) -> Tuple[str, Dict[str, Any]]:
    round_data = doc["rounds"][round_index]
    request_started = time.time()
    messages, request_context = build_messages(spec, doc, round_index, media_root, args, previous_predictions)
    prediction, usage, error, error_info = runner.generate_round(messages)
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
    }
    if error:
        round_record["error"] = error
    return prediction, round_record


def process_dialogue_work_item(
    args: argparse.Namespace,
    spec: TaskSpec,
    media_root: Path,
    work_item: Phase1DialogueWorkItem,
    shared_runner: Optional[OpenAICompatiblePhase1Runner] = None,
) -> Dict[str, Any]:
    owns_runner = shared_runner is None
    runner = shared_runner or build_runner(args)
    try:
        doc = work_item.doc
        dialogue_id = work_item.dialogue_id
        all_rounds = doc.get("rounds", [])
        resume_round_index = min(max(work_item.resume_round_index, 0), len(all_rounds))
        existing_payload = work_item.existing_payload if isinstance(work_item.existing_payload, dict) else None
        use_current_turn_only_repair = args.repair_failed and args.repair_mode == "current_turn_only" and existing_payload is not None and bool(work_item.failed_round_indices)

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
        payload = {
            "benchmark_name": "RUBRIC-MME",
            "phase": "phase1_generation",
            "provider": "gpt_openai_compatible_api",
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
            "rounds": final_round_records,
        }
        return {"dialogue_id": dialogue_id, "doc_index": work_item.doc_index, "payload": payload, "failed_rounds": failed_rounds}
    finally:
        if owns_runner:
            runner.close()


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
        active_runner: Optional[OpenAICompatiblePhase1Runner] = None
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
        "provider": "gpt_openai_compatible_api",
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
        "video_history_mode": args.video_history_mode if spec.media_mode == "video" else "",
        "video_frame_count": args.video_frame_count if spec.media_mode == "video" else 0,
        "video_frame_max_side": args.video_frame_max_side if spec.media_mode == "video" else 0,
        "video_frame_jpeg_quality": args.video_frame_jpeg_quality if spec.media_mode == "video" else 0,
        "video_frame_max_inline_bytes": args.video_frame_max_inline_bytes if spec.media_mode == "video" else 0,
        "video_frame_sampling_strategy": args.video_frame_sampling_strategy if spec.media_mode == "video" else "",
        "video_prepared_frame_mode": args.video_prepared_frame_mode if spec.media_mode == "video" else "",
        "repair_mode": args.repair_mode if args.repair_failed else "",
    }
    dump_json(summary_path, summary)
    return summary


def main() -> None:
    args = parse_args()
    task_names = resolve_task_names(args.tasks)
    summaries: List[Dict[str, Any]] = []
    for task_name in task_names:
        summaries.append(run_task(args, TASK_SPECS[task_name]))
    output_dir = Path(args.output_dir).resolve()
    run_summary = {
        "benchmark_name": "RUBRIC-MME",
        "phase": "phase1_generation",
        "provider": "gpt_openai_compatible_api",
        "model_name": args.model,
        "generated_at": utc_now(),
        "tasks": summaries,
    }
    dump_json(output_dir / f"{safe_name(args.model)}_run_summary.json", run_summary)


if __name__ == "__main__":
    main()
