from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional, Sequence, Tuple

import requests

from phase1_common import (
    DEFAULT_DATA_ROOT,
    DEFAULT_MEDIA_ROOT,
    Phase1DialogueWorkItem,
    REPO_ROOT,
    TASK_SPECS,
    TaskSpec,
    build_phase1_work_items,
    dump_json,
    is_phase1_round_success,
    load_docs_for_task,
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
    attach_video_precompressed_metadata,
    build_media_content,
    encode_file_to_data_url,
    extract_error_message,
    extract_text_response,
    resolve_local_path,
)


DEFAULT_MODEL = "ernie-5.0-thinking-preview"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "logs" / "rubric_mme_ernie_phase1"
PROVIDER_NAME = "ernie_openai_compatible_api"
SUPPORTED_TASKS = ["omnibench_image_multi_text"]

QUESTION_MODE_TEXT = "text"
QUESTION_MODE_TRANSCRIPT = "transcript_text_fallback"
QUESTION_MODE_AUDIO_INPUT = "audio_input_audio"
QUESTION_MODE_AUDIO_URL = "audio_audio_url"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行 ERNIE OpenAI 兼容接口版本 RUBRIC-MME Phase 1。当前稳定支持 image_multi_text。")
    parser.add_argument("--tasks", default="rubric-mme", help="逗号分隔的任务名；rubric-mme/omnibench/all 当前仅映射到 omnibench_image_multi_text。")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="被测试模型名称。")
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT if DEFAULT_DATA_ROOT.exists() else REPO_ROOT), help="RUBRIC-MME JSON 数据根目录。")
    parser.add_argument("--media-root", default=str(DEFAULT_MEDIA_ROOT if DEFAULT_MEDIA_ROOT.exists() else REPO_ROOT), help="图片、视频、音频等相对路径所对应的媒体根目录。")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="输出目录。")
    parser.add_argument("--limit", type=int, default=None, help="每个任务最多处理多少个 dialogue。")
    parser.add_argument("--dialogue-id", default=None, help="只运行指定的 dialogue_id。")
    parser.add_argument("--resume", action="store_true", help="基于已有 samples.jsonl 跳过已完成 dialogue。")
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
    parser.add_argument("--max-output-tokens", type=int, default=2048, help="单轮最大输出 token。对 ERNIE thinking 模型建议至少 1024，默认 2048。")
    parser.add_argument(
        "--tts-input-mode",
        choices=["auto", "audio_data_url", "audio_url", "text_fallback"],
        default="auto",
        help="TTS 任务处理方式。auto 先尝试音频直输，失败时回退到文本转写。",
    )
    parser.add_argument("--image-input-mode", choices=["local_data_url", "remote_url", "auto"], default="auto", help="图片输入策略。")
    parser.add_argument("--video-input-mode", choices=["local_data_url", "remote_url", "auto"], default="auto", help="视频输入策略。不会做抽帧。")
    parser.add_argument("--video-history-mode", choices=["text_only", "full"], default="text_only", help="视频多轮历史处理方式。")
    parser.add_argument("--video-precompressed-mode", choices=["prefer", "off"], default="prefer", help="是否优先使用数据集中预先压缩好的视频路径。")
    parser.add_argument("--video-precompressed-field", default=DEFAULT_VIDEO_PRECOMPRESSED_FIELD, help="视频数据集中记录预压缩视频相对路径的字段名。")
    parser.add_argument("--video-compress-mode", choices=["auto", "off"], default="auto", help="超大本地视频的处理方式。")
    parser.add_argument("--video-max-inline-bytes", type=int, default=DEFAULT_VIDEO_MAX_INLINE_BYTES, help="单个视频内联发送的目标原始体积上限（字节）。")
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
        raise ValueError(
            "ERNIE 当前在这条 internal chat/completions 路线上只稳定支持 "
            f"{SUPPORTED_TASKS}；不支持的任务: {unknown}"
        )
    return raw


def audio_format_from_path(file_path: Path) -> str:
    suffix = file_path.suffix.lower().lstrip(".")
    if suffix in {"mp3", "wav", "flac", "ogg", "aac", "m4a", "webm", "opus"}:
        return suffix
    mime_type, _ = mimetypes.guess_type(str(file_path))
    if mime_type and "/" in mime_type:
        return mime_type.split("/", 1)[1]
    return "mp3"


def requested_tts_mode_to_internal(tts_input_mode: str) -> str:
    if tts_input_mode == "audio_data_url":
        return QUESTION_MODE_AUDIO_INPUT
    if tts_input_mode == "audio_url":
        return QUESTION_MODE_AUDIO_URL
    if tts_input_mode == "text_fallback":
        return QUESTION_MODE_TRANSCRIPT
    return QUESTION_MODE_TEXT


def is_direct_audio_mode(mode: str) -> bool:
    return mode in {QUESTION_MODE_AUDIO_INPUT, QUESTION_MODE_AUDIO_URL}


def should_fallback_audio(error: str) -> bool:
    text = (error or "").lower()
    if not text:
        return False
    has_audio_hint = any(token in text for token in ["input_audio", "audio_url", "audio"])
    has_unsupported_hint = any(
        token in text
        for token in [
            "unsupported",
            "not supported",
            "not support",
            "unknown parameter",
            "unexpected field",
            "extra inputs are not permitted",
            "invalid",
            "schema",
            "content type",
            "message content",
        ]
    )
    return has_audio_hint and has_unsupported_hint


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
    usage = response_payload.get("usage", {})
    if not isinstance(usage, dict):
        return False
    completion_details = usage.get("completion_tokens_details", {})
    if not isinstance(completion_details, dict):
        return False
    reasoning_tokens = int(completion_details.get("reasoning_tokens", 0) or 0)
    return finish_reason == "length" and reasoning_tokens > 0 and max_output_tokens < 4096


def build_system_prompt(spec: TaskSpec, question_delivery_mode: str) -> str:
    media_desc = "完整视频片段" if spec.media_mode == "video" else "单张图像"
    if spec.question_mode == "tts":
        if question_delivery_mode == QUESTION_MODE_TRANSCRIPT:
            question_desc = "用户问题原本以语音给出；当前请求中使用语音转写文本代替音频输入"
        else:
            question_desc = "用户问题以语音给出，请结合语音与视觉信息理解问题"
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


def build_user_text(round_index: int, round_data: Dict[str, Any], question_delivery_mode: str, *, is_history: bool) -> str:
    prefix = f"这是第{round_index + 1}轮历史对话。" if is_history else f"这是当前第{round_index + 1}轮。"
    question_text = str(round_data.get("question_text", "") or "").strip()
    if question_delivery_mode in {QUESTION_MODE_TEXT, QUESTION_MODE_TRANSCRIPT}:
        if question_delivery_mode == QUESTION_MODE_TRANSCRIPT:
            return (
                f"{prefix}这一轮用户问题原本通过语音给出；当前接口改用转写文本提供问题。"
                f"\n用户问题转写：{question_text}\n请只围绕这一轮作答。"
            )
        return f"{prefix}请结合提供的视觉信息回答这一轮问题。\n用户问题：{question_text}\n请只围绕这一轮作答。"
    spoken_placeholder = round_data.get("spoken_question_placeholder") or "当前这一轮的问题以语音形式给出，请结合附带音频和视觉信息直接作答。"
    return f"{prefix}{spoken_placeholder}\n请只围绕这一轮作答。"


def build_audio_content(round_data: Dict[str, Any], media_root: Path, question_delivery_mode: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    if question_delivery_mode == QUESTION_MODE_TRANSCRIPT:
        return [], []

    remote_url = str(round_data.get("question_audio_remote_url", "") or "").strip()
    local_path = resolve_local_path(media_root, str(round_data.get("question_audio_local_path", "") or ""))

    if question_delivery_mode == QUESTION_MODE_AUDIO_INPUT:
        if local_path is None:
            raise FileNotFoundError(f"当前轮缺少可用于 input_audio 的本地音频文件: {round_data}")
        raw = local_path.read_bytes()
        audio_format = audio_format_from_path(local_path)
        return (
            [{"type": "input_audio", "input_audio": {"data": base64.b64encode(raw).decode("ascii"), "format": audio_format}}],
            [{"kind": "audio", "transport": "input_audio", "local_path": str(local_path), "format": audio_format, "size_bytes": len(raw)}],
        )

    if question_delivery_mode == QUESTION_MODE_AUDIO_URL:
        if local_path is not None:
            data_url, mime_type, size = encode_file_to_data_url(local_path)
            return (
                [{"type": "audio_url", "audio_url": {"url": data_url}}],
                [{"kind": "audio", "transport": "audio_url_data_url", "local_path": str(local_path), "mime_type": mime_type, "size_bytes": size}],
            )
        if remote_url:
            return (
                [{"type": "audio_url", "audio_url": {"url": remote_url}}],
                [{"kind": "audio", "transport": "audio_url_remote_url", "remote_url": remote_url}],
            )
        raise FileNotFoundError(f"当前轮缺少可用于 audio_url 的音频资源: {round_data}")

    return [], []


class ErnieCompatiblePhase1Runner(OpenAICompatiblePhase1Runner):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._tts_auto_lock = Lock()
        self._tts_resolved_mode: Optional[str] = None
        self._tts_unsupported_modes: set[str] = set()

    def get_tts_candidate_modes(self, requested_mode: str) -> List[str]:
        if requested_mode != "auto":
            return [requested_tts_mode_to_internal(requested_mode)]
        with self._tts_auto_lock:
            if self._tts_resolved_mode:
                return [self._tts_resolved_mode]
            ordered = [QUESTION_MODE_AUDIO_INPUT, QUESTION_MODE_AUDIO_URL, QUESTION_MODE_TRANSCRIPT]
            candidates = [mode for mode in ordered if mode not in self._tts_unsupported_modes]
            if QUESTION_MODE_TRANSCRIPT not in candidates:
                candidates.append(QUESTION_MODE_TRANSCRIPT)
            return candidates

    def note_tts_mode_success(self, question_delivery_mode: str) -> None:
        if not is_direct_audio_mode(question_delivery_mode):
            return
        with self._tts_auto_lock:
            self._tts_resolved_mode = question_delivery_mode

    def note_tts_mode_unsupported(self, question_delivery_mode: str) -> None:
        if not is_direct_audio_mode(question_delivery_mode):
            return
        with self._tts_auto_lock:
            self._tts_unsupported_modes.add(question_delivery_mode)
            if {QUESTION_MODE_AUDIO_INPUT, QUESTION_MODE_AUDIO_URL}.issubset(self._tts_unsupported_modes):
                self._tts_resolved_mode = QUESTION_MODE_TRANSCRIPT

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

            for profile_index, profile in enumerate(deduped_profiles):
                payload: Dict[str, Any] = {"model": self.model_name, "messages": messages, "stream": False, **profile}

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
                            }
                        )
                        last_error_info = {
                            "status_code": None,
                            "error_type": error_type,
                            "retriable": attempt + 1 < self.max_retries,
                            "retry_trace": retry_trace,
                        }
                        if attempt + 1 < self.max_retries:
                            time.sleep(sleep_seconds)
                            continue
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
                            }
                        )
                        last_error_info = {
                            "status_code": 429,
                            "error_type": "rate_limit",
                            "retriable": attempt + 1 < self.max_retries,
                            "retry_trace": retry_trace,
                        }
                        last_error = f"HTTP 429: {extract_error_message(response)}"
                        if attempt + 1 < self.max_retries:
                            time.sleep(sleep_seconds)
                            continue
                        return "", {}, last_error, last_error_info

                    if response.status_code >= 400:
                        error_text = extract_error_message(response)
                        error_text_lower = error_text.lower()
                        unsupported_parameter = any(
                            token in error_text_lower
                            for token in [
                                "unsupported parameter",
                                "unknown parameter",
                                "not supported",
                                "extra inputs are not permitted",
                                "unexpected field",
                            ]
                        )
                        retriable = response.status_code >= 500
                        last_error = f"HTTP {response.status_code}: {error_text}"
                        last_error_info = {
                            "status_code": response.status_code,
                            "error_type": "http_error",
                            "retriable": retriable or (unsupported_parameter and profile_index + 1 < len(deduped_profiles)),
                            "retry_trace": list(last_error_info.get("retry_trace", [])),
                        }
                        if retriable and attempt + 1 < self.max_retries:
                            time.sleep(self.retry_sleep)
                            continue
                        if unsupported_parameter and profile_index + 1 < len(deduped_profiles):
                            break
                        return "", {}, last_error, last_error_info

                    response_payload = response.json()
                    prediction = extract_text_response(response_payload)
                    usage = response_payload.get("usage", {}) if isinstance(response_payload.get("usage"), dict) else {}
                    if should_retry_for_reasoning_exhaustion(prediction, response_payload, completion_budget) and budget_index + 1 < len(completion_budgets):
                        break
                    return prediction, usage, "", {"status_code": None, "error_type": "", "retriable": False, "retry_trace": []}

        return "", {}, last_error, last_error_info


def build_runner(args: argparse.Namespace) -> ErnieCompatiblePhase1Runner:
    return ErnieCompatiblePhase1Runner(
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


def build_messages(
    runner: ErnieCompatiblePhase1Runner,
    args: argparse.Namespace,
    spec: TaskSpec,
    doc: Dict[str, Any],
    round_index: int,
    media_root: Path,
    previous_predictions: Sequence[str],
    question_delivery_mode: str,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    rounds = doc.get("rounds", [])
    system_prompt = build_system_prompt(spec, question_delivery_mode)
    messages: List[Dict[str, Any]] = [{"role": "system", "content": system_prompt}]
    blueprint: List[Dict[str, Any]] = [{"role": "system", "text": system_prompt}]

    for history_index in range(min(len(previous_predictions), round_index)):
        history_round = rounds[history_index]
        history_mode = question_delivery_mode
        include_history_visual = True
        include_history_audio = spec.question_mode == "tts" and is_direct_audio_mode(history_mode)

        if spec.media_mode == "video" and args.video_history_mode == "text_only":
            include_history_visual = False
            include_history_audio = False
            if spec.question_mode == "tts":
                history_mode = QUESTION_MODE_TRANSCRIPT

        user_text = build_user_text(history_index, history_round, history_mode, is_history=True)
        media_content: List[Dict[str, Any]] = []
        media_refs: List[Dict[str, Any]] = []
        if include_history_visual:
            visual_content, visual_refs = build_media_content(spec, history_round, media_root, args)
            media_content.extend(visual_content)
            media_refs.extend(visual_refs)
        else:
            media_refs.append({"kind": spec.media_mode, "transport": "omitted_history_media", "note": "为控制请求体大小未重复发送历史视觉输入，仅保留历史文本与历史回答。"})
        if include_history_audio:
            audio_content, audio_refs = build_audio_content(history_round, media_root, history_mode)
            media_content.extend(audio_content)
            media_refs.extend(audio_refs)
        elif spec.question_mode == "tts" and history_mode == QUESTION_MODE_TRANSCRIPT:
            media_refs.append({"kind": "audio", "transport": "transcript_text_fallback", "note": "历史语音问题改用文本转写提供。"})

        user_content = [{"type": "text", "text": user_text}] + media_content
        messages.append({"role": "user", "content": user_content})
        messages.append({"role": "assistant", "content": str(previous_predictions[history_index])})
        blueprint.append({"role": "user", "round_index": history_index, "is_history": True, "text": user_text, "media_refs": media_refs})
        blueprint.append({"role": "assistant", "round_index": history_index, "is_history": True, "text": str(previous_predictions[history_index])})

    current_round = rounds[round_index]
    current_text = build_user_text(round_index, current_round, question_delivery_mode, is_history=False)
    current_visual_content, current_visual_refs = build_media_content(spec, current_round, media_root, args)
    current_audio_content, current_audio_refs = build_audio_content(current_round, media_root, question_delivery_mode) if spec.question_mode == "tts" else ([], [])
    current_content = [{"type": "text", "text": current_text}] + current_visual_content + current_audio_content
    messages.append({"role": "user", "content": current_content})
    current_refs = list(current_visual_refs) + list(current_audio_refs)
    blueprint.append({"role": "user", "round_index": round_index, "is_history": False, "text": current_text, "media_refs": current_refs})

    request_context: Dict[str, Any] = {
        "history_round_count": min(len(previous_predictions), round_index),
        "current_round_index": round_index,
        "question_mode": spec.question_mode,
        "media_mode": spec.media_mode,
        "question_delivery_mode": question_delivery_mode,
        "tts_input_mode_requested": args.tts_input_mode if spec.question_mode == "tts" else "",
        "image_input_mode": args.image_input_mode if spec.media_mode == "image" else "",
        "video_input_mode": args.video_input_mode if spec.media_mode == "video" else "",
        "video_history_mode": args.video_history_mode if spec.media_mode == "video" else "",
        "video_precompressed_mode": args.video_precompressed_mode if spec.media_mode == "video" else "",
    }
    if current_visual_refs:
        request_context["current_media_ref"] = current_visual_refs[0]
    if current_audio_refs:
        request_context["current_audio_ref"] = current_audio_refs[0]
    if args.save_request_blueprint:
        request_context["message_blueprint"] = blueprint
    return messages, request_context


def generate_round_record(
    args: argparse.Namespace,
    spec: TaskSpec,
    runner: ErnieCompatiblePhase1Runner,
    media_root: Path,
    doc: Dict[str, Any],
    round_index: int,
    previous_predictions: Sequence[str],
) -> Tuple[str, Dict[str, Any]]:
    round_data = doc["rounds"][round_index]
    request_started = time.time()
    if spec.question_mode == "tts":
        candidate_modes = runner.get_tts_candidate_modes(args.tts_input_mode)
    else:
        candidate_modes = [QUESTION_MODE_TEXT]

    prediction = ""
    usage: Dict[str, Any] = {}
    error = ""
    error_info: Dict[str, Any] = {"status_code": None, "error_type": "", "retriable": False, "retry_trace": []}
    request_context: Dict[str, Any] = {}
    selected_mode = QUESTION_MODE_TEXT

    for candidate_mode in candidate_modes:
        selected_mode = candidate_mode
        messages, request_context = build_messages(runner, args, spec, doc, round_index, media_root, previous_predictions, candidate_mode)
        prediction, usage, error, error_info = runner.generate_round(messages)
        if error and spec.question_mode == "tts" and args.tts_input_mode == "auto" and is_direct_audio_mode(candidate_mode) and should_fallback_audio(error):
            runner.note_tts_mode_unsupported(candidate_mode)
            continue
        if not error and spec.question_mode == "tts" and args.tts_input_mode == "auto":
            runner.note_tts_mode_success(candidate_mode)
        break

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
        "tts_input_mode_effective": selected_mode if spec.question_mode == "tts" else "",
    }
    if error:
        round_record["error"] = error
    return prediction, round_record


def process_dialogue_work_item(
    args: argparse.Namespace,
    spec: TaskSpec,
    media_root: Path,
    work_item: Phase1DialogueWorkItem,
    shared_runner: Optional[ErnieCompatiblePhase1Runner] = None,
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
            round_records = []
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
            "rounds": final_round_records,
        }
        return {"dialogue_id": dialogue_id, "doc_index": work_item.doc_index, "payload": payload, "failed_rounds": failed_rounds}
    finally:
        if owns_runner:
            runner.close()


def summarize_effective_tts_modes(payloads: Sequence[Dict[str, Any]]) -> List[str]:
    modes: set[str] = set()
    for payload in payloads:
        for mode in payload.get("tts_input_modes_effective", []) or []:
            if isinstance(mode, str) and mode.strip():
                modes.add(mode.strip())
    return sorted(modes)


def run_task(args: argparse.Namespace, spec: TaskSpec) -> Dict[str, Any]:
    data_root = Path(args.data_root).resolve()
    media_root = Path(args.media_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    task_output_dir = output_dir / spec.name
    samples_path = task_output_dir / f"{safe_name(args.model)}_samples.jsonl"
    summary_path = task_output_dir / f"{safe_name(args.model)}_summary.json"
    docs = load_docs_for_task(spec, data_root, args.dialogue_id, args.limit)
    if spec.media_mode == "video" and args.video_precompressed_mode == "prefer":
        docs = attach_video_precompressed_metadata(docs, (data_root / spec.dataset_file).resolve(), args.video_precompressed_field)
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
        active_runner: Optional[ErnieCompatiblePhase1Runner] = None
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
        "tts_input_modes_effective": summarize_effective_tts_modes(final_payloads) if spec.question_mode == "tts" else [],
        "video_precompressed_mode": args.video_precompressed_mode if spec.media_mode == "video" else "",
        "video_precompressed_field": args.video_precompressed_field if spec.media_mode == "video" else "",
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
