from __future__ import annotations

import argparse
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
    merge_phase1_payloads,
    safe_name,
    utc_now,
    write_jsonl,
)
from run_gpt_openai_compatible_phase1 import (
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
    SUPPORTED_TASKS,
    build_messages,
    generate_round_record,
    load_docs_for_task,
    resolve_task_names,
)
from run_openai_compatible_phase1 import (
    DEFAULT_API_KEY_ENV,
    DEFAULT_API_URL,
    OpenAICompatiblePhase1Runner,
    extract_error_message,
    extract_text_response,
)


CLAUDE_PROVIDER = "claude_openai_compatible_api"
DEFAULT_MODEL = "claude-sonnet-4-6"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "logs" / "rubric_mme_claude_phase1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行 Claude 系列 OpenAI 兼容接口版本的 RUBRIC-MME Phase 1。")
    parser.add_argument("--tasks", default="rubric-mme", help="逗号分隔的任务名；rubric-mme/omnibench/all 在 Claude 脚本中会映射为 image_text 与 video_text。")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="被测试模型名称。")
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT if DEFAULT_DATA_ROOT.exists() else REPO_ROOT), help="RUBRIC-MME JSON 数据根目录。")
    parser.add_argument("--media-root", default=str(DEFAULT_MEDIA_ROOT if DEFAULT_MEDIA_ROOT.exists() else REPO_ROOT), help="图片与视频相对路径所对应的媒体根目录。")
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
    parser.add_argument("--max-output-tokens", type=int, default=1024, help="单轮最大输出 token。")
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
    parser.add_argument("--video-frame-jpeg-quality", type=int, default=DEFAULT_VIDEO_FRAME_JPEG_QUALITY, help="JPEG 质量参数。")
    parser.add_argument("--video-frame-max-inline-bytes", type=int, default=DEFAULT_VIDEO_FRAME_MAX_INLINE_BYTES, help="单轮所有帧图总大小预算。")
    parser.add_argument("--video-frame-sampling-strategy", choices=["uniform", "hybrid_tail"], default=DEFAULT_VIDEO_FRAME_SAMPLING_STRATEGY, help="视频抽帧采样策略。")
    parser.add_argument("--video-max-images-per-request", type=int, default=50, help="单次请求允许的最大图片数量。Claude 官方通常允许更高，但内部网关未知，因此默认保守设置为 50。")
    parser.add_argument("--video-history-max-frames-per-round", type=int, default=4, help="启用历史视频帧时，每个历史轮次最多保留多少帧。")
    parser.add_argument("--save-request-blueprint", action="store_true", help="将请求蓝图写入每轮记录，方便调试。")
    return parser.parse_args()


def build_runner(args: argparse.Namespace) -> OpenAICompatiblePhase1Runner:
    class ClaudeCompatibleRunner(OpenAICompatiblePhase1Runner):
        def generate_round(self, messages: List[Dict[str, Any]]) -> Tuple[str, Dict[str, Any], str, Dict[str, Any]]:
            parameter_profiles: List[Dict[str, Any]] = [
                {"temperature": self.temperature, "top_p": self.top_p, "max_tokens": self.max_output_tokens},
                {"temperature": self.temperature, "max_tokens": self.max_output_tokens},
                {"max_tokens": self.max_output_tokens},
                {"max_completion_tokens": self.max_output_tokens},
            ]
            deduped_profiles: List[Dict[str, Any]] = []
            seen_profiles: set[Tuple[Tuple[str, Any], ...]] = set()
            for profile in parameter_profiles:
                key = tuple(sorted(profile.items()))
                if key in seen_profiles:
                    continue
                seen_profiles.add(key)
                deduped_profiles.append(profile)

            last_error = ""
            last_error_info: Dict[str, Any] = {"status_code": None, "error_type": "", "retriable": False, "retry_trace": []}

            for profile_index, profile in enumerate(deduped_profiles):
                payload: Dict[str, Any] = {
                    "model": self.model_name,
                    "messages": messages,
                    "stream": False,
                    **profile,
                }

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
                                "deprecated",
                                "is deprecated",
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
                    return prediction, usage, "", {"status_code": None, "error_type": "", "retriable": False, "retry_trace": []}

            return "", {}, last_error, last_error_info

    return ClaudeCompatibleRunner(
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


def process_dialogue_work_item(
    args: argparse.Namespace,
    spec: TaskSpec,
    media_root: Path,
    work_item: Phase1DialogueWorkItem,
    shared_runner: Optional[OpenAICompatiblePhase1Runner] = None,
    persist_callback: Optional[Any] = None,
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
                if persist_callback is not None:
                    persist_callback(
                        {
                            "benchmark_name": "RUBRIC-MME",
                            "phase": "phase1_generation",
                            "provider": CLAUDE_PROVIDER,
                            "task_name": spec.name,
                            "task_alias": spec.task_alias,
                            "model_name": args.model,
                            "dialogue_id": dialogue_id,
                            "source_type": doc.get("source_type", ""),
                            "environment": doc.get("environment", ""),
                            "interaction_setup": doc.get("interaction_setup"),
                            "conversation_meta": doc.get("conversation_meta"),
                            "round_count": len(all_rounds),
                            "question_mode": spec.question_mode,
                            "media_mode": spec.media_mode,
                            "source_dataset_file": spec.dataset_file,
                            "source_data_root": str(Path(args.data_root).resolve()),
                            "generated_at": utc_now(),
                            "rounds": list(round_records),
                        }
                    )
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
                if persist_callback is not None:
                    persist_callback(
                        {
                            "benchmark_name": "RUBRIC-MME",
                            "phase": "phase1_generation",
                            "provider": CLAUDE_PROVIDER,
                            "task_name": spec.name,
                            "task_alias": spec.task_alias,
                            "model_name": args.model,
                            "dialogue_id": dialogue_id,
                            "source_type": doc.get("source_type", ""),
                            "environment": doc.get("environment", ""),
                            "interaction_setup": doc.get("interaction_setup"),
                            "conversation_meta": doc.get("conversation_meta"),
                            "round_count": len(all_rounds),
                            "question_mode": spec.question_mode,
                            "media_mode": spec.media_mode,
                            "source_dataset_file": spec.dataset_file,
                            "source_data_root": str(Path(args.data_root).resolve()),
                            "generated_at": utc_now(),
                            "rounds": list(round_records),
                        }
                    )
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
            "provider": CLAUDE_PROVIDER,
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
                executor.submit(process_dialogue_work_item, args, spec, media_root, work_item, None, persist_payload): work_item.doc_index
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
        "provider": CLAUDE_PROVIDER,
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
        "video_max_images_per_request": args.video_max_images_per_request if spec.media_mode == "video" else 0,
        "video_history_max_frames_per_round": args.video_history_max_frames_per_round if spec.media_mode == "video" else 0,
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
        "provider": CLAUDE_PROVIDER,
        "model_name": args.model,
        "generated_at": utc_now(),
        "tasks": summaries,
    }
    dump_json(output_dir / f"{safe_name(args.model)}_run_summary.json", run_summary)


if __name__ == "__main__":
    main()
