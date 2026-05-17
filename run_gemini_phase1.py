from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
import re
from threading import Lock
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:
    from PIL import Image
except Exception as exc:  # pragma: no cover - environment dependent
    Image = None
    PIL_IMPORT_ERROR = exc
else:
    PIL_IMPORT_ERROR = None

try:
    import google.generativeai as genai
    from google.generativeai.types import HarmBlockThreshold, HarmCategory

except Exception as exc:  # pragma: no cover - environment dependent
    genai = None
    HarmBlockThreshold = None
    HarmCategory = None
    IMPORT_ERROR = exc
else:
    IMPORT_ERROR = None

from judge_runner import OfficialGeminiJudgeBackend
from phase1_common import (
    AUDIO_EXTENSIONS,
    Phase1DialogueWorkItem,
    DEFAULT_DATA_ROOT,
    REPO_ROOT,
    TASK_SPECS,
    VIDEO_EXTENSIONS,
    TaskSpec,
    build_phase1_work_items,
    merge_phase1_payloads,
    append_jsonl,
    convert_doc_for_official_runner,
    dump_json,
    is_phase1_round_success,
    load_docs_for_task,
    read_completed_dialogue_ids,
    resolve_task_names,
    run_phase1_followups,
    safe_name,
    write_jsonl,
    utc_now,
)


DEFAULT_OUTPUT_DIR = REPO_ROOT / "logs" / "rubric_mme_gemini_phase1"
DEFAULT_SPOKEN_PROMPT = "请结合提供的视觉信息与问题音频，直接回答当前这一轮用户问题。"
DEFAULT_HISTORY_SPOKEN_PROMPT = "这是一轮历史语音问题，请结合附带的视觉信息、问题音频和随后给出的助手回答来理解上下文。"

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run RUBRIC-MME Phase 1 generation with Gemini API.")
    parser.add_argument(
        "--tasks",
        default="rubric-mme",
        help="Comma-separated RUBRIC-MME task names, or 'rubric-mme' / 'omnibench' / 'all' for all 4 tasks.",
    )
    parser.add_argument("--model", default="gemini-2.5-flash", help="Gemini model name.")
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT if DEFAULT_DATA_ROOT.exists() else REPO_ROOT), help="Root directory for local RUBRIC-MME JSON/media data.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Directory to store JSONL/JSON outputs.")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit per task.")
    parser.add_argument("--dialogue-id", default=None, help="Run only one dialogue id.")
    parser.add_argument("--resume", action="store_true", help="Resume from an existing samples JSONL file.")
    parser.add_argument("--repair-failed", action="store_true", help="Only repair failed or incomplete dialogues from an existing samples JSONL file.")
    parser.add_argument("--max-workers", type=int, default=1, help="Session-level parallel worker count; 1 keeps the original sequential behavior.")
    parser.add_argument("--repair-mode", choices=["resume_from_failure", "current_turn_only"], default="resume_from_failure", help="Repair strategy: resume_from_failure continues from the first failed round; current_turn_only only reruns failed rounds without history.")
    parser.add_argument("--max-retries", type=int, default=5, help="Retry count per round.")
    parser.add_argument("--retry-sleep", type=float, default=5.0, help="Seconds to sleep between retries.")
    parser.add_argument("--poll-interval", type=float, default=5.0, help="Seconds to poll uploaded Gemini files.")
    parser.add_argument("--timeout-seconds", type=float, default=600.0, help="Max wait time for uploaded Gemini files.")
    parser.add_argument(
        "--api-key-env",
        default="GOOGLE_API_KEY",
        help="Environment variable name containing the Gemini API key.",
    )
    parser.add_argument("--temperature", type=float, default=0.0, help="Generation temperature.")
    parser.add_argument("--max-output-tokens", type=int, default=512, help="Max output tokens per round.")
    parser.add_argument(
        "--save-raw-response",
        action="store_true",
        help="Include the raw Gemini text in the saved per-round payload.",
    )
    parser.add_argument("--phase2-output-dir", default="", help="Optional directory to also write standardized RUBRIC-MME Phase 2 outputs.")
    parser.add_argument("--phase2-keep-existing", action="store_true", help="Keep existing Phase 2 files instead of clearing them before writing.")
    parser.add_argument("--phase3-output-dir", default="", help="Optional directory to also write RUBRIC-MME Phase 3 judge outputs after Phase 2.")
    parser.add_argument("--phase3-judge-model", default="gemini-3.1-pro-preview", help="Gemini model name used for Phase 3 judging.")
    parser.add_argument("--phase3-judge-api-key-env", default="GOOGLE_API_KEY", help="Environment variable name containing the Phase 3 judge Gemini API key.")
    parser.add_argument("--phase3-keep-existing", action="store_true", help="Keep existing Phase 3 files instead of clearing them before writing.")
    parser.add_argument("--phase3-save-prompt-text", action="store_true", help="Also save full Phase 3 judge prompts into the output records.")
    parser.add_argument("--phase3-allow-incomplete-dialogues", action="store_true", help="Allow session-level judging even when some round predictions are missing.")
    return parser.parse_args()


def build_task_kwargs(spec: TaskSpec, data_root: Path) -> Dict[str, Any]:
    task_kwargs: Dict[str, Any] = {
        "data_root": str(data_root),
        "benchmark_name": "RUBRIC-MME",
        "media_mode": spec.media_mode,
        "question_mode": spec.question_mode,
        "include_history": True,
        "task_instruction": spec.task_instruction,
        "answer_instruction": spec.answer_instruction,
    }
    if spec.spoken_question_placeholder:
        task_kwargs["spoken_question_placeholder"] = spec.spoken_question_placeholder
    return task_kwargs


def _resolve_local_path(relative_path: Optional[str], task_kwargs: Dict[str, Any]) -> Optional[Path]:
    if not relative_path:
        return None
    return (Path(task_kwargs["data_root"]) / relative_path).resolve()


def _clean_text(text: Any) -> str:
    if text is None:
        return ""
    normalized = str(text).strip()
    normalized = THINK_PATTERN.sub("", normalized)
    normalized = WHITESPACE_PATTERN.sub(" ", normalized)
    return normalized.strip()


def _build_round_prompt(
    round_data: Dict[str, Any],
    task_kwargs: Dict[str, Any],
    *,
    include_task_instruction: bool,
    include_answer_instruction: bool,
) -> str:
    question_mode = task_kwargs.get("question_mode", "text")
    task_instruction = task_kwargs.get("task_instruction", "").strip()
    answer_instruction = task_kwargs.get("answer_instruction", "").strip()

    sections: List[str] = []
    if include_task_instruction and task_instruction:
        sections.append(task_instruction)

    if question_mode == "text":
        sections.append(f"当前用户问题：\n{_clean_text(round_data.get('question_text', ''))}")
    else:
        prompt_key = "spoken_question_placeholder" if include_task_instruction else "history_spoken_question_placeholder"
        fallback = DEFAULT_SPOKEN_PROMPT if include_task_instruction else DEFAULT_HISTORY_SPOKEN_PROMPT
        sections.append(task_kwargs.get(prompt_key, fallback).strip())

    if include_answer_instruction and answer_instruction:
        sections.append(answer_instruction)

    return "\n\n".join(section for section in sections if section).strip()


def _build_round_media(doc: Dict[str, Any], round_index: int, task_kwargs: Dict[str, Any]) -> List[Any]:
    rounds = doc.get("rounds", [])
    if round_index >= len(rounds):
        return []

    round_data = rounds[round_index]
    media_mode = task_kwargs.get("media_mode", "video")
    media_path = _resolve_local_path(round_data.get("media_rel_path"), task_kwargs)
    media_payload: List[Any] = []

    if media_mode == "image":
        if media_path is None:
            raise FileNotFoundError(f"Missing image path for RUBRIC-MME round: {round_data}")
        if Image is None:
            raise RuntimeError(f"Pillow is required for image tasks but is not available: {PIL_IMPORT_ERROR}")
        with Image.open(media_path) as image:
            media_payload.append(image.convert("RGB"))
    else:
        if media_path is None:
            raise FileNotFoundError(f"Missing video path for RUBRIC-MME round: {round_data}")
        media_payload.append(str(media_path))

    if task_kwargs.get("question_mode") == "tts":
        audio_path = _resolve_local_path(round_data.get("question_audio_rel_path"), task_kwargs)
        if audio_path is not None:
            media_payload.append({"type": "audio", "url": str(audio_path)})

    return media_payload


def _build_history_turns(doc: Dict[str, Any], previous_output: Optional[List[str]], task_kwargs: Dict[str, Any]) -> List[Dict[str, Any]]:
    outputs = list(previous_output or [])
    history_turns: List[Dict[str, Any]] = []
    for round_index, round_data in enumerate(doc.get("rounds", [])):
        if round_index >= len(outputs):
            break
        history_turns.append(
            {
                "round_index": round_index,
                "timestamp": round_data.get("timestamp", ""),
                "category": round_data.get("category", ""),
                "prompt": _build_round_prompt(
                    round_data,
                    task_kwargs,
                    include_task_instruction=False,
                    include_answer_instruction=False,
                ),
                "media": _build_round_media(doc, round_index, task_kwargs),
                "assistant_response": outputs[round_index],
            }
        )
    return history_turns


def iter_processed_docs(spec: TaskSpec, data_root: Path, dialogue_id: Optional[str], limit: Optional[int]) -> List[Dict[str, Any]]:
    shared_docs = load_docs_for_task(spec, data_root, dialogue_id, limit)
    return [convert_doc_for_official_runner(doc) for doc in shared_docs]


safe_model_name = safe_name


def build_runner(args: argparse.Namespace) -> "GeminiPhase1Runner":
    return GeminiPhase1Runner(
        model_name=args.model,
        api_key_env=args.api_key_env,
        max_retries=args.max_retries,
        retry_sleep=args.retry_sleep,
        poll_interval=args.poll_interval,
        timeout_seconds=args.timeout_seconds,
        temperature=args.temperature,
        max_output_tokens=args.max_output_tokens,
    )


def _generate_round_record(
    args: argparse.Namespace,
    spec: TaskSpec,
    task_kwargs: Dict[str, Any],
    runner: "GeminiPhase1Runner",
    doc: Dict[str, Any],
    round_index: int,
    previous_predictions: Sequence[str],
) -> Tuple[str, Dict[str, Any]]:
    round_data = doc["rounds"][round_index]
    request_started = time.time()
    contents, request_context = runner._build_contents(doc, round_index, task_kwargs, list(previous_predictions))
    prediction, usage, error = runner.generate_round(contents)
    latency_seconds = round(time.time() - request_started, 3)

    round_record = {
        "round_index": round_index,
        "timestamp": round_data.get("timestamp", ""),
        "category": round_data.get("category", ""),
        "question_text": round_data.get("question_text", ""),
        "question_audio_rel_path": round_data.get("question_audio_rel_path", ""),
        "question_audio_local_path": round_data.get("question_audio_rel_path", ""),
        "question_audio_remote_url": "",
        "question_audio_cloud_path": "",
        "media_rel_path": round_data.get("media_rel_path", ""),
        "media_local_path": round_data.get("media_rel_path", ""),
        "media_remote_url": "",
        "media_cloud_path": "",
        "reference_answer": round_data.get("answer_text", ""),
        "prediction": prediction,
        "usage": usage,
        "attempt_count": 0,
        "latency_seconds": latency_seconds,
        "request_context": request_context,
    }
    if args.save_raw_response:
        round_record["raw_response_text"] = prediction
    if error:
        round_record["error"] = error
    return prediction, round_record


class GeminiPhase1Runner:
    def __init__(
        self,
        *,
        model_name: str,
        api_key_env: str,
        max_retries: int,
        retry_sleep: float,
        poll_interval: float,
        timeout_seconds: float,
        temperature: float,
        max_output_tokens: int,
    ) -> None:
        if genai is None:
            raise RuntimeError(f"google.generativeai import failed: {IMPORT_ERROR}")
        api_key = os.getenv(api_key_env)
        if not api_key:
            raise RuntimeError(f"Missing Gemini API key. Set environment variable {api_key_env}.")

        genai.configure(api_key=api_key)
        self.model_name = model_name
        self.model = genai.GenerativeModel(model_name)
        self.max_retries = max_retries
        self.retry_sleep = retry_sleep
        self.poll_interval = poll_interval
        self.timeout_seconds = timeout_seconds
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.upload_cache: Dict[str, Any] = {}
        self.uploaded_files: List[Any] = []

    def close(self) -> None:
        for uploaded in self.uploaded_files:
            try:
                uploaded.delete()
            except Exception:
                pass
        self.uploaded_files.clear()
        self.upload_cache.clear()

    def _wait_until_ready(self, uploaded: Any) -> Any:
        deadline = time.time() + self.timeout_seconds
        current = uploaded
        while time.time() < deadline:
            state = getattr(getattr(current, "state", None), "name", "")
            state_normalized = str(state).lower()
            if state_normalized in {"", "active", "succeeded", "ready"}:
                return current
            if state_normalized in {"failed", "error"}:
                raise RuntimeError(f"Gemini file processing failed for {getattr(current, 'name', '<unknown>')}")
            time.sleep(self.poll_interval)
            current = genai.get_file(current.name)
        raise TimeoutError(f"Timed out waiting for Gemini file to become ready: {getattr(uploaded, 'name', '<unknown>')}")

    def _upload_file(self, file_path: str) -> Any:
        resolved = str(Path(file_path).resolve())
        cached = self.upload_cache.get(resolved)
        if cached is not None:
            return cached
        uploaded = genai.upload_file(path=resolved)
        uploaded = self._wait_until_ready(uploaded)
        self.upload_cache[resolved] = uploaded
        self.uploaded_files.append(uploaded)
        return uploaded

    def _convert_media_item(self, media_item: Any) -> List[Any]:
        if media_item is None:
            return []
        if Image is not None and isinstance(media_item, Image.Image):
            return [media_item.copy()]
        if isinstance(media_item, dict) and media_item.get("type") == "audio":
            return [self._upload_file(media_item["url"])]
        if isinstance(media_item, str):
            lower = media_item.lower()
            if lower.endswith(VIDEO_EXTENSIONS) or lower.endswith(AUDIO_EXTENSIONS):
                return [self._upload_file(media_item)]
            return [media_item]
        if isinstance(media_item, (list, tuple)):
            converted: List[Any] = []
            for nested in media_item:
                converted.extend(self._convert_media_item(nested))
            return converted
        raise TypeError(f"Unsupported RUBRIC-MME media item type: {type(media_item)}")

    def _flatten_media(self, media_items: Sequence[Any]) -> List[Any]:
        converted: List[Any] = []
        for media_item in media_items:
            converted.extend(self._convert_media_item(media_item))
        return converted

    def _build_contents(self, doc: Dict[str, Any], round_index: int, task_kwargs: Dict[str, Any], predictions: List[str]) -> Tuple[List[Any], Dict[str, Any]]:
        round_data = doc["rounds"][round_index]
        current_prompt = _build_round_prompt(
            round_data,
            task_kwargs,
            include_task_instruction=True,
            include_answer_instruction=True,
        )
        current_media = _build_round_media(doc, round_index, task_kwargs)
        history_turns = _build_history_turns(doc, predictions, task_kwargs)

        contents: List[Any] = [
            "你正在执行 RUBRIC-MME 第一阶段多轮作答。请基于历史对话与当前轮输入，只回答当前这一轮问题。",
        ]

        history_trace: List[Dict[str, Any]] = []
        for turn in history_turns:
            contents.append(f"历史第{turn['round_index'] + 1}轮：")
            contents.extend(self._flatten_media(turn["media"]))
            contents.append(turn["prompt"])
            contents.append(f"历史第{turn['round_index'] + 1}轮助手回答：{turn['assistant_response']}")
            history_trace.append(
                {
                    "round_index": turn["round_index"],
                    "timestamp": turn.get("timestamp", ""),
                    "category": turn.get("category", ""),
                    "prompt": turn.get("prompt", ""),
                    "assistant_response": turn.get("assistant_response", ""),
                    "media_rel_path": doc["rounds"][turn["round_index"]].get("media_rel_path", ""),
                    "question_audio_rel_path": doc["rounds"][turn["round_index"]].get("question_audio_rel_path", ""),
                }
            )

        contents.append(f"当前第{round_index + 1}轮：")
        contents.extend(self._flatten_media(current_media))
        contents.append(current_prompt)

        request_context = {
            "history_round_count": len(history_turns),
            "history_context": history_trace,
            "current_prompt": current_prompt,
            "question_mode": task_kwargs.get("question_mode", "text"),
            "media_mode": task_kwargs.get("media_mode", "video"),
        }
        return contents, request_context

    def _extract_usage(self, response: Any) -> Dict[str, int]:
        usage = getattr(response, "usage_metadata", None)
        if usage is None:
            return {}
        return {
            "prompt_token_count": int(getattr(usage, "prompt_token_count", 0) or 0),
            "candidates_token_count": int(getattr(usage, "candidates_token_count", 0) or 0),
            "total_token_count": int(getattr(usage, "total_token_count", 0) or 0),
        }

    def generate_round(self, contents: Sequence[Any]) -> Tuple[str, Dict[str, int], str]:
        config = genai.GenerationConfig(
            temperature=self.temperature,
            max_output_tokens=self.max_output_tokens,
        )
        last_error = ""
        for attempt in range(self.max_retries):
            try:
                response = self.model.generate_content(
                    list(contents),
                    generation_config=config,
                    safety_settings={
                        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
                        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                    },
                )
                text = getattr(response, "text", "") or ""
                return text.strip(), self._extract_usage(response), ""
            except Exception as exc:  # pragma: no cover - depends on remote API
                last_error = str(exc)
                if attempt + 1 < self.max_retries:
                    time.sleep(self.retry_sleep)
        return "", {}, last_error


def process_dialogue_work_item(
    args: argparse.Namespace,
    spec: TaskSpec,
    task_kwargs: Dict[str, Any],
    work_item: Phase1DialogueWorkItem,
    shared_runner: Optional[GeminiPhase1Runner] = None,
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
                _prediction, round_record = _generate_round_record(
                    args,
                    spec,
                    task_kwargs,
                    runner,
                    doc,
                    round_index,
                    [],
                )
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
                prediction, round_record = _generate_round_record(
                    args,
                    spec,
                    task_kwargs,
                    runner,
                    doc,
                    round_index,
                    predictions,
                )
                predictions.append(prediction)
                round_records.append(round_record)

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
            "provider": "gemini_official",
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
        return {
            "dialogue_id": dialogue_id,
            "doc_index": work_item.doc_index,
            "payload": payload,
            "failed_rounds": failed_rounds,
        }
    finally:
        if owns_runner:
            runner.close()


def run_task(args: argparse.Namespace, spec: TaskSpec, runner: Optional[GeminiPhase1Runner]) -> Dict[str, Any]:
    data_root = Path(args.data_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    task_output_dir = output_dir / spec.name
    samples_path = task_output_dir / f"{safe_model_name(args.model)}_samples.jsonl"
    summary_path = task_output_dir / f"{safe_model_name(args.model)}_summary.json"

    docs = iter_processed_docs(spec, data_root, args.dialogue_id, args.limit)
    work_items, samples_index, skipped = build_phase1_work_items(
        docs,
        samples_path,
        resume=args.resume,
        repair_failed=args.repair_failed,
        repair_mode=args.repair_mode,
    )
    task_kwargs = build_task_kwargs(spec, data_root)

    started_at = utc_now()
    attempted = len(work_items)
    completed = 0
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
        active_runner = runner or build_runner(args)
        owns_runner = runner is None
        try:
            for work_item in work_items:
                result = process_dialogue_work_item(args, spec, task_kwargs, work_item, active_runner)
                results_by_index[work_item.doc_index] = result
                persist_payload(result["payload"])
        finally:
            if owns_runner:
                active_runner.close()
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_index = {
                executor.submit(process_dialogue_work_item, args, spec, task_kwargs, work_item, None): work_item.doc_index
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
        failed_rounds += int(result.get("failed_rounds", 0) or 0)

    if args.repair_failed or args.resume:
        merged_payloads = merge_phase1_payloads(samples_index, updated_payloads_by_id, docs)
        write_jsonl(samples_path, merged_payloads)
    else:
        write_jsonl(samples_path, ordered_payloads)

    summary = {
        "benchmark_name": "RUBRIC-MME",
        "phase": "phase1_generation",
        "provider": "gemini_official",
        "task_name": spec.name,
        "task_alias": spec.task_alias,
        "model_name": args.model,
        "started_at": started_at,
        "completed_at": utc_now(),
        "data_root": str(data_root),
        "output_dir": str(task_output_dir),
        "attempted_dialogues": attempted,
        "completed_dialogues": completed,
        "skipped_dialogues": skipped,
        "failed_rounds": failed_rounds,
        "samples_path": str(samples_path),
        "repair_mode": args.repair_mode if args.repair_failed else "",
    }
    dump_json(summary_path, summary)
    return summary


def main() -> None:
    args = parse_args()
    task_names = resolve_task_names(args.tasks)
    shared_runner: Optional[GeminiPhase1Runner] = None
    summaries: List[Dict[str, Any]] = []
    try:
        if max(1, int(args.max_workers)) == 1:
            shared_runner = build_runner(args)
        for task_name in task_names:
            summaries.append(run_task(args, TASK_SPECS[task_name], shared_runner))
    finally:
        if shared_runner is not None:
            shared_runner.close()

    combined_summary = {
        "benchmark_name": "RUBRIC-MME",
        "phase": "phase1_generation",
        "provider": "gemini_official",
        "model_name": args.model,
        "tasks": summaries,
        "generated_at": utc_now(),
    }
    combined_summary = run_phase1_followups(
        summaries=summaries,
        run_summary=combined_summary,
        phase2_output_dir=args.phase2_output_dir,
        phase2_keep_existing=args.phase2_keep_existing,
        phase3_output_dir=args.phase3_output_dir,
        phase3_keep_existing=args.phase3_keep_existing,
        dialogue_id=args.dialogue_id or "",
        limit=args.limit,
        phase3_save_prompt_text=args.phase3_save_prompt_text,
        phase3_allow_incomplete_dialogues=args.phase3_allow_incomplete_dialogues,
        build_phase3_backend=lambda: OfficialGeminiJudgeBackend(
            api_key_env=args.phase3_judge_api_key_env,
            use_response_schema=True,
            model_name=args.phase3_judge_model,
            max_retries=args.max_retries,
            retry_sleep=args.retry_sleep,
            temperature=args.temperature,
            max_output_tokens=max(args.max_output_tokens, 2048),
        ),
    )
    output_dir = Path(args.output_dir).resolve()
    dump_json(output_dir / f"{safe_model_name(args.model)}_run_summary.json", combined_summary)


if __name__ == "__main__":
    main()

