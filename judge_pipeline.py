from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from aggregation import (
    build_category_summary,
    build_session_summary,
    build_task_summary,
    build_turn_summary,
    build_validation_summary,
)
from judge_media import build_session_judge_request, build_turn_judge_request
from judge_parsing import (
    PHASE3_SCHEMA_VERSION,
    SESSION_METRICS,
    TURN_METRICS,
    build_session_response_schema,
    build_turn_response_schema,
    parse_session_judgement_text,
    parse_turn_judgement_text,
)
from judge_prompts import SESSION_PROMPT_VERSION, TURN_PROMPT_VERSION, build_session_prompt, build_turn_prompt

OUTPUT_FILENAMES = {
    "turn_judgements": "turn_judgements.jsonl",
    "session_judgements": "session_judgements.jsonl",
    "judge_errors": "judge_errors.jsonl",
    "turn_summary": "turn_summary.json",
    "session_summary": "session_summary.json",
    "task_summary": "task_summary.json",
    "category_summary": "category_summary.json",
    "validation_summary": "validation_summary.json",
    "manifest": "manifest.json",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for raw_line in file:
            line = raw_line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if isinstance(payload, dict):
                records.append(payload)
    return records


def append_jsonl(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(payload, ensure_ascii=False) + "\n")


def write_jsonl(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for payload in rows:
            file.write(json.dumps(payload, ensure_ascii=False) + "\n")


def dump_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)


def resolve_phase2_paths(input_dir: Path) -> Tuple[Path, Path]:
    resolved = input_dir.expanduser().resolve()
    rounds_path = resolved / "rounds.jsonl"
    dialogues_path = resolved / "dialogues.jsonl"
    if not rounds_path.exists():
        raise FileNotFoundError(f"未找到 Phase 2 的 rounds.jsonl: {rounds_path}")
    if not dialogues_path.exists():
        raise FileNotFoundError(f"未找到 Phase 2 的 dialogues.jsonl: {dialogues_path}")
    return rounds_path, dialogues_path


def dialogue_scope_key(record: Dict[str, Any]) -> str:
    return "::".join(
        [
            str(record.get("provider", "") or ""),
            str(record.get("model_name", "") or ""),
            str(record.get("task_name", "") or ""),
            str(record.get("dialogue_id", "") or ""),
        ]
    )


def turn_judgement_key(record: Dict[str, Any]) -> str:
    return f"{dialogue_scope_key(record)}::{str(record.get('round_id', '') or '')}"


def session_judgement_key(record: Dict[str, Any]) -> str:
    return dialogue_scope_key(record)


def latest_records_by_key(records: Sequence[Dict[str, Any]], key_name: str) -> List[Dict[str, Any]]:
    latest: Dict[str, Dict[str, Any]] = {}
    ordered_without_key: List[Dict[str, Any]] = []
    for record in records:
        key = str(record.get(key_name, "") or "")
        if not key:
            ordered_without_key.append(record)
            continue
        latest[key] = record
    return list(latest.values()) + ordered_without_key


def load_existing_success_ids(path: Path, key_name: str) -> set[str]:
    if not path.exists():
        return set()
    latest_records = latest_records_by_key(read_jsonl(path), key_name)
    return {
        str(record.get(key_name, "") or "")
        for record in latest_records
        if str(record.get(key_name, "") or "") and record.get("status") == "success"
    }


def load_latest_records_map(path: Path, key_name: str) -> Dict[str, Dict[str, Any]]:
    if not path.exists():
        return {}
    latest_records = latest_records_by_key(read_jsonl(path), key_name)
    result: Dict[str, Dict[str, Any]] = {}
    for record in latest_records:
        key = str(record.get(key_name, "") or "")
        if key:
            result[key] = record
    return result


@dataclass(frozen=True)
class Phase3DialogueWorkItem:
    dialogue_record: Dict[str, Any]
    dialogue_rounds: List[Dict[str, Any]]
    pending_turns: List[Dict[str, Any]]
    needs_session: bool


def close_backend(backend: Any) -> None:
    close = getattr(backend, "close", None)
    if callable(close):
        close()


def load_all_judgement_records(output_paths: Dict[str, Path]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    raw_turn_records = read_jsonl(output_paths["turn_judgements"]) if output_paths["turn_judgements"].exists() else []
    raw_session_records = read_jsonl(output_paths["session_judgements"]) if output_paths["session_judgements"].exists() else []
    turn_records = latest_records_by_key(raw_turn_records, "judgement_id")
    session_records = latest_records_by_key(raw_session_records, "judgement_id")
    error_records = [
        record
        for record in list(turn_records) + list(session_records)
        if record.get("status") != "success"
    ]
    return turn_records, session_records, error_records


def _sorted_turn_records(records: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return sorted(
        records,
        key=lambda record: (
            str(record.get("task_name", "") or ""),
            str(record.get("dialogue_id", "") or ""),
            int(record.get("round_index", 0) or 0),
            str(record.get("judgement_id", "") or ""),
        ),
    )


def _sorted_session_records(records: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return sorted(
        records,
        key=lambda record: (
            str(record.get("task_name", "") or ""),
            str(record.get("dialogue_id", "") or ""),
            str(record.get("judgement_id", "") or ""),
        ),
    )


def _sorted_error_records(records: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return sorted(
        records,
        key=lambda record: (
            str(record.get("record_type", "") or ""),
            str(record.get("task_name", "") or ""),
            str(record.get("dialogue_id", "") or ""),
            int(record.get("round_index", -1) or -1),
            str(record.get("judgement_id", "") or ""),
        ),
    )


def canonicalize_phase3_output_files(output_paths: Dict[str, Path]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    turn_records, session_records, error_records = load_all_judgement_records(output_paths)
    turn_records = _sorted_turn_records(turn_records)
    session_records = _sorted_session_records(session_records)
    error_records = _sorted_error_records(error_records)
    write_jsonl(output_paths["turn_judgements"], turn_records)
    write_jsonl(output_paths["session_judgements"], session_records)
    write_jsonl(output_paths["judge_errors"], error_records)
    return turn_records, session_records, error_records

def build_round_lookup(rounds: Sequence[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for record in rounds:
        grouped.setdefault(dialogue_scope_key(record), []).append(record)
    for records in grouped.values():
        records.sort(key=lambda item: int(item.get("round_index", 0) or 0))
    return grouped


def filter_records(
    rounds: Sequence[Dict[str, Any]],
    dialogues: Sequence[Dict[str, Any]],
    *,
    dialogue_id: str = "",
    limit_dialogues: Optional[int] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    filtered_dialogues = list(dialogues)
    if dialogue_id:
        filtered_dialogues = [record for record in filtered_dialogues if str(record.get("dialogue_id", "") or "") == dialogue_id]
    if limit_dialogues is not None:
        filtered_dialogues = filtered_dialogues[:limit_dialogues]
    allowed_scope_keys = {dialogue_scope_key(record) for record in filtered_dialogues}
    filtered_rounds = [record for record in rounds if dialogue_scope_key(record) in allowed_scope_keys]
    return filtered_rounds, filtered_dialogues


def make_empty_score_vector(metric_names: Sequence[str]) -> Dict[str, Optional[int]]:
    return {metric_name: None for metric_name in metric_names}


def make_empty_reason_vector(metric_names: Sequence[str]) -> Dict[str, str]:
    return {metric_name: "" for metric_name in metric_names}


def build_turn_skip_record(round_record: Dict[str, Any], *, status: str, reason: str) -> Dict[str, Any]:
    return {
        "schema_version": PHASE3_SCHEMA_VERSION,
        "record_type": "turn_judgement",
        "score_scope": "turn_core",
        "dialogue_scope_key": dialogue_scope_key(round_record),
        "judgement_id": turn_judgement_key(round_record),
        "status": status,
        "judge_backend": "",
        "judge_model_name": "",
        "judge_prompt_version": TURN_PROMPT_VERSION,
        "judge_input_mode": "text_plus_visual",
        "judge_media_refs": [],
        "judge_attempt_count": 0,
        "judge_usage": {},
        "schema_mode": "",
        "judged_at": utc_now(),
        "tested_provider": round_record.get("provider", ""),
        "tested_model_name": round_record.get("model_name", ""),
        "task_name": round_record.get("task_name", ""),
        "task_alias": round_record.get("task_alias", ""),
        "task_mode": round_record.get("task_mode", ""),
        "question_mode": round_record.get("question_mode", ""),
        "media_mode": round_record.get("media_mode", ""),
        "dialogue_id": round_record.get("dialogue_id", ""),
        "round_id": round_record.get("round_id", ""),
        "round_index": round_record.get("round_index", 0),
        "primary_category": round_record.get("primary_category", ""),
        "secondary_categories": round_record.get("secondary_categories", []),
        "question_text": round_record.get("question_text", ""),
        "reference_answer": round_record.get("reference_answer", ""),
        "prediction": round_record.get("prediction", ""),
        "raw_judge_text": "",
        "parse_success": False,
        "parse_error": reason,
        "parse_issues": [reason],
        "scores": {metric_name: {"score": None, "reason": ""} for metric_name in TURN_METRICS},
        "score_vector": make_empty_score_vector(TURN_METRICS),
        "reason_vector": make_empty_reason_vector(TURN_METRICS),
        "avg_score": None,
        "overall_summary": "",
        "noted_user_state_cues": [],
        "error": reason,
        "error_type": status,
        "status_code": None,
    }


def build_session_skip_record(dialogue_record: Dict[str, Any], *, status: str, reason: str) -> Dict[str, Any]:
    return {
        "schema_version": PHASE3_SCHEMA_VERSION,
        "record_type": "session_judgement",
        "score_scope": "session_core",
        "dialogue_scope_key": dialogue_scope_key(dialogue_record),
        "judgement_id": session_judgement_key(dialogue_record),
        "status": status,
        "judge_backend": "",
        "judge_model_name": "",
        "judge_prompt_version": SESSION_PROMPT_VERSION,
        "judge_input_mode": "text_plus_visual",
        "judge_media_refs": [],
        "judge_attempt_count": 0,
        "judge_usage": {},
        "schema_mode": "",
        "judged_at": utc_now(),
        "tested_provider": dialogue_record.get("provider", ""),
        "tested_model_name": dialogue_record.get("model_name", ""),
        "task_name": dialogue_record.get("task_name", ""),
        "task_alias": dialogue_record.get("task_alias", ""),
        "task_mode": dialogue_record.get("task_mode", ""),
        "question_mode": dialogue_record.get("question_mode", ""),
        "media_mode": dialogue_record.get("media_mode", ""),
        "dialogue_id": dialogue_record.get("dialogue_id", ""),
        "round_count": dialogue_record.get("round_count", 0),
        "raw_judge_text": "",
        "parse_success": False,
        "parse_error": reason,
        "parse_issues": [reason],
        "scores": {metric_name: {"score": None, "reason": ""} for metric_name in SESSION_METRICS},
        "score_vector": make_empty_score_vector(SESSION_METRICS),
        "reason_vector": make_empty_reason_vector(SESSION_METRICS),
        "avg_score": None,
        "overall_summary": "",
        "key_dialogue_signals": [],
        "error": reason,
        "error_type": status,
        "status_code": None,
    }


def build_turn_success_record(
    round_record: Dict[str, Any],
    judge_result: Any,
    parsed: Dict[str, Any],
    *,
    save_prompt_text: bool,
    prompt_text: str,
    judge_input_mode: str,
    judge_media_refs: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    payload = {
        "schema_version": PHASE3_SCHEMA_VERSION,
        "record_type": "turn_judgement",
        "score_scope": "turn_core",
        "dialogue_scope_key": dialogue_scope_key(round_record),
        "judgement_id": turn_judgement_key(round_record),
        "status": "success" if parsed.get("parse_success") else "parse_error",
        "judge_backend": judge_result.backend_name,
        "judge_model_name": judge_result.model_name,
        "judge_prompt_version": TURN_PROMPT_VERSION,
        "judge_input_mode": judge_input_mode,
        "judge_media_refs": list(judge_media_refs),
        "judge_attempt_count": judge_result.attempt_count,
        "judge_usage": judge_result.usage,
        "schema_mode": judge_result.schema_mode,
        "judged_at": utc_now(),
        "tested_provider": round_record.get("provider", ""),
        "tested_model_name": round_record.get("model_name", ""),
        "task_name": round_record.get("task_name", ""),
        "task_alias": round_record.get("task_alias", ""),
        "task_mode": round_record.get("task_mode", ""),
        "question_mode": round_record.get("question_mode", ""),
        "media_mode": round_record.get("media_mode", ""),
        "dialogue_id": round_record.get("dialogue_id", ""),
        "round_id": round_record.get("round_id", ""),
        "round_index": round_record.get("round_index", 0),
        "primary_category": round_record.get("primary_category", ""),
        "secondary_categories": round_record.get("secondary_categories", []),
        "question_text": round_record.get("question_text", ""),
        "reference_answer": round_record.get("reference_answer", ""),
        "prediction": round_record.get("prediction", ""),
        "raw_judge_text": judge_result.raw_text,
        "parse_success": parsed.get("parse_success", False),
        "parse_error": parsed.get("parse_error", ""),
        "parse_issues": parsed.get("parse_issues", []),
        "scores": parsed.get("scores", {}),
        "score_vector": parsed.get("score_vector", {}),
        "reason_vector": parsed.get("reason_vector", {}),
        "avg_score": parsed.get("avg_score"),
        "overall_summary": parsed.get("overall_summary", ""),
        "noted_user_state_cues": parsed.get("noted_user_state_cues", []),
        "error": judge_result.error,
        "error_type": judge_result.error_type,
        "status_code": judge_result.status_code,
    }
    if save_prompt_text:
        payload["judge_prompt_text"] = prompt_text
    return payload


def build_session_success_record(
    dialogue_record: Dict[str, Any],
    judge_result: Any,
    parsed: Dict[str, Any],
    *,
    save_prompt_text: bool,
    prompt_text: str,
    judge_input_mode: str,
    judge_media_refs: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    payload = {
        "schema_version": PHASE3_SCHEMA_VERSION,
        "record_type": "session_judgement",
        "score_scope": "session_core",
        "dialogue_scope_key": dialogue_scope_key(dialogue_record),
        "judgement_id": session_judgement_key(dialogue_record),
        "status": "success" if parsed.get("parse_success") else "parse_error",
        "judge_backend": judge_result.backend_name,
        "judge_model_name": judge_result.model_name,
        "judge_prompt_version": SESSION_PROMPT_VERSION,
        "judge_input_mode": judge_input_mode,
        "judge_media_refs": list(judge_media_refs),
        "judge_attempt_count": judge_result.attempt_count,
        "judge_usage": judge_result.usage,
        "schema_mode": judge_result.schema_mode,
        "judged_at": utc_now(),
        "tested_provider": dialogue_record.get("provider", ""),
        "tested_model_name": dialogue_record.get("model_name", ""),
        "task_name": dialogue_record.get("task_name", ""),
        "task_alias": dialogue_record.get("task_alias", ""),
        "task_mode": dialogue_record.get("task_mode", ""),
        "question_mode": dialogue_record.get("question_mode", ""),
        "media_mode": dialogue_record.get("media_mode", ""),
        "dialogue_id": dialogue_record.get("dialogue_id", ""),
        "round_count": dialogue_record.get("round_count", 0),
        "raw_judge_text": judge_result.raw_text,
        "parse_success": parsed.get("parse_success", False),
        "parse_error": parsed.get("parse_error", ""),
        "parse_issues": parsed.get("parse_issues", []),
        "scores": parsed.get("scores", {}),
        "score_vector": parsed.get("score_vector", {}),
        "reason_vector": parsed.get("reason_vector", {}),
        "avg_score": parsed.get("avg_score"),
        "overall_summary": parsed.get("overall_summary", ""),
        "key_dialogue_signals": parsed.get("key_dialogue_signals", []),
        "error": judge_result.error,
        "error_type": judge_result.error_type,
        "status_code": judge_result.status_code,
    }
    if save_prompt_text:
        payload["judge_prompt_text"] = prompt_text
    return payload


def _sleep_between_requests(seconds: float) -> None:
    if seconds > 0:
        time.sleep(seconds)


def build_phase3_work_items(
    rounds: Sequence[Dict[str, Any]],
    dialogues: Sequence[Dict[str, Any]],
    output_paths: Dict[str, Path],
    *,
    resume: bool,
    repair_failed: bool,
) -> Tuple[List[Phase3DialogueWorkItem], int]:
    round_lookup = build_round_lookup(rounds)
    use_existing = resume or repair_failed
    existing_turn_map = load_latest_records_map(output_paths["turn_judgements"], "judgement_id") if use_existing else {}
    existing_session_map = load_latest_records_map(output_paths["session_judgements"], "judgement_id") if use_existing else {}

    work_items: List[Phase3DialogueWorkItem] = []
    skipped_dialogues = 0

    for dialogue_record in dialogues:
        scope_key = dialogue_scope_key(dialogue_record)
        dialogue_rounds = list(round_lookup.get(scope_key, []))
        session_key = session_judgement_key(dialogue_record)
        has_existing_records = session_key in existing_session_map or any(
            turn_judgement_key(round_record) in existing_turn_map
            for round_record in dialogue_rounds
        )

        if repair_failed and not has_existing_records:
            skipped_dialogues += 1
            continue

        if resume or repair_failed:
            pending_turns = [
                round_record
                for round_record in dialogue_rounds
                if (existing_turn_map.get(turn_judgement_key(round_record)) or {}).get("status") != "success"
            ]
            needs_session = (existing_session_map.get(session_key) or {}).get("status") != "success"
        else:
            pending_turns = dialogue_rounds
            needs_session = True

        if not pending_turns and not needs_session:
            skipped_dialogues += 1
            continue

        work_items.append(
            Phase3DialogueWorkItem(
                dialogue_record=dialogue_record,
                dialogue_rounds=dialogue_rounds,
                pending_turns=pending_turns,
                needs_session=needs_session,
            )
        )

    return work_items, skipped_dialogues


def process_phase3_dialogue_work_item(
    work_item: Phase3DialogueWorkItem,
    backend: Any,
    *,
    save_prompt_text: bool,
    allow_incomplete_dialogues: bool,
    inter_request_sleep: float,
    turn_schema: Dict[str, Any],
    session_schema: Dict[str, Any],
    use_repair_modes: bool,
    turn_repair_mode: str,
    session_repair_mode: str,
    on_turn_record: Optional[Callable[[Dict[str, Any]], None]] = None,
    on_session_record: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Dict[str, Any]:
    turn_results: List[Dict[str, Any]] = []
    session_result: Optional[Dict[str, Any]] = None

    for round_record in work_item.pending_turns:
        issued_request = False
        if round_record.get("has_error"):
            record = build_turn_skip_record(round_record, status="skipped_input_error", reason="phase1_or_phase2_input_error")
        elif not str(round_record.get("prediction", "") or "").strip():
            record = build_turn_skip_record(round_record, status="skipped_empty_prediction", reason="empty_prediction")
        else:
            history_rounds = [
                item
                for item in work_item.dialogue_rounds
                if int(item.get("round_index", 0) or 0) < int(round_record.get("round_index", 0) or 0)
            ]
            effective_history_rounds = [] if (use_repair_modes and turn_repair_mode == "current_turn_only") else history_rounds
            prompt_text = build_turn_prompt(round_record, effective_history_rounds)
            judge_request = build_turn_judge_request(prompt_text, round_record, effective_history_rounds)
            judge_result = backend.judge(judge_request, turn_schema)
            issued_request = True
            if judge_result.error:
                record = build_turn_skip_record(round_record, status="api_error", reason=judge_result.error)
                record["judge_backend"] = judge_result.backend_name
                record["judge_model_name"] = judge_result.model_name
                record["judge_input_mode"] = judge_request.input_mode
                record["judge_media_refs"] = judge_request.media_refs
                record["judge_attempt_count"] = judge_result.attempt_count
                record["schema_mode"] = judge_result.schema_mode
                record["status_code"] = judge_result.status_code
                record["error_type"] = judge_result.error_type
                record["error"] = judge_result.error
                record["raw_judge_text"] = judge_result.raw_text
                if save_prompt_text:
                    record["judge_prompt_text"] = prompt_text
            else:
                parsed = parse_turn_judgement_text(judge_result.raw_text)
                record = build_turn_success_record(
                    round_record,
                    judge_result,
                    parsed,
                    save_prompt_text=save_prompt_text,
                    prompt_text=prompt_text,
                    judge_input_mode=judge_request.input_mode,
                    judge_media_refs=judge_request.media_refs,
                )
        turn_results.append(record)
        if on_turn_record is not None:
            on_turn_record(record)
        if issued_request:
            _sleep_between_requests(inter_request_sleep)

    if work_item.needs_session:
        dialogue_record = work_item.dialogue_record
        issued_request = False
        incomplete_rounds = [
            item
            for item in work_item.dialogue_rounds
            if item.get("has_error") or not str(item.get("prediction", "") or "").strip()
        ]
        if incomplete_rounds and not allow_incomplete_dialogues:
            session_result = build_session_skip_record(dialogue_record, status="skipped_incomplete_dialogue", reason="incomplete_dialogue_predictions")
        else:
            prompt_text = build_session_prompt(dialogue_record)
            session_visual_mode = "light_context" if (use_repair_modes and session_repair_mode == "light_context") else "full_context"
            judge_request = build_session_judge_request(prompt_text, dialogue_record, visual_mode=session_visual_mode)
            judge_result = backend.judge(judge_request, session_schema)
            issued_request = True
            if judge_result.error:
                session_result = build_session_skip_record(dialogue_record, status="api_error", reason=judge_result.error)
                session_result["judge_backend"] = judge_result.backend_name
                session_result["judge_model_name"] = judge_result.model_name
                session_result["judge_input_mode"] = judge_request.input_mode
                session_result["judge_media_refs"] = judge_request.media_refs
                session_result["judge_attempt_count"] = judge_result.attempt_count
                session_result["schema_mode"] = judge_result.schema_mode
                session_result["status_code"] = judge_result.status_code
                session_result["error_type"] = judge_result.error_type
                session_result["error"] = judge_result.error
                session_result["raw_judge_text"] = judge_result.raw_text
                if save_prompt_text:
                    session_result["judge_prompt_text"] = prompt_text
            else:
                parsed = parse_session_judgement_text(judge_result.raw_text)
                session_result = build_session_success_record(
                    dialogue_record,
                    judge_result,
                    parsed,
                    save_prompt_text=save_prompt_text,
                    prompt_text=prompt_text,
                    judge_input_mode=judge_request.input_mode,
                    judge_media_refs=judge_request.media_refs,
                )
        if issued_request:
            _sleep_between_requests(inter_request_sleep)
        if isinstance(session_result, dict) and on_session_record is not None:
            on_session_record(session_result)

    return {
        "dialogue_id": work_item.dialogue_record.get("dialogue_id", ""),
        "turn_records": turn_results,
        "session_record": session_result,
    }


def run_phase3_pipeline(
    phase2_dir: Path,
    output_dir: Path,
    backend: Any,
    *,
    dialogue_id: str = "",
    limit_dialogues: Optional[int] = None,
    clear_output: bool = True,
    resume: bool = False,
    repair_failed: bool = False,
    max_workers: int = 1,
    save_prompt_text: bool = False,
    allow_incomplete_dialogues: bool = False,
    inter_request_sleep: float = 0.0,
    repair_passes: int = 0,
    repair_pass_cooldown: float = 0.0,
    turn_repair_mode: str = "full_context",
    session_repair_mode: str = "full_context",
    backend_factory: Optional[Callable[[], Any]] = None,
) -> Dict[str, Any]:
    valid_turn_repair_modes = {"full_context", "current_turn_only"}
    valid_session_repair_modes = {"full_context", "light_context"}
    if turn_repair_mode not in valid_turn_repair_modes:
        raise ValueError(f"不支持的 turn repair 模式: {turn_repair_mode}")
    if session_repair_mode not in valid_session_repair_modes:
        raise ValueError(f"不支持的 session repair 模式: {session_repair_mode}")

    rounds_path, dialogues_path = resolve_phase2_paths(phase2_dir)
    rounds = read_jsonl(rounds_path)
    dialogues = read_jsonl(dialogues_path)
    rounds, dialogues = filter_records(rounds, dialogues, dialogue_id=dialogue_id, limit_dialogues=limit_dialogues)

    output_dir = output_dir.expanduser().resolve()
    output_paths = {name: output_dir / filename for name, filename in OUTPUT_FILENAMES.items()}
    effective_clear_output = clear_output and not resume and not repair_failed
    if effective_clear_output:
        for path in output_paths.values():
            if path.exists():
                path.unlink()
    output_dir.mkdir(parents=True, exist_ok=True)
    for key in ["turn_judgements", "session_judgements", "judge_errors"]:
        if not output_paths[key].exists():
            output_paths[key].write_text("", encoding="utf-8")

    turn_schema = build_turn_response_schema()
    session_schema = build_session_response_schema()
    total_passes = 1 + max(0, int(repair_passes))
    executed_passes = 0
    selected_dialogue_count = len(dialogues)
    skipped_dialogues_total = 0

    for pass_index in range(total_passes):
        executed_passes += 1
        pass_resume = resume or repair_failed or pass_index > 0
        work_items, skipped_dialogues = build_phase3_work_items(
            rounds,
            dialogues,
            output_paths,
            resume=pass_resume,
            repair_failed=repair_failed,
        )
        skipped_dialogues_total = skipped_dialogues
        if not work_items:
            current_turn_records, current_session_records, _ = canonicalize_phase3_output_files(output_paths)
            turn_success_count = sum(1 for record in current_turn_records if record.get("status") == "success")
            session_success_count = sum(1 for record in current_session_records if record.get("status") == "success")
            if turn_success_count >= len(rounds) and session_success_count >= len(dialogues):
                break
            if pass_index < total_passes - 1:
                _sleep_between_requests(repair_pass_cooldown)
            continue

        write_lock = Lock()

        def _append_turn_record(record: Dict[str, Any]) -> None:
            with write_lock:
                append_jsonl(output_paths["turn_judgements"], record)

        def _append_session_record(record: Dict[str, Any]) -> None:
            with write_lock:
                append_jsonl(output_paths["session_judgements"], record)

        use_repair_modes = repair_failed or pass_index > 0
        worker_count = max(1, int(max_workers))
        if worker_count == 1 or len(work_items) <= 1:
            for work_item in work_items:
                process_phase3_dialogue_work_item(
                    work_item,
                    backend,
                    save_prompt_text=save_prompt_text,
                    allow_incomplete_dialogues=allow_incomplete_dialogues,
                    inter_request_sleep=inter_request_sleep,
                    turn_schema=turn_schema,
                    session_schema=session_schema,
                    use_repair_modes=use_repair_modes,
                    turn_repair_mode=turn_repair_mode,
                    session_repair_mode=session_repair_mode,
                    on_turn_record=_append_turn_record,
                    on_session_record=_append_session_record,
                )
        else:
            if backend_factory is None:
                raise ValueError("max_workers > 1 时必须提供 backend_factory，以便每个 worker 创建独立的 judge backend。")

            def _worker(work_item: Phase3DialogueWorkItem) -> Dict[str, Any]:
                worker_backend = backend_factory()
                try:
                    return process_phase3_dialogue_work_item(
                        work_item,
                        worker_backend,
                        save_prompt_text=save_prompt_text,
                        allow_incomplete_dialogues=allow_incomplete_dialogues,
                        inter_request_sleep=inter_request_sleep,
                        turn_schema=turn_schema,
                        session_schema=session_schema,
                        use_repair_modes=use_repair_modes,
                        turn_repair_mode=turn_repair_mode,
                        session_repair_mode=session_repair_mode,
                        on_turn_record=_append_turn_record,
                        on_session_record=_append_session_record,
                    )
                finally:
                    close_backend(worker_backend)

            with ThreadPoolExecutor(max_workers=worker_count) as executor:
                future_to_dialogue = {executor.submit(_worker, work_item): str(work_item.dialogue_record.get("dialogue_id", "")) for work_item in work_items}
                for future in as_completed(future_to_dialogue):
                    future.result()

        current_turn_records, current_session_records, _ = canonicalize_phase3_output_files(output_paths)
        turn_success_count = sum(1 for record in current_turn_records if record.get("status") == "success")
        session_success_count = sum(1 for record in current_session_records if record.get("status") == "success")
        if turn_success_count >= len(rounds) and session_success_count >= len(dialogues):
            break
        if pass_index < total_passes - 1:
            _sleep_between_requests(repair_pass_cooldown)

    turn_records, session_records, error_records = canonicalize_phase3_output_files(output_paths)
    turn_summary = build_turn_summary(turn_records)
    session_summary = build_session_summary(session_records)
    task_summary = build_task_summary(turn_records, session_records)
    category_summary = build_category_summary(turn_records)
    validation_summary = build_validation_summary(
        len(rounds),
        len(dialogues),
        turn_records,
        session_records,
        error_records,
    )

    dump_json(output_paths["turn_summary"], turn_summary)
    dump_json(output_paths["session_summary"], session_summary)
    dump_json(output_paths["task_summary"], task_summary)
    dump_json(output_paths["category_summary"], category_summary)
    dump_json(output_paths["validation_summary"], validation_summary)

    manifest = {
        "schema_version": PHASE3_SCHEMA_VERSION,
        "generated_at": utc_now(),
        "phase2_dir": str(phase2_dir.expanduser().resolve()),
        "rounds_input_path": str(rounds_path),
        "dialogues_input_path": str(dialogues_path),
        "output_dir": str(output_dir),
        "judge_backend": getattr(backend, "backend_name", ""),
        "judge_model_name": getattr(backend, "model_name", ""),
        "judge_evidence_mode": "text_plus_visual",
        "inter_request_sleep": inter_request_sleep,
        "repair_failed": repair_failed,
        "repair_passes_requested": int(repair_passes),
        "repair_passes_executed": executed_passes,
        "repair_pass_cooldown": repair_pass_cooldown,
        "turn_repair_mode": turn_repair_mode,
        "session_repair_mode": session_repair_mode,
        "max_workers": max(1, int(max_workers)),
        "selected_dialogue_count": selected_dialogue_count,
        "skipped_dialogues": skipped_dialogues_total,
        "turn_judgements_path": str(output_paths["turn_judgements"]),
        "session_judgements_path": str(output_paths["session_judgements"]),
        "judge_errors_path": str(output_paths["judge_errors"]),
        "turn_summary_path": str(output_paths["turn_summary"]),
        "session_summary_path": str(output_paths["session_summary"]),
        "task_summary_path": str(output_paths["task_summary"]),
        "category_summary_path": str(output_paths["category_summary"]),
        "validation_summary_path": str(output_paths["validation_summary"]),
        "expected_round_count": len(rounds),
        "expected_dialogue_count": len(dialogues),
        "turn_record_count": len(turn_records),
        "session_record_count": len(session_records),
        "turn_success_count": sum(1 for record in turn_records if record.get("status") == "success"),
        "session_success_count": sum(1 for record in session_records if record.get("status") == "success"),
        "error_record_count": len(error_records),
    }
    dump_json(output_paths["manifest"], manifest)
    return manifest
