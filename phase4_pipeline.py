from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from attribution_aggregation import (
    build_ability_score_summary,
    build_error_category_summary,
    build_error_reason_by_ability_summary,
    build_error_reason_by_mode_summary,
    build_error_reason_by_task_summary,
    build_high_score_summary,
    build_metric_error_cross_summary,
    build_metric_failure_summary,
    build_phase4_validation_summary,
    build_representative_cases,
    build_score_summary,
    build_session_attribution_summary,
    build_task_error_summary,
    build_turn_attribution_summary,
)
from attribution_parsing import (
    PHASE4_SCHEMA_VERSION,
    build_session_attribution_response_schema,
    build_turn_attribution_response_schema,
    parse_session_attribution_text,
    parse_turn_attribution_text,
)
from attribution_prompts import (
    SESSION_PROMPT_VERSION,
    TURN_PROMPT_VERSION,
    build_session_attribution_prompt,
    build_turn_attribution_prompt,
)
from attribution_runner import build_text_only_request
from low_score_selector import (
    DEFAULT_AVG_THRESHOLD,
    DEFAULT_CRITICAL_THRESHOLD,
    DEFAULT_METRIC_THRESHOLD,
    select_session_low_score_candidates,
    select_turn_low_score_candidates,
)

OUTPUT_FILENAMES = {
    "low_score_turns": "low_score_turns.jsonl",
    "low_score_sessions": "low_score_sessions.jsonl",
    "turn_attributions": "turn_error_attributions.jsonl",
    "session_attributions": "session_error_attributions.jsonl",
    "phase4_errors": "phase4_errors.jsonl",
    "turn_summary": "turn_attribution_summary.json",
    "session_summary": "session_attribution_summary.json",
    "error_category_summary": "error_category_summary.json",
    "task_error_summary": "task_error_summary.json",
    "metric_failure_summary": "metric_failure_summary.json",
    "error_reason_by_task_summary": "error_reason_by_task_summary.json",
    "error_reason_by_mode_summary": "error_reason_by_mode_summary.json",
    "error_reason_by_ability_summary": "error_reason_by_ability_summary.json",
    "metric_error_cross_summary": "metric_error_cross_summary.json",
    "high_score_summary": "high_score_summary.json",
    "representative_cases": "representative_cases.json",
    "score_summary": "score_summary.json",
    "ability_score_summary": "ability_score_summary.json",
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



def write_jsonl(path: Path, records: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for payload in records:
            file.write(json.dumps(payload, ensure_ascii=False) + "\n")



def append_jsonl(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(payload, ensure_ascii=False) + "\n")



def dump_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)



def resolve_phase3_paths(input_dir: Path) -> Tuple[Path, Path]:
    resolved = input_dir.expanduser().resolve()
    turn_path = resolved / "turn_judgements.jsonl"
    session_path = resolved / "session_judgements.jsonl"
    if not turn_path.exists():
        raise FileNotFoundError(f"未找到 Phase 3 的 turn_judgements.jsonl: {turn_path}")
    if not session_path.exists():
        raise FileNotFoundError(f"未找到 Phase 3 的 session_judgements.jsonl: {session_path}")
    return turn_path, session_path



def resolve_phase2_dialogues_path(phase2_dir: Optional[Path]) -> Optional[Path]:
    if phase2_dir is None:
        return None
    resolved = phase2_dir.expanduser().resolve()
    candidate = resolved / "dialogues.jsonl"
    return candidate if candidate.exists() else None



def attribution_scope_key(record: Dict[str, Any]) -> str:
    return "::".join(
        [
            str(record.get("tested_provider", record.get("provider", "")) or ""),
            str(record.get("tested_model_name", record.get("model_name", "")) or ""),
            str(record.get("task_name", "") or ""),
            str(record.get("dialogue_id", "") or ""),
        ]
    )



def latest_records_by_key(records: Sequence[Dict[str, Any]], key_name: str) -> List[Dict[str, Any]]:
    latest: Dict[str, Dict[str, Any]] = {}
    order: List[str] = []
    without_key: List[Dict[str, Any]] = []
    for record in records:
        key = str(record.get(key_name, "") or "")
        if not key:
            without_key.append(record)
            continue
        if key not in latest:
            order.append(key)
        latest[key] = record
    return [latest[key] for key in order] + without_key



def load_existing_success_ids(path: Path, key_name: str) -> set[str]:
    if not path.exists():
        return set()
    latest_records = latest_records_by_key(read_jsonl(path), key_name)
    return {
        str(record.get(key_name, "") or "")
        for record in latest_records
        if str(record.get(key_name, "") or "") and str(record.get("status", "") or "") == "success"
    }



def _ordered_dialogue_ids(session_records: Sequence[Dict[str, Any]]) -> List[str]:
    seen: set[str] = set()
    ordered: List[str] = []
    for record in session_records:
        dialogue_id = str(record.get("dialogue_id", "") or "")
        if dialogue_id and dialogue_id not in seen:
            seen.add(dialogue_id)
            ordered.append(dialogue_id)
    return ordered



def filter_phase3_records(
    turn_records: Sequence[Dict[str, Any]],
    session_records: Sequence[Dict[str, Any]],
    *,
    dialogue_id: str = "",
    limit_dialogues: Optional[int] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    filtered_sessions = list(session_records)
    if dialogue_id:
        filtered_sessions = [record for record in filtered_sessions if str(record.get("dialogue_id", "") or "") == dialogue_id]
    if limit_dialogues is not None:
        ordered_ids = _ordered_dialogue_ids(filtered_sessions)[:limit_dialogues]
        allowed_ids = set(ordered_ids)
        filtered_sessions = [record for record in filtered_sessions if str(record.get("dialogue_id", "") or "") in allowed_ids]
    allowed_dialogue_ids = {str(record.get("dialogue_id", "") or "") for record in filtered_sessions}
    filtered_turns = [record for record in turn_records if str(record.get("dialogue_id", "") or "") in allowed_dialogue_ids]
    return filtered_turns, filtered_sessions



def build_dialogue_context_index(dialogues: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    index: Dict[str, Dict[str, Any]] = {}
    for dialogue_record in dialogues:
        interaction_setup = dialogue_record.get("interaction_setup") or {}
        key = attribution_scope_key(dialogue_record)
        index[key] = {
            "dialogue_scope_key": key,
            "interaction_goal": interaction_setup.get("interaction_goal", {}),
            "user_persona": interaction_setup.get("user_persona", {}),
            "environment": dialogue_record.get("environment", ""),
        }
    return index



def load_phase2_dialogue_contexts(phase2_dir: Optional[Path]) -> Dict[str, Dict[str, Any]]:
    path = resolve_phase2_dialogues_path(phase2_dir)
    if path is None:
        return {}
    return build_dialogue_context_index(read_jsonl(path))



def _sleep_between_requests(seconds: float) -> None:
    if seconds > 0:
        time.sleep(seconds)



def build_turn_api_error_record(candidate: Dict[str, Any], judge_result: Any, *, save_prompt_text: bool, prompt_text: str) -> Dict[str, Any]:
    record = {
        "schema_version": PHASE4_SCHEMA_VERSION,
        "record_type": "turn_error_attribution",
        "candidate_scope": "turn",
        "candidate_id": candidate.get("candidate_id", ""),
        "judgement_id": candidate.get("judgement_id", ""),
        "dialogue_scope_key": candidate.get("dialogue_scope_key", ""),
        "dialogue_id": candidate.get("dialogue_id", ""),
        "round_id": candidate.get("round_id", ""),
        "round_index": candidate.get("round_index", 0),
        "task_name": candidate.get("task_name", ""),
        "task_alias": candidate.get("task_alias", ""),
        "task_mode": candidate.get("task_mode", ""),
        "question_mode": candidate.get("question_mode", ""),
        "media_mode": candidate.get("media_mode", ""),
        "primary_category": candidate.get("primary_category", ""),
        "secondary_categories": candidate.get("secondary_categories", []),
        "avg_score": candidate.get("avg_score"),
        "score_vector": candidate.get("score_vector", {}),
        "reason_vector": candidate.get("reason_vector", {}),
        "overall_summary": candidate.get("overall_summary", ""),
        "question_text": candidate.get("question_text", ""),
        "reference_answer": candidate.get("reference_answer", ""),
        "prediction": candidate.get("prediction", ""),
        "source_judge_status": candidate.get("status", ""),
        "severity": candidate.get("severity", ""),
        "trigger_reasons": candidate.get("trigger_reasons", []),
        "low_score_metrics": candidate.get("low_score_metrics", []),
        "critical_metrics": candidate.get("critical_metrics", []),
        "judge_backend": judge_result.backend_name,
        "judge_model_name": judge_result.model_name,
        "judge_prompt_version": TURN_PROMPT_VERSION,
        "judge_input_mode": "text_only",
        "judge_attempt_count": judge_result.attempt_count,
        "judge_usage": judge_result.usage,
        "schema_mode": judge_result.schema_mode,
        "attributed_at": utc_now(),
        "raw_attribution_text": judge_result.raw_text,
        "parse_success": False,
        "parse_error": judge_result.error,
        "parse_issues": [judge_result.error],
        "primary_error_category": "",
        "secondary_error_categories": [],
        "affected_metrics": [],
        "attribution_summary": "",
        "error": judge_result.error,
        "error_type": judge_result.error_type or "api_error",
        "status_code": judge_result.status_code,
        "status": "api_error",
    }
    if save_prompt_text:
        record["judge_prompt_text"] = prompt_text
    return record



def build_session_api_error_record(candidate: Dict[str, Any], judge_result: Any, *, save_prompt_text: bool, prompt_text: str) -> Dict[str, Any]:
    record = {
        "schema_version": PHASE4_SCHEMA_VERSION,
        "record_type": "session_error_attribution",
        "candidate_scope": "session",
        "candidate_id": candidate.get("candidate_id", ""),
        "judgement_id": candidate.get("judgement_id", ""),
        "dialogue_scope_key": candidate.get("dialogue_scope_key", ""),
        "dialogue_id": candidate.get("dialogue_id", ""),
        "task_name": candidate.get("task_name", ""),
        "task_alias": candidate.get("task_alias", ""),
        "task_mode": candidate.get("task_mode", ""),
        "question_mode": candidate.get("question_mode", ""),
        "media_mode": candidate.get("media_mode", ""),
        "primary_category": candidate.get("primary_category", ""),
        "secondary_categories": candidate.get("secondary_categories", []),
        "avg_score": candidate.get("avg_score"),
        "score_vector": candidate.get("score_vector", {}),
        "reason_vector": candidate.get("reason_vector", {}),
        "overall_summary": candidate.get("overall_summary", ""),
        "question_text": candidate.get("question_text", ""),
        "reference_answer": candidate.get("reference_answer", ""),
        "prediction": candidate.get("prediction", ""),
        "source_judge_status": candidate.get("status", ""),
        "severity": candidate.get("severity", ""),
        "trigger_reasons": candidate.get("trigger_reasons", []),
        "low_score_metrics": candidate.get("low_score_metrics", []),
        "critical_metrics": candidate.get("critical_metrics", []),
        "judge_backend": judge_result.backend_name,
        "judge_model_name": judge_result.model_name,
        "judge_prompt_version": SESSION_PROMPT_VERSION,
        "judge_input_mode": "text_only",
        "judge_attempt_count": judge_result.attempt_count,
        "judge_usage": judge_result.usage,
        "schema_mode": judge_result.schema_mode,
        "attributed_at": utc_now(),
        "raw_attribution_text": judge_result.raw_text,
        "parse_success": False,
        "parse_error": judge_result.error,
        "parse_issues": [judge_result.error],
        "primary_error_category": "",
        "secondary_error_categories": [],
        "affected_metrics": [],
        "attribution_summary": "",
        "improvement_focus": [],
        "error": judge_result.error,
        "error_type": judge_result.error_type or "api_error",
        "status_code": judge_result.status_code,
        "status": "api_error",
    }
    if save_prompt_text:
        record["judge_prompt_text"] = prompt_text
    return record



def build_turn_success_record(candidate: Dict[str, Any], judge_result: Any, parsed: Dict[str, Any], *, save_prompt_text: bool, prompt_text: str) -> Dict[str, Any]:
    record = {
        "schema_version": PHASE4_SCHEMA_VERSION,
        "record_type": "turn_error_attribution",
        "candidate_scope": "turn",
        "candidate_id": candidate.get("candidate_id", ""),
        "judgement_id": candidate.get("judgement_id", ""),
        "dialogue_scope_key": candidate.get("dialogue_scope_key", ""),
        "dialogue_id": candidate.get("dialogue_id", ""),
        "round_id": candidate.get("round_id", ""),
        "round_index": candidate.get("round_index", 0),
        "task_name": candidate.get("task_name", ""),
        "task_alias": candidate.get("task_alias", ""),
        "task_mode": candidate.get("task_mode", ""),
        "question_mode": candidate.get("question_mode", ""),
        "media_mode": candidate.get("media_mode", ""),
        "primary_category": candidate.get("primary_category", ""),
        "secondary_categories": candidate.get("secondary_categories", []),
        "question_text": candidate.get("question_text", ""),
        "reference_answer": candidate.get("reference_answer", ""),
        "prediction": candidate.get("prediction", ""),
        "round_count": candidate.get("round_count", 0),
        "avg_score": candidate.get("avg_score"),
        "score_vector": candidate.get("score_vector", {}),
        "reason_vector": candidate.get("reason_vector", {}),
        "overall_summary": candidate.get("overall_summary", ""),
        "key_dialogue_signals": candidate.get("key_dialogue_signals", []),
        "source_judge_status": candidate.get("status", ""),
        "severity": candidate.get("severity", ""),
        "trigger_reasons": candidate.get("trigger_reasons", []),
        "low_score_metrics": candidate.get("low_score_metrics", []),
        "critical_metrics": candidate.get("critical_metrics", []),
        "judge_backend": judge_result.backend_name,
        "judge_model_name": judge_result.model_name,
        "judge_prompt_version": TURN_PROMPT_VERSION,
        "judge_input_mode": "text_only",
        "judge_attempt_count": judge_result.attempt_count,
        "judge_usage": judge_result.usage,
        "schema_mode": judge_result.schema_mode,
        "attributed_at": utc_now(),
        "raw_attribution_text": judge_result.raw_text,
        "parse_success": parsed.get("parse_success", False),
        "parse_error": parsed.get("parse_error", ""),
        "parse_issues": parsed.get("parse_issues", []),
        "primary_error_category": parsed.get("primary_error_category", ""),
        "secondary_error_categories": parsed.get("secondary_error_categories", []),
        "affected_metrics": parsed.get("affected_metrics", []),
        "attribution_summary": parsed.get("attribution_summary", ""),
        "error": judge_result.error,
        "error_type": judge_result.error_type,
        "status_code": judge_result.status_code,
        "status": "success" if parsed.get("parse_success") else "parse_error",
    }
    if save_prompt_text:
        record["judge_prompt_text"] = prompt_text
    return record



def build_session_success_record(candidate: Dict[str, Any], judge_result: Any, parsed: Dict[str, Any], *, save_prompt_text: bool, prompt_text: str) -> Dict[str, Any]:
    record = {
        "schema_version": PHASE4_SCHEMA_VERSION,
        "record_type": "session_error_attribution",
        "candidate_scope": "session",
        "candidate_id": candidate.get("candidate_id", ""),
        "judgement_id": candidate.get("judgement_id", ""),
        "dialogue_scope_key": candidate.get("dialogue_scope_key", ""),
        "dialogue_id": candidate.get("dialogue_id", ""),
        "task_name": candidate.get("task_name", ""),
        "task_alias": candidate.get("task_alias", ""),
        "task_mode": candidate.get("task_mode", ""),
        "question_mode": candidate.get("question_mode", ""),
        "media_mode": candidate.get("media_mode", ""),
        "primary_category": candidate.get("primary_category", ""),
        "secondary_categories": candidate.get("secondary_categories", []),
        "question_text": candidate.get("question_text", ""),
        "reference_answer": candidate.get("reference_answer", ""),
        "prediction": candidate.get("prediction", ""),
        "round_count": candidate.get("round_count", 0),
        "avg_score": candidate.get("avg_score"),
        "score_vector": candidate.get("score_vector", {}),
        "reason_vector": candidate.get("reason_vector", {}),
        "overall_summary": candidate.get("overall_summary", ""),
        "key_dialogue_signals": candidate.get("key_dialogue_signals", []),
        "source_judge_status": candidate.get("status", ""),
        "severity": candidate.get("severity", ""),
        "trigger_reasons": candidate.get("trigger_reasons", []),
        "low_score_metrics": candidate.get("low_score_metrics", []),
        "critical_metrics": candidate.get("critical_metrics", []),
        "judge_backend": judge_result.backend_name,
        "judge_model_name": judge_result.model_name,
        "judge_prompt_version": SESSION_PROMPT_VERSION,
        "judge_input_mode": "text_only",
        "judge_attempt_count": judge_result.attempt_count,
        "judge_usage": judge_result.usage,
        "schema_mode": judge_result.schema_mode,
        "attributed_at": utc_now(),
        "raw_attribution_text": judge_result.raw_text,
        "parse_success": parsed.get("parse_success", False),
        "parse_error": parsed.get("parse_error", ""),
        "parse_issues": parsed.get("parse_issues", []),
        "primary_error_category": parsed.get("primary_error_category", ""),
        "secondary_error_categories": parsed.get("secondary_error_categories", []),
        "affected_metrics": parsed.get("affected_metrics", []),
        "attribution_summary": parsed.get("attribution_summary", ""),
        "improvement_focus": parsed.get("improvement_focus", []),
        "error": judge_result.error,
        "error_type": judge_result.error_type,
        "status_code": judge_result.status_code,
        "status": "success" if parsed.get("parse_success") else "parse_error",
    }
    if save_prompt_text:
        record["judge_prompt_text"] = prompt_text
    return record



def load_all_attribution_records(output_paths: Dict[str, Path]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    raw_turn = read_jsonl(output_paths["turn_attributions"]) if output_paths["turn_attributions"].exists() else []
    raw_session = read_jsonl(output_paths["session_attributions"]) if output_paths["session_attributions"].exists() else []
    turn_records = latest_records_by_key(raw_turn, "candidate_id")
    session_records = latest_records_by_key(raw_session, "candidate_id")
    error_records = [record for record in list(turn_records) + list(session_records) if str(record.get("status", "") or "") != "success"]
    return turn_records, session_records, error_records


def _sorted_turn_attributions(records: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return sorted(
        records,
        key=lambda record: (
            str(record.get("task_name", "") or ""),
            str(record.get("dialogue_id", "") or ""),
            int(record.get("round_index", 0) or 0),
            str(record.get("candidate_id", "") or ""),
        ),
    )


def _sorted_session_attributions(records: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return sorted(
        records,
        key=lambda record: (
            str(record.get("task_name", "") or ""),
            str(record.get("dialogue_id", "") or ""),
            str(record.get("candidate_id", "") or ""),
        ),
    )


def _sorted_phase4_errors(records: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return sorted(
        records,
        key=lambda record: (
            str(record.get("record_type", "") or ""),
            str(record.get("task_name", "") or ""),
            str(record.get("dialogue_id", "") or ""),
            int(record.get("round_index", -1) or -1),
            str(record.get("candidate_id", "") or ""),
        ),
    )


def canonicalize_phase4_output_files(output_paths: Dict[str, Path]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    turn_records, session_records, error_records = load_all_attribution_records(output_paths)
    turn_records = _sorted_turn_attributions(turn_records)
    session_records = _sorted_session_attributions(session_records)
    error_records = _sorted_phase4_errors(error_records)
    write_jsonl(output_paths["turn_attributions"], turn_records)
    write_jsonl(output_paths["session_attributions"], session_records)
    write_jsonl(output_paths["phase4_errors"], error_records)
    return turn_records, session_records, error_records


def close_backend(backend: Any) -> None:
    close = getattr(backend, "close", None)
    if callable(close):
        close()


def process_turn_candidate(
    candidate: Dict[str, Any],
    backend: Any,
    *,
    turn_schema: Dict[str, Any],
    save_prompt_text: bool,
) -> Dict[str, Any]:
    prompt_text = build_turn_attribution_prompt(candidate)
    judge_request = build_text_only_request(prompt_text)
    judge_result = backend.judge(judge_request, turn_schema)
    if judge_result.error:
        return build_turn_api_error_record(candidate, judge_result, save_prompt_text=save_prompt_text, prompt_text=prompt_text)
    parsed = parse_turn_attribution_text(judge_result.raw_text)
    return build_turn_success_record(candidate, judge_result, parsed, save_prompt_text=save_prompt_text, prompt_text=prompt_text)


def process_session_candidate(
    candidate: Dict[str, Any],
    dialogue_context_index: Dict[str, Dict[str, Any]],
    backend: Any,
    *,
    session_schema: Dict[str, Any],
    save_prompt_text: bool,
) -> Dict[str, Any]:
    dialogue_scope_key = str(candidate.get("dialogue_scope_key", "") or "")
    dialogue_context = dialogue_context_index.get(dialogue_scope_key, {})
    prompt_text = build_session_attribution_prompt(candidate, dialogue_context)
    judge_request = build_text_only_request(prompt_text)
    judge_result = backend.judge(judge_request, session_schema)
    if judge_result.error:
        return build_session_api_error_record(candidate, judge_result, save_prompt_text=save_prompt_text, prompt_text=prompt_text)
    parsed = parse_session_attribution_text(judge_result.raw_text)
    return build_session_success_record(candidate, judge_result, parsed, save_prompt_text=save_prompt_text, prompt_text=prompt_text)


def run_phase4_pipeline(
    phase3_dir: Path,
    output_dir: Path,
    backend: Any,
    *,
    phase2_dir: Optional[Path] = None,
    dialogue_id: str = "",
    limit_dialogues: Optional[int] = None,
    clear_output: bool = True,
    resume: bool = False,
    save_prompt_text: bool = False,
    avg_threshold: float = DEFAULT_AVG_THRESHOLD,
    metric_threshold: int = DEFAULT_METRIC_THRESHOLD,
    critical_threshold: int = DEFAULT_CRITICAL_THRESHOLD,
    inter_request_sleep: float = 0.0,
    repair_passes: int = 0,
    repair_pass_cooldown: float = 0.0,
    max_workers: int = 1,
    backend_factory: Optional[Callable[[], Any]] = None,
) -> Dict[str, Any]:
    turn_path, session_path = resolve_phase3_paths(phase3_dir)
    phase3_turn_records = latest_records_by_key(read_jsonl(turn_path), "judgement_id")
    phase3_session_records = latest_records_by_key(read_jsonl(session_path), "judgement_id")
    phase3_turn_records, phase3_session_records = filter_phase3_records(
        phase3_turn_records,
        phase3_session_records,
        dialogue_id=dialogue_id,
        limit_dialogues=limit_dialogues,
    )

    dialogue_context_index = load_phase2_dialogue_contexts(phase2_dir)
    turn_candidates = select_turn_low_score_candidates(
        phase3_turn_records,
        avg_threshold=avg_threshold,
        metric_threshold=metric_threshold,
        critical_threshold=critical_threshold,
    )
    session_candidates = select_session_low_score_candidates(
        phase3_session_records,
        avg_threshold=avg_threshold,
        metric_threshold=metric_threshold,
        critical_threshold=critical_threshold,
    )

    output_dir = output_dir.expanduser().resolve()
    output_paths = {name: output_dir / filename for name, filename in OUTPUT_FILENAMES.items()}
    if clear_output and not resume:
        for path in output_paths.values():
            if path.exists():
                path.unlink()
    output_dir.mkdir(parents=True, exist_ok=True)
    for key in ["turn_attributions", "session_attributions", "phase4_errors"]:
        if not output_paths[key].exists():
            output_paths[key].write_text("", encoding="utf-8")

    write_jsonl(output_paths["low_score_turns"], turn_candidates)
    write_jsonl(output_paths["low_score_sessions"], session_candidates)

    turn_schema = build_turn_attribution_response_schema()
    session_schema = build_session_attribution_response_schema()
    total_passes = 1 + max(0, int(repair_passes))
    executed_passes = 0

    for pass_index in range(total_passes):
        executed_passes += 1
        pass_resume = resume or pass_index > 0
        existing_turn_success_ids = load_existing_success_ids(output_paths["turn_attributions"], "candidate_id") if pass_resume else set()
        existing_session_success_ids = load_existing_success_ids(output_paths["session_attributions"], "candidate_id") if pass_resume else set()
        pending_turn_candidates = [
            candidate
            for candidate in turn_candidates
            if str(candidate.get("candidate_id", "") or "") not in existing_turn_success_ids
        ]
        pending_session_candidates = [
            candidate
            for candidate in session_candidates
            if str(candidate.get("candidate_id", "") or "") not in existing_session_success_ids
        ]

        write_lock = Lock()

        def _append_turn_record(record: Dict[str, Any]) -> None:
            with write_lock:
                append_jsonl(output_paths["turn_attributions"], record)
                if str(record.get("status", "") or "") != "success":
                    append_jsonl(output_paths["phase4_errors"], record)

        def _append_session_record(record: Dict[str, Any]) -> None:
            with write_lock:
                append_jsonl(output_paths["session_attributions"], record)
                if str(record.get("status", "") or "") != "success":
                    append_jsonl(output_paths["phase4_errors"], record)

        worker_count = max(1, int(max_workers))

        if worker_count == 1 or len(pending_turn_candidates) <= 1:
            for candidate in pending_turn_candidates:
                record = process_turn_candidate(
                    candidate,
                    backend,
                    turn_schema=turn_schema,
                    save_prompt_text=save_prompt_text,
                )
                _append_turn_record(record)
                _sleep_between_requests(inter_request_sleep)
        else:
            if backend_factory is None:
                raise ValueError("max_workers > 1 时必须提供 backend_factory，以便每个 worker 创建独立的归因 backend。")

            def _turn_worker(candidate: Dict[str, Any]) -> Dict[str, Any]:
                worker_backend = backend_factory()
                try:
                    return process_turn_candidate(
                        candidate,
                        worker_backend,
                        turn_schema=turn_schema,
                        save_prompt_text=save_prompt_text,
                    )
                finally:
                    close_backend(worker_backend)

            with ThreadPoolExecutor(max_workers=worker_count) as executor:
                future_to_candidate = {executor.submit(_turn_worker, candidate): str(candidate.get("candidate_id", "") or "") for candidate in pending_turn_candidates}
                for future in as_completed(future_to_candidate):
                    record = future.result()
                    _append_turn_record(record)
                    _sleep_between_requests(inter_request_sleep)

        if worker_count == 1 or len(pending_session_candidates) <= 1:
            for candidate in pending_session_candidates:
                record = process_session_candidate(
                    candidate,
                    dialogue_context_index,
                    backend,
                    session_schema=session_schema,
                    save_prompt_text=save_prompt_text,
                )
                _append_session_record(record)
                _sleep_between_requests(inter_request_sleep)
        else:
            if backend_factory is None:
                raise ValueError("max_workers > 1 时必须提供 backend_factory，以便每个 worker 创建独立的归因 backend。")

            def _session_worker(candidate: Dict[str, Any]) -> Dict[str, Any]:
                worker_backend = backend_factory()
                try:
                    return process_session_candidate(
                        candidate,
                        dialogue_context_index,
                        worker_backend,
                        session_schema=session_schema,
                        save_prompt_text=save_prompt_text,
                    )
                finally:
                    close_backend(worker_backend)

            with ThreadPoolExecutor(max_workers=worker_count) as executor:
                future_to_candidate = {executor.submit(_session_worker, candidate): str(candidate.get("candidate_id", "") or "") for candidate in pending_session_candidates}
                for future in as_completed(future_to_candidate):
                    record = future.result()
                    _append_session_record(record)
                    _sleep_between_requests(inter_request_sleep)

        current_turn_records, current_session_records, _ = canonicalize_phase4_output_files(output_paths)
        turn_success_count = sum(1 for record in current_turn_records if str(record.get("status", "") or "") == "success")
        session_success_count = sum(1 for record in current_session_records if str(record.get("status", "") or "") == "success")
        if turn_success_count >= len(turn_candidates) and session_success_count >= len(session_candidates):
            break
        if pass_index < total_passes - 1:
            _sleep_between_requests(repair_pass_cooldown)

    turn_records, session_records, error_records = canonicalize_phase4_output_files(output_paths)
    turn_summary = build_turn_attribution_summary(turn_records)
    session_summary = build_session_attribution_summary(session_records)
    error_category_summary = build_error_category_summary(turn_records, session_records)
    task_error_summary = build_task_error_summary(turn_records, session_records)
    metric_failure_summary = build_metric_failure_summary(turn_candidates, session_candidates, turn_records, session_records)
    error_reason_by_task_summary = build_error_reason_by_task_summary(turn_records, session_records)
    error_reason_by_mode_summary = build_error_reason_by_mode_summary(turn_records, session_records)
    error_reason_by_ability_summary = build_error_reason_by_ability_summary(turn_records)
    metric_error_cross_summary = build_metric_error_cross_summary(turn_records, session_records)
    high_score_summary = build_high_score_summary(phase3_turn_records, phase3_session_records)
    representative_cases = build_representative_cases(phase3_turn_records, phase3_session_records, turn_candidates, session_candidates, turn_records, session_records)
    score_summary = build_score_summary(phase3_turn_records, phase3_session_records)
    ability_score_summary = build_ability_score_summary(phase3_turn_records)
    validation_summary = build_phase4_validation_summary(
        len(turn_candidates),
        len(session_candidates),
        turn_records,
        session_records,
        error_records,
    )

    dump_json(output_paths["turn_summary"], turn_summary)
    dump_json(output_paths["session_summary"], session_summary)
    dump_json(output_paths["error_category_summary"], error_category_summary)
    dump_json(output_paths["task_error_summary"], task_error_summary)
    dump_json(output_paths["metric_failure_summary"], metric_failure_summary)
    dump_json(output_paths["error_reason_by_task_summary"], error_reason_by_task_summary)
    dump_json(output_paths["error_reason_by_mode_summary"], error_reason_by_mode_summary)
    dump_json(output_paths["error_reason_by_ability_summary"], error_reason_by_ability_summary)
    dump_json(output_paths["metric_error_cross_summary"], metric_error_cross_summary)
    dump_json(output_paths["high_score_summary"], high_score_summary)
    dump_json(output_paths["representative_cases"], representative_cases)
    dump_json(output_paths["score_summary"], score_summary)
    dump_json(output_paths["ability_score_summary"], ability_score_summary)
    dump_json(output_paths["validation_summary"], validation_summary)

    manifest = {
        "schema_version": PHASE4_SCHEMA_VERSION,
        "generated_at": utc_now(),
        "phase3_dir": str(phase3_dir.expanduser().resolve()),
        "phase2_dir": str(phase2_dir.expanduser().resolve()) if phase2_dir else "",
        "turn_input_path": str(turn_path),
        "session_input_path": str(session_path),
        "output_dir": str(output_dir),
        "attribution_backend": getattr(backend, "backend_name", ""),
        "attribution_model_name": getattr(backend, "model_name", ""),
        "attribution_input_mode": "text_only",
        "avg_threshold": avg_threshold,
        "metric_threshold": metric_threshold,
        "critical_threshold": critical_threshold,
        "inter_request_sleep": inter_request_sleep,
        "repair_passes_requested": int(repair_passes),
        "repair_passes_executed": executed_passes,
        "repair_pass_cooldown": repair_pass_cooldown,
        "max_workers": max(1, int(max_workers)),
        "low_score_turns_path": str(output_paths["low_score_turns"]),
        "low_score_sessions_path": str(output_paths["low_score_sessions"]),
        "turn_attributions_path": str(output_paths["turn_attributions"]),
        "session_attributions_path": str(output_paths["session_attributions"]),
        "phase4_errors_path": str(output_paths["phase4_errors"]),
        "turn_summary_path": str(output_paths["turn_summary"]),
        "session_summary_path": str(output_paths["session_summary"]),
        "error_category_summary_path": str(output_paths["error_category_summary"]),
        "task_error_summary_path": str(output_paths["task_error_summary"]),
        "metric_failure_summary_path": str(output_paths["metric_failure_summary"]),
        "error_reason_by_task_summary_path": str(output_paths["error_reason_by_task_summary"]),
        "error_reason_by_mode_summary_path": str(output_paths["error_reason_by_mode_summary"]),
        "error_reason_by_ability_summary_path": str(output_paths["error_reason_by_ability_summary"]),
        "metric_error_cross_summary_path": str(output_paths["metric_error_cross_summary"]),
        "high_score_summary_path": str(output_paths["high_score_summary"]),
        "representative_cases_path": str(output_paths["representative_cases"]),
        "score_summary_path": str(output_paths["score_summary"]),
        "ability_score_summary_path": str(output_paths["ability_score_summary"]),
        "validation_summary_path": str(output_paths["validation_summary"]),
        "turn_candidate_count": len(turn_candidates),
        "session_candidate_count": len(session_candidates),
        "turn_success_count": sum(1 for record in turn_records if str(record.get("status", "") or "") == "success"),
        "session_success_count": sum(1 for record in session_records if str(record.get("status", "") or "") == "success"),
        "error_record_count": len(error_records),
    }
    dump_json(output_paths["manifest"], manifest)
    return manifest
