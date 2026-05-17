from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

SCHEMA_VERSION = "rubric_mme_phase2_v2"
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = REPO_ROOT / "logs"
DEFAULT_OUTPUT = REPO_ROOT / "logs" / "rubric_mme_phase2_normalized"
OUTPUT_FILENAMES = {
    "dialogues": "dialogues.jsonl",
    "rounds": "rounds.jsonl",
    "errors": "errors.jsonl",
    "task_summary": "task_summary.json",
    "category_summary": "category_summary.json",
    "validation_summary": "validation_summary.json",
    "manifest": "manifest.json",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def first_nonempty(*values: Any) -> str:
    for value in values:
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def discover_input_files(input_args: Sequence[str], pattern: str = "*_samples.jsonl") -> List[Path]:
    files: List[Path] = []
    for raw in input_args:
        path = Path(raw).expanduser().resolve()
        if path.is_file():
            files.append(path)
            continue
        if path.is_dir():
            files.extend(sorted(candidate.resolve() for candidate in path.rglob(pattern) if candidate.is_file()))
            continue
        raise FileNotFoundError(f"Input path not found: {path}")
    unique: List[Path] = []
    seen = set()
    for path in files:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        unique.append(path)
    return unique


def read_jsonl(path: Path, *, strict: bool) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as file:
        for line_number, raw_line in enumerate(file, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                if strict:
                    raise
                continue
            if not isinstance(payload, dict):
                if strict:
                    raise ValueError(f"Expected JSON object at {path}:{line_number}, got {type(payload)}")
                continue
            yield payload


def append_jsonl(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(payload, ensure_ascii=False) + "\n")


def dump_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)


def extract_primary_category(category: Any) -> str:
    if isinstance(category, dict):
        return str(category.get("primary_category", "") or "")
    if isinstance(category, str):
        return category
    return ""


def extract_secondary_categories(category: Any) -> List[str]:
    if isinstance(category, dict):
        values = category.get("secondary_categories", [])
        if isinstance(values, list):
            return [str(item) for item in values if str(item).strip()]
    return []


def extract_requires_history(category: Any) -> Optional[bool]:
    if isinstance(category, dict) and "requires_history" in category:
        return bool(category.get("requires_history"))
    return None


def extract_category_reasoning(category: Any) -> str:
    if isinstance(category, dict):
        return str(category.get("reasoning", "") or "")
    return ""


def build_task_mode(media_mode: str, question_mode: str) -> str:
    return f"{media_mode}_{question_mode}"


def build_dialogue_key(dialogue: Dict[str, Any]) -> str:
    return "::".join(
        [
            str(dialogue.get("provider", "")),
            str(dialogue.get("model_name", "")),
            str(dialogue.get("task_name", "")),
            str(dialogue.get("dialogue_id", "")),
        ]
    )


def build_round_id(dialogue_id: str, round_index: int) -> str:
    return f"{dialogue_id}:round_{round_index:03d}"


def normalize_round(
    dialogue: Dict[str, Any],
    round_payload: Dict[str, Any],
    *,
    source_phase1_file: str,
) -> Dict[str, Any]:
    round_index = int(round_payload.get("round_index", 0) or 0)
    category = round_payload.get("category")
    request_context = round_payload.get("request_context") or {}
    error_info = round_payload.get("error_info") or {}
    question_mode = str(dialogue.get("question_mode", "") or "")
    media_mode = str(dialogue.get("media_mode", "") or "")
    prediction = str(round_payload.get("prediction", "") or "")
    normalized = {
        "schema_version": SCHEMA_VERSION,
        "record_type": "round",
        "source_phase1_file": source_phase1_file,
        "benchmark_name": dialogue.get("benchmark_name", "RUBRIC-MME"),
        "source_phase": dialogue.get("phase", "phase1_generation"),
        "normalized_phase": "phase2_standardized",
        "provider": dialogue.get("provider", ""),
        "model_name": dialogue.get("model_name", ""),
        "task_name": dialogue.get("task_name", ""),
        "task_alias": dialogue.get("task_alias", ""),
        "task_mode": build_task_mode(media_mode, question_mode),
        "question_mode": question_mode,
        "media_mode": media_mode,
        "source_type": dialogue.get("source_type", ""),
        "source_dataset_file": dialogue.get("source_dataset_file", ""),
        "source_data_root": dialogue.get("source_data_root", ""),
        "dialogue_id": dialogue.get("dialogue_id", ""),
        "round_id": build_round_id(str(dialogue.get("dialogue_id", "")), round_index),
        "round_index": round_index,
        "round_number": round_index + 1,
        "timestamp": round_payload.get("timestamp", ""),
        "environment": dialogue.get("environment", ""),
        "question_text": round_payload.get("question_text", ""),
        "reference_answer": first_nonempty(round_payload.get("reference_answer", ""), round_payload.get("answer_text", "")),
        "prediction": prediction,
        "prediction_is_empty": prediction == "",
        "category": category,
        "primary_category": extract_primary_category(category),
        "secondary_categories": extract_secondary_categories(category),
        "category_requires_history": extract_requires_history(category),
        "category_reasoning": extract_category_reasoning(category),
        "media_local_path": first_nonempty(round_payload.get("media_local_path", ""), round_payload.get("media_rel_path", "")),
        "media_remote_url": round_payload.get("media_remote_url", ""),
        "media_cloud_path": round_payload.get("media_cloud_path", ""),
        "question_audio_local_path": first_nonempty(round_payload.get("question_audio_local_path", ""), round_payload.get("question_audio_rel_path", "")),
        "question_audio_remote_url": round_payload.get("question_audio_remote_url", ""),
        "question_audio_cloud_path": round_payload.get("question_audio_cloud_path", ""),
        "history_round_count": request_context.get("history_round_count", 0),
        "attempt_count": round_payload.get("attempt_count", 0),
        "latency_seconds": round_payload.get("latency_seconds", None),
        "usage": round_payload.get("usage", {}),
        "request_context": request_context,
        "request_fallback": round_payload.get("request_fallback"),
        "error": round_payload.get("error", ""),
        "error_info": error_info,
        "status_code": error_info.get("status_code"),
        "error_type": error_info.get("error_type", ""),
        "has_error": bool(round_payload.get("error", "")),
        "generated_at": dialogue.get("generated_at", ""),
    }
    return normalized


def normalize_dialogue(
    payload: Dict[str, Any],
    *,
    source_phase1_file: str,
) -> Dict[str, Any]:
    question_mode = str(payload.get("question_mode", "") or "")
    media_mode = str(payload.get("media_mode", "") or "")
    normalized_rounds: List[Dict[str, Any]] = []
    for round_payload in payload.get("rounds", []):
        if not isinstance(round_payload, dict):
            continue
        normalized_rounds.append(normalize_round(payload, round_payload, source_phase1_file=source_phase1_file))

    failed_round_count = sum(1 for item in normalized_rounds if item.get("has_error"))
    return {
        "schema_version": SCHEMA_VERSION,
        "record_type": "dialogue",
        "source_phase1_file": source_phase1_file,
        "benchmark_name": payload.get("benchmark_name", "RUBRIC-MME"),
        "source_phase": payload.get("phase", "phase1_generation"),
        "normalized_phase": "phase2_standardized",
        "provider": payload.get("provider", ""),
        "model_name": payload.get("model_name", ""),
        "task_name": payload.get("task_name", ""),
        "task_alias": payload.get("task_alias", ""),
        "task_mode": build_task_mode(media_mode, question_mode),
        "question_mode": question_mode,
        "media_mode": media_mode,
        "source_type": payload.get("source_type", ""),
        "source_dataset_file": payload.get("source_dataset_file", ""),
        "source_data_root": payload.get("source_data_root", ""),
        "dialogue_id": payload.get("dialogue_id", ""),
        "environment": payload.get("environment", ""),
        "interaction_setup": payload.get("interaction_setup"),
        "conversation_meta": payload.get("conversation_meta"),
        "round_count": len(normalized_rounds),
        "successful_round_count": len(normalized_rounds) - failed_round_count,
        "failed_round_count": failed_round_count,
        "has_errors": failed_round_count > 0,
        "generated_at": payload.get("generated_at", ""),
        "rounds": normalized_rounds,
    }


def validate_dialogue(record: Dict[str, Any]) -> List[str]:
    issues: List[str] = []
    for field_name in ["dialogue_id", "task_name", "model_name", "question_mode", "media_mode"]:
        if not str(record.get(field_name, "") or "").strip():
            issues.append(field_name)
    if int(record.get("round_count", 0) or 0) != len(record.get("rounds", [])):
        issues.append("round_count_mismatch")
    return issues


def validate_round(record: Dict[str, Any]) -> List[str]:
    issues: List[str] = []
    for field_name in ["dialogue_id", "round_id", "task_name", "question_mode", "media_mode"]:
        if not str(record.get(field_name, "") or "").strip():
            issues.append(field_name)
    if record.get("question_mode") == "text" and not str(record.get("question_text", "") or "").strip():
        issues.append("question_text")
    if not str(record.get("reference_answer", "") or "").strip():
        issues.append("reference_answer")
    if not any(
        str(record.get(field_name, "") or "").strip()
        for field_name in ["media_local_path", "media_remote_url", "media_cloud_path"]
    ):
        issues.append("media_reference")
    if record.get("question_mode") == "tts":
        if not any(
            str(record.get(field_name, "") or "").strip()
            for field_name in ["question_audio_local_path", "question_audio_remote_url", "question_audio_cloud_path"]
        ):
            issues.append("question_audio_reference")
    return issues


def build_task_summary(dialogues: Sequence[Dict[str, Any]], rounds: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    by_task: Dict[str, Dict[str, Any]] = {}
    by_task_mode: Dict[str, Dict[str, Any]] = {}
    for dialogue in dialogues:
        task_name = str(dialogue.get("task_name", "") or "")
        task_mode = str(dialogue.get("task_mode", "") or "")
        task_entry = by_task.setdefault(
            task_name,
            {
                "task_name": task_name,
                "task_alias": dialogue.get("task_alias", ""),
                "task_mode": task_mode,
                "dialogue_count": 0,
                "round_count": 0,
                "failed_dialogue_count": 0,
                "failed_round_count": 0,
                "providers": Counter(),
                "models": Counter(),
            },
        )
        task_entry["dialogue_count"] += 1
        task_entry["round_count"] += int(dialogue.get("round_count", 0) or 0)
        task_entry["failed_dialogue_count"] += int(bool(dialogue.get("has_errors")))
        task_entry["failed_round_count"] += int(dialogue.get("failed_round_count", 0) or 0)
        task_entry["providers"].update([str(dialogue.get("provider", "") or "")])
        task_entry["models"].update([str(dialogue.get("model_name", "") or "")])

        mode_entry = by_task_mode.setdefault(
            task_mode,
            {
                "task_mode": task_mode,
                "dialogue_count": 0,
                "round_count": 0,
                "failed_dialogue_count": 0,
                "failed_round_count": 0,
                "tasks": Counter(),
            },
        )
        mode_entry["dialogue_count"] += 1
        mode_entry["round_count"] += int(dialogue.get("round_count", 0) or 0)
        mode_entry["failed_dialogue_count"] += int(bool(dialogue.get("has_errors")))
        mode_entry["failed_round_count"] += int(dialogue.get("failed_round_count", 0) or 0)
        mode_entry["tasks"].update([task_name])

    for payload in by_task.values():
        payload["providers"] = dict(sorted(payload["providers"].items()))
        payload["models"] = dict(sorted(payload["models"].items()))
    for payload in by_task_mode.values():
        payload["tasks"] = dict(sorted(payload["tasks"].items()))

    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": utc_now(),
        "dialogue_count": len(dialogues),
        "round_count": len(rounds),
        "by_task": dict(sorted(by_task.items())),
        "by_task_mode": dict(sorted(by_task_mode.items())),
    }


def build_category_summary(rounds: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    primary: Dict[str, Dict[str, Any]] = {}
    secondary: Dict[str, Dict[str, Any]] = {}
    for round_record in rounds:
        primary_category = str(round_record.get("primary_category", "") or "")
        if primary_category:
            primary_entry = primary.setdefault(
                primary_category,
                {
                    "round_count": 0,
                    "failed_round_count": 0,
                    "tasks": Counter(),
                    "task_modes": Counter(),
                    "requires_history_count": 0,
                },
            )
            primary_entry["round_count"] += 1
            primary_entry["failed_round_count"] += int(bool(round_record.get("has_error")))
            primary_entry["tasks"].update([str(round_record.get("task_name", "") or "")])
            primary_entry["task_modes"].update([str(round_record.get("task_mode", "") or "")])
            primary_entry["requires_history_count"] += int(bool(round_record.get("category_requires_history")))

        for secondary_category in round_record.get("secondary_categories", []):
            secondary_entry = secondary.setdefault(
                secondary_category,
                {
                    "round_count": 0,
                    "failed_round_count": 0,
                    "tasks": Counter(),
                    "task_modes": Counter(),
                },
            )
            secondary_entry["round_count"] += 1
            secondary_entry["failed_round_count"] += int(bool(round_record.get("has_error")))
            secondary_entry["tasks"].update([str(round_record.get("task_name", "") or "")])
            secondary_entry["task_modes"].update([str(round_record.get("task_mode", "") or "")])

    for payload in primary.values():
        payload["tasks"] = dict(sorted(payload["tasks"].items()))
        payload["task_modes"] = dict(sorted(payload["task_modes"].items()))
    for payload in secondary.values():
        payload["tasks"] = dict(sorted(payload["tasks"].items()))
        payload["task_modes"] = dict(sorted(payload["task_modes"].items()))

    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": utc_now(),
        "primary_categories": dict(sorted(primary.items())),
        "secondary_categories": dict(sorted(secondary.items())),
    }


def build_validation_summary(
    dialogues: Sequence[Dict[str, Any]],
    rounds: Sequence[Dict[str, Any]],
    *,
    duplicate_dialogue_count: int,
) -> Dict[str, Any]:
    dialogue_issue_counter: Counter[str] = Counter()
    round_issue_counter: Counter[str] = Counter()
    dialogue_samples: List[Dict[str, Any]] = []
    round_samples: List[Dict[str, Any]] = []

    for dialogue in dialogues:
        issues = validate_dialogue(dialogue)
        if not issues:
            continue
        dialogue_issue_counter.update(issues)
        if len(dialogue_samples) < 20:
            dialogue_samples.append(
                {
                    "dialogue_id": dialogue.get("dialogue_id", ""),
                    "task_name": dialogue.get("task_name", ""),
                    "issues": issues,
                }
            )

    for round_record in rounds:
        issues = validate_round(round_record)
        if not issues:
            continue
        round_issue_counter.update(issues)
        if len(round_samples) < 40:
            round_samples.append(
                {
                    "round_id": round_record.get("round_id", ""),
                    "dialogue_id": round_record.get("dialogue_id", ""),
                    "task_name": round_record.get("task_name", ""),
                    "issues": issues,
                }
            )

    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": utc_now(),
        "duplicate_dialogue_count": duplicate_dialogue_count,
        "dialogue_issue_counts": dict(sorted(dialogue_issue_counter.items())),
        "round_issue_counts": dict(sorted(round_issue_counter.items())),
        "dialogue_issue_samples": dialogue_samples,
        "round_issue_samples": round_samples,
    }


def normalize_phase1_outputs(
    input_files: Sequence[Path],
    output_dir: Path,
    *,
    strict: bool = False,
    clear_output: bool = True,
) -> Dict[str, Any]:
    resolved_inputs = [Path(path).resolve() for path in input_files]
    if not resolved_inputs:
        raise FileNotFoundError("No phase-1 sample files were provided.")

    output_dir = Path(output_dir).expanduser().resolve()
    output_paths = {name: output_dir / filename for name, filename in OUTPUT_FILENAMES.items()}
    if clear_output:
        for path in output_paths.values():
            if path.exists():
                path.unlink()
    output_dir.mkdir(parents=True, exist_ok=True)
    for key in ["dialogues", "rounds", "errors"]:
        output_paths[key].write_text("", encoding="utf-8")

    dialogues: List[Dict[str, Any]] = []
    rounds: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []
    seen_dialogues = set()
    duplicate_dialogue_count = 0
    input_counter: Counter[str] = Counter()
    provider_counter: Counter[str] = Counter()
    model_counter: Counter[str] = Counter()
    task_counter: Counter[str] = Counter()

    for input_file in resolved_inputs:
        for payload in read_jsonl(input_file, strict=strict):
            dialogue = normalize_dialogue(payload, source_phase1_file=str(input_file))
            dialogue_key = build_dialogue_key(dialogue)
            if dialogue_key in seen_dialogues:
                duplicate_dialogue_count += 1
                continue
            seen_dialogues.add(dialogue_key)
            dialogues.append(dialogue)
            input_counter.update([str(input_file)])
            provider_counter.update([str(dialogue.get("provider", "") or "")])
            model_counter.update([str(dialogue.get("model_name", "") or "")])
            task_counter.update([str(dialogue.get("task_name", "") or "")])
            for round_record in dialogue.get("rounds", []):
                rounds.append(round_record)
                if round_record.get("has_error"):
                    errors.append(round_record)

    for dialogue in dialogues:
        append_jsonl(output_paths["dialogues"], dialogue)
    for round_record in rounds:
        append_jsonl(output_paths["rounds"], round_record)
    for round_record in errors:
        append_jsonl(output_paths["errors"], round_record)

    task_summary = build_task_summary(dialogues, rounds)
    category_summary = build_category_summary(rounds)
    validation_summary = build_validation_summary(
        dialogues,
        rounds,
        duplicate_dialogue_count=duplicate_dialogue_count,
    )
    dump_json(output_paths["task_summary"], task_summary)
    dump_json(output_paths["category_summary"], category_summary)
    dump_json(output_paths["validation_summary"], validation_summary)

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "normalized_at": utc_now(),
        "input_files": sorted(input_counter.keys()),
        "output_dir": str(output_dir),
        "dialogues_path": str(output_paths["dialogues"]),
        "rounds_path": str(output_paths["rounds"]),
        "errors_path": str(output_paths["errors"]),
        "task_summary_path": str(output_paths["task_summary"]),
        "category_summary_path": str(output_paths["category_summary"]),
        "validation_summary_path": str(output_paths["validation_summary"]),
        "dialogue_count": len(dialogues),
        "round_count": len(rounds),
        "error_round_count": len(errors),
        "duplicate_dialogue_count": duplicate_dialogue_count,
        "task_counts": dict(sorted(task_counter.items())),
        "model_counts": dict(sorted(model_counter.items())),
        "provider_counts": dict(sorted(provider_counter.items())),
    }
    dump_json(output_paths["manifest"], manifest)
    return manifest
