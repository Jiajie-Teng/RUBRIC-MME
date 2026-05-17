from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Sequence

from judge_parsing import PHASE3_SCHEMA_VERSION, SESSION_METRICS, TURN_METRICS


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _successful(records: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [record for record in records if record.get("status") == "success"]


def _metric_average(records: Sequence[Dict[str, Any]], metric_names: Sequence[str]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for metric_name in metric_names:
        values = [record.get("score_vector", {}).get(metric_name) for record in records]
        valid = [float(value) for value in values if isinstance(value, int)]
        result[metric_name] = {
            "count": len(valid),
            "avg_score": round(sum(valid) / len(valid), 4) if valid else None,
        }
    return result


def build_turn_summary(turn_records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    successful = _successful(turn_records)
    return {
        "schema_version": PHASE3_SCHEMA_VERSION,
        "generated_at": utc_now(),
        "record_count": len(turn_records),
        "success_count": len(successful),
        "failed_count": len(turn_records) - len(successful),
        "avg_score": round(sum(record.get("avg_score", 0.0) for record in successful) / len(successful), 4) if successful else None,
        "metrics": _metric_average(successful, TURN_METRICS),
    }


def build_session_summary(session_records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    successful = _successful(session_records)
    return {
        "schema_version": PHASE3_SCHEMA_VERSION,
        "generated_at": utc_now(),
        "record_count": len(session_records),
        "success_count": len(successful),
        "failed_count": len(session_records) - len(successful),
        "avg_score": round(sum(record.get("avg_score", 0.0) for record in successful) / len(successful), 4) if successful else None,
        "metrics": _metric_average(successful, SESSION_METRICS),
    }


def build_task_summary(turn_records: Sequence[Dict[str, Any]], session_records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    tasks: Dict[str, Dict[str, Any]] = {}
    task_modes: Dict[str, Dict[str, Any]] = {}

    for record in turn_records:
        task_name = str(record.get("task_name", "") or "")
        task_mode = str(record.get("task_mode", "") or "")
        task_entry = tasks.setdefault(
            task_name,
            {
                "task_name": task_name,
                "task_alias": record.get("task_alias", ""),
                "turn_record_count": 0,
                "turn_success_count": 0,
                "turn_metrics": defaultdict(list),
                "session_record_count": 0,
                "session_success_count": 0,
                "session_metrics": defaultdict(list),
            },
        )
        task_entry["turn_record_count"] += 1
        if record.get("status") == "success":
            task_entry["turn_success_count"] += 1
            for metric_name, score in (record.get("score_vector") or {}).items():
                if isinstance(score, int):
                    task_entry["turn_metrics"][metric_name].append(float(score))

        mode_entry = task_modes.setdefault(
            task_mode,
            {
                "task_mode": task_mode,
                "turn_record_count": 0,
                "turn_success_count": 0,
                "turn_metrics": defaultdict(list),
                "session_record_count": 0,
                "session_success_count": 0,
                "session_metrics": defaultdict(list),
                "tasks": Counter(),
            },
        )
        mode_entry["turn_record_count"] += 1
        mode_entry["tasks"].update([task_name])
        if record.get("status") == "success":
            mode_entry["turn_success_count"] += 1
            for metric_name, score in (record.get("score_vector") or {}).items():
                if isinstance(score, int):
                    mode_entry["turn_metrics"][metric_name].append(float(score))

    for record in session_records:
        task_name = str(record.get("task_name", "") or "")
        task_mode = str(record.get("task_mode", "") or "")
        task_entry = tasks.setdefault(
            task_name,
            {
                "task_name": task_name,
                "task_alias": record.get("task_alias", ""),
                "turn_record_count": 0,
                "turn_success_count": 0,
                "turn_metrics": defaultdict(list),
                "session_record_count": 0,
                "session_success_count": 0,
                "session_metrics": defaultdict(list),
            },
        )
        task_entry["session_record_count"] += 1
        if record.get("status") == "success":
            task_entry["session_success_count"] += 1
            for metric_name, score in (record.get("score_vector") or {}).items():
                if isinstance(score, int):
                    task_entry["session_metrics"][metric_name].append(float(score))

        mode_entry = task_modes.setdefault(
            task_mode,
            {
                "task_mode": task_mode,
                "turn_record_count": 0,
                "turn_success_count": 0,
                "turn_metrics": defaultdict(list),
                "session_record_count": 0,
                "session_success_count": 0,
                "session_metrics": defaultdict(list),
                "tasks": Counter(),
            },
        )
        mode_entry["session_record_count"] += 1
        mode_entry["tasks"].update([task_name])
        if record.get("status") == "success":
            mode_entry["session_success_count"] += 1
            for metric_name, score in (record.get("score_vector") or {}).items():
                if isinstance(score, int):
                    mode_entry["session_metrics"][metric_name].append(float(score))

    def _finalize_metrics(metric_map: Dict[str, List[float]]) -> Dict[str, Any]:
        return {
            metric_name: {
                "count": len(values),
                "avg_score": round(sum(values) / len(values), 4) if values else None,
            }
            for metric_name, values in sorted(metric_map.items())
        }

    finalized_tasks: Dict[str, Any] = {}
    for task_name, payload in sorted(tasks.items()):
        finalized_tasks[task_name] = {
            "task_name": payload["task_name"],
            "task_alias": payload["task_alias"],
            "turn_record_count": payload["turn_record_count"],
            "turn_success_count": payload["turn_success_count"],
            "turn_metrics": _finalize_metrics(payload["turn_metrics"]),
            "session_record_count": payload["session_record_count"],
            "session_success_count": payload["session_success_count"],
            "session_metrics": _finalize_metrics(payload["session_metrics"]),
        }

    finalized_task_modes: Dict[str, Any] = {}
    for task_mode, payload in sorted(task_modes.items()):
        finalized_task_modes[task_mode] = {
            "task_mode": payload["task_mode"],
            "tasks": dict(sorted(payload["tasks"].items())),
            "turn_record_count": payload["turn_record_count"],
            "turn_success_count": payload["turn_success_count"],
            "turn_metrics": _finalize_metrics(payload["turn_metrics"]),
            "session_record_count": payload["session_record_count"],
            "session_success_count": payload["session_success_count"],
            "session_metrics": _finalize_metrics(payload["session_metrics"]),
        }

    return {
        "schema_version": PHASE3_SCHEMA_VERSION,
        "generated_at": utc_now(),
        "by_task": finalized_tasks,
        "by_task_mode": finalized_task_modes,
    }


def build_category_summary(turn_records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    primary_categories: Dict[str, Dict[str, List[float]]] = {}
    secondary_categories: Dict[str, Dict[str, List[float]]] = {}

    for record in _successful(turn_records):
        primary = str(record.get("primary_category", "") or "")
        secondary_list = record.get("secondary_categories") or []
        if primary:
            bucket = primary_categories.setdefault(primary, {metric_name: [] for metric_name in TURN_METRICS})
            for metric_name, score in (record.get("score_vector") or {}).items():
                if isinstance(score, int):
                    bucket.setdefault(metric_name, []).append(float(score))
        for secondary in secondary_list:
            secondary_name = str(secondary or "").strip()
            if not secondary_name:
                continue
            bucket = secondary_categories.setdefault(secondary_name, {metric_name: [] for metric_name in TURN_METRICS})
            for metric_name, score in (record.get("score_vector") or {}).items():
                if isinstance(score, int):
                    bucket.setdefault(metric_name, []).append(float(score))

    def _finalize(bucket_map: Dict[str, Dict[str, List[float]]]) -> Dict[str, Any]:
        finalized: Dict[str, Any] = {}
        for category_name, metric_map in sorted(bucket_map.items()):
            finalized[category_name] = {
                metric_name: {
                    "count": len(values),
                    "avg_score": round(sum(values) / len(values), 4) if values else None,
                }
                for metric_name, values in sorted(metric_map.items())
            }
        return finalized

    return {
        "schema_version": PHASE3_SCHEMA_VERSION,
        "generated_at": utc_now(),
        "primary_categories": _finalize(primary_categories),
        "secondary_categories": _finalize(secondary_categories),
    }


def build_validation_summary(
    expected_round_count: int,
    expected_dialogue_count: int,
    turn_records: Sequence[Dict[str, Any]],
    session_records: Sequence[Dict[str, Any]],
    error_records: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    turn_status_counts = Counter(str(record.get("status", "")) for record in turn_records)
    session_status_counts = Counter(str(record.get("status", "")) for record in session_records)

    return {
        "schema_version": PHASE3_SCHEMA_VERSION,
        "generated_at": utc_now(),
        "expected_round_count": expected_round_count,
        "expected_dialogue_count": expected_dialogue_count,
        "turn_record_count": len(turn_records),
        "session_record_count": len(session_records),
        "turn_status_counts": dict(sorted(turn_status_counts.items())),
        "session_status_counts": dict(sorted(session_status_counts.items())),
        "error_record_count": len(error_records),
        "turn_coverage_complete": len(turn_records) == expected_round_count,
        "session_coverage_complete": len(session_records) == expected_dialogue_count,
    }
