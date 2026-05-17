from __future__ import annotations

from typing import Any, Dict, List, Sequence

DEFAULT_AVG_THRESHOLD = 4.0
DEFAULT_METRIC_THRESHOLD = 3
DEFAULT_CRITICAL_THRESHOLD = 2


def _as_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _metric_scores(record: Dict[str, Any]) -> Dict[str, int]:
    scores = record.get("score_vector") or {}
    normalized: Dict[str, int] = {}
    if not isinstance(scores, dict):
        return normalized
    for metric_name, value in scores.items():
        try:
            if value is not None:
                normalized[str(metric_name)] = int(value)
        except (TypeError, ValueError):
            continue
    return normalized


def classify_low_score_candidate(
    record: Dict[str, Any],
    *,
    avg_threshold: float = DEFAULT_AVG_THRESHOLD,
    metric_threshold: int = DEFAULT_METRIC_THRESHOLD,
    critical_threshold: int = DEFAULT_CRITICAL_THRESHOLD,
) -> Dict[str, Any]:
    status = str(record.get("status", "") or "")
    avg_score = _as_float(record.get("avg_score"))
    metric_scores = _metric_scores(record)

    low_metrics = [metric for metric, score in metric_scores.items() if score <= metric_threshold]
    critical_metrics = [metric for metric, score in metric_scores.items() if score <= critical_threshold]
    trigger_reasons: List[str] = []

    if status != "success":
        trigger_reasons.append(f"status:{status}")
    if avg_score is not None and avg_score < avg_threshold:
        trigger_reasons.append(f"avg_score_below:{avg_threshold}")
    if low_metrics:
        trigger_reasons.append(f"metric_below_or_equal:{metric_threshold}")

    should_attribute = bool(trigger_reasons)
    if status != "success" or critical_metrics:
        severity = "critical"
    elif should_attribute:
        severity = "warning"
    else:
        severity = "ok"

    return {
        "should_attribute": should_attribute,
        "severity": severity,
        "low_score_metrics": low_metrics,
        "critical_metrics": critical_metrics,
        "trigger_reasons": trigger_reasons,
        "avg_score": avg_score,
    }


def build_turn_candidate_record(
    record: Dict[str, Any],
    *,
    avg_threshold: float = DEFAULT_AVG_THRESHOLD,
    metric_threshold: int = DEFAULT_METRIC_THRESHOLD,
    critical_threshold: int = DEFAULT_CRITICAL_THRESHOLD,
) -> Dict[str, Any] | None:
    classification = classify_low_score_candidate(
        record,
        avg_threshold=avg_threshold,
        metric_threshold=metric_threshold,
        critical_threshold=critical_threshold,
    )
    if not classification["should_attribute"]:
        return None
    return {
        "schema_version": "rubric_mme_phase4_v1",
        "record_type": "turn_low_score_candidate",
        "candidate_id": str(record.get("judgement_id", "") or ""),
        "judgement_id": str(record.get("judgement_id", "") or ""),
        "dialogue_scope_key": str(record.get("dialogue_scope_key", "") or ""),
        "dialogue_id": str(record.get("dialogue_id", "") or ""),
        "round_id": str(record.get("round_id", "") or ""),
        "round_index": int(record.get("round_index", 0) or 0),
        "task_name": str(record.get("task_name", "") or ""),
        "task_alias": str(record.get("task_alias", "") or ""),
        "task_mode": str(record.get("task_mode", "") or ""),
        "question_mode": str(record.get("question_mode", "") or ""),
        "media_mode": str(record.get("media_mode", "") or ""),
        "primary_category": str(record.get("primary_category", "") or ""),
        "secondary_categories": list(record.get("secondary_categories") or []),
        "status": str(record.get("status", "") or ""),
        "avg_score": classification["avg_score"],
        "score_vector": dict(record.get("score_vector") or {}),
        "reason_vector": dict(record.get("reason_vector") or {}),
        "overall_summary": str(record.get("overall_summary", "") or ""),
        "question_text": str(record.get("question_text", "") or ""),
        "reference_answer": str(record.get("reference_answer", "") or ""),
        "prediction": str(record.get("prediction", "") or ""),
        "severity": classification["severity"],
        "low_score_metrics": classification["low_score_metrics"],
        "critical_metrics": classification["critical_metrics"],
        "trigger_reasons": classification["trigger_reasons"],
        "error": str(record.get("error", "") or ""),
        "error_type": str(record.get("error_type", "") or ""),
    }


def build_session_candidate_record(
    record: Dict[str, Any],
    *,
    avg_threshold: float = DEFAULT_AVG_THRESHOLD,
    metric_threshold: int = DEFAULT_METRIC_THRESHOLD,
    critical_threshold: int = DEFAULT_CRITICAL_THRESHOLD,
) -> Dict[str, Any] | None:
    classification = classify_low_score_candidate(
        record,
        avg_threshold=avg_threshold,
        metric_threshold=metric_threshold,
        critical_threshold=critical_threshold,
    )
    if not classification["should_attribute"]:
        return None
    return {
        "schema_version": "rubric_mme_phase4_v1",
        "record_type": "session_low_score_candidate",
        "candidate_id": str(record.get("judgement_id", "") or ""),
        "judgement_id": str(record.get("judgement_id", "") or ""),
        "dialogue_scope_key": str(record.get("dialogue_scope_key", "") or ""),
        "dialogue_id": str(record.get("dialogue_id", "") or ""),
        "task_name": str(record.get("task_name", "") or ""),
        "task_alias": str(record.get("task_alias", "") or ""),
        "task_mode": str(record.get("task_mode", "") or ""),
        "question_mode": str(record.get("question_mode", "") or ""),
        "media_mode": str(record.get("media_mode", "") or ""),
        "status": str(record.get("status", "") or ""),
        "round_count": int(record.get("round_count", 0) or 0),
        "avg_score": classification["avg_score"],
        "score_vector": dict(record.get("score_vector") or {}),
        "reason_vector": dict(record.get("reason_vector") or {}),
        "overall_summary": str(record.get("overall_summary", "") or ""),
        "key_dialogue_signals": list(record.get("key_dialogue_signals") or []),
        "severity": classification["severity"],
        "low_score_metrics": classification["low_score_metrics"],
        "critical_metrics": classification["critical_metrics"],
        "trigger_reasons": classification["trigger_reasons"],
        "error": str(record.get("error", "") or ""),
        "error_type": str(record.get("error_type", "") or ""),
    }


def select_turn_low_score_candidates(records: Sequence[Dict[str, Any]], **kwargs: Any) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    for record in records:
        candidate = build_turn_candidate_record(record, **kwargs)
        if candidate is not None:
            results.append(candidate)
    return results


def select_session_low_score_candidates(records: Sequence[Dict[str, Any]], **kwargs: Any) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    for record in records:
        candidate = build_session_candidate_record(record, **kwargs)
        if candidate is not None:
            results.append(candidate)
    return results
