from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Sequence


TURN_METRICS = [
    "accuracy",
    "completeness",
    "relevance",
    "conciseness",
    "naturalness",
    "proactiveness_helpfulness",
    "intent_understanding_depth",
    "user_state_adaptation",
]

SESSION_METRICS = [
    "session_consistency",
    "intent_fulfillment",
    "persona_adaptation",
    "overall_helpfulness_trustworthiness",
]


def _counter_to_dict(counter: Counter) -> Dict[str, int]:
    return {str(key): int(value) for key, value in sorted(counter.items(), key=lambda item: (-item[1], item[0]))}



def _status_counts(records: Sequence[Dict[str, Any]]) -> Dict[str, int]:
    return _counter_to_dict(Counter(str(record.get("status", "") or "unknown") for record in records))



def _successful(records: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [record for record in records if str(record.get("status", "") or "") == "success"]



def _numeric_score(value: Any) -> Optional[float]:
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value))
    except (TypeError, ValueError):
        return None



def _round_score(value: float) -> float:
    return round(float(value), 4)



def _avg_score_band_distribution(scores: Sequence[float]) -> Dict[str, int]:
    counter = Counter()
    for score in scores:
        if score < 3.0:
            counter["lt_3.0"] += 1
        elif score < 4.0:
            counter["3.0_to_lt_4.0"] += 1
        elif score < 4.5:
            counter["4.0_to_lt_4.5"] += 1
        else:
            counter["4.5_to_5.0"] += 1
    return _counter_to_dict(counter)



def _overall_score_summary(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    success_records = _successful(records)
    scores = [score for score in (_numeric_score(record.get("avg_score")) for record in success_records) if score is not None]
    if not scores:
        return {
            "count": 0,
            "avg_score": 0.0,
            "min_score": 0.0,
            "max_score": 0.0,
            "avg_score_band_distribution": {},
        }
    return {
        "count": len(scores),
        "avg_score": _round_score(sum(scores) / len(scores)),
        "min_score": _round_score(min(scores)),
        "max_score": _round_score(max(scores)),
        "avg_score_band_distribution": _avg_score_band_distribution(scores),
    }



def _metric_summary(records: Sequence[Dict[str, Any]], metric_names: Sequence[str]) -> Dict[str, Any]:
    success_records = _successful(records)
    metric_to_scores: Dict[str, List[float]] = {metric_name: [] for metric_name in metric_names}
    for record in success_records:
        score_vector = record.get("score_vector") or {}
        for metric_name in metric_names:
            score = _numeric_score(score_vector.get(metric_name))
            if score is not None:
                metric_to_scores[metric_name].append(score)

    summary: Dict[str, Any] = {}
    for metric_name in metric_names:
        scores = metric_to_scores[metric_name]
        if not scores:
            summary[metric_name] = {
                "count": 0,
                "avg_score": 0.0,
                "min_score": 0.0,
                "max_score": 0.0,
                "score_distribution": {str(score): 0 for score in range(1, 6)},
                "low_score_count": 0,
                "low_score_rate": 0.0,
            }
            continue
        distribution_counter = Counter(str(int(round(score))) for score in scores)
        low_score_count = sum(1 for score in scores if score <= 3.0)
        summary[metric_name] = {
            "count": len(scores),
            "avg_score": _round_score(sum(scores) / len(scores)),
            "min_score": _round_score(min(scores)),
            "max_score": _round_score(max(scores)),
            "score_distribution": {str(score): int(distribution_counter.get(str(score), 0)) for score in range(1, 6)},
            "low_score_count": low_score_count,
            "low_score_rate": _round_score(low_score_count / len(scores)),
        }
    return summary



def _base_group_summary(records: Sequence[Dict[str, Any]], metric_names: Sequence[str]) -> Dict[str, Any]:
    success_records = _successful(records)
    return {
        "record_count": len(records),
        "success_count": len(success_records),
        "success_rate": _round_score(len(success_records) / len(records)) if records else 0.0,
        "status_counts": _status_counts(records),
        "overall_avg_score": _overall_score_summary(records),
        "metrics": _metric_summary(records, metric_names),
    }



def _group_records(records: Sequence[Dict[str, Any]], key_name: str) -> Dict[str, List[Dict[str, Any]]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[str(record.get(key_name, "") or "unknown")].append(record)
    return dict(grouped)



def _multi_group_records(records: Sequence[Dict[str, Any]], key_name: str) -> Dict[str, List[Dict[str, Any]]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for record in records:
        values = list(record.get(key_name) or [])
        if not values:
            grouped["unknown"].append(record)
            continue
        for value in values:
            grouped[str(value or "unknown")].append(record)
    return dict(grouped)



def build_score_summary(
    turn_phase3_records: Sequence[Dict[str, Any]],
    session_phase3_records: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    by_task_turn = _group_records(turn_phase3_records, "task_name")
    by_task_session = _group_records(session_phase3_records, "task_name")
    by_task_mode_turn = _group_records(turn_phase3_records, "task_mode")
    by_task_mode_session = _group_records(session_phase3_records, "task_mode")
    by_question_mode_turn = _group_records(turn_phase3_records, "question_mode")
    by_question_mode_session = _group_records(session_phase3_records, "question_mode")
    by_media_mode_turn = _group_records(turn_phase3_records, "media_mode")
    by_media_mode_session = _group_records(session_phase3_records, "media_mode")

    def _paired_group_summary(turn_groups: Dict[str, List[Dict[str, Any]]], session_groups: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        keys = sorted(set(turn_groups) | set(session_groups))
        payload: Dict[str, Any] = {}
        for key in keys:
            payload[key] = {
                "turn": _base_group_summary(turn_groups.get(key, []), TURN_METRICS),
                "session": _base_group_summary(session_groups.get(key, []), SESSION_METRICS),
            }
        return payload

    return {
        "record_type": "phase4_score_summary",
        "overall": {
            "turn": _base_group_summary(turn_phase3_records, TURN_METRICS),
            "session": _base_group_summary(session_phase3_records, SESSION_METRICS),
        },
        "by_task": _paired_group_summary(by_task_turn, by_task_session),
        "by_task_mode": _paired_group_summary(by_task_mode_turn, by_task_mode_session),
        "by_question_mode": _paired_group_summary(by_question_mode_turn, by_question_mode_session),
        "by_media_mode": _paired_group_summary(by_media_mode_turn, by_media_mode_session),
    }



def build_ability_score_summary(turn_phase3_records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    by_primary = _group_records(turn_phase3_records, "primary_category")
    by_secondary = _multi_group_records(turn_phase3_records, "secondary_categories")

    primary_payload: Dict[str, Any] = {}
    for category_name, records in sorted(by_primary.items()):
        primary_payload[category_name] = {
            **_base_group_summary(records, TURN_METRICS),
            "task_counts": _counter_to_dict(Counter(str(record.get("task_name", "") or "unknown") for record in records)),
            "task_mode_counts": _counter_to_dict(Counter(str(record.get("task_mode", "") or "unknown") for record in records)),
            "secondary_category_counts": _counter_to_dict(
                Counter(
                    str(secondary)
                    for record in records
                    for secondary in list(record.get("secondary_categories") or [])
                    if str(secondary).strip()
                )
            ),
        }

    secondary_payload: Dict[str, Any] = {}
    for category_name, records in sorted(by_secondary.items()):
        secondary_payload[category_name] = {
            **_base_group_summary(records, TURN_METRICS),
            "task_counts": _counter_to_dict(Counter(str(record.get("task_name", "") or "unknown") for record in records)),
            "task_mode_counts": _counter_to_dict(Counter(str(record.get("task_mode", "") or "unknown") for record in records)),
            "primary_category_counts": _counter_to_dict(Counter(str(record.get("primary_category", "") or "unknown") for record in records)),
        }

    return {
        "record_type": "phase4_ability_score_summary",
        "turn_record_count": len(turn_phase3_records),
        "turn_success_count": len(_successful(turn_phase3_records)),
        "by_primary_category": primary_payload,
        "by_secondary_category": secondary_payload,
    }



def build_turn_attribution_summary(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    success_records = _successful(records)
    return {
        "record_type": "turn_attribution_summary",
        "record_count": len(records),
        "success_count": len(success_records),
        "status_counts": _status_counts(records),
        "primary_error_category_counts": _counter_to_dict(
            Counter(str(record.get("primary_error_category", "") or "") for record in success_records if str(record.get("primary_error_category", "") or ""))
        ),
        "secondary_error_category_counts": _counter_to_dict(
            Counter(
                secondary
                for record in success_records
                for secondary in list(record.get("secondary_error_categories") or [])
                if str(secondary).strip()
            )
        ),
    }



def build_session_attribution_summary(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    success_records = _successful(records)
    return {
        "record_type": "session_attribution_summary",
        "record_count": len(records),
        "success_count": len(success_records),
        "status_counts": _status_counts(records),
        "primary_error_category_counts": _counter_to_dict(
            Counter(str(record.get("primary_error_category", "") or "") for record in success_records if str(record.get("primary_error_category", "") or ""))
        ),
        "secondary_error_category_counts": _counter_to_dict(
            Counter(
                secondary
                for record in success_records
                for secondary in list(record.get("secondary_error_categories") or [])
                if str(secondary).strip()
            )
        ),
    }



def build_error_category_summary(
    turn_records: Sequence[Dict[str, Any]],
    session_records: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    turn_success = _successful(turn_records)
    session_success = _successful(session_records)
    return {
        "turn": {
            "primary": _counter_to_dict(Counter(str(record.get("primary_error_category", "") or "") for record in turn_success if str(record.get("primary_error_category", "") or ""))),
            "secondary": _counter_to_dict(
                Counter(
                    secondary
                    for record in turn_success
                    for secondary in list(record.get("secondary_error_categories") or [])
                    if str(secondary).strip()
                )
            ),
        },
        "session": {
            "primary": _counter_to_dict(Counter(str(record.get("primary_error_category", "") or "") for record in session_success if str(record.get("primary_error_category", "") or ""))),
            "secondary": _counter_to_dict(
                Counter(
                    secondary
                    for record in session_success
                    for secondary in list(record.get("secondary_error_categories") or [])
                    if str(secondary).strip()
                )
            ),
        },
    }



def build_task_error_summary(
    turn_records: Sequence[Dict[str, Any]],
    session_records: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    task_summary: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
        "turn_candidate_count": 0,
        "turn_success_count": 0,
        "session_candidate_count": 0,
        "session_success_count": 0,
        "turn_primary_error_category_counts": Counter(),
        "session_primary_error_category_counts": Counter(),
        "turn_low_metric_counts": Counter(),
        "session_low_metric_counts": Counter(),
    })

    for record in turn_records:
        task_name = str(record.get("task_name", "") or "unknown")
        item = task_summary[task_name]
        item["turn_candidate_count"] += 1
        for metric_name in list(record.get("low_score_metrics") or []):
            item["turn_low_metric_counts"][str(metric_name)] += 1
        if str(record.get("status", "") or "") == "success":
            item["turn_success_count"] += 1
            primary = str(record.get("primary_error_category", "") or "")
            if primary:
                item["turn_primary_error_category_counts"][primary] += 1

    for record in session_records:
        task_name = str(record.get("task_name", "") or "unknown")
        item = task_summary[task_name]
        item["session_candidate_count"] += 1
        for metric_name in list(record.get("low_score_metrics") or []):
            item["session_low_metric_counts"][str(metric_name)] += 1
        if str(record.get("status", "") or "") == "success":
            item["session_success_count"] += 1
            primary = str(record.get("primary_error_category", "") or "")
            if primary:
                item["session_primary_error_category_counts"][primary] += 1

    normalized: Dict[str, Any] = {}
    for task_name, item in task_summary.items():
        normalized[task_name] = {
            "turn_candidate_count": item["turn_candidate_count"],
            "turn_success_count": item["turn_success_count"],
            "session_candidate_count": item["session_candidate_count"],
            "session_success_count": item["session_success_count"],
            "turn_primary_error_category_counts": _counter_to_dict(item["turn_primary_error_category_counts"]),
            "session_primary_error_category_counts": _counter_to_dict(item["session_primary_error_category_counts"]),
            "turn_low_metric_counts": _counter_to_dict(item["turn_low_metric_counts"]),
            "session_low_metric_counts": _counter_to_dict(item["session_low_metric_counts"]),
        }
    return normalized



def build_metric_failure_summary(
    turn_candidates: Sequence[Dict[str, Any]],
    session_candidates: Sequence[Dict[str, Any]],
    turn_records: Sequence[Dict[str, Any]],
    session_records: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    turn_metric_counts = Counter(
        metric_name
        for record in turn_candidates
        for metric_name in list(record.get("low_score_metrics") or [])
        if str(metric_name).strip()
    )
    session_metric_counts = Counter(
        metric_name
        for record in session_candidates
        for metric_name in list(record.get("low_score_metrics") or [])
        if str(metric_name).strip()
    )

    turn_metric_to_primary: Dict[str, Counter] = defaultdict(Counter)
    for record in _successful(turn_records):
        primary = str(record.get("primary_error_category", "") or "")
        for metric_name in list(record.get("affected_metrics") or []):
            if primary and str(metric_name).strip():
                turn_metric_to_primary[str(metric_name)][primary] += 1

    session_metric_to_primary: Dict[str, Counter] = defaultdict(Counter)
    for record in _successful(session_records):
        primary = str(record.get("primary_error_category", "") or "")
        for metric_name in list(record.get("affected_metrics") or []):
            if primary and str(metric_name).strip():
                session_metric_to_primary[str(metric_name)][primary] += 1

    return {
        "turn_low_metric_counts": _counter_to_dict(turn_metric_counts),
        "session_low_metric_counts": _counter_to_dict(session_metric_counts),
        "turn_metric_to_primary_error_counts": {
            metric_name: _counter_to_dict(counter)
            for metric_name, counter in sorted(turn_metric_to_primary.items())
        },
        "session_metric_to_primary_error_counts": {
            metric_name: _counter_to_dict(counter)
            for metric_name, counter in sorted(session_metric_to_primary.items())
        },
    }



def build_phase4_validation_summary(
    expected_turn_candidates: int,
    expected_session_candidates: int,
    turn_records: Sequence[Dict[str, Any]],
    session_records: Sequence[Dict[str, Any]],
    error_records: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    return {
        "expected_turn_candidate_count": int(expected_turn_candidates),
        "expected_session_candidate_count": int(expected_session_candidates),
        "actual_turn_record_count": len(turn_records),
        "actual_session_record_count": len(session_records),
        "turn_status_counts": _status_counts(turn_records),
        "session_status_counts": _status_counts(session_records),
        "error_record_count": len(error_records),
        "turn_coverage_complete": len(turn_records) >= int(expected_turn_candidates),
        "session_coverage_complete": len(session_records) >= int(expected_session_candidates),
    }

def _task_counts(records: Sequence[Dict[str, Any]]) -> Dict[str, int]:
    return _counter_to_dict(Counter(str(record.get("task_name", "") or "unknown") for record in records))



def _task_mode_counts(records: Sequence[Dict[str, Any]]) -> Dict[str, int]:
    return _counter_to_dict(Counter(str(record.get("task_mode", "") or "unknown") for record in records))



def _question_mode_counts(records: Sequence[Dict[str, Any]]) -> Dict[str, int]:
    return _counter_to_dict(Counter(str(record.get("question_mode", "") or "unknown") for record in records))



def _media_mode_counts(records: Sequence[Dict[str, Any]]) -> Dict[str, int]:
    return _counter_to_dict(Counter(str(record.get("media_mode", "") or "unknown") for record in records))



def _primary_error_counter(records: Sequence[Dict[str, Any]]) -> Counter:
    return Counter(
        str(record.get("primary_error_category", "") or "")
        for record in _successful(records)
        if str(record.get("primary_error_category", "") or "")
    )



def _secondary_error_counter(records: Sequence[Dict[str, Any]]) -> Counter:
    return Counter(
        secondary
        for record in _successful(records)
        for secondary in list(record.get("secondary_error_categories") or [])
        if str(secondary).strip()
    )



def _low_metric_counter(records: Sequence[Dict[str, Any]]) -> Counter:
    return Counter(
        str(metric_name)
        for record in records
        for metric_name in list(record.get("low_score_metrics") or [])
        if str(metric_name).strip()
    )



def _severity_counts(records: Sequence[Dict[str, Any]]) -> Dict[str, int]:
    return _counter_to_dict(Counter(str(record.get("severity", "") or "unknown") for record in records))



def _safe_rate(numerator: int, denominator: int) -> float:
    return _round_score(numerator / denominator) if denominator else 0.0



def _error_distribution_block(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    success_records = _successful(records)
    return {
        "candidate_count": len(records),
        "success_count": len(success_records),
        "success_rate": _safe_rate(len(success_records), len(records)),
        "status_counts": _status_counts(records),
        "severity_counts": _severity_counts(records),
        "primary_error_category_counts": _counter_to_dict(_primary_error_counter(records)),
        "secondary_error_category_counts": _counter_to_dict(_secondary_error_counter(records)),
        "low_metric_counts": _counter_to_dict(_low_metric_counter(records)),
    }



def _paired_error_distribution_summary(
    turn_groups: Dict[str, List[Dict[str, Any]]],
    session_groups: Dict[str, List[Dict[str, Any]]],
) -> Dict[str, Any]:
    keys = sorted(set(turn_groups) | set(session_groups))
    payload: Dict[str, Any] = {}
    for key in keys:
        payload[key] = {
            "turn": _error_distribution_block(turn_groups.get(key, [])),
            "session": _error_distribution_block(session_groups.get(key, [])),
        }
    return payload



def build_error_reason_by_task_summary(
    turn_records: Sequence[Dict[str, Any]],
    session_records: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    return {
        "record_type": "phase4_error_reason_by_task_summary",
        "overall": {
            "turn": _error_distribution_block(turn_records),
            "session": _error_distribution_block(session_records),
        },
        "by_task": _paired_error_distribution_summary(
            _group_records(turn_records, "task_name"),
            _group_records(session_records, "task_name"),
        ),
    }



def build_error_reason_by_mode_summary(
    turn_records: Sequence[Dict[str, Any]],
    session_records: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    return {
        "record_type": "phase4_error_reason_by_mode_summary",
        "by_task_mode": _paired_error_distribution_summary(
            _group_records(turn_records, "task_mode"),
            _group_records(session_records, "task_mode"),
        ),
        "by_question_mode": _paired_error_distribution_summary(
            _group_records(turn_records, "question_mode"),
            _group_records(session_records, "question_mode"),
        ),
        "by_media_mode": _paired_error_distribution_summary(
            _group_records(turn_records, "media_mode"),
            _group_records(session_records, "media_mode"),
        ),
    }



def build_error_reason_by_ability_summary(turn_records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    by_primary = _group_records(turn_records, "primary_category")
    by_secondary = _multi_group_records(turn_records, "secondary_categories")

    primary_payload: Dict[str, Any] = {}
    for category_name, records in sorted(by_primary.items()):
        primary_payload[category_name] = {
            **_error_distribution_block(records),
            "task_counts": _task_counts(records),
            "task_mode_counts": _task_mode_counts(records),
            "question_mode_counts": _question_mode_counts(records),
            "media_mode_counts": _media_mode_counts(records),
            "secondary_category_counts": _counter_to_dict(
                Counter(
                    str(secondary)
                    for record in records
                    for secondary in list(record.get("secondary_categories") or [])
                    if str(secondary).strip()
                )
            ),
        }

    secondary_payload: Dict[str, Any] = {}
    for category_name, records in sorted(by_secondary.items()):
        secondary_payload[category_name] = {
            **_error_distribution_block(records),
            "task_counts": _task_counts(records),
            "task_mode_counts": _task_mode_counts(records),
            "question_mode_counts": _question_mode_counts(records),
            "media_mode_counts": _media_mode_counts(records),
            "primary_category_counts": _counter_to_dict(Counter(str(record.get("primary_category", "") or "unknown") for record in records)),
        }

    return {
        "record_type": "phase4_error_reason_by_ability_summary",
        "overall": _error_distribution_block(turn_records),
        "by_primary_category": primary_payload,
        "by_secondary_category": secondary_payload,
    }

HIGH_SCORE_AVG_THRESHOLD = 4.75
HIGH_SCORE_METRIC_FLOOR = 4.0
LOW_TURN_CASE_LIMIT = 12
LOW_SESSION_CASE_LIMIT = 8
HIGH_TURN_CASE_LIMIT = 12
HIGH_SESSION_CASE_LIMIT = 8



def _shorten_text(value: Any, limit: int = 160) -> str:
    text = str(value or "").strip().replace("\r", " ").replace("\n", " ")
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)] + "..."



def _score_vector_min(record: Dict[str, Any]) -> float:
    score_vector = record.get("score_vector") or {}
    scores = [_numeric_score(value) for value in score_vector.values()]
    numeric = [score for score in scores if score is not None]
    return min(numeric) if numeric else 0.0



def _is_high_score_record(
    record: Dict[str, Any],
    metric_names: Sequence[str],
    *,
    avg_threshold: float = HIGH_SCORE_AVG_THRESHOLD,
    metric_floor: float = HIGH_SCORE_METRIC_FLOOR,
) -> bool:
    if str(record.get("status", "") or "") != "success":
        return False
    avg_score = _numeric_score(record.get("avg_score"))
    if avg_score is None or avg_score < avg_threshold:
        return False
    score_vector = record.get("score_vector") or {}
    for metric_name in metric_names:
        score = _numeric_score(score_vector.get(metric_name))
        if score is None or score < metric_floor:
            return False
    return True



def build_metric_error_cross_summary(
    turn_records: Sequence[Dict[str, Any]],
    session_records: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    def _build(records: Sequence[Dict[str, Any]], metric_names: Sequence[str], *, include_ability: bool) -> Dict[str, Any]:
        success_records = _successful(records)
        payload: Dict[str, Any] = {}
        for metric_name in metric_names:
            matched = [record for record in success_records if metric_name in list(record.get("affected_metrics") or [])]
            item = {
                "record_count": len(matched),
                "primary_error_category_counts": _counter_to_dict(_primary_error_counter(matched)),
                "secondary_error_category_counts": _counter_to_dict(_secondary_error_counter(matched)),
                "task_counts": _task_counts(matched),
                "task_mode_counts": _task_mode_counts(matched),
                "question_mode_counts": _question_mode_counts(matched),
                "media_mode_counts": _media_mode_counts(matched),
                "sample_candidate_ids": [str(record.get("candidate_id", "") or "") for record in matched[:10]],
            }
            if include_ability:
                item["primary_category_counts"] = _counter_to_dict(Counter(str(record.get("primary_category", "") or "unknown") for record in matched))
                item["secondary_category_counts"] = _counter_to_dict(
                    Counter(
                        str(secondary)
                        for record in matched
                        for secondary in list(record.get("secondary_categories") or [])
                        if str(secondary).strip()
                    )
                )
            payload[metric_name] = item
        return payload

    return {
        "record_type": "phase4_metric_error_cross_summary",
        "turn": _build(turn_records, TURN_METRICS, include_ability=True),
        "session": _build(session_records, SESSION_METRICS, include_ability=False),
    }



def build_high_score_summary(
    turn_phase3_records: Sequence[Dict[str, Any]],
    session_phase3_records: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    high_turn_records = [record for record in _successful(turn_phase3_records) if _is_high_score_record(record, TURN_METRICS)]
    high_session_records = [record for record in _successful(session_phase3_records) if _is_high_score_record(record, SESSION_METRICS)]

    def _block(records: Sequence[Dict[str, Any]], all_records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        success_records = _successful(all_records)
        avg_scores = [_numeric_score(record.get("avg_score")) or 0.0 for record in records]
        return {
            "success_record_count": len(success_records),
            "high_score_count": len(records),
            "high_score_rate": _safe_rate(len(records), len(success_records)),
            "avg_score_threshold": HIGH_SCORE_AVG_THRESHOLD,
            "metric_floor": HIGH_SCORE_METRIC_FLOOR,
            "avg_score": _round_score(sum(avg_scores) / len(avg_scores)) if avg_scores else 0.0,
            "task_counts": _task_counts(records),
            "task_mode_counts": _task_mode_counts(records),
            "question_mode_counts": _question_mode_counts(records),
            "media_mode_counts": _media_mode_counts(records),
        }

    return {
        "record_type": "phase4_high_score_summary",
        "overall": {
            "turn": {
                **_block(high_turn_records, turn_phase3_records),
                "primary_category_counts": _counter_to_dict(Counter(str(record.get("primary_category", "") or "unknown") for record in high_turn_records)),
                "secondary_category_counts": _counter_to_dict(
                    Counter(
                        str(secondary)
                        for record in high_turn_records
                        for secondary in list(record.get("secondary_categories") or [])
                        if str(secondary).strip()
                    )
                ),
            },
            "session": _block(high_session_records, session_phase3_records),
        },
        "by_task": {
            key: {
                "turn_high_score_count": len(value),
                "session_high_score_count": len(_group_records(high_session_records, "task_name").get(key, [])),
            }
            for key, value in sorted(_group_records(high_turn_records, "task_name").items())
        },
    }


def _build_low_turn_case(record: Dict[str, Any], candidate: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    merged = dict(candidate or {})
    merged.update(record)
    return {
        "candidate_id": str(merged.get("candidate_id", "") or ""),
        "dialogue_id": str(merged.get("dialogue_id", "") or ""),
        "round_id": str(merged.get("round_id", "") or ""),
        "round_index": int(merged.get("round_index", 0) or 0),
        "task_name": str(merged.get("task_name", "") or ""),
        "task_mode": str(merged.get("task_mode", "") or ""),
        "question_mode": str(merged.get("question_mode", "") or ""),
        "media_mode": str(merged.get("media_mode", "") or ""),
        "primary_category": str(merged.get("primary_category", "") or ""),
        "secondary_categories": list(merged.get("secondary_categories") or []),
        "severity": str(record.get("severity", "") or ""),
        "avg_score": _numeric_score(merged.get("avg_score")) or 0.0,
        "low_score_metrics": list(merged.get("low_score_metrics") or []),
        "primary_error_category": str(merged.get("primary_error_category", "") or ""),
        "secondary_error_categories": list(merged.get("secondary_error_categories") or []),
        "attribution_summary": _shorten_text(merged.get("attribution_summary", ""), 220),
        "overall_summary": _shorten_text(merged.get("overall_summary", ""), 220),
        "question_text": _shorten_text(merged.get("question_text", ""), 220),
        "reference_answer": _shorten_text(merged.get("reference_answer", ""), 220),
        "prediction": _shorten_text(merged.get("prediction", ""), 220),
    }


def _build_low_session_case(record: Dict[str, Any], candidate: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    merged = dict(candidate or {})
    merged.update(record)
    return {
        "candidate_id": str(merged.get("candidate_id", "") or ""),
        "dialogue_id": str(merged.get("dialogue_id", "") or ""),
        "task_name": str(merged.get("task_name", "") or ""),
        "task_mode": str(merged.get("task_mode", "") or ""),
        "question_mode": str(merged.get("question_mode", "") or ""),
        "media_mode": str(merged.get("media_mode", "") or ""),
        "severity": str(record.get("severity", "") or ""),
        "avg_score": _numeric_score(merged.get("avg_score")) or 0.0,
        "low_score_metrics": list(merged.get("low_score_metrics") or []),
        "primary_error_category": str(merged.get("primary_error_category", "") or ""),
        "secondary_error_categories": list(merged.get("secondary_error_categories") or []),
        "attribution_summary": _shorten_text(merged.get("attribution_summary", ""), 260),
        "overall_summary": _shorten_text(merged.get("overall_summary", ""), 260),
        "key_dialogue_signals": [_shorten_text(item, 120) for item in list(merged.get("key_dialogue_signals") or [])[:5]],
    }


def _build_high_turn_case(record: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "judgement_id": str(record.get("judgement_id", "") or ""),
        "dialogue_id": str(record.get("dialogue_id", "") or ""),
        "round_id": str(record.get("round_id", "") or ""),
        "round_index": int(record.get("round_index", 0) or 0),
        "task_name": str(record.get("task_name", "") or ""),
        "task_mode": str(record.get("task_mode", "") or ""),
        "question_mode": str(record.get("question_mode", "") or ""),
        "media_mode": str(record.get("media_mode", "") or ""),
        "primary_category": str(record.get("primary_category", "") or ""),
        "secondary_categories": list(record.get("secondary_categories") or []),
        "avg_score": _numeric_score(record.get("avg_score")) or 0.0,
        "score_vector": dict(record.get("score_vector") or {}),
        "overall_summary": _shorten_text(record.get("overall_summary", ""), 220),
        "question_text": _shorten_text(record.get("question_text", ""), 220),
        "reference_answer": _shorten_text(record.get("reference_answer", ""), 220),
        "prediction": _shorten_text(record.get("prediction", ""), 220),
    }



def _build_high_session_case(record: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "judgement_id": str(record.get("judgement_id", "") or ""),
        "dialogue_id": str(record.get("dialogue_id", "") or ""),
        "task_name": str(record.get("task_name", "") or ""),
        "task_mode": str(record.get("task_mode", "") or ""),
        "question_mode": str(record.get("question_mode", "") or ""),
        "media_mode": str(record.get("media_mode", "") or ""),
        "avg_score": _numeric_score(record.get("avg_score")) or 0.0,
        "score_vector": dict(record.get("score_vector") or {}),
        "overall_summary": _shorten_text(record.get("overall_summary", ""), 260),
        "key_dialogue_signals": [_shorten_text(item, 120) for item in list(record.get("key_dialogue_signals") or [])[:5]],
    }



def _select_diverse_top_records(records: Sequence[Dict[str, Any]], key_name: str, limit: int) -> List[Dict[str, Any]]:
    if not records or limit <= 0:
        return []
    selected: List[Dict[str, Any]] = []
    seen_keys: set[str] = set()
    for record in records:
        key = str(record.get(key_name, "") or "unknown")
        if key not in seen_keys:
            selected.append(record)
            seen_keys.add(key)
        if len(selected) >= limit:
            return selected
    for record in records:
        if record not in selected:
            selected.append(record)
        if len(selected) >= limit:
            break
    return selected



def build_representative_cases(
    turn_phase3_records: Sequence[Dict[str, Any]],
    session_phase3_records: Sequence[Dict[str, Any]],
    turn_candidates: Sequence[Dict[str, Any]],
    session_candidates: Sequence[Dict[str, Any]],
    turn_records: Sequence[Dict[str, Any]],
    session_records: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    low_turn_records = sorted(
        _successful(turn_records),
        key=lambda record: (
            0 if str(record.get("severity", "") or "") == "critical" else 1,
            _numeric_score(record.get("avg_score")) if _numeric_score(record.get("avg_score")) is not None else 999.0,
            -len(list(record.get("low_score_metrics") or [])),
        ),
    )
    low_session_records = sorted(
        _successful(session_records),
        key=lambda record: (
            0 if str(record.get("severity", "") or "") == "critical" else 1,
            _numeric_score(record.get("avg_score")) if _numeric_score(record.get("avg_score")) is not None else 999.0,
            -len(list(record.get("low_score_metrics") or [])),
        ),
    )
    high_turn_records = sorted(
        [record for record in _successful(turn_phase3_records) if _is_high_score_record(record, TURN_METRICS)],
        key=lambda record: (
            -(_numeric_score(record.get("avg_score")) or 0.0),
            -_score_vector_min(record),
        ),
    )
    high_session_records = sorted(
        [record for record in _successful(session_phase3_records) if _is_high_score_record(record, SESSION_METRICS)],
        key=lambda record: (
            -(_numeric_score(record.get("avg_score")) or 0.0),
            -_score_vector_min(record),
        ),
    )

    selected_low_turn = _select_diverse_top_records(low_turn_records, "task_name", LOW_TURN_CASE_LIMIT)
    selected_low_session = _select_diverse_top_records(low_session_records, "task_name", LOW_SESSION_CASE_LIMIT)
    turn_candidate_map = {str(record.get("candidate_id", "") or ""): record for record in turn_candidates}
    session_candidate_map = {str(record.get("candidate_id", "") or ""): record for record in session_candidates}
    selected_high_turn = _select_diverse_top_records(high_turn_records, "task_name", HIGH_TURN_CASE_LIMIT)
    selected_high_session = _select_diverse_top_records(high_session_records, "task_name", HIGH_SESSION_CASE_LIMIT)

    return {
        "record_type": "phase4_representative_cases",
        "thresholds": {
            "high_score_avg_threshold": HIGH_SCORE_AVG_THRESHOLD,
            "high_score_metric_floor": HIGH_SCORE_METRIC_FLOOR,
        },
        "low_turn_cases": [
            _build_low_turn_case(record, turn_candidate_map.get(str(record.get("candidate_id", "") or "")))
            for record in selected_low_turn
        ],
        "low_session_cases": [
            _build_low_session_case(record, session_candidate_map.get(str(record.get("candidate_id", "") or "")))
            for record in selected_low_session
        ],
        "high_turn_cases": [_build_high_turn_case(record) for record in selected_high_turn],
        "high_session_cases": [_build_high_session_case(record) for record in selected_high_session],
    }
