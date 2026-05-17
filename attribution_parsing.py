from __future__ import annotations

from typing import Any, Dict, List, Sequence

from error_taxonomy import (
    SESSION_ERROR_CATEGORIES_CN,
    TURN_ERROR_CATEGORIES_CN,
    valid_secondary_categories,
)
from judge_parsing import load_json_object

PHASE4_SCHEMA_VERSION = "rubric_mme_phase4_v1"

TURN_AFFECTED_METRICS = (
    "accuracy",
    "completeness",
    "relevance",
    "conciseness",
    "naturalness",
    "proactiveness_helpfulness",
    "intent_understanding_depth",
    "user_state_adaptation",
)

SESSION_AFFECTED_METRICS = (
    "session_consistency",
    "intent_fulfillment",
    "persona_adaptation",
    "overall_helpfulness_trustworthiness",
)


def build_turn_attribution_response_schema() -> Dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "primary_error_category": {"type": "string", "enum": list(TURN_ERROR_CATEGORIES_CN.keys())},
            "secondary_error_categories": {
                "type": "array",
                "items": {"type": "string", "enum": valid_secondary_categories(TURN_ERROR_CATEGORIES_CN)},
                "minItems": 1,
                "maxItems": 3,
            },
            "affected_metrics": {
                "type": "array",
                "items": {"type": "string", "enum": list(TURN_AFFECTED_METRICS)},
                "minItems": 1,
                "maxItems": 4,
            },
            "attribution_summary": {"type": "string"},
        },
        "required": [
            "primary_error_category",
            "secondary_error_categories",
            "affected_metrics",
            "attribution_summary",
        ],
    }



def build_session_attribution_response_schema() -> Dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "primary_error_category": {"type": "string", "enum": list(SESSION_ERROR_CATEGORIES_CN.keys())},
            "secondary_error_categories": {
                "type": "array",
                "items": {"type": "string", "enum": valid_secondary_categories(SESSION_ERROR_CATEGORIES_CN)},
                "minItems": 1,
                "maxItems": 3,
            },
            "affected_metrics": {
                "type": "array",
                "items": {"type": "string", "enum": list(SESSION_AFFECTED_METRICS)},
                "minItems": 1,
                "maxItems": 4,
            },
            "attribution_summary": {"type": "string"},
            "improvement_focus": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 1,
                "maxItems": 3,
            },
        },
        "required": [
            "primary_error_category",
            "secondary_error_categories",
            "affected_metrics",
            "attribution_summary",
            "improvement_focus",
        ],
    }



def _normalize_string_list(value: Any) -> List[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]



def _validate_secondary_categories(
    primary_category: str,
    secondary_categories: Sequence[str],
    taxonomy: Dict[str, Dict[str, str]],
) -> List[str]:
    issues: List[str] = []
    allowed_secondaries = set((taxonomy.get(primary_category) or {}).keys())
    if not primary_category:
        issues.append("primary_error_category:missing")
        return issues
    if primary_category not in taxonomy:
        issues.append("primary_error_category:invalid")
    for secondary_category in secondary_categories:
        if secondary_category not in allowed_secondaries:
            issues.append(f"secondary_error_category:not_under_primary:{secondary_category}")
    return issues



def _validate_affected_metrics(affected_metrics: Sequence[str], allowed_metrics: Sequence[str]) -> List[str]:
    issues: List[str] = []
    allowed = set(allowed_metrics)
    if not affected_metrics:
        issues.append("affected_metrics:empty")
    for metric_name in affected_metrics:
        if metric_name not in allowed:
            issues.append(f"affected_metrics:invalid:{metric_name}")
    return issues



def parse_turn_attribution_text(raw_text: str) -> Dict[str, Any]:
    payload, load_error = load_json_object(raw_text)
    if payload is None:
        return {
            "schema_version": PHASE4_SCHEMA_VERSION,
            "parse_success": False,
            "parse_error": load_error or "unknown_parse_error",
            "parse_issues": [load_error or "unknown_parse_error"],
            "primary_error_category": "",
            "secondary_error_categories": [],
            "affected_metrics": [],
            "attribution_summary": "",
        }

    primary_category = str(payload.get("primary_error_category", "") or "").strip()
    secondary_categories = _normalize_string_list(payload.get("secondary_error_categories", []))
    affected_metrics = _normalize_string_list(payload.get("affected_metrics", []))
    attribution_summary = str(payload.get("attribution_summary", "") or "").strip()

    issues: List[str] = []
    issues.extend(_validate_secondary_categories(primary_category, secondary_categories, TURN_ERROR_CATEGORIES_CN))
    issues.extend(_validate_affected_metrics(affected_metrics, TURN_AFFECTED_METRICS))
    if not attribution_summary:
        issues.append("attribution_summary:missing")

    return {
        "schema_version": PHASE4_SCHEMA_VERSION,
        "parse_success": len(issues) == 0,
        "parse_error": "" if not issues else "invalid_turn_attribution_schema",
        "parse_issues": issues,
        "primary_error_category": primary_category,
        "secondary_error_categories": secondary_categories,
        "affected_metrics": affected_metrics,
        "attribution_summary": attribution_summary,
    }



def parse_session_attribution_text(raw_text: str) -> Dict[str, Any]:
    payload, load_error = load_json_object(raw_text)
    if payload is None:
        return {
            "schema_version": PHASE4_SCHEMA_VERSION,
            "parse_success": False,
            "parse_error": load_error or "unknown_parse_error",
            "parse_issues": [load_error or "unknown_parse_error"],
            "primary_error_category": "",
            "secondary_error_categories": [],
            "affected_metrics": [],
            "attribution_summary": "",
            "improvement_focus": [],
        }

    primary_category = str(payload.get("primary_error_category", "") or "").strip()
    secondary_categories = _normalize_string_list(payload.get("secondary_error_categories", []))
    affected_metrics = _normalize_string_list(payload.get("affected_metrics", []))
    attribution_summary = str(payload.get("attribution_summary", "") or "").strip()
    improvement_focus = _normalize_string_list(payload.get("improvement_focus", []))

    issues: List[str] = []
    issues.extend(_validate_secondary_categories(primary_category, secondary_categories, SESSION_ERROR_CATEGORIES_CN))
    issues.extend(_validate_affected_metrics(affected_metrics, SESSION_AFFECTED_METRICS))
    if not attribution_summary:
        issues.append("attribution_summary:missing")
    if not improvement_focus:
        issues.append("improvement_focus:empty")

    return {
        "schema_version": PHASE4_SCHEMA_VERSION,
        "parse_success": len(issues) == 0,
        "parse_error": "" if not issues else "invalid_session_attribution_schema",
        "parse_issues": issues,
        "primary_error_category": primary_category,
        "secondary_error_categories": secondary_categories,
        "affected_metrics": affected_metrics,
        "attribution_summary": attribution_summary,
        "improvement_focus": improvement_focus,
    }
