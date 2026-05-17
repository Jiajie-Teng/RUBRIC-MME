from __future__ import annotations

import json
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

PHASE3_SCHEMA_VERSION = "rubric_mme_phase3_v2"

TURN_METRICS: Tuple[str, ...] = (
    "accuracy",
    "completeness",
    "relevance",
    "conciseness",
    "naturalness",
    "proactiveness_helpfulness",
    "intent_understanding_depth",
    "user_state_adaptation",
)

SESSION_METRICS: Tuple[str, ...] = (
    "session_consistency",
    "intent_fulfillment",
    "persona_adaptation",
    "overall_helpfulness_trustworthiness",
)

TURN_METRIC_DEFINITIONS: Dict[str, str] = {
    "accuracy": "Whether the answer is factually accurate for the current round.",
    "completeness": "Whether the answer covers the key information the user needs in the current round.",
    "relevance": "Whether the answer stays focused on the current round question and context.",
    "conciseness": "Whether the answer is concise and avoids unnecessary information.",
    "naturalness": "Whether the answer sounds natural and fluent in Chinese.",
    "proactiveness_helpfulness": "Whether the answer proactively provides useful extra help without drifting off-topic.",
    "intent_understanding_depth": "Whether the answer captures the deeper user need based only on visible dialogue and media evidence.",
    "user_state_adaptation": "Whether the answer adapts to explicit user state cues visible in the dialogue.",
}

SESSION_METRIC_DEFINITIONS: Dict[str, str] = {
    "session_consistency": "Whether the model stays consistent across the whole dialogue.",
    "intent_fulfillment": "Whether the whole dialogue helps fulfill the interaction goal.",
    "persona_adaptation": "Whether the model adapts to explicit user needs and state changes across the dialogue.",
    "overall_helpfulness_trustworthiness": "Whether the whole dialogue feels helpful and trustworthy.",
}

JSON_FENCE_PATTERN = re.compile(r"```(?:json)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)


def build_metric_object_schema(description: str) -> Dict[str, Any]:
    return {
        "type": "object",
        "description": description,
        "properties": {
            "score": {
                "type": "integer",
                "minimum": 1,
                "maximum": 5,
                "description": "Integer score from 1 to 5.",
            },
            "reason": {
                "type": "string",
                "description": "Short explanation for the score.",
            },
        },
        "required": ["score", "reason"],
    }


def build_scores_schema(metric_definitions: Dict[str, str]) -> Dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            metric_name: build_metric_object_schema(metric_description)
            for metric_name, metric_description in metric_definitions.items()
        },
        "required": list(metric_definitions.keys()),
    }


def build_turn_response_schema() -> Dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "scores": build_scores_schema(TURN_METRIC_DEFINITIONS),
            "overall_summary": {
                "type": "string",
                "description": "Brief overall summary for the current round.",
            },
            "noted_user_state_cues": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Explicit user-state cues noticed from the dialogue.",
            },
        },
        "required": ["scores", "overall_summary"],
    }


def build_session_response_schema() -> Dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "scores": build_scores_schema(SESSION_METRIC_DEFINITIONS),
            "overall_summary": {
                "type": "string",
                "description": "Brief overall summary for the whole dialogue.",
            },
            "key_dialogue_signals": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Key signals that support the dialogue-level judgement.",
            },
        },
        "required": ["scores", "overall_summary"],
    }


def strip_json_fence(text: str) -> str:
    match = JSON_FENCE_PATTERN.search(text)
    if match:
        return match.group(1).strip()
    return text.strip()


def extract_first_json_object(text: str) -> str:
    stripped = strip_json_fence(text)
    if not stripped:
        return ""
    if stripped.startswith("{") and stripped.endswith("}"):
        return stripped

    start = stripped.find("{")
    if start < 0:
        return stripped

    depth = 0
    in_string = False
    escape = False
    for index in range(start, len(stripped)):
        char = stripped[index]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return stripped[start : index + 1]
    return stripped


def load_json_object(text: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    candidate = extract_first_json_object(text)
    if not candidate:
        return None, "empty_response"
    try:
        payload = json.loads(candidate)
    except json.JSONDecodeError as exc:
        return None, f"json_decode_error:{exc}"
    if not isinstance(payload, dict):
        return None, f"json_root_not_object:{type(payload).__name__}"
    return payload, None


def coerce_score(value: Any) -> Optional[int]:
    if isinstance(value, bool):
        return None
    try:
        score = int(value)
    except Exception:
        return None
    if 1 <= score <= 5:
        return score
    return None


def normalize_metric_block(metric_name: str, payload: Any) -> Tuple[Dict[str, Any], List[str]]:
    issues: List[str] = []
    if not isinstance(payload, dict):
        issues.append(f"{metric_name}:not_object")
        return {"score": None, "reason": ""}, issues

    score = coerce_score(payload.get("score"))
    reason = str(payload.get("reason", "") or "").strip()
    if score is None:
        issues.append(f"{metric_name}:invalid_score")
    if not reason:
        issues.append(f"{metric_name}:missing_reason")
    return {"score": score, "reason": reason}, issues


def average_scores(score_values: Iterable[Optional[int]]) -> Optional[float]:
    valid = [float(score) for score in score_values if isinstance(score, int)]
    if not valid:
        return None
    return round(sum(valid) / len(valid), 4)


def parse_metric_payload(
    payload: Dict[str, Any],
    metric_names: Sequence[str],
    *,
    extra_list_key: str,
) -> Dict[str, Any]:
    issues: List[str] = []
    scores_payload = payload.get("scores")
    if not isinstance(scores_payload, dict):
        issues.append("scores:not_object")
        scores_payload = {}

    normalized_scores: Dict[str, Dict[str, Any]] = {}
    score_vector: Dict[str, Optional[int]] = {}
    reason_vector: Dict[str, str] = {}

    for metric_name in metric_names:
        normalized_block, block_issues = normalize_metric_block(metric_name, scores_payload.get(metric_name))
        normalized_scores[metric_name] = normalized_block
        score_vector[metric_name] = normalized_block["score"]
        reason_vector[metric_name] = normalized_block["reason"]
        issues.extend(block_issues)

    overall_summary = str(payload.get("overall_summary", "") or "").strip()
    if not overall_summary:
        issues.append("overall_summary:missing")

    extra_values = payload.get(extra_list_key, [])
    if isinstance(extra_values, list):
        normalized_extra = [str(item).strip() for item in extra_values if str(item).strip()]
    else:
        normalized_extra = []
        if extra_values not in ("", None):
            issues.append(f"{extra_list_key}:not_list")

    avg_score = average_scores(score_vector.values())
    return {
        "scores": normalized_scores,
        "score_vector": score_vector,
        "reason_vector": reason_vector,
        "overall_summary": overall_summary,
        extra_list_key: normalized_extra,
        "avg_score": avg_score,
        "parse_issues": issues,
        "parse_success": len(issues) == 0,
    }


def parse_turn_judgement_text(raw_text: str) -> Dict[str, Any]:
    payload, load_error = load_json_object(raw_text)
    if payload is None:
        return {
            "schema_version": PHASE3_SCHEMA_VERSION,
            "parse_success": False,
            "parse_error": load_error or "unknown_parse_error",
            "scores": {metric_name: {"score": None, "reason": ""} for metric_name in TURN_METRICS},
            "score_vector": {metric_name: None for metric_name in TURN_METRICS},
            "reason_vector": {metric_name: "" for metric_name in TURN_METRICS},
            "overall_summary": "",
            "noted_user_state_cues": [],
            "avg_score": None,
            "parse_issues": [load_error or "unknown_parse_error"],
        }
    normalized = parse_metric_payload(payload, TURN_METRICS, extra_list_key="noted_user_state_cues")
    normalized["schema_version"] = PHASE3_SCHEMA_VERSION
    normalized["parse_error"] = "" if normalized["parse_success"] else "invalid_turn_schema"
    return normalized


def parse_session_judgement_text(raw_text: str) -> Dict[str, Any]:
    payload, load_error = load_json_object(raw_text)
    if payload is None:
        return {
            "schema_version": PHASE3_SCHEMA_VERSION,
            "parse_success": False,
            "parse_error": load_error or "unknown_parse_error",
            "scores": {metric_name: {"score": None, "reason": ""} for metric_name in SESSION_METRICS},
            "score_vector": {metric_name: None for metric_name in SESSION_METRICS},
            "reason_vector": {metric_name: "" for metric_name in SESSION_METRICS},
            "overall_summary": "",
            "key_dialogue_signals": [],
            "avg_score": None,
            "parse_issues": [load_error or "unknown_parse_error"],
        }
    normalized = parse_metric_payload(payload, SESSION_METRICS, extra_list_key="key_dialogue_signals")
    normalized["schema_version"] = PHASE3_SCHEMA_VERSION
    normalized["parse_error"] = "" if normalized["parse_success"] else "invalid_session_schema"
    return normalized


