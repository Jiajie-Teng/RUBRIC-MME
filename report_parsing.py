from __future__ import annotations

import re
from typing import Any, Dict, List

_JSON_FENCE_PATTERN = re.compile(r"```(?:json)?\s*(\{.*\})\s*```", re.DOTALL)
_TRAILING_COMMA_PATTERN = re.compile(r",\s*([}\]])")
_STRAY_KEY_PREFIX_PATTERN = re.compile(r"(?m)^(\s*)[A-Za-z]\s+(?=\"[^\"]+\"\s*:)")

from judge_parsing import load_json_object

PHASE5_SCHEMA_VERSION = "rubric_mme_phase5_v4_refined"
_LIST_SPLIT_PATTERN = re.compile(r"[,;\n]+")


def _metric_finding_schema() -> Dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "metric": {"type": "string"},
            "assessment": {"type": "string"},
            "evidence": {"type": "string"},
            "common_errors": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["metric", "assessment", "evidence", "common_errors"],
    }


def _scoped_finding_schema() -> Dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "scope": {"type": "string"},
            "strengths": {"type": "string"},
            "weaknesses": {"type": "string"},
            "evidence": {"type": "string"},
        },
        "required": ["scope", "strengths", "weaknesses", "evidence"],
    }


def _ability_finding_schema() -> Dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "scope": {"type": "string"},
            "assessment": {"type": "string"},
            "common_errors": {"type": "array", "items": {"type": "string"}},
            "evidence": {"type": "string"},
        },
        "required": ["scope", "assessment", "common_errors", "evidence"],
    }


def _root_cause_schema() -> Dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "category": {"type": "string"},
            "explanation": {"type": "string"},
            "affected_metrics": {"type": "array", "items": {"type": "string"}},
            "affected_scopes": {"type": "array", "items": {"type": "string"}},
            "evidence": {"type": "string"},
        },
        "required": ["category", "explanation", "affected_metrics", "affected_scopes", "evidence"],
    }


def _case_block_schema() -> Dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "case_id": {"type": "string"},
            "why_representative": {"type": "string"},
            "lesson": {"type": "string"},
        },
        "required": ["case_id", "why_representative", "lesson"],
    }


def _recommendation_schema() -> Dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "priority": {"type": "string"},
            "title": {"type": "string"},
            "rationale": {"type": "string"},
            "actions": {"type": "array", "items": {"type": "string"}},
            "expected_gain": {"type": "string"},
            "target_metrics": {"type": "array", "items": {"type": "string"}},
            "target_scopes": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["priority", "title", "rationale", "actions", "expected_gain", "target_metrics", "target_scopes"],
    }


def build_report_step1_response_schema() -> Dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "executive_summary": {"type": "string"},
            "benchmark_overview": {
                "type": "object",
                "properties": {
                    "scope_summary": {"type": "string"},
                    "coverage_summary": {"type": "string"},
                    "evaluation_note": {"type": "string"},
                },
                "required": ["scope_summary", "coverage_summary", "evaluation_note"],
            },
            "overall_assessment": {
                "type": "object",
                "properties": {
                    "verdict": {"type": "string"},
                    "turn_level_summary": {"type": "string"},
                    "session_level_summary": {"type": "string"},
                    "strongest_signals": {"type": "array", "items": {"type": "string"}},
                    "weakest_signals": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["verdict", "turn_level_summary", "session_level_summary", "strongest_signals", "weakest_signals"],
            },
            "turn_level_findings": {"type": "array", "items": _metric_finding_schema()},
            "session_level_findings": {"type": "array", "items": _metric_finding_schema()},
        },
        "required": ["executive_summary", "benchmark_overview", "overall_assessment", "turn_level_findings", "session_level_findings"],
    }


def build_report_step2_scope_response_schema() -> Dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "task_findings": {"type": "array", "items": _scoped_finding_schema()},
            "mode_findings": {"type": "array", "items": _scoped_finding_schema()},
            "ability_findings": {"type": "array", "items": _ability_finding_schema()},
        },
        "required": ["task_findings", "mode_findings", "ability_findings"],
    }


def build_report_step3_causes_response_schema() -> Dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "root_causes": {"type": "array", "items": _root_cause_schema()},
            "representative_case_analysis": {
                "type": "object",
                "properties": {
                    "low_turn_cases": {"type": "array", "items": _case_block_schema()},
                    "low_session_cases": {"type": "array", "items": _case_block_schema()},
                    "high_turn_cases": {"type": "array", "items": _case_block_schema()},
                    "high_session_cases": {"type": "array", "items": _case_block_schema()},
                },
                "required": ["low_turn_cases", "low_session_cases", "high_turn_cases", "high_session_cases"],
            },
        },
        "required": ["root_causes", "representative_case_analysis"],
    }


def build_report_step4_recommendations_response_schema() -> Dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "recommendations": {"type": "array", "items": _recommendation_schema()},
            "priority_roadmap": {
                "type": "object",
                "properties": {
                    "p0": {"type": "array", "items": {"type": "string"}},
                    "p1": {"type": "array", "items": {"type": "string"}},
                    "p2": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["p0", "p1", "p2"],
            },
            "report_closing": {"type": "string"},
        },
        "required": ["recommendations", "priority_roadmap", "report_closing"],
    }


def _split_string_values(value: str) -> List[str]:
    return [part.strip() for part in _LIST_SPLIT_PATTERN.split(value) if part.strip()]


def _string_list(values: Any) -> List[str]:
    if values is None:
        return []
    if isinstance(values, list):
        result: List[str] = []
        for item in values:
            if isinstance(item, str):
                result.extend(_split_string_values(item))
            else:
                text = str(item).strip()
                if text:
                    result.append(text)
        return result
    if isinstance(values, str):
        return _split_string_values(values)
    text = str(values).strip()
    return [text] if text else []


def _text(value: Any) -> str:
    return str(value or "").strip()


def _normalize_metric_findings(values: Any) -> List[Dict[str, Any]]:
    result: List[Dict[str, Any]] = []
    if not isinstance(values, list):
        return result
    for item in values:
        if not isinstance(item, dict):
            continue
        metric = _text(item.get("metric"))
        assessment = _text(item.get("assessment"))
        evidence = _text(item.get("evidence"))
        common_errors = _string_list(item.get("common_errors", []))
        if metric and assessment and evidence:
            result.append({
                "metric": metric,
                "assessment": assessment,
                "evidence": evidence,
                "common_errors": common_errors,
            })
    return result


def _normalize_scoped_findings(values: Any) -> List[Dict[str, str]]:
    result: List[Dict[str, str]] = []
    if not isinstance(values, list):
        return result
    for item in values:
        if not isinstance(item, dict):
            continue
        scope = _text(item.get("scope"))
        strengths = _text(item.get("strengths"))
        weaknesses = _text(item.get("weaknesses"))
        evidence = _text(item.get("evidence"))
        if scope and (strengths or weaknesses or evidence):
            result.append({
                "scope": scope,
                "strengths": strengths,
                "weaknesses": weaknesses,
                "evidence": evidence,
            })
    return result


def _normalize_ability_findings(values: Any) -> List[Dict[str, Any]]:
    result: List[Dict[str, Any]] = []
    if not isinstance(values, list):
        return result
    for item in values:
        if not isinstance(item, dict):
            continue
        scope = _text(item.get("scope"))
        assessment = _text(item.get("assessment"))
        evidence = _text(item.get("evidence"))
        common_errors = _string_list(item.get("common_errors", []))
        if scope and assessment and evidence:
            result.append({
                "scope": scope,
                "assessment": assessment,
                "common_errors": common_errors,
                "evidence": evidence,
            })
    return result


def _normalize_root_causes(values: Any) -> List[Dict[str, Any]]:
    result: List[Dict[str, Any]] = []
    if not isinstance(values, list):
        return result
    for item in values:
        if not isinstance(item, dict):
            continue
        category = _text(item.get("category"))
        explanation = _text(item.get("explanation"))
        evidence = _text(item.get("evidence"))
        affected_metrics = _string_list(item.get("affected_metrics", []))
        affected_scopes = _string_list(item.get("affected_scopes", []))
        if category and explanation and evidence:
            result.append({
                "category": category,
                "explanation": explanation,
                "affected_metrics": affected_metrics,
                "affected_scopes": affected_scopes,
                "evidence": evidence,
            })
    return result


def _normalize_case_blocks(values: Any) -> List[Dict[str, str]]:
    result: List[Dict[str, str]] = []
    if not isinstance(values, list):
        return result
    for item in values:
        if not isinstance(item, dict):
            continue
        case_id = _text(item.get("case_id"))
        why_representative = _text(item.get("why_representative"))
        lesson = _text(item.get("lesson"))
        if case_id and why_representative and lesson:
            result.append({
                "case_id": case_id,
                "why_representative": why_representative,
                "lesson": lesson,
            })
    return result


def _normalize_representative_case_analysis(value: Any) -> Dict[str, List[Dict[str, str]]]:
    if not isinstance(value, dict):
        return {
            "low_turn_cases": [],
            "low_session_cases": [],
            "high_turn_cases": [],
            "high_session_cases": [],
        }
    return {
        "low_turn_cases": _normalize_case_blocks(value.get("low_turn_cases", [])),
        "low_session_cases": _normalize_case_blocks(value.get("low_session_cases", [])),
        "high_turn_cases": _normalize_case_blocks(value.get("high_turn_cases", [])),
        "high_session_cases": _normalize_case_blocks(value.get("high_session_cases", [])),
    }


def _normalize_recommendations(values: Any) -> List[Dict[str, Any]]:
    result: List[Dict[str, Any]] = []
    if not isinstance(values, list):
        return result
    for item in values:
        if not isinstance(item, dict):
            continue
        priority = _text(item.get("priority"))
        title = _text(item.get("title"))
        rationale = _text(item.get("rationale"))
        expected_gain = _text(item.get("expected_gain"))
        actions = _string_list(item.get("actions", []))
        target_metrics = _string_list(item.get("target_metrics", []))
        target_scopes = _string_list(item.get("target_scopes", []))
        if priority and title and rationale and expected_gain and actions:
            result.append({
                "priority": priority,
                "title": title,
                "rationale": rationale,
                "actions": actions,
                "expected_gain": expected_gain,
                "target_metrics": target_metrics,
                "target_scopes": target_scopes,
            })
    return result


def _normalize_priority_roadmap(value: Any) -> Dict[str, List[str]]:
    if not isinstance(value, dict):
        return {"p0": [], "p1": [], "p2": []}
    return {
        "p0": _string_list(value.get("p0", [])),
        "p1": _string_list(value.get("p1", [])),
        "p2": _string_list(value.get("p2", [])),
    }


def _extract_json_candidate(raw_text: str) -> str:
    text = _text(raw_text)
    if not text:
        return ""
    match = _JSON_FENCE_PATTERN.search(text)
    if match:
        return match.group(1).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        return text[start : end + 1].strip()
    return text



def _repair_json_candidate(text: str) -> str:
    repaired = (text or "").lstrip("﻿").strip()
    repaired = _STRAY_KEY_PREFIX_PATTERN.sub(r"\1", repaired)
    repaired = _TRAILING_COMMA_PATTERN.sub(r"\1", repaired)
    return repaired



def _parse_root_object(raw_text: str, *, allow_repair: bool = False) -> Dict[str, Any]:
    payload, load_error = load_json_object(raw_text)
    parse_issues: List[str] = []
    if load_error and allow_repair:
        candidate = _extract_json_candidate(raw_text)
        if candidate and candidate != _text(raw_text):
            repaired_payload, repaired_error = load_json_object(candidate)
            if not repaired_error:
                payload, load_error = repaired_payload, ""
                parse_issues.append("repair_applied:extracted_json_candidate")
            else:
                parse_issues.append(f"repair_failed:extracted_json_candidate:{repaired_error}")
        if load_error:
            normalized = _repair_json_candidate(candidate or raw_text)
            if normalized and normalized not in {_text(raw_text), candidate}:
                repaired_payload, repaired_error = load_json_object(normalized)
                if not repaired_error:
                    payload, load_error = repaired_payload, ""
                    parse_issues.append("repair_applied:normalized_json_candidate")
                else:
                    parse_issues.append(f"repair_failed:normalized_json_candidate:{repaired_error}")
    if load_error:
        return {
            "ok": False,
            "payload": {},
            "parse_error": load_error,
            "parse_issues": [load_error, *parse_issues],
        }
    if not isinstance(payload, dict):
        return {
            "ok": False,
            "payload": {},
            "parse_error": "Report analysis payload is not a JSON object.",
            "parse_issues": ["root_not_object", *parse_issues],
        }
    return {"ok": True, "payload": payload, "parse_error": "", "parse_issues": parse_issues}


def parse_report_step1_text(raw_text: str) -> Dict[str, Any]:
    root = _parse_root_object(raw_text)
    if not root["ok"]:
        return {"parse_success": False, "parse_error": root["parse_error"], "parse_issues": root["parse_issues"], "analysis": {}}
    payload = root["payload"]
    benchmark_overview = payload.get("benchmark_overview") if isinstance(payload.get("benchmark_overview"), dict) else {}
    overall_assessment = payload.get("overall_assessment") if isinstance(payload.get("overall_assessment"), dict) else {}
    analysis = {
        "executive_summary": _text(payload.get("executive_summary")),
        "benchmark_overview": {
            "scope_summary": _text(benchmark_overview.get("scope_summary")),
            "coverage_summary": _text(benchmark_overview.get("coverage_summary")),
            "evaluation_note": _text(benchmark_overview.get("evaluation_note")),
        },
        "overall_assessment": {
            "verdict": _text(overall_assessment.get("verdict")),
            "turn_level_summary": _text(overall_assessment.get("turn_level_summary")),
            "session_level_summary": _text(overall_assessment.get("session_level_summary")),
            "strongest_signals": _string_list(overall_assessment.get("strongest_signals", [])),
            "weakest_signals": _string_list(overall_assessment.get("weakest_signals", [])),
        },
        "turn_level_findings": _normalize_metric_findings(payload.get("turn_level_findings", [])),
        "session_level_findings": _normalize_metric_findings(payload.get("session_level_findings", [])),
    }
    parse_issues: List[str] = []
    if not analysis["executive_summary"]:
        parse_issues.append("missing_or_empty:executive_summary")
    for field in ["scope_summary", "coverage_summary", "evaluation_note"]:
        if not analysis["benchmark_overview"].get(field):
            parse_issues.append(f"missing_or_empty:benchmark_overview.{field}")
    for field in ["verdict", "turn_level_summary", "session_level_summary"]:
        if not analysis["overall_assessment"].get(field):
            parse_issues.append(f"missing_or_empty:overall_assessment.{field}")
    if len(analysis["turn_level_findings"]) < 2:
        parse_issues.append("missing_or_empty:turn_level_findings")
    if len(analysis["session_level_findings"]) < 2:
        parse_issues.append("missing_or_empty:session_level_findings")
    fatal_fields = {"executive_summary", "benchmark_overview.scope_summary", "overall_assessment.verdict", "turn_level_findings", "session_level_findings"}
    fatal_issues = [issue for issue in parse_issues if issue.split(":", 1)[-1] in fatal_fields]
    return {"parse_success": len(fatal_issues) == 0, "parse_error": "; ".join(parse_issues), "parse_issues": parse_issues, "analysis": analysis}


def parse_report_step2_scope_text(raw_text: str) -> Dict[str, Any]:
    root = _parse_root_object(raw_text)
    if not root["ok"]:
        return {"parse_success": False, "parse_error": root["parse_error"], "parse_issues": root["parse_issues"], "analysis": {}}
    payload = root["payload"]
    analysis = {
        "task_findings": _normalize_scoped_findings(payload.get("task_findings", [])),
        "mode_findings": _normalize_scoped_findings(payload.get("mode_findings", [])),
        "ability_findings": _normalize_ability_findings(payload.get("ability_findings", [])),
    }
    parse_issues: List[str] = []
    for field in ["task_findings", "mode_findings", "ability_findings"]:
        if len(analysis[field]) < 2:
            parse_issues.append(f"missing_or_empty:{field}")
    fatal_fields = {"task_findings", "mode_findings", "ability_findings"}
    fatal_issues = [issue for issue in parse_issues if issue.split(":", 1)[-1] in fatal_fields]
    return {"parse_success": len(fatal_issues) == 0, "parse_error": "; ".join(parse_issues), "parse_issues": parse_issues, "analysis": analysis}


def parse_report_step3_causes_text(raw_text: str) -> Dict[str, Any]:
    root = _parse_root_object(raw_text)
    if not root["ok"]:
        return {"parse_success": False, "parse_error": root["parse_error"], "parse_issues": root["parse_issues"], "analysis": {}}
    payload = root["payload"]
    analysis = {
        "root_causes": _normalize_root_causes(payload.get("root_causes", [])),
        "representative_case_analysis": _normalize_representative_case_analysis(payload.get("representative_case_analysis", {})),
    }
    parse_issues: List[str] = []
    if len(analysis["root_causes"]) < 2:
        parse_issues.append("missing_or_empty:root_causes")
    if not any(analysis["representative_case_analysis"].values()):
        parse_issues.append("missing_or_empty:representative_case_analysis")
    fatal_fields = {"root_causes", "representative_case_analysis"}
    fatal_issues = [issue for issue in parse_issues if issue.split(":", 1)[-1] in fatal_fields]
    return {"parse_success": len(fatal_issues) == 0, "parse_error": "; ".join(parse_issues), "parse_issues": parse_issues, "analysis": analysis}


def parse_report_step4_recommendations_text(raw_text: str) -> Dict[str, Any]:
    root = _parse_root_object(raw_text, allow_repair=True)
    if not root["ok"]:
        return {"parse_success": False, "parse_error": root["parse_error"], "parse_issues": root["parse_issues"], "analysis": {}}
    payload = root["payload"]
    analysis = {
        "recommendations": _normalize_recommendations(payload.get("recommendations", [])),
        "priority_roadmap": _normalize_priority_roadmap(payload.get("priority_roadmap", {})),
        "report_closing": _text(payload.get("report_closing")),
    }
    parse_issues: List[str] = []
    if len(analysis["recommendations"]) < 2:
        parse_issues.append("missing_or_empty:recommendations")
    if not any(analysis["priority_roadmap"].values()):
        parse_issues.append("missing_or_empty:priority_roadmap")
    if not analysis["report_closing"]:
        parse_issues.append("missing_or_empty:report_closing")
    fatal_fields = {"recommendations", "priority_roadmap", "report_closing"}
    fatal_issues = [issue for issue in parse_issues if issue.split(":", 1)[-1] in fatal_fields]
    return {"parse_success": len(fatal_issues) == 0, "parse_error": "; ".join(parse_issues), "parse_issues": parse_issues, "analysis": analysis}
