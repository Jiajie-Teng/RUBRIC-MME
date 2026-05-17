from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence, Set

from attribution_runner import build_text_only_request
from report_parsing import (
    PHASE5_SCHEMA_VERSION,
    build_report_step1_response_schema,
    build_report_step2_scope_response_schema,
    build_report_step3_causes_response_schema,
    build_report_step4_recommendations_response_schema,
    parse_report_step1_text,
    parse_report_step2_scope_text,
    parse_report_step3_causes_text,
    parse_report_step4_recommendations_text,
)
from report_prompts import REPORT_PROMPT_VERSION, build_report_step_prompt
from report_render import build_html_report, build_markdown_report

OUTPUT_FILENAMES = {
    "report_payload": "report_payload.json",
    "report_digest": "report_digest.json",
    "report_step_results": "report_step_results.json",
    "report_raw_text": "report_analysis_raw.txt",
    "report_analysis": "report_analysis.json",
    "report_markdown": "benchmark_report.md",
    "report_html": "benchmark_report.html",
    "manifest": "manifest.json",
}

STEP_SEQUENCE = (
    "step1_overview",
    "step2_scope_findings",
    "step3_root_causes_cases",
    "step4_recommendations",
)
STEP_LABELS = {
    "step1_overview": "第一步：整体表现分析",
    "step2_scope_findings": "第二步：任务、模态与能力分析",
    "step3_root_causes_cases": "第三步：根因与代表性案例分析",
    "step4_recommendations": "第四步：改进建议与路线图",
}
STEP_DEPENDENCIES = {
    "step1_overview": set(),
    "step2_scope_findings": set(),
    "step3_root_causes_cases": set(),
    "step4_recommendations": {"step1_overview", "step2_scope_findings", "step3_root_causes_cases"},
}
STEP_DOWNSTREAM = {
    "step1_overview": {"step4_recommendations"},
    "step2_scope_findings": {"step4_recommendations"},
    "step3_root_causes_cases": {"step4_recommendations"},
    "step4_recommendations": set(),
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def dump_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)


def dump_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _iter_jsonl_dicts(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            rows.append(payload)
    return rows


def _text(value: Any) -> str:
    return "" if value is None else str(value).strip()


def _trim_text(value: Any, *, limit: int = 220) -> str:
    text = _text(value)
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 1)].rstrip() + "…"


def _counter_items(counter_payload: Dict[str, Any], *, key_name: str, limit: int = 5) -> List[Dict[str, Any]]:
    items = [{key_name: name, "count": int(count)} for name, count in (counter_payload or {}).items()]
    items.sort(key=lambda item: (-item["count"], item[key_name]))
    return items[:limit]


def _top_metric_items(metric_summary: Dict[str, Any], *, reverse: bool, limit: int = 6) -> List[Dict[str, Any]]:
    items = []
    for metric_name, payload in metric_summary.items():
        items.append({
            "metric": metric_name,
            "count": int(payload.get("count", 0) or 0),
            "avg_score": float(payload.get("avg_score", 0.0) or 0.0),
            "low_score_rate": float(payload.get("low_score_rate", 0.0) or 0.0),
        })
    items.sort(key=lambda item: (item["avg_score"], -item["low_score_rate"], item["metric"]), reverse=reverse)
    return items[:limit]


def _top_group_items(group_summary: Dict[str, Any], *, key_name: str) -> List[Dict[str, Any]]:
    items = []
    for name, payload in group_summary.items():
        turn_payload = payload.get("turn", payload)
        overall = turn_payload.get("overall_avg_score", {})
        items.append({
            key_name: name,
            "avg_score": float(overall.get("avg_score", 0.0) or 0.0),
            "count": int(turn_payload.get("record_count", 0) or 0),
        })
    items.sort(key=lambda item: (item["avg_score"], item["count"], item[key_name]))
    return items


def _unique(values: Sequence[str]) -> List[str]:
    seen: Set[str] = set()
    result: List[str] = []
    for value in values:
        text = _text(value)
        if text and text not in seen:
            seen.add(text)
            result.append(text)
    return result


def _pick_names(items: Sequence[Dict[str, Any]], key_name: str, *, limit: int) -> List[str]:
    return _unique([str(item.get(key_name, "") or "") for item in list(items)[:limit]])


def _slice_mapping(mapping: Dict[str, Any], names: Sequence[str]) -> Dict[str, Any]:
    return {name: mapping[name] for name in names if name in mapping}


def _infer_tested_model_metadata(phase4_manifest: Dict[str, Any]) -> Dict[str, str]:
    phase2_dir_text = str(phase4_manifest.get("phase2_dir", "") or "").strip()
    if not phase2_dir_text:
        return {"tested_provider": "", "tested_model_name": ""}
    dialogues_path = Path(phase2_dir_text) / "dialogues.jsonl"
    rows = _iter_jsonl_dicts(dialogues_path)
    if not rows:
        return {"tested_provider": "", "tested_model_name": ""}
    first = rows[0]
    return {
        "tested_provider": str(first.get("provider", "") or ""),
        "tested_model_name": str(first.get("model_name", "") or ""),
    }


def _load_phase2_dialogue_map(phase4_manifest: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    phase2_dir_text = str(phase4_manifest.get("phase2_dir", "") or "").strip()
    if not phase2_dir_text:
        return {}
    dialogues_path = Path(phase2_dir_text) / "dialogues.jsonl"
    dialogue_map: Dict[str, Dict[str, Any]] = {}
    for payload in _iter_jsonl_dicts(dialogues_path):
        dialogue_id = _text(payload.get("dialogue_id"))
        if dialogue_id:
            dialogue_map[dialogue_id] = payload
    return dialogue_map


def _build_dialogue_preview_fields(dialogue_record: Dict[str, Any]) -> Dict[str, Any]:
    rounds = list(dialogue_record.get("rounds") or [])
    first_round = rounds[0] if rounds else {}
    last_round = rounds[-1] if rounds else {}
    preview: List[str] = []
    first_q = _trim_text(first_round.get("question_text"), limit=140)
    first_pred = _trim_text(first_round.get("prediction"), limit=140)
    last_q = _trim_text(last_round.get("question_text"), limit=140)
    last_pred = _trim_text(last_round.get("prediction"), limit=140)
    if first_q:
        preview.append(f"首轮问题：{first_q}")
    if first_pred:
        preview.append(f"首轮回答：{first_pred}")
    if len(rounds) > 1 and last_q and last_q != first_q:
        preview.append(f"末轮问题：{last_q}")
    if len(rounds) > 1 and last_pred and last_pred != first_pred:
        preview.append(f"末轮回答：{last_pred}")
    return {
        "question_text": _text(first_round.get("question_text")),
        "reference_answer": _text(first_round.get("reference_answer")),
        "prediction": _text(first_round.get("prediction")),
        "dialogue_preview": preview,
    }


def _normalize_representative_cases(representative_cases: Dict[str, Any], phase4_manifest: Dict[str, Any]) -> Dict[str, Any]:
    dialogue_map = _load_phase2_dialogue_map(phase4_manifest)
    normalized: Dict[str, Any] = {
        "record_type": representative_cases.get("record_type", "phase4_representative_cases"),
        "thresholds": representative_cases.get("thresholds", {}),
    }
    for group_name in ["low_turn_cases", "low_session_cases", "high_turn_cases", "high_session_cases"]:
        items: List[Dict[str, Any]] = []
        for item in list(representative_cases.get(group_name, []) or []):
            row = dict(item)
            row["case_id"] = _text(
                row.get("case_id")
                or row.get("candidate_id")
                or row.get("judgement_id")
                or row.get("dialogue_id")
                or row.get("round_id")
            )
            if group_name.endswith("session_cases"):
                preview_fields = _build_dialogue_preview_fields(dialogue_map.get(_text(row.get("dialogue_id")), {}))
                for field in ["question_text", "reference_answer", "prediction"]:
                    if not _text(row.get(field)):
                        row[field] = preview_fields.get(field, "")
                if not list(row.get("dialogue_preview") or []):
                    row["dialogue_preview"] = preview_fields.get("dialogue_preview", [])
            items.append(row)
        normalized[group_name] = items
    return normalized


def build_report_payload(phase4_dir: Path) -> Dict[str, Any]:
    manifest = load_json(phase4_dir / "manifest.json")
    score_summary = load_json(phase4_dir / "score_summary.json")
    ability_score_summary = load_json(phase4_dir / "ability_score_summary.json")
    error_category_summary = load_json(phase4_dir / "error_category_summary.json")
    task_error_summary = load_json(phase4_dir / "task_error_summary.json")
    metric_failure_summary = load_json(phase4_dir / "metric_failure_summary.json")
    turn_attribution_summary = load_json(phase4_dir / "turn_attribution_summary.json")
    session_attribution_summary = load_json(phase4_dir / "session_attribution_summary.json")
    error_reason_by_task_summary = load_json(phase4_dir / "error_reason_by_task_summary.json")
    error_reason_by_mode_summary = load_json(phase4_dir / "error_reason_by_mode_summary.json")
    error_reason_by_ability_summary = load_json(phase4_dir / "error_reason_by_ability_summary.json")
    metric_error_cross_summary = load_json(phase4_dir / "metric_error_cross_summary.json")
    high_score_summary = load_json(phase4_dir / "high_score_summary.json")
    representative_cases = _normalize_representative_cases(load_json(phase4_dir / "representative_cases.json"), manifest)
    validation_summary = load_json(phase4_dir / "validation_summary.json")
    tested_meta = _infer_tested_model_metadata(manifest)

    task_ranked = _top_group_items(score_summary.get("by_task", {}), key_name="task_name")
    task_mode_ranked = _top_group_items(score_summary.get("by_task_mode", {}), key_name="task_mode")
    question_mode_ranked = _top_group_items(score_summary.get("by_question_mode", {}), key_name="question_mode")
    media_mode_ranked = _top_group_items(score_summary.get("by_media_mode", {}), key_name="media_mode")
    primary_ranked = _top_group_items(ability_score_summary.get("by_primary_category", {}), key_name="ability_name")
    secondary_ranked = _top_group_items(ability_score_summary.get("by_secondary_category", {}), key_name="ability_name")

    return {
        "benchmark_name": "RUBRIC-MME",
        "tested_provider": tested_meta.get("tested_provider", ""),
        "tested_model_name": tested_meta.get("tested_model_name", ""),
        "phase4_manifest": manifest,
        "validation_summary": validation_summary,
        "score_summary": score_summary,
        "ability_score_summary": ability_score_summary,
        "error_category_summary": error_category_summary,
        "task_error_summary": task_error_summary,
        "metric_failure_summary": metric_failure_summary,
        "turn_attribution_summary": turn_attribution_summary,
        "session_attribution_summary": session_attribution_summary,
        "error_reason_by_task_summary": error_reason_by_task_summary,
        "error_reason_by_mode_summary": error_reason_by_mode_summary,
        "error_reason_by_ability_summary": error_reason_by_ability_summary,
        "metric_error_cross_summary": metric_error_cross_summary,
        "high_score_summary": high_score_summary,
        "representative_cases": representative_cases,
        "quick_insights": {
            "best_turn_metrics": _top_metric_items(score_summary.get("overall", {}).get("turn", {}).get("metrics", {}), reverse=True),
            "weakest_turn_metrics": _top_metric_items(score_summary.get("overall", {}).get("turn", {}).get("metrics", {}), reverse=False),
            "best_session_metrics": _top_metric_items(score_summary.get("overall", {}).get("session", {}).get("metrics", {}), reverse=True),
            "weakest_session_metrics": _top_metric_items(score_summary.get("overall", {}).get("session", {}).get("metrics", {}), reverse=False),
            "best_tasks": list(reversed(task_ranked[-5:])),
            "weakest_tasks": task_ranked[:5],
            "best_task_modes": list(reversed(task_mode_ranked[-5:])),
            "weakest_task_modes": task_mode_ranked[:5],
            "best_question_modes": list(reversed(question_mode_ranked[-4:])),
            "weakest_question_modes": question_mode_ranked[:4],
            "best_media_modes": list(reversed(media_mode_ranked[-4:])),
            "weakest_media_modes": media_mode_ranked[:4],
            "best_primary_abilities": list(reversed(primary_ranked[-8:])),
            "weakest_primary_abilities": primary_ranked[:8],
            "best_secondary_abilities": list(reversed(secondary_ranked[-8:])),
            "weakest_secondary_abilities": secondary_ranked[:8],
            "top_turn_primary_errors": _counter_items(error_category_summary.get("turn", {}).get("primary", {}), key_name="error_category", limit=6),
            "top_session_primary_errors": _counter_items(error_category_summary.get("session", {}).get("primary", {}), key_name="error_category", limit=6),
        },
    }

def _summarize_error_scope(scope_summary: Dict[str, Any], *, include_session: bool = False, limit: int = 8) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for scope_name, payload in scope_summary.items():
        turn_payload = payload.get("turn", payload)
        session_payload = payload.get("session", {}) if isinstance(payload.get("session", {}), dict) else {}
        row = {
            "scope": scope_name,
            "turn_candidate_count": int(turn_payload.get("candidate_count", 0) or 0),
            "turn_primary_errors": _counter_items(turn_payload.get("primary_error_category_counts", {}), key_name="error_category", limit=4),
            "turn_low_metrics": _counter_items(turn_payload.get("low_metric_counts", {}), key_name="metric", limit=5),
        }
        if include_session:
            row.update({
                "session_candidate_count": int(session_payload.get("candidate_count", 0) or 0),
                "session_primary_errors": _counter_items(session_payload.get("primary_error_category_counts", {}), key_name="error_category", limit=4),
                "session_low_metrics": _counter_items(session_payload.get("low_metric_counts", {}), key_name="metric", limit=4),
            })
        rows.append(row)
    rows.sort(key=lambda item: (-(item.get("turn_candidate_count", 0) + item.get("session_candidate_count", 0)), item["scope"]))
    return rows[:limit]


def _summarize_metric_cross(metric_summary: Dict[str, Any], *, limit: int = 10) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for metric_name, payload in metric_summary.items():
        rows.append({
            "metric": metric_name,
            "record_count": int(payload.get("record_count", 0) or 0),
            "primary_errors": _counter_items(payload.get("primary_error_category_counts", {}), key_name="error_category", limit=4),
            "top_tasks": _counter_items(payload.get("task_counts", {}), key_name="task_name", limit=5),
            "top_modes": _counter_items(payload.get("task_mode_counts", {}), key_name="task_mode", limit=5),
        })
    rows.sort(key=lambda item: (-item["record_count"], item["metric"]))
    return rows[:limit]


def _summarize_high_score_by_task(high_score_summary: Dict[str, Any], *, limit: int = 8) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for task_name, payload in high_score_summary.get("by_task", {}).items():
        rows.append({
            "task_name": task_name,
            "turn_high_score_count": int(payload.get("turn_high_score_count", 0) or 0),
            "session_high_score_count": int(payload.get("session_high_score_count", 0) or 0),
        })
    rows.sort(key=lambda item: (-(item["turn_high_score_count"] + item["session_high_score_count"]), item["task_name"]))
    return rows[:limit]


def _digest_case_items(items: Sequence[Dict[str, Any]], *, limit: int, text_limit: int = 320) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for item in list(items)[:limit]:
        rows.append({
            "case_id": _text(item.get("case_id") or item.get("candidate_id") or item.get("judgement_id") or item.get("dialogue_id") or item.get("round_id")),
            "dialogue_id": _text(item.get("dialogue_id")),
            "round_id": _text(item.get("round_id")),
            "task_name": _text(item.get("task_name")),
            "task_mode": _text(item.get("task_mode")),
            "question_mode": _text(item.get("question_mode")),
            "media_mode": _text(item.get("media_mode")),
            "primary_category": _text(item.get("primary_category")),
            "secondary_categories": list(item.get("secondary_categories", []) or []),
            "severity": _text(item.get("severity")),
            "avg_score": float(item.get("avg_score", 0.0) or 0.0),
            "low_score_metrics": list(item.get("low_score_metrics", []) or []),
            "primary_error_category": _text(item.get("primary_error_category")),
            "secondary_error_categories": list(item.get("secondary_error_categories", []) or []),
            "attribution_summary": _trim_text(item.get("attribution_summary"), limit=text_limit),
            "overall_summary": _trim_text(item.get("overall_summary"), limit=max(text_limit, 360)),
            "question_text": _trim_text(item.get("question_text"), limit=text_limit),
            "reference_answer": _trim_text(item.get("reference_answer"), limit=text_limit),
            "prediction": _trim_text(item.get("prediction"), limit=text_limit),
            "dialogue_preview": [_trim_text(v, limit=text_limit) for v in list(item.get("dialogue_preview", []) or [])[:4]],
            "key_dialogue_signals": [_trim_text(v, limit=text_limit) for v in list(item.get("key_dialogue_signals", []) or [])[:5]],
            "dialogue_preview": [_trim_text(v, limit=text_limit) for v in list(item.get("dialogue_preview", []) or [])[:4]],
            "key_dialogue_signals": [_trim_text(v, limit=text_limit) for v in list(item.get("key_dialogue_signals", []) or [])[:5]],
        })
    return rows


def build_report_digest(payload: Dict[str, Any]) -> Dict[str, Any]:
    quick = payload.get("quick_insights", {})
    phase4_manifest = payload.get("phase4_manifest", {})
    validation_summary = payload.get("validation_summary", {})
    score_summary = payload.get("score_summary", {})
    error_reason_by_task_summary = payload.get("error_reason_by_task_summary", {})
    error_reason_by_mode_summary = payload.get("error_reason_by_mode_summary", {})
    error_reason_by_ability_summary = payload.get("error_reason_by_ability_summary", {})
    metric_error_cross_summary = payload.get("metric_error_cross_summary", {})
    high_score_summary = payload.get("high_score_summary", {})
    representative_cases = payload.get("representative_cases", {})

    return {
        "benchmark_name": payload.get("benchmark_name", "RUBRIC-MME"),
        "tested_provider": payload.get("tested_provider", ""),
        "tested_model_name": payload.get("tested_model_name", ""),
        "coverage": {
            "turn_candidate_count": int(phase4_manifest.get("turn_candidate_count", 0) or 0),
            "session_candidate_count": int(phase4_manifest.get("session_candidate_count", 0) or 0),
            "turn_success_count": int(phase4_manifest.get("turn_success_count", 0) or 0),
            "session_success_count": int(phase4_manifest.get("session_success_count", 0) or 0),
            "phase4_error_record_count": int(phase4_manifest.get("error_record_count", 0) or 0),
            "validation_error_record_count": int(validation_summary.get("error_record_count", 0) or 0),
        },
        "overview": {
            "turn_overall_avg": float(score_summary.get("overall", {}).get("turn", {}).get("overall_avg_score", {}).get("avg_score", 0.0) or 0.0),
            "session_overall_avg": float(score_summary.get("overall", {}).get("session", {}).get("overall_avg_score", {}).get("avg_score", 0.0) or 0.0),
            "best_turn_metrics": quick.get("best_turn_metrics", [])[:5],
            "weakest_turn_metrics": quick.get("weakest_turn_metrics", [])[:5],
            "best_session_metrics": quick.get("best_session_metrics", [])[:5],
            "weakest_session_metrics": quick.get("weakest_session_metrics", [])[:5],
            "best_tasks": quick.get("best_tasks", [])[:5],
            "weakest_tasks": quick.get("weakest_tasks", [])[:5],
            "best_task_modes": quick.get("best_task_modes", [])[:5],
            "weakest_task_modes": quick.get("weakest_task_modes", [])[:5],
            "best_question_modes": quick.get("best_question_modes", [])[:4],
            "weakest_question_modes": quick.get("weakest_question_modes", [])[:4],
            "best_media_modes": quick.get("best_media_modes", [])[:4],
            "weakest_media_modes": quick.get("weakest_media_modes", [])[:4],
            "best_primary_abilities": quick.get("best_primary_abilities", [])[:8],
            "weakest_primary_abilities": quick.get("weakest_primary_abilities", [])[:8],
            "best_secondary_abilities": quick.get("best_secondary_abilities", [])[:8],
            "weakest_secondary_abilities": quick.get("weakest_secondary_abilities", [])[:8],
        },
        "diagnosis": {
            "top_turn_primary_errors": quick.get("top_turn_primary_errors", [])[:6],
            "top_session_primary_errors": quick.get("top_session_primary_errors", [])[:6],
            "task_error_patterns": _summarize_error_scope(error_reason_by_task_summary.get("by_task", {}), include_session=True, limit=8),
            "task_mode_error_patterns": _summarize_error_scope(error_reason_by_mode_summary.get("by_task_mode", {}), include_session=True, limit=6),
            "question_mode_error_patterns": _summarize_error_scope(error_reason_by_mode_summary.get("by_question_mode", {}), include_session=True, limit=4),
            "media_mode_error_patterns": _summarize_error_scope(error_reason_by_mode_summary.get("by_media_mode", {}), include_session=True, limit=4),
            "primary_ability_error_patterns": _summarize_error_scope(error_reason_by_ability_summary.get("by_primary_category", {}), include_session=False, limit=8),
            "secondary_ability_error_patterns": _summarize_error_scope(error_reason_by_ability_summary.get("by_secondary_category", {}), include_session=False, limit=8),
            "turn_metric_error_patterns": _summarize_metric_cross(metric_error_cross_summary.get("turn", {}), limit=10),
            "session_metric_error_patterns": _summarize_metric_cross(metric_error_cross_summary.get("session", {}), limit=6),
        },
        "high_score_patterns": {
            "overall": high_score_summary.get("overall", {}),
            "by_task_top": _summarize_high_score_by_task(high_score_summary, limit=8),
        },
        "representative_cases": {
            "low_turn_cases": _digest_case_items(representative_cases.get("low_turn_cases", []), limit=6, text_limit=360),
            "low_session_cases": _digest_case_items(representative_cases.get("low_session_cases", []), limit=4, text_limit=360),
            "high_turn_cases": _digest_case_items(representative_cases.get("high_turn_cases", []), limit=6, text_limit=360),
            "high_session_cases": _digest_case_items(representative_cases.get("high_session_cases", []), limit=4, text_limit=360),
        },
    }


def _build_step1_payload(payload: Dict[str, Any], digest: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "benchmark_name": digest.get("benchmark_name", "RUBRIC-MME"),
        "tested_provider": digest.get("tested_provider", ""),
        "tested_model_name": digest.get("tested_model_name", ""),
        "coverage": digest.get("coverage", {}),
        "overview": digest.get("overview", {}),
        "overall_turn_metrics": payload.get("score_summary", {}).get("overall", {}).get("turn", {}),
        "overall_session_metrics": payload.get("score_summary", {}).get("overall", {}).get("session", {}),
        "high_score_patterns": digest.get("high_score_patterns", {}),
    }


def _build_step2_scope_payload(payload: Dict[str, Any], digest: Dict[str, Any]) -> Dict[str, Any]:
    overview = digest.get("overview", {})
    weakest_tasks = _pick_names(overview.get("weakest_tasks", []), "task_name", limit=5)
    strongest_tasks = _pick_names(overview.get("best_tasks", []), "task_name", limit=4)
    weakest_task_modes = _pick_names(overview.get("weakest_task_modes", []), "task_mode", limit=5)
    strongest_task_modes = _pick_names(overview.get("best_task_modes", []), "task_mode", limit=4)
    weakest_question_modes = _pick_names(overview.get("weakest_question_modes", []), "question_mode", limit=4)
    strongest_question_modes = _pick_names(overview.get("best_question_modes", []), "question_mode", limit=3)
    weakest_media_modes = _pick_names(overview.get("weakest_media_modes", []), "media_mode", limit=4)
    strongest_media_modes = _pick_names(overview.get("best_media_modes", []), "media_mode", limit=3)
    weakest_primary = _pick_names(overview.get("weakest_primary_abilities", []), "ability_name", limit=6)
    strongest_primary = _pick_names(overview.get("best_primary_abilities", []), "ability_name", limit=5)
    weakest_secondary = _pick_names(overview.get("weakest_secondary_abilities", []), "ability_name", limit=6)
    strongest_secondary = _pick_names(overview.get("best_secondary_abilities", []), "ability_name", limit=5)

    return {
        "benchmark_name": digest.get("benchmark_name", "RUBRIC-MME"),
        "tested_provider": digest.get("tested_provider", ""),
        "tested_model_name": digest.get("tested_model_name", ""),
        "coverage": digest.get("coverage", {}),
        "overview_focus": {
            "weakest_tasks": overview.get("weakest_tasks", [])[:5],
            "best_tasks": overview.get("best_tasks", [])[:4],
            "weakest_task_modes": overview.get("weakest_task_modes", [])[:5],
            "best_task_modes": overview.get("best_task_modes", [])[:4],
            "weakest_primary_abilities": overview.get("weakest_primary_abilities", [])[:6],
            "best_primary_abilities": overview.get("best_primary_abilities", [])[:5],
        },
        "task_score_slices": {
            "weakest": _slice_mapping(payload.get("score_summary", {}).get("by_task", {}), weakest_tasks),
            "strongest": _slice_mapping(payload.get("score_summary", {}).get("by_task", {}), strongest_tasks),
        },
        "mode_score_slices": {
            "task_mode_weakest": _slice_mapping(payload.get("score_summary", {}).get("by_task_mode", {}), weakest_task_modes),
            "task_mode_strongest": _slice_mapping(payload.get("score_summary", {}).get("by_task_mode", {}), strongest_task_modes),
            "question_mode_weakest": _slice_mapping(payload.get("score_summary", {}).get("by_question_mode", {}), weakest_question_modes),
            "question_mode_strongest": _slice_mapping(payload.get("score_summary", {}).get("by_question_mode", {}), strongest_question_modes),
            "media_mode_weakest": _slice_mapping(payload.get("score_summary", {}).get("by_media_mode", {}), weakest_media_modes),
            "media_mode_strongest": _slice_mapping(payload.get("score_summary", {}).get("by_media_mode", {}), strongest_media_modes),
        },
        "ability_score_slices": {
            "primary_weakest": _slice_mapping(payload.get("ability_score_summary", {}).get("by_primary_category", {}), weakest_primary),
            "primary_strongest": _slice_mapping(payload.get("ability_score_summary", {}).get("by_primary_category", {}), strongest_primary),
            "secondary_weakest": _slice_mapping(payload.get("ability_score_summary", {}).get("by_secondary_category", {}), weakest_secondary),
            "secondary_strongest": _slice_mapping(payload.get("ability_score_summary", {}).get("by_secondary_category", {}), strongest_secondary),
        },
        "task_error_slices": _slice_mapping(payload.get("error_reason_by_task_summary", {}).get("by_task", {}), _unique(weakest_tasks + strongest_tasks)[:8]),
        "task_mode_error_slices": _slice_mapping(payload.get("error_reason_by_mode_summary", {}).get("by_task_mode", {}), _unique(weakest_task_modes + strongest_task_modes)[:8]),
        "question_mode_error_slices": _slice_mapping(payload.get("error_reason_by_mode_summary", {}).get("by_question_mode", {}), _unique(weakest_question_modes + strongest_question_modes)[:6]),
        "media_mode_error_slices": _slice_mapping(payload.get("error_reason_by_mode_summary", {}).get("by_media_mode", {}), _unique(weakest_media_modes + strongest_media_modes)[:6]),
        "primary_ability_error_slices": _slice_mapping(payload.get("error_reason_by_ability_summary", {}).get("by_primary_category", {}), _unique(weakest_primary + strongest_primary)[:10]),
        "secondary_ability_error_slices": _slice_mapping(payload.get("error_reason_by_ability_summary", {}).get("by_secondary_category", {}), _unique(weakest_secondary + strongest_secondary)[:10]),
        "high_score_patterns": digest.get("high_score_patterns", {}),
    }


def _build_step3_causes_payload(payload: Dict[str, Any], digest: Dict[str, Any]) -> Dict[str, Any]:
    diagnosis = digest.get("diagnosis", {})
    weakest_turn_metrics = _pick_names(digest.get("overview", {}).get("weakest_turn_metrics", []), "metric", limit=5)
    weakest_session_metrics = _pick_names(digest.get("overview", {}).get("weakest_session_metrics", []), "metric", limit=4)

    return {
        "benchmark_name": digest.get("benchmark_name", "RUBRIC-MME"),
        "tested_provider": digest.get("tested_provider", ""),
        "tested_model_name": digest.get("tested_model_name", ""),
        "coverage": digest.get("coverage", {}),
        "top_turn_primary_errors": diagnosis.get("top_turn_primary_errors", [])[:6],
        "top_session_primary_errors": diagnosis.get("top_session_primary_errors", [])[:6],
        "metric_failure_turn": _slice_mapping(payload.get("metric_failure_summary", {}).get("turn", {}), weakest_turn_metrics),
        "metric_failure_session": _slice_mapping(payload.get("metric_failure_summary", {}).get("session", {}), weakest_session_metrics),
        "metric_error_patterns": {
            "turn": diagnosis.get("turn_metric_error_patterns", [])[:10],
            "session": diagnosis.get("session_metric_error_patterns", [])[:6],
        },
        "task_error_patterns": diagnosis.get("task_error_patterns", [])[:8],
        "task_mode_error_patterns": diagnosis.get("task_mode_error_patterns", [])[:6],
        "ability_error_patterns": diagnosis.get("primary_ability_error_patterns", [])[:8],
        "representative_cases": {
            "low_turn_cases": _digest_case_items(payload.get("representative_cases", {}).get("low_turn_cases", []), limit=6, text_limit=420),
            "low_session_cases": _digest_case_items(payload.get("representative_cases", {}).get("low_session_cases", []), limit=4, text_limit=420),
            "high_turn_cases": _digest_case_items(payload.get("representative_cases", {}).get("high_turn_cases", []), limit=6, text_limit=420),
            "high_session_cases": _digest_case_items(payload.get("representative_cases", {}).get("high_session_cases", []), limit=4, text_limit=420),
        },
    }


def _build_step4_payload(payload: Dict[str, Any], digest: Dict[str, Any], step1_analysis: Dict[str, Any], step2_analysis: Dict[str, Any], step3_analysis: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "benchmark_name": digest.get("benchmark_name", "RUBRIC-MME"),
        "tested_provider": digest.get("tested_provider", ""),
        "tested_model_name": digest.get("tested_model_name", ""),
        "coverage": digest.get("coverage", {}),
        "overall_assessment": step1_analysis.get("overall_assessment", {}),
        "turn_level_findings": step1_analysis.get("turn_level_findings", []),
        "session_level_findings": step1_analysis.get("session_level_findings", []),
        "task_findings": step2_analysis.get("task_findings", []),
        "mode_findings": step2_analysis.get("mode_findings", []),
        "ability_findings": step2_analysis.get("ability_findings", []),
        "root_causes": step3_analysis.get("root_causes", []),
        "representative_case_analysis": step3_analysis.get("representative_case_analysis", {}),
        "overview_focus": {
            "strongest_signals": step1_analysis.get("overall_assessment", {}).get("strongest_signals", []),
            "weakest_signals": step1_analysis.get("overall_assessment", {}).get("weakest_signals", []),
            "weakest_tasks": digest.get("overview", {}).get("weakest_tasks", [])[:5],
            "weakest_task_modes": digest.get("overview", {}).get("weakest_task_modes", [])[:5],
            "weakest_primary_abilities": digest.get("overview", {}).get("weakest_primary_abilities", [])[:6],
        },
        "high_score_patterns": digest.get("high_score_patterns", {}),
        "top_primary_errors": {
            "turn": digest.get("diagnosis", {}).get("top_turn_primary_errors", [])[:5],
            "session": digest.get("diagnosis", {}).get("top_session_primary_errors", [])[:5],
        },
    }

def _empty_analysis() -> Dict[str, Any]:
    return {
        "executive_summary": "",
        "benchmark_overview": {"scope_summary": "", "coverage_summary": "", "evaluation_note": ""},
        "overall_assessment": {"verdict": "", "turn_level_summary": "", "session_level_summary": "", "strongest_signals": [], "weakest_signals": []},
        "turn_level_findings": [],
        "session_level_findings": [],
        "task_findings": [],
        "mode_findings": [],
        "ability_findings": [],
        "root_causes": [],
        "representative_case_analysis": {"low_turn_cases": [], "low_session_cases": [], "high_turn_cases": [], "high_session_cases": []},
        "recommendations": [],
        "priority_roadmap": {"p0": [], "p1": [], "p2": []},
        "report_closing": "",
    }


def build_fallback_analysis(payload: Dict[str, Any], digest: Dict[str, Any], error_message: str = "") -> Dict[str, Any]:
    quick = payload.get("quick_insights", {})
    weakest_turn = quick.get("weakest_turn_metrics", [])[:4]
    weakest_session = quick.get("weakest_session_metrics", [])[:4]
    weakest_tasks = quick.get("weakest_tasks", [])[:4]
    weakest_modes = quick.get("weakest_task_modes", [])[:4]
    weakest_abilities = quick.get("weakest_primary_abilities", [])[:5]
    top_turn_errors = quick.get("top_turn_primary_errors", [])[:4]
    top_session_errors = quick.get("top_session_primary_errors", [])[:4]
    representative = payload.get("representative_cases", {})

    def _error_names(items: Sequence[Dict[str, Any]], limit: int = 2) -> List[str]:
        return [item.get("error_category", "") for item in list(items)[:limit] if item.get("error_category")]

    turn_findings = [{"metric": item.get("metric", ""), "assessment": f"该 turn-level 指标当前属于相对弱项，平均分约为 {item.get('avg_score', 0.0):.4g}。", "evidence": f"该指标的低分率约为 {item.get('low_score_rate', 0.0):.4g}，说明它会在低分样本中反复出现。", "common_errors": _error_names(top_turn_errors)} for item in weakest_turn]
    session_findings = [{"metric": item.get("metric", ""), "assessment": f"该 session-level 指标表现相对偏弱，平均分约为 {item.get('avg_score', 0.0):.4g}。", "evidence": "从 session-level 统计来看，这个维度会在整段对话中反复成为不稳定来源。", "common_errors": _error_names(top_session_errors)} for item in weakest_session]
    task_findings = [{"scope": item.get("task_name", ""), "strengths": "该任务下模型仍能保持基本可用的回答质量，说明底层多模态链路没有完全失效。", "weaknesses": f"但该任务的 turn-level 平均分仅约为 {item.get('avg_score', 0.0):.4g}，说明在这类场景下稳定性和细节处理仍然不足。", "evidence": "这一判断主要依据 Phase 4 的 by-task 分数统计和低分归因汇总。"} for item in weakest_tasks]
    mode_findings = [{"scope": item.get("task_mode", ""), "strengths": "在这种模式下，模型仍能完成基本交互流程。", "weaknesses": f"但该模式的整体平均分仅约为 {item.get('avg_score', 0.0):.4g}，说明这种模式对系统仍然构成明显压力。", "evidence": "这一判断主要基于 task_mode 分数统计和 mode 层的低分原因分布。"} for item in weakest_modes]
    ability_findings = [{"scope": item.get("ability_name", ""), "assessment": f"这个能力维度当前表现偏弱，平均分约为 {item.get('avg_score', 0.0):.4g}。", "common_errors": _error_names(top_turn_errors), "evidence": "这一结论同时参考了 Phase 4 的 ability score summary 和 ability-level error reason summary。"} for item in weakest_abilities]
    root_causes = [{"category": item.get("error_category", ""), "explanation": "这类错误出现频率足够高，说明它更像是结构性短板，而不是孤立的偶发样本。", "affected_metrics": [entry.get("metric", "") for entry in weakest_turn[:3] if entry.get("metric")], "affected_scopes": [entry.get("task_name", "") for entry in weakest_tasks[:3] if entry.get("task_name")], "evidence": f"在 Phase 4 的错误类型汇总中，这一原因出现了 {item.get('count', 0)} 次。"} for item in top_turn_errors[:4]]

    def _fallback_cases(items: Sequence[Dict[str, Any]]) -> List[Dict[str, str]]:
        result: List[Dict[str, str]] = []
        for item in items[:3]:
            case_id = item.get("candidate_id") or item.get("round_id") or item.get("dialogue_id") or item.get("case_id")
            if case_id:
                result.append({"case_id": _text(case_id), "why_representative": "该案例能够集中体现当前模型在 benchmark 下的典型优势或典型短板。", "lesson": "它可以作为后续人工回看、定向排查和数据迭代的重要锚点。"})
        return result

    recommendations = [
        {"priority": "P0", "title": "优先修复最弱的 turn-level 指标与对应错误簇", "rationale": "最弱的 turn-level 指标会直接拉低整体交互质量，因此应被作为最高优先级的修复对象。", "actions": ["回看最低分的 turn-level 指标，并结合其对应的低分原因一起分析。", "按照重复出现的 primary error 将失败样本分组，以簇的方式排查。", "优先在压力最大的任务和模式上做定向优化。"], "expected_gain": "这有望最快提升核心可用性和 turn-level 分数稳定性。", "target_metrics": [item.get("metric", "") for item in weakest_turn[:3] if item.get("metric")], "target_scopes": [item.get("task_name", "") for item in weakest_tasks[:3] if item.get("task_name")]},
        {"priority": "P1", "title": "补强长上下文一致性与目标推进能力", "rationale": "session-level 弱项往往说明模型在完整多轮交互中还不够稳定可靠。", "actions": ["检查低分 session 案例，识别最常见的崩溃模式。", "检查上下文保留和对话状态跟踪是否能在长 session 中保持稳定。", "增强对用户显式状态变化和意图演化的响应能力。"], "expected_gain": "这有望提升端到端交互质量和用户信任感。", "target_metrics": [item.get("metric", "") for item in weakest_session[:2] if item.get("metric")], "target_scopes": [item.get("task_mode", "") for item in weakest_modes[:3] if item.get("task_mode")]},
        {"priority": "P1", "title": "围绕脆弱能力维度做针对性补强", "rationale": "能力层面的稳定短板会反复映射到多个任务和模式中。", "actions": ["优先针对最弱 primary_category 组织复盘样本。", "梳理这些能力上最常见的低分错误和触发条件。", "把能力弱点映射到后续数据构造和训练迭代中。"], "expected_gain": "这有助于提升 benchmark 的泛化表现，而不是只修单一任务。", "target_metrics": [], "target_scopes": [item.get("ability_name", "") for item in weakest_abilities[:3] if item.get("ability_name")]},
        {"priority": "P2", "title": "把高分模式沉淀成稳定的成功模板", "rationale": "高分样本能提供更直接的对照证据，帮助我们知道哪些行为模式值得保留和放大。", "actions": ["对照高分与低分案例，提炼成功模式与失败模式的差异。", "把高分任务、模式和能力的成功信号沉淀为常规检查项。", "让 benchmark 发现回流到后续迭代规划与门槛设定中。"], "expected_gain": "这会让 benchmark 结果更直接服务于持续优化闭环。", "target_metrics": [], "target_scopes": ["整体 benchmark 迭代"]},
    ]

    return {
        "executive_summary": "这份 Phase 5 回退分析建立在 Phase 4 的结构化统计、低分归因、多维交叉汇总和代表性案例之上。虽然本次运行没有完全拿到报告模型的正式分析对象，但已有证据仍然足以支撑一份有用的 benchmark 级诊断。结果表明，模型已经具备一定的多模态多轮交互能力，但少数高压力任务、模式和能力短板仍在反复拉低稳定性。",
        "benchmark_overview": {"scope_summary": "这次分析同时覆盖了 turn-level 和 session-level 的 benchmark 信号，并结合了 task、mode、ability、低分原因和代表性案例证据。", "coverage_summary": f"当前 Phase 4 结果中包含 {payload.get('phase4_manifest', {}).get('turn_candidate_count', 0)} 条 turn 低分候选样本和 {payload.get('phase4_manifest', {}).get('session_candidate_count', 0)} 条 session 低分候选样本。", "evaluation_note": "这次运行进入了 fallback analysis 模式，因此报告结论主要基于 Phase 4 的结构化证据合成，而不是来自报告模型的完整成功输出。"},
        "overall_assessment": {"verdict": "模型已具备一定的多模态多轮交互能力，但仍有几个弱指标和重复出现的错误模式在实质性地限制其稳健性。", "turn_level_summary": "turn-level 弱项主要集中在少数会在低分样本中反复出现的指标上。", "session_level_summary": "session-level 弱项更多体现为交互稳定性、目标推进能力以及整段对话中的可信性不足。", "strongest_signals": [item.get("metric", "") for item in quick.get("best_turn_metrics", [])[:3] if item.get("metric")], "weakest_signals": [item.get("metric", "") for item in weakest_turn[:3] if item.get("metric")]},
        "turn_level_findings": turn_findings,
        "session_level_findings": session_findings,
        "task_findings": task_findings,
        "mode_findings": mode_findings,
        "ability_findings": ability_findings,
        "root_causes": root_causes,
        "representative_case_analysis": {"low_turn_cases": _fallback_cases(representative.get("low_turn_cases", [])), "low_session_cases": _fallback_cases(representative.get("low_session_cases", [])), "high_turn_cases": _fallback_cases(representative.get("high_turn_cases", [])), "high_session_cases": _fallback_cases(representative.get("high_session_cases", []))},
        "recommendations": recommendations,
        "priority_roadmap": {"p0": ["优先修复最弱的 turn-level 指标及其对应的主导错误类型。", "优先回看压力最大的任务和模式。"], "p1": ["提升长上下文稳定性和多轮目标推进能力。", "围绕脆弱能力维度建立定向优化闭环。"], "p2": ["通过高分与低分对照提炼可复用的成功模式。", "把 benchmark 发现回流到常规迭代规划中。"]},
        "report_closing": "虽然这次运行使用了 fallback analysis，但前四个阶段已经提供了足够的结构化证据，能够支撑一份有用的 benchmark 报告。目前这份输出已经能够指出模型在哪些地方表现不足、哪些错误类型在驱动这些失败，以及哪些任务或能力范围值得优先关注。下一步应该继续提升 Phase 5 结构化响应的稳定性，让最终报告在保留相同证据基础的同时拥有更完整的模型生成叙事。",
        "fallback_reason": error_message,
    }


def _extract_step_from_analysis(full_analysis: Dict[str, Any], step_name: str) -> Dict[str, Any]:
    if step_name == "step1_overview":
        return {"executive_summary": full_analysis.get("executive_summary", ""), "benchmark_overview": full_analysis.get("benchmark_overview", {}), "overall_assessment": full_analysis.get("overall_assessment", {}), "turn_level_findings": full_analysis.get("turn_level_findings", []), "session_level_findings": full_analysis.get("session_level_findings", [])}
    if step_name == "step2_scope_findings":
        return {"task_findings": full_analysis.get("task_findings", []), "mode_findings": full_analysis.get("mode_findings", []), "ability_findings": full_analysis.get("ability_findings", [])}
    if step_name == "step3_root_causes_cases":
        return {"root_causes": full_analysis.get("root_causes", []), "representative_case_analysis": full_analysis.get("representative_case_analysis", {})}
    return {"recommendations": full_analysis.get("recommendations", []), "priority_roadmap": full_analysis.get("priority_roadmap", {}), "report_closing": full_analysis.get("report_closing", "")}


def _merge_analysis_sections(step_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    analysis = _empty_analysis()
    for step_name in STEP_SEQUENCE:
        analysis.update(step_results.get(step_name, {}).get("analysis", {}))
    return analysis


def _build_prompt_bundle(step_prompts: Dict[str, str]) -> str:
    blocks: List[str] = []
    for step_name in STEP_SEQUENCE:
        blocks.append(f"===== {STEP_LABELS[step_name]} =====")
        blocks.append(step_prompts.get(step_name, ""))
        blocks.append("")
    return "\n".join(blocks).strip() + "\n"


def _build_raw_bundle(step_results: Dict[str, Dict[str, Any]]) -> str:
    blocks: List[str] = []
    for step_name in STEP_SEQUENCE:
        step = step_results.get(step_name, {})
        blocks.append(f"===== {STEP_LABELS[step_name]} | status={step.get('status', '')} =====")
        raw_text = _text(step.get("raw_text"))
        if raw_text:
            blocks.append(raw_text)
        else:
            blocks.append(f"[无模型原始输出；错误：{_text(step.get('raw_error')) or _text(step.get('parse_error')) or '无'}]")
        blocks.append("")
    return "\n".join(blocks).strip() + "\n"


def _compute_overall_status(step_results: Dict[str, Dict[str, Any]]) -> str:
    statuses = [step_results.get(step_name, {}).get("status", "") for step_name in STEP_SEQUENCE]
    if statuses and all(status == "success" for status in statuses):
        return "success"
    if statuses and all(status.endswith("fallback") for status in statuses):
        return "full_fallback"
    return "partial_fallback"


def _default_step_result(step_name: str) -> Dict[str, Any]:
    return {"step_name": step_name, "label": STEP_LABELS[step_name], "status": "not_started", "parse_success": False, "parse_error": "", "parse_issues": [], "raw_error": "", "usage": {}, "attempt_count": 0, "schema_mode": "", "prompt_text": "", "raw_text": "", "analysis": {}}


def _can_reuse_step(step: Dict[str, Any]) -> bool:
    return bool(step) and step.get("status") == "success" and isinstance(step.get("analysis"), dict) and bool(step.get("analysis"))


def _load_existing_step_results(path: Path) -> Dict[str, Dict[str, Any]]:
    if not path.exists():
        return {}
    payload = load_json(path)
    steps = payload.get("steps", {}) if isinstance(payload, dict) else {}
    if not isinstance(steps, dict):
        return {}
    result: Dict[str, Dict[str, Any]] = {}
    for step_name in STEP_SEQUENCE:
        step = steps.get(step_name)
        if isinstance(step, dict):
            normalized = _default_step_result(step_name)
            normalized.update(step)
            result[step_name] = normalized
    return result


def _normalize_requested_steps(repair_steps: Sequence[str] | None) -> Set[str]:
    valid = set(STEP_SEQUENCE)
    selected: Set[str] = set()
    for step_name in repair_steps or []:
        if step_name in valid:
            selected.add(step_name)
    return selected


def _expand_steps(selected: Set[str], existing_steps: Dict[str, Dict[str, Any]]) -> Set[str]:
    expanded = set(selected)
    changed = True
    while changed:
        changed = False
        current = list(expanded)
        for step_name in current:
            for downstream in STEP_DOWNSTREAM.get(step_name, set()):
                if downstream not in expanded:
                    expanded.add(downstream)
                    changed = True
            for dependency in STEP_DEPENDENCIES.get(step_name, set()):
                if dependency not in expanded and not _can_reuse_step(existing_steps.get(dependency, {})):
                    expanded.add(dependency)
                    changed = True
    return expanded


def _determine_steps_to_run(existing_steps: Dict[str, Dict[str, Any]], *, resume: bool, repair_failed: bool, repair_steps: Sequence[str] | None) -> Set[str]:
    explicit = _normalize_requested_steps(repair_steps)
    if explicit:
        return _expand_steps(explicit, existing_steps)
    if repair_failed:
        if existing_steps:
            initial = {step_name for step_name in STEP_SEQUENCE if existing_steps.get(step_name, {}).get("status") != "success"}
        else:
            initial = set(STEP_SEQUENCE)
        return _expand_steps(initial, existing_steps)
    if resume:
        initial = {step_name for step_name in STEP_SEQUENCE if not _can_reuse_step(existing_steps.get(step_name, {}))}
        return _expand_steps(initial, existing_steps)
    return set(STEP_SEQUENCE)


def _run_report_step(step_name: str, step_payload: Dict[str, Any], backend: Any, fallback_full_analysis: Dict[str, Any], *, save_prompt_text: bool) -> Dict[str, Any]:
    prompt_text = build_report_step_prompt(step_name, step_payload)
    request = build_text_only_request(prompt_text)
    if step_name == "step1_overview":
        response_schema = build_report_step1_response_schema()
        parser = parse_report_step1_text
    elif step_name == "step2_scope_findings":
        response_schema = build_report_step2_scope_response_schema()
        parser = parse_report_step2_scope_text
    elif step_name == "step3_root_causes_cases":
        response_schema = build_report_step3_causes_response_schema()
        parser = parse_report_step3_causes_text
    else:
        response_schema = build_report_step4_recommendations_response_schema()
        parser = parse_report_step4_recommendations_text

    result = backend.judge(request, response_schema)
    raw_text = result.raw_text
    parse_success = False
    parse_error = ""
    parse_issues: List[str] = []
    if result.error:
        status = "api_fallback"
        analysis = _extract_step_from_analysis(fallback_full_analysis, step_name)
        parse_error = result.error
        parse_issues = [result.error]
        raw_text = ""
    else:
        parsed = parser(result.raw_text)
        parse_success = bool(parsed.get("parse_success", False))
        parse_error = _text(parsed.get("parse_error"))
        parse_issues = list(parsed.get("parse_issues", []))
        if parse_success:
            status = "success"
            analysis = parsed.get("analysis", {})
        else:
            status = "parse_fallback"
            analysis = _extract_step_from_analysis(fallback_full_analysis, step_name)

    return {"step_name": step_name, "label": STEP_LABELS[step_name], "status": status, "prompt_text": prompt_text if save_prompt_text else "", "raw_text": raw_text, "parse_success": parse_success, "parse_error": parse_error, "parse_issues": parse_issues, "raw_error": result.error, "usage": result.usage, "attempt_count": result.attempt_count, "schema_mode": result.schema_mode, "analysis": analysis}


def run_phase5_pipeline(phase4_dir: Path, output_dir: Path, backend: Any, *, clear_output: bool = True, save_prompt_text: bool = False, resume: bool = False, repair_failed: bool = False, repair_steps: Sequence[str] | None = None) -> Dict[str, Any]:
    phase4_dir = phase4_dir.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    output_paths = {name: output_dir / filename for name, filename in OUTPUT_FILENAMES.items()}
    if clear_output and not (resume or repair_failed or repair_steps):
        for path in output_paths.values():
            if path.exists():
                path.unlink()
    output_dir.mkdir(parents=True, exist_ok=True)

    payload = build_report_payload(phase4_dir)
    digest = build_report_digest(payload)
    dump_json(output_paths["report_payload"], payload)
    dump_json(output_paths["report_digest"], digest)

    fallback_full_analysis = build_fallback_analysis(payload, digest)
    existing_steps = _load_existing_step_results(output_paths["report_step_results"]) if (resume or repair_failed or repair_steps) else {}
    steps_to_run = _determine_steps_to_run(existing_steps, resume=resume, repair_failed=repair_failed, repair_steps=repair_steps)

    step_results: Dict[str, Dict[str, Any]] = {}
    step_prompts: Dict[str, str] = {}
    for step_name in STEP_SEQUENCE:
        if step_name == "step1_overview":
            step_payload = _build_step1_payload(payload, digest)
        elif step_name == "step2_scope_findings":
            step_payload = _build_step2_scope_payload(payload, digest)
        elif step_name == "step3_root_causes_cases":
            step_payload = _build_step3_causes_payload(payload, digest)
        else:
            step_payload = _build_step4_payload(payload, digest, step_results.get("step1_overview", {}).get("analysis", existing_steps.get("step1_overview", {}).get("analysis", {})), step_results.get("step2_scope_findings", {}).get("analysis", existing_steps.get("step2_scope_findings", {}).get("analysis", {})), step_results.get("step3_root_causes_cases", {}).get("analysis", existing_steps.get("step3_root_causes_cases", {}).get("analysis", {})))
        if save_prompt_text:
            step_prompts[step_name] = build_report_step_prompt(step_name, step_payload)
        if step_name in steps_to_run:
            step = _run_report_step(step_name, step_payload, backend, fallback_full_analysis, save_prompt_text=save_prompt_text)
        else:
            step = _default_step_result(step_name)
            step.update(existing_steps.get(step_name, {}))
            if save_prompt_text:
                step["prompt_text"] = step_prompts[step_name]
        step_results[step_name] = step

    analysis_status = _compute_overall_status(step_results)
    final_analysis = _merge_analysis_sections(step_results)
    step_result_summary = {step_name: {"label": step.get("label", ""), "status": step.get("status", ""), "parse_success": bool(step.get("parse_success", False)), "parse_error": _text(step.get("parse_error")), "parse_issues": list(step.get("parse_issues", [])), "raw_error": _text(step.get("raw_error")), "usage": step.get("usage", {}), "attempt_count": int(step.get("attempt_count", 0) or 0), "schema_mode": _text(step.get("schema_mode"))} for step_name, step in step_results.items()}

    step_results_payload = {"schema_version": PHASE5_SCHEMA_VERSION, "prompt_version": REPORT_PROMPT_VERSION, "generated_at": utc_now(), "analysis_status": analysis_status, "steps": step_results}
    report_analysis = {"schema_version": PHASE5_SCHEMA_VERSION, "prompt_version": REPORT_PROMPT_VERSION, "analysis_status": analysis_status, "backend_name": getattr(backend, "backend_name", ""), "analysis_model_name": getattr(backend, "model_name", ""), "input_mode": "text_only_multistep_refined", "generated_at": utc_now(), "step_results": step_result_summary, "analysis": final_analysis}
    dump_json(output_paths["report_step_results"], step_results_payload)
    dump_json(output_paths["report_analysis"], report_analysis)
    dump_text(output_paths["report_raw_text"], _build_raw_bundle(step_results))

    prompt_path = output_dir / "report_prompt.txt"
    if save_prompt_text:
        prompts_for_bundle = {step_name: step_prompts.get(step_name) or step_results.get(step_name, {}).get("prompt_text", "") for step_name in STEP_SEQUENCE}
        dump_text(prompt_path, _build_prompt_bundle(prompts_for_bundle))

    report_manifest = {
        "schema_version": PHASE5_SCHEMA_VERSION,
        "generated_at": utc_now(),
        "phase4_dir": str(phase4_dir),
        "output_dir": str(output_dir),
        "tested_provider": payload.get("tested_provider", ""),
        "tested_model_name": payload.get("tested_model_name", ""),
        "analysis_backend": getattr(backend, "backend_name", ""),
        "analysis_model_name": getattr(backend, "model_name", ""),
        "analysis_vs_tested_same_model": payload.get("tested_model_name", "") == getattr(backend, "model_name", ""),
        "analysis_input_mode": "text_only_multistep_refined",
        "analysis_stage_count": len(STEP_SEQUENCE),
        "analysis_stage_names": list(STEP_SEQUENCE),
        "analysis_status": analysis_status,
        "resume": bool(resume),
        "repair_failed": bool(repair_failed),
        "repair_steps": list(_normalize_requested_steps(repair_steps)),
        "step_statuses": {step_name: step_result_summary.get(step_name, {}).get("status", "") for step_name in STEP_SEQUENCE},
        "report_payload_path": str(output_paths["report_payload"]),
        "report_digest_path": str(output_paths["report_digest"]),
        "report_step_results_path": str(output_paths["report_step_results"]),
        "report_raw_text_path": str(output_paths["report_raw_text"]),
        "report_analysis_path": str(output_paths["report_analysis"]),
        "report_markdown_path": str(output_paths["report_markdown"]),
        "report_html_path": str(output_paths["report_html"]),
        "report_prompt_path": str(prompt_path) if save_prompt_text else "",
    }

    dump_text(output_paths["report_markdown"], build_markdown_report(report_manifest, payload, final_analysis))
    dump_text(output_paths["report_html"], build_html_report(report_manifest, payload, final_analysis))
    dump_json(output_paths["manifest"], report_manifest)
    return report_manifest
