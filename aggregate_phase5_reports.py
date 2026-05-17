#!/usr/bin/env python3
"""Aggregate phase5 benchmark_report.md files into comparison artifacts.

The script is intentionally read-only for model result folders. It only writes
new aggregate files under the requested output directory.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from html import escape
from pathlib import Path
from statistics import mean, median, pstdev
from typing import Any

try:
    from error_taxonomy import SESSION_ERROR_CATEGORIES_CN, TURN_ERROR_CATEGORIES_CN
except Exception:
    TURN_ERROR_CATEGORIES_CN = {}
    SESSION_ERROR_CATEGORIES_CN = {}


TARGET_SECTION_PREFIXES = (
    "三、关键看点速览",
    "四、整体分数概览",
    "七、任务与模态分析",
    "八、能力维度分析",
)

SERIES_ORDER = {
    "GPT": 10,
    "Claude": 20,
    "Gemini": 30,
    "Doubao": 40,
    "Qwen3-VL": 50,
    "Qwen3.5": 60,
    "Qwen2.5-VL": 70,
    "Other": 999,
}

TASK_ORDER = [
    "omnibench_image_multi_text",
    "omnibench_video_stream_text",
    "omnibench_image_multi_tts",
    "omnibench_video_stream_tts",
]

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

LOW_SCORE_THRESHOLDS = [
    ("lt4", 4.0, "<4分"),
    ("lt3", 3.0, "<3分"),
    ("lt2", 2.0, "<2分"),
]

SERIES_COLORS = {
    "GPT": "#2563eb",
    "Claude": "#059669",
    "Gemini": "#d97706",
    "Doubao": "#dc2626",
    "Qwen3-VL": "#7c3aed",
    "Qwen3.5": "#0891b2",
    "Qwen2.5-VL": "#64748b",
    "Other": "#334155",
}


@dataclass(frozen=True)
class ModelInfo:
    model_dir: str
    model_name: str
    series: str
    sort_key: tuple[Any, ...]
    report_path: Path


@dataclass
class MarkdownTable:
    section: str
    subsection: str
    headers: list[str]
    rows: list[dict[str, str]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate phase5 benchmark_report.md files for model comparison."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("result_judge(2.5pro)"),
        help="Root directory containing one subdirectory per model result.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output directory. Defaults to <root>/_phase5_model_comparison.",
    )
    parser.add_argument(
        "--include-nested",
        action="store_true",
        help=(
            "Also include nested */phase5/benchmark_report.md files. "
            "By default only <root>/<model>/phase5/benchmark_report.md is used."
        ),
    )
    parser.add_argument(
        "--gemini-text-only",
        action="store_true",
        help=(
            "Create a comparison variant where Gemini aggregates are recomputed "
            "from raw phase2/phase3/phase4 records after excluding tasks whose name "
            "contains 'tts'. Defaults to <root>/_phase5_model_comparison_gemini_text_only."
        ),
    )
    return parser.parse_args()


def clean_cell(value: str) -> str:
    return (
        value.strip()
        .replace("<br>", " ")
        .replace("<br/>", " ")
        .replace("<br />", " ")
    )


def split_markdown_row(line: str) -> list[str]:
    stripped = line.strip()
    if stripped.startswith("|"):
        stripped = stripped[1:]
    if stripped.endswith("|"):
        stripped = stripped[:-1]

    cells: list[str] = []
    current: list[str] = []
    escaped = False
    for char in stripped:
        if char == "\\" and not escaped:
            escaped = True
            current.append(char)
            continue
        if char == "|" and not escaped:
            cells.append(clean_cell("".join(current)))
            current = []
        else:
            current.append(char)
        escaped = False
    cells.append(clean_cell("".join(current)))
    return cells


def is_separator_row(line: str) -> bool:
    cells = [cell.strip() for cell in split_markdown_row(line)]
    if not cells:
        return False
    return all(re.fullmatch(r":?-{3,}:?", cell or "---") for cell in cells)


def heading_text(line: str) -> str:
    return re.sub(r"^#+\s*", "", line.strip()).strip()


def is_target_section(section: str) -> bool:
    return any(section.startswith(prefix) for prefix in TARGET_SECTION_PREFIXES)


def parse_markdown_report(text: str) -> tuple[list[MarkdownTable], list[dict[str, str]]]:
    lines = text.splitlines()
    tables: list[MarkdownTable] = []
    bullets: list[dict[str, str]] = []
    section = ""
    subsection = ""
    i = 0

    while i < len(lines):
        stripped = lines[i].strip()

        if stripped.startswith("## "):
            section = heading_text(stripped)
            subsection = ""
            i += 1
            continue

        if stripped.startswith("### "):
            subsection = heading_text(stripped)
            i += 1
            continue

        if section.startswith("三、关键看点速览") and stripped.startswith("- "):
            body = stripped[2:].strip()
            label, sep, value = body.partition("：")
            bullets.append(
                {
                    "label": label.strip() if sep else "",
                    "text": value.strip() if sep else body,
                }
            )
            i += 1
            continue

        if (
            is_target_section(section)
            and stripped.startswith("|")
            and i + 1 < len(lines)
            and is_separator_row(lines[i + 1])
        ):
            headers = split_markdown_row(stripped)
            rows: list[dict[str, str]] = []
            i += 2
            while i < len(lines) and lines[i].strip().startswith("|"):
                cells = split_markdown_row(lines[i])
                if len(cells) < len(headers):
                    cells.extend([""] * (len(headers) - len(cells)))
                if len(cells) > len(headers):
                    cells = cells[: len(headers) - 1] + [" | ".join(cells[len(headers) - 1 :])]
                rows.append(dict(zip(headers, cells)))
                i += 1
            tables.append(
                MarkdownTable(
                    section=section,
                    subsection=subsection or section,
                    headers=headers,
                    rows=rows,
                )
            )
            continue

        i += 1

    return tables, bullets


def maybe_float(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip().replace(",", "")
    if not text or text in {"-", "N/A", "nan", "None"}:
        return None
    match = re.search(r"[-+]?\d+(?:\.\d+)?", text)
    if not match:
        return None
    number = float(match.group(0))
    if "%" in text:
        number /= 100.0
    return number


def maybe_int(value: Any) -> int | None:
    number = maybe_float(value)
    if number is None:
        return None
    return int(round(number))


def iter_jsonl(path: Path) -> Any:
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def safe_read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def strip_result_suffix(name: str) -> str:
    result = name
    if result.startswith("logs_"):
        result = result[5:]
    suffixes = [
        "_antchat_full_video",
        "_full_video",
        "_antchat_video_frames",
        "_video_frames_full",
        "_video_frames",
    ]
    changed = True
    while changed:
        changed = False
        for suffix in suffixes:
            if result.endswith(suffix):
                result = result[: -len(suffix)]
                changed = True
    return result


def model_info_for(model_dir: str, report_path: Path) -> ModelInfo:
    key = strip_result_suffix(model_dir)
    lower = key.lower()

    if lower.startswith("gpt4o"):
        return ModelInfo(model_dir, "GPT-4o", "GPT", (SERIES_ORDER["GPT"], 4.0), report_path)
    if lower == "gpt5":
        return ModelInfo(model_dir, "GPT-5", "GPT", (SERIES_ORDER["GPT"], 5.0), report_path)
    match = re.fullmatch(r"gpt5(\d+)", lower)
    if match:
        version = float(f"5.{match.group(1)}")
        return ModelInfo(
            model_dir,
            f"GPT-5.{match.group(1)}",
            "GPT",
            (SERIES_ORDER["GPT"], version),
            report_path,
        )

    match = re.fullmatch(r"claude_(opus|sonnet)(\d)(\d)", lower)
    if match:
        family = match.group(1).title()
        version = f"{match.group(2)}.{match.group(3)}"
        return ModelInfo(
            model_dir,
            f"Claude {family} {version}",
            "Claude",
            (SERIES_ORDER["Claude"], family, float(version)),
            report_path,
        )

    match = re.fullmatch(r"gemini(\d)(\d)(pro|flash(?:_lite)?)", lower)
    if match:
        version = f"{match.group(1)}.{match.group(2)}"
        variant = match.group(3).replace("_", " ").title()
        return ModelInfo(
            model_dir,
            f"Gemini {version} {variant}",
            "Gemini",
            (SERIES_ORDER["Gemini"], float(version), variant),
            report_path,
        )
    match = re.fullmatch(r"gemini(\d)(pro|flash(?:_lite)?)", lower)
    if match:
        variant = match.group(2).replace("_", " ").title()
        return ModelInfo(
            model_dir,
            f"Gemini {match.group(1)} {variant}",
            "Gemini",
            (SERIES_ORDER["Gemini"], float(match.group(1)), variant),
            report_path,
        )

    match = re.fullmatch(r"doubao(\d+(?:\.\d+)?)(pro)?", lower)
    if match:
        version = match.group(1)
        variant = " Pro" if match.group(2) else ""
        return ModelInfo(
            model_dir,
            f"Doubao {version}{variant}",
            "Doubao",
            (SERIES_ORDER["Doubao"], float(version), variant),
            report_path,
        )

    match = re.fullmatch(r"qwen25vl(\d+)b", lower)
    if match:
        size = int(match.group(1))
        return ModelInfo(
            model_dir,
            f"Qwen2.5-VL {size}B",
            "Qwen2.5-VL",
            (SERIES_ORDER["Qwen2.5-VL"], size),
            report_path,
        )

    match = re.fullmatch(r"qwen35_(\d+)b", lower)
    if match:
        size = int(match.group(1))
        return ModelInfo(
            model_dir,
            f"Qwen3.5 {size}B",
            "Qwen3.5",
            (SERIES_ORDER["Qwen3.5"], size),
            report_path,
        )

    match = re.fullmatch(r"qwen3vl_?(\d+)b(?:_(instruct|thinking))?", lower)
    if match:
        size = int(match.group(1))
        variant = match.group(2)
        label = f"Qwen3-VL {size}B"
        if variant:
            label += f" {variant.title()}"
        return ModelInfo(
            model_dir,
            label,
            "Qwen3-VL",
            (SERIES_ORDER["Qwen3-VL"], size, variant or ""),
            report_path,
        )

    title = key.replace("_", " ").replace("-", " ").title()
    return ModelInfo(model_dir, title, "Other", (SERIES_ORDER["Other"], title), report_path)


def discover_reports(root: Path, include_nested: bool = False) -> list[ModelInfo]:
    reports: list[ModelInfo] = []
    if include_nested:
        for report_path in root.glob("**/phase5/benchmark_report.md"):
            relative = report_path.relative_to(root)
            model_dir = relative.parts[0]
            if model_dir.startswith("_"):
                continue
            reports.append(model_info_for(model_dir, report_path))
    else:
        for model_dir in sorted(path for path in root.iterdir() if path.is_dir()):
            if model_dir.name.startswith("_"):
                continue
            report_path = model_dir / "phase5" / "benchmark_report.md"
            if report_path.exists():
                reports.append(model_info_for(model_dir.name, report_path))

    deduped: dict[Path, ModelInfo] = {}
    for info in reports:
        deduped[info.report_path.resolve()] = info
    return sorted(deduped.values(), key=lambda item: item.sort_key)


def relpath(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def rows_to_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    if fieldnames is None:
        keys: list[str] = []
        seen: set[str] = set()
        for row in rows:
            for key in row:
                if key not in seen:
                    seen.add(key)
                    keys.append(key)
        fieldnames = keys

    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def add_model_metadata(row: dict[str, Any], info: ModelInfo, root: Path) -> dict[str, Any]:
    return {
        "model": info.model_name,
        "model_dir": info.model_dir,
        "series": info.series,
        "report_path": relpath(info.report_path, root),
        **row,
    }


def collect_data(root: Path, reports: list[ModelInfo]) -> dict[str, Any]:
    all_tables: list[dict[str, Any]] = []
    key_takeaways: list[dict[str, Any]] = []
    task_rows: list[dict[str, Any]] = []
    task_dimension_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    ability_primary_rows: list[dict[str, Any]] = []
    ability_secondary_rows: list[dict[str, Any]] = []

    for info in reports:
        text = info.report_path.read_text(encoding="utf-8")
        tables, bullets = parse_markdown_report(text)

        for bullet in bullets:
            key_takeaways.append(add_model_metadata(bullet, info, root))

        for table in tables:
            table_payload = {
                "model": info.model_name,
                "model_dir": info.model_dir,
                "series": info.series,
                "report_path": relpath(info.report_path, root),
                "section": table.section,
                "table": table.subsection,
                "headers": table.headers,
                "rows": table.rows,
            }
            all_tables.append(table_payload)

            if table.section.startswith("三、") and "任务" in table.headers:
                for source_row in table.rows:
                    task_rows.append(
                        add_model_metadata(
                            {
                                "source_table": table.subsection,
                                "task": source_row.get("任务", ""),
                                "avg_score": maybe_float(source_row.get("均分")),
                                "sample_count": maybe_int(source_row.get("样本数")),
                            },
                            info,
                            root,
                        )
                    )

            if table.section.startswith("四、") and "指标" in table.headers:
                level = "Turn-Level" if "Turn" in table.subsection else "Session-Level"
                for source_row in table.rows:
                    metric_rows.append(
                        add_model_metadata(
                            {
                                "level": level,
                                "metric": source_row.get("指标", ""),
                                "avg_score": maybe_float(source_row.get("平均分")),
                                "min_score": maybe_float(source_row.get("最低分")),
                                "max_score": maybe_float(source_row.get("最高分")),
                                "low_count": maybe_int(source_row.get("低分数")),
                                "low_rate": maybe_float(source_row.get("低分率")),
                            },
                            info,
                            root,
                        )
                    )

            if (
                table.section.startswith("七、")
                and table.subsection == "任务维度"
                and "任务" in table.headers
            ):
                for source_row in table.rows:
                    task_dimension_rows.append(
                        add_model_metadata(
                            {
                                "task": source_row.get("任务", ""),
                                "turn_avg_score": maybe_float(source_row.get("Turn 均分")),
                                "turn_sample_count": maybe_int(source_row.get("Turn 样本数")),
                                "session_avg_score": maybe_float(source_row.get("Session 均分")),
                                "session_sample_count": maybe_int(
                                    source_row.get("Session 样本数")
                                ),
                            },
                            info,
                            root,
                        )
                    )

            if table.section.startswith("八、") and table.subsection in {
                "Primary Category",
                "Secondary Category",
            }:
                target = ability_primary_rows if table.subsection == "Primary Category" else ability_secondary_rows
                for source_row in table.rows:
                    target.append(
                        add_model_metadata(
                            {
                                "ability": source_row.get("能力", ""),
                                "overall_score": maybe_float(source_row.get("整体均分")),
                                "sample_count": maybe_int(source_row.get("样本数")),
                                "accuracy": maybe_float(source_row.get("Accuracy")),
                                "proactiveness": maybe_float(source_row.get("Proactiveness")),
                                "intent_depth": maybe_float(source_row.get("Intent Depth")),
                            },
                            info,
                            root,
                        )
                    )

    return {
        "all_tables": all_tables,
        "key_takeaways": key_takeaways,
        "task_rows": task_rows,
        "task_dimension_rows": task_dimension_rows,
        "metric_rows": metric_rows,
        "ability_primary_rows": ability_primary_rows,
        "ability_secondary_rows": ability_secondary_rows,
    }


def task_name_from_record(obj: dict[str, Any]) -> str:
    return str(obj.get("task_name") or obj.get("task") or obj.get("task_id") or "")


def include_task_for_variant(
    info: ModelInfo,
    task_name: str,
    gemini_text_only: bool,
) -> bool:
    if not gemini_text_only or info.series != "Gemini":
        return True
    return "tts" not in str(task_name or "").lower()


def coerce_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, set):
        return list(value)
    if isinstance(value, str) and value.strip():
        return [value]
    return []


def record_avg_score(record: dict[str, Any], metric_order: list[str] | None = None) -> float | None:
    avg_score = maybe_float(record.get("avg_score"))
    if avg_score is not None:
        return avg_score
    score_vector = record.get("score_vector") or {}
    metrics = metric_order or list(score_vector)
    values = finite_values([maybe_float(score_vector.get(metric)) for metric in metrics])
    return mean(values) if values else None


def metric_vector_mean(record: dict[str, Any], metric_order: list[str]) -> float | None:
    score_vector = record.get("score_vector") or {}
    values = finite_values([maybe_float(score_vector.get(metric)) for metric in metric_order])
    if values:
        return mean(values)
    return maybe_float(record.get("avg_score"))


def successful_records_for_task(path: Path, info: ModelInfo, gemini_text_only: bool) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for obj in iter_jsonl(path) or []:
        if obj.get("status") != "success":
            continue
        if not include_task_for_variant(info, task_name_from_record(obj), gemini_text_only):
            continue
        records.append(obj)
    return records


def metric_summary_row(
    values: list[float | None],
) -> dict[str, Any]:
    scores = finite_values(values)
    low_count = sum(1 for value in scores if value < 4.0)
    return {
        "avg_score": round(mean(scores), 4) if scores else None,
        "min_score": min(scores) if scores else None,
        "max_score": max(scores) if scores else None,
        "low_count": low_count,
        "low_rate": (low_count / len(scores)) if scores else None,
    }


def recompute_gemini_text_only_data(
    root: Path,
    reports: list[ModelInfo],
    data: dict[str, Any],
) -> dict[str, Any]:
    gemini_infos = [info for info in reports if info.series == "Gemini"]
    if not gemini_infos:
        return data

    gemini_models = {info.model_name for info in gemini_infos}
    recomputed: dict[str, Any] = {
        "all_tables": [row for row in data["all_tables"] if row.get("model") not in gemini_models],
        "key_takeaways": [
            row for row in data["key_takeaways"] if row.get("model") not in gemini_models
        ],
        "task_rows": [row for row in data["task_rows"] if row.get("model") not in gemini_models],
        "task_dimension_rows": [
            row for row in data["task_dimension_rows"] if row.get("model") not in gemini_models
        ],
        "metric_rows": [row for row in data["metric_rows"] if row.get("model") not in gemini_models],
        "ability_primary_rows": [
            row for row in data["ability_primary_rows"] if row.get("model") not in gemini_models
        ],
        "ability_secondary_rows": [
            row for row in data["ability_secondary_rows"] if row.get("model") not in gemini_models
        ],
    }

    for info in gemini_infos:
        model_root = root / info.model_dir
        turn_records = successful_records_for_task(
            model_root / "phase3" / "turn_judgements.jsonl",
            info,
            gemini_text_only=True,
        )
        session_records = successful_records_for_task(
            model_root / "phase3" / "session_judgements.jsonl",
            info,
            gemini_text_only=True,
        )

        recomputed["key_takeaways"].append(
            add_model_metadata(
                {
                    "label": "Text-only recalculation",
                    "text": (
                        "Gemini TTS tasks were excluded; task, metric, ability, scene, "
                        "and failure-attribution values are recomputed from raw text-task records."
                    ),
                },
                info,
                root,
            )
        )

        for level, records, metrics in (
            ("Turn-Level", turn_records, TURN_METRICS),
            ("Session-Level", session_records, SESSION_METRICS),
        ):
            for metric in metrics:
                summary = metric_summary_row(
                    [maybe_float((record.get("score_vector") or {}).get(metric)) for record in records]
                )
                recomputed["metric_rows"].append(
                    add_model_metadata(
                        {
                            "level": level,
                            "metric": metric,
                            **summary,
                        },
                        info,
                        root,
                    )
                )

        turn_by_task: dict[str, list[float]] = defaultdict(list)
        session_by_task: dict[str, list[float]] = defaultdict(list)
        for record in turn_records:
            task_name = task_name_from_record(record)
            score = record_avg_score(record, TURN_METRICS)
            if task_name and score is not None:
                turn_by_task[task_name].append(score)
        for record in session_records:
            task_name = task_name_from_record(record)
            score = record_avg_score(record, SESSION_METRICS)
            if task_name and score is not None:
                session_by_task[task_name].append(score)

        task_names = ordered_unique(
            list(turn_by_task) + list(session_by_task),
            [task for task in TASK_ORDER if "tts" not in task.lower()],
        )
        for task_name in task_names:
            turn_scores = turn_by_task.get(task_name, [])
            session_scores = session_by_task.get(task_name, [])
            combined_scores = turn_scores + session_scores
            recomputed["task_dimension_rows"].append(
                add_model_metadata(
                    {
                        "task": task_name,
                        "turn_avg_score": round(mean(turn_scores), 4) if turn_scores else None,
                        "turn_sample_count": len(turn_scores),
                        "session_avg_score": round(mean(session_scores), 4) if session_scores else None,
                        "session_sample_count": len(session_scores),
                    },
                    info,
                    root,
                )
            )
            recomputed["task_rows"].append(
                add_model_metadata(
                    {
                        "source_table": "text_only_recomputed",
                        "task": task_name,
                        "avg_score": round(mean(combined_scores), 4) if combined_scores else None,
                        "sample_count": len(combined_scores),
                    },
                    info,
                    root,
                )
            )

        primary_buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
        secondary_buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for record in turn_records:
            primary = str(record.get("primary_category") or "").strip()
            if primary:
                primary_buckets[primary].append(record)
            for secondary in coerce_list(record.get("secondary_categories")):
                secondary_name = str(secondary or "").strip()
                if secondary_name:
                    secondary_buckets[secondary_name].append(record)

        for target_key, buckets in (
            ("ability_primary_rows", primary_buckets),
            ("ability_secondary_rows", secondary_buckets),
        ):
            for ability, records in sorted(buckets.items()):
                overall_scores = [record_avg_score(record, TURN_METRICS) for record in records]
                recomputed[target_key].append(
                    add_model_metadata(
                        {
                            "ability": ability,
                            "overall_score": (
                                round(mean(finite_values(overall_scores)), 4)
                                if finite_values(overall_scores)
                                else None
                            ),
                            "sample_count": len(records),
                            "accuracy": round(
                                mean(
                                    finite_values(
                                        [
                                            maybe_float((record.get("score_vector") or {}).get("accuracy"))
                                            for record in records
                                        ]
                                    )
                                ),
                                4,
                            )
                            if finite_values(
                                [
                                    maybe_float((record.get("score_vector") or {}).get("accuracy"))
                                    for record in records
                                ]
                            )
                            else None,
                            "proactiveness": round(
                                mean(
                                    finite_values(
                                        [
                                            maybe_float(
                                                (record.get("score_vector") or {}).get(
                                                    "proactiveness_helpfulness"
                                                )
                                            )
                                            for record in records
                                        ]
                                    )
                                ),
                                4,
                            )
                            if finite_values(
                                [
                                    maybe_float(
                                        (record.get("score_vector") or {}).get(
                                            "proactiveness_helpfulness"
                                        )
                                    )
                                    for record in records
                                ]
                            )
                            else None,
                            "intent_depth": round(
                                mean(
                                    finite_values(
                                        [
                                            maybe_float(
                                                (record.get("score_vector") or {}).get(
                                                    "intent_understanding_depth"
                                                )
                                            )
                                            for record in records
                                        ]
                                    )
                                ),
                                4,
                            )
                            if finite_values(
                                [
                                    maybe_float(
                                        (record.get("score_vector") or {}).get(
                                            "intent_understanding_depth"
                                        )
                                    )
                                    for record in records
                                ]
                            )
                            else None,
                        },
                        info,
                        root,
                    )
                )

    return recomputed


def finite_values(values: list[float | None]) -> list[float]:
    return [value for value in values if value is not None and math.isfinite(value)]


def percentile(sorted_values: list[float], q: float) -> float | None:
    if not sorted_values:
        return None
    if len(sorted_values) == 1:
        return sorted_values[0]
    pos = (len(sorted_values) - 1) * q
    lower = math.floor(pos)
    upper = math.ceil(pos)
    if lower == upper:
        return sorted_values[int(pos)]
    return sorted_values[lower] * (upper - pos) + sorted_values[upper] * (pos - lower)


def describe_values(values: list[float | None]) -> dict[str, Any]:
    nums = sorted(finite_values(values))
    if not nums:
        return {
            "n": 0,
            "mean": None,
            "median": None,
            "variance": None,
            "std": None,
            "min": None,
            "max": None,
            "range": None,
            "q1": None,
            "q3": None,
            "iqr": None,
            "cv": None,
        }
    avg = mean(nums)
    std = pstdev(nums) if len(nums) > 1 else 0.0
    variance = std * std
    q1 = percentile(nums, 0.25)
    q3 = percentile(nums, 0.75)
    return {
        "n": len(nums),
        "mean": avg,
        "median": median(nums),
        "variance": variance,
        "std": std,
        "min": nums[0],
        "max": nums[-1],
        "range": nums[-1] - nums[0],
        "q1": q1,
        "q3": q3,
        "iqr": (q3 - q1) if q1 is not None and q3 is not None else None,
        "cv": (std / avg) if avg else None,
    }


def estimate_sample_count(rows: list[dict[str, Any]]) -> int | None:
    estimates: list[float] = []
    for row in rows:
        low_count = row.get("low_count")
        low_rate = row.get("low_rate")
        if low_count is None or low_rate in (None, 0):
            continue
        estimates.append(float(low_count) / float(low_rate))
    if not estimates:
        return None
    return int(round(median(estimates)))


def weighted_task_average(rows: list[dict[str, Any]], tasks: set[str] | None = None) -> float | None:
    numerator = 0.0
    denominator = 0
    for row in rows:
        task = row.get("task")
        if tasks is not None and task not in tasks:
            continue
        score = row.get("avg_score")
        count = row.get("sample_count")
        if score is None or count is None:
            continue
        numerator += float(score) * int(count)
        denominator += int(count)
    if not denominator:
        return None
    return numerator / denominator


def weighted_turn_session_average(
    turn_metric_avg: float | None,
    session_metric_avg: float | None,
    turn_count: int,
    session_count: int,
) -> float | None:
    if turn_metric_avg is None or session_metric_avg is None:
        return None
    return (turn_metric_avg * turn_count + session_metric_avg * session_count) / (
        turn_count + session_count
    )


def weighted_task_dimension_average(
    rows: list[dict[str, Any]],
    keyword: str | None = None,
) -> float | None:
    numerator = 0.0
    denominator = 0
    keyword_lower = keyword.lower() if keyword else None
    for row in rows:
        task = str(row.get("task", "")).lower()
        if keyword_lower and keyword_lower not in task:
            continue
        turn_score = row.get("turn_avg_score")
        turn_count = row.get("turn_sample_count")
        session_score = row.get("session_avg_score")
        session_count = row.get("session_sample_count")
        if turn_score is not None and turn_count is not None:
            numerator += float(turn_score) * int(turn_count)
            denominator += int(turn_count)
        if session_score is not None and session_count is not None:
            numerator += float(session_score) * int(session_count)
            denominator += int(session_count)
    if not denominator:
        return None
    return numerator / denominator


def weighted_task_dimension_level_average(
    rows: list[dict[str, Any]],
    score_key: str,
    count_key: str,
) -> float | None:
    numerator = 0.0
    denominator = 0
    for row in rows:
        score = row.get(score_key)
        count = row.get(count_key)
        if score is None or count is None:
            continue
        numerator += float(score) * int(count)
        denominator += int(count)
    if not denominator:
        return None
    return numerator / denominator


def compute_model_summary(
    reports: list[ModelInfo],
    metric_rows: list[dict[str, Any]],
    ability_primary_rows: list[dict[str, Any]],
    task_rows: list[dict[str, Any]],
    task_dimension_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    metrics_by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in metric_rows:
        metrics_by_model[row["model"]].append(row)

    primary_by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in ability_primary_rows:
        primary_by_model[row["model"]].append(row)

    tasks_by_model: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in task_rows:
        task = row.get("task") or ""
        if not task:
            continue
        existing = tasks_by_model[row["model"]].get(task)
        if existing is None or row.get("source_table", "").startswith("最强"):
            tasks_by_model[row["model"]][task] = row

    task_dimension_by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in task_dimension_rows:
        task_dimension_by_model[row["model"]].append(row)

    summaries: list[dict[str, Any]] = []
    info_by_name = {info.model_name: info for info in reports}
    for model_name, info in info_by_name.items():
        rows = metrics_by_model.get(model_name, [])
        turn_scores = finite_values([row.get("avg_score") for row in rows if row.get("level") == "Turn-Level"])
        session_scores = finite_values(
            [row.get("avg_score") for row in rows if row.get("level") == "Session-Level"]
        )
        all_scores = finite_values([row.get("avg_score") for row in rows])
        low_rates = finite_values([row.get("low_rate") for row in rows])
        dimension_rows = task_dimension_by_model.get(model_name, [])
        turn_sample_count = sum(
            int(row.get("turn_sample_count") or 0) for row in dimension_rows
        )
        session_sample_count = sum(
            int(row.get("session_sample_count") or 0) for row in dimension_rows
        )
        turn_metric_avg = weighted_task_dimension_level_average(
            dimension_rows,
            "turn_avg_score",
            "turn_sample_count",
        )
        session_metric_avg = weighted_task_dimension_level_average(
            dimension_rows,
            "session_avg_score",
            "session_sample_count",
        )
        text_task_weighted_avg = weighted_task_dimension_average(dimension_rows, "text")
        tts_task_weighted_avg = weighted_task_dimension_average(dimension_rows, "tts")
        all_task_weighted_avg = weighted_task_dimension_average(dimension_rows)

        metric_lookup = {row.get("metric"): row for row in rows}
        accuracy = metric_lookup.get("accuracy", {})
        overall_helpfulness = metric_lookup.get("overall_helpfulness_trustworthiness", {})

        primary_rows = primary_by_model.get(model_name, [])
        weighted_primary_score = None
        weighted_numerator = 0.0
        weighted_denominator = 0
        for row in primary_rows:
            score = row.get("overall_score")
            count = row.get("sample_count")
            if score is None or count is None:
                continue
            weighted_numerator += score * count
            weighted_denominator += count
        if weighted_denominator:
            weighted_primary_score = weighted_numerator / weighted_denominator

        scored_primary = [
            row
            for row in primary_rows
            if row.get("overall_score") is not None and row.get("ability")
        ]
        best_primary = max(scored_primary, key=lambda row: row["overall_score"], default={})
        worst_primary = min(scored_primary, key=lambda row: row["overall_score"], default={})

        task_lookup = tasks_by_model.get(model_name, {})
        image_task = task_lookup.get("omnibench_image_multi_text", {})
        video_task = task_lookup.get("omnibench_video_stream_text", {})
        task_values = list(task_lookup.values())
        covered_tasks = sorted(task_lookup)
        text_tasks = {task for task in covered_tasks if task.endswith("_text")}
        tts_tasks = {task for task in covered_tasks if task.endswith("_tts")}

        summaries.append(
            {
                "model": model_name,
                "model_dir": info.model_dir,
                "series": info.series,
                "report_path": str(info.report_path),
                "all_metric_avg": round(mean(all_scores), 4) if all_scores else None,
                "turn_metric_avg": round(turn_metric_avg, 4)
                if turn_metric_avg is not None
                else None,
                "session_metric_avg": round(session_metric_avg, 4)
                if session_metric_avg is not None
                else None,
                "avg_low_rate": round(mean(low_rates), 4) if low_rates else None,
                "turn_sample_count_est": turn_sample_count,
                "session_sample_count_est": session_sample_count,
                "task_coverage_count": len(covered_tasks),
                "task_coverage": f"{len(covered_tasks)}/{len(TASK_ORDER)}",
                "covered_tasks": ", ".join(covered_tasks),
                "text_task_weighted_avg": round(text_task_weighted_avg, 4)
                if text_task_weighted_avg is not None
                else None,
                "tts_task_weighted_avg": round(tts_task_weighted_avg, 4)
                if tts_task_weighted_avg is not None
                else None,
                "all_task_weighted_avg": round(all_task_weighted_avg, 4)
                if all_task_weighted_avg is not None
                else None,
                "accuracy": accuracy.get("avg_score"),
                "accuracy_low_rate": accuracy.get("low_rate"),
                "overall_helpfulness_trustworthiness": overall_helpfulness.get("avg_score"),
                "overall_helpfulness_low_rate": overall_helpfulness.get("low_rate"),
                "image_task_score": image_task.get("avg_score"),
                "video_task_score": video_task.get("avg_score"),
                "image_minus_video": (
                    round(image_task.get("avg_score") - video_task.get("avg_score"), 4)
                    if image_task.get("avg_score") is not None
                    and video_task.get("avg_score") is not None
                    else None
                ),
                "primary_weighted_score": round(weighted_primary_score, 4)
                if weighted_primary_score is not None
                else None,
                "primary_ability_count": len(scored_primary),
                "best_primary_ability": best_primary.get("ability"),
                "best_primary_score": best_primary.get("overall_score"),
                "worst_primary_ability": worst_primary.get("ability"),
                "worst_primary_score": worst_primary.get("overall_score"),
            }
        )

    macro_ranked = sorted(
        summaries,
        key=lambda row: (row.get("all_metric_avg") is None, -(row.get("all_metric_avg") or 0)),
    )
    for index, row in enumerate(macro_ranked, start=1):
        row["metric_macro_rank"] = index
        row["overall_rank"] = index

    info_sort = {info.model_name: info.sort_key for info in reports}
    summaries.sort(key=lambda row: info_sort.get(row["model"], (999, row["model"])))
    for index, row in enumerate(summaries, start=1):
        row["display_order"] = index
    return summaries


def compute_series_summary(model_summary: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_series: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in model_summary:
        by_series[row["series"]].append(row)

    rows: list[dict[str, Any]] = []
    for series, items in sorted(by_series.items(), key=lambda item: SERIES_ORDER.get(item[0], 999)):
        ranked = sorted(items, key=lambda row: (row.get("all_metric_avg") is None, -(row.get("all_metric_avg") or 0)))
        best = ranked[0] if ranked else {}
        best_score = best.get("all_metric_avg")
        for rank, item in enumerate(ranked, start=1):
            score = item.get("all_metric_avg")
            rows.append(
                {
                    "series": series,
                    "series_rank": rank,
                    "model": item.get("model"),
                    "overall_rank": item.get("overall_rank"),
                    "all_metric_avg": score,
                    "turn_metric_avg": item.get("turn_metric_avg"),
                    "session_metric_avg": item.get("session_metric_avg"),
                    "accuracy": item.get("accuracy"),
                    "overall_helpfulness_trustworthiness": item.get(
                        "overall_helpfulness_trustworthiness"
                    ),
                    "best_model_in_series": best.get("model"),
                    "gap_to_series_best": round(best_score - score, 4)
                    if best_score is not None and score is not None
                    else None,
                }
            )
    return rows


def compute_ability_comparison(
    ability_rows: list[dict[str, Any]],
    model_order: list[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_ability: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in ability_rows:
        if row.get("ability"):
            by_ability[row["ability"]].append(row)

    pivot_rows: list[dict[str, Any]] = []
    leader_rows: list[dict[str, Any]] = []
    for ability, rows in by_ability.items():
        by_model = {row["model"]: row for row in rows}
        scores = finite_values([row.get("overall_score") for row in rows])
        sample_counts = finite_values([float(row.get("sample_count")) for row in rows])

        best = max(
            [row for row in rows if row.get("overall_score") is not None],
            key=lambda row: row["overall_score"],
            default={},
        )
        worst = min(
            [row for row in rows if row.get("overall_score") is not None],
            key=lambda row: row["overall_score"],
            default={},
        )
        spread = (
            round(best["overall_score"] - worst["overall_score"], 4)
            if best and worst
            else None
        )

        base = {
            "ability": ability,
            "model_count": len(scores),
            "mean_score": round(mean(scores), 4) if scores else None,
            "std_score": round(pstdev(scores), 4) if len(scores) > 1 else 0.0,
            "mean_sample_count": round(mean(sample_counts), 2) if sample_counts else None,
            "best_model": best.get("model"),
            "best_score": best.get("overall_score"),
            "worst_model": worst.get("model"),
            "worst_score": worst.get("overall_score"),
            "spread": spread,
        }
        pivot = dict(base)
        for model in model_order:
            pivot[model] = by_model.get(model, {}).get("overall_score")
        pivot_rows.append(pivot)
        leader_rows.append(base)

    pivot_rows.sort(key=lambda row: (row.get("mean_score") is None, row.get("mean_score") or 0))
    leader_rows.sort(key=lambda row: (row.get("spread") is None, -(row.get("spread") or 0)))
    return pivot_rows, leader_rows


def filter_primary_by_non_gemini_sample_count(
    ability_rows: list[dict[str, Any]],
    model_summary: list[dict[str, Any]],
    min_sample_count: int = 30,
) -> tuple[list[dict[str, Any]], list[str]]:
    non_gemini_models = {
        row["model"]
        for row in model_summary
        if row.get("series") != "Gemini" and row.get("model")
    }
    by_ability: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in ability_rows:
        ability = row.get("ability")
        model = row.get("model")
        if ability and model:
            by_ability[str(ability)][str(model)] = row

    kept: list[str] = []
    for ability, by_model in by_ability.items():
        counts = [
            maybe_int(by_model.get(model, {}).get("sample_count"))
            for model in non_gemini_models
        ]
        if counts and all(count is not None and count >= min_sample_count for count in counts):
            kept.append(ability)
    kept_set = set(kept)
    filtered_rows = [row for row in ability_rows if row.get("ability") in kept_set]
    return filtered_rows, sorted(kept)


def distribution_stats_for_pivot(
    rows: list[dict[str, Any]],
    model_order: list[str],
    label_keys: list[str],
    value_label: str,
    lower_is_better: bool = False,
) -> list[dict[str, Any]]:
    stats_rows: list[dict[str, Any]] = []
    for row in rows:
        values = [maybe_float(row.get(model)) for model in model_order]
        stats = describe_values(values)
        scored = [
            (model, maybe_float(row.get(model)))
            for model in model_order
            if maybe_float(row.get(model)) is not None
        ]
        if lower_is_better:
            best = min(scored, key=lambda item: item[1], default=(None, None))
            worst = max(scored, key=lambda item: item[1], default=(None, None))
        else:
            best = max(scored, key=lambda item: item[1], default=(None, None))
            worst = min(scored, key=lambda item: item[1], default=(None, None))
        label_values = {key: row.get(key) for key in label_keys}
        stats_rows.append(
            {
                **label_values,
                "value_label": value_label,
                "n": stats["n"],
                "mean": round(stats["mean"], 4) if stats["mean"] is not None else None,
                "median": round(stats["median"], 4) if stats["median"] is not None else None,
                "variance": round(stats["variance"], 6) if stats["variance"] is not None else None,
                "std": round(stats["std"], 4) if stats["std"] is not None else None,
                "min": round(stats["min"], 4) if stats["min"] is not None else None,
                "max": round(stats["max"], 4) if stats["max"] is not None else None,
                "range": round(stats["range"], 4) if stats["range"] is not None else None,
                "iqr": round(stats["iqr"], 4) if stats["iqr"] is not None else None,
                "cv": round(stats["cv"], 4) if stats["cv"] is not None else None,
                "best_model": best[0],
                "best_value": round(best[1], 4) if best[1] is not None else None,
                "worst_model": worst[0],
                "worst_value": round(worst[1], 4) if worst[1] is not None else None,
            }
        )
    return stats_rows


def complement_rate_pivot(
    rows: list[dict[str, Any]],
    model_order: list[str],
    label_keys: list[str],
) -> list[dict[str, Any]]:
    complemented: list[dict[str, Any]] = []
    for row in rows:
        new_row = {key: row.get(key) for key in label_keys if key in row}
        for key, value in row.items():
            if key.endswith("__count") or key in {"model_count", "total_count"}:
                new_row[key] = value
        for model in model_order:
            value = maybe_float(row.get(model))
            new_row[model] = (1.0 - value) if value is not None else None
        complemented.append(new_row)
    return complemented


def max_matrix_value(rows: list[dict[str, Any]], columns: list[str]) -> float:
    values = finite_values([maybe_float(row.get(column)) for row in rows for column in columns])
    return max(max(values), 1.0) if values else 1.0


def count_matrix_from_ratio_rows(
    rows: list[dict[str, Any]],
    columns: list[str],
    row_key: str,
) -> list[dict[str, Any]]:
    count_rows: list[dict[str, Any]] = []
    for row in rows:
        count_row: dict[str, Any] = {
            row_key: row.get(row_key),
            "total_count": row.get("total_count"),
        }
        for column in columns:
            count_row[column] = row.get(f"{column}__count")
        count_rows.append(count_row)
    return count_rows


def prepend_model_total_row(
    rows: list[dict[str, Any]],
    model_order: list[str],
    row_key: str,
    label: str = "Total low cases",
) -> list[dict[str, Any]]:
    if not rows:
        return rows
    total_row: dict[str, Any] = {row_key: label}
    for model in model_order:
        total_row[model] = sum(int(maybe_float(row.get(model)) or 0) for row in rows)
    return [total_row, *rows]


def pass_rate_pivot_from_rows(
    rows: list[dict[str, Any]],
    model_order: list[str],
    label_key: str,
    value_key: str = "pass_rate",
    preferred_order: list[str] | None = None,
) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    model_counts: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        label = str(row.get(label_key) or "")
        model = str(row.get("model") or "")
        if not label or not model:
            continue
        grouped.setdefault(label, {label_key: label})
        grouped[label][model] = row.get(value_key)
        if row.get(value_key) is not None:
            model_counts[label].add(model)
    ordered_labels = ordered_unique(list(grouped), preferred_order)
    return [
        {
            **{label_key: label, "model_count": len(model_counts.get(label, set()))},
            **{model: grouped[label].get(model) for model in model_order},
        }
        for label in ordered_labels
    ]


def compute_model_metric_self_stats(metric_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in metric_rows:
        model = str(row.get("model") or "")
        if not model:
            continue
        level = str(row.get("level") or "")
        grouped[(model, level)].append(row)
        grouped[(model, "All 12 Metrics")].append(row)

    scope_order = {"Turn-Level": 0, "Session-Level": 1, "All 12 Metrics": 2}
    output: list[dict[str, Any]] = []
    for (model, scope), rows in grouped.items():
        if scope not in scope_order:
            continue
        score_stats = describe_values([maybe_float(row.get("avg_score")) for row in rows])
        low_stats = describe_values([maybe_float(row.get("low_rate")) for row in rows])
        pass_stats = describe_values(
            [
                1.0 - low_rate
                for low_rate in [maybe_float(row.get("low_rate")) for row in rows]
                if low_rate is not None
            ]
        )
        series = next((str(row.get("series") or "") for row in rows if row.get("series")), "")
        output.append(
            {
                "model": model,
                "series": series,
                "scope": scope,
                "metric_count": score_stats["n"],
                "score_mean": round(score_stats["mean"], 4) if score_stats["mean"] is not None else None,
                "score_variance": round(score_stats["variance"], 6) if score_stats["variance"] is not None else None,
                "score_std": round(score_stats["std"], 4) if score_stats["std"] is not None else None,
                "score_min": round(score_stats["min"], 4) if score_stats["min"] is not None else None,
                "score_max": round(score_stats["max"], 4) if score_stats["max"] is not None else None,
                "score_range": round(score_stats["range"], 4) if score_stats["range"] is not None else None,
                "score_iqr": round(score_stats["iqr"], 4) if score_stats["iqr"] is not None else None,
                "low_rate_mean": round(low_stats["mean"], 4) if low_stats["mean"] is not None else None,
                "low_rate_variance": round(low_stats["variance"], 6) if low_stats["variance"] is not None else None,
                "low_rate_std": round(low_stats["std"], 4) if low_stats["std"] is not None else None,
                "low_rate_min": round(low_stats["min"], 4) if low_stats["min"] is not None else None,
                "low_rate_max": round(low_stats["max"], 4) if low_stats["max"] is not None else None,
                "low_rate_range": round(low_stats["range"], 4) if low_stats["range"] is not None else None,
                "pass_rate_mean": round(pass_stats["mean"], 4) if pass_stats["mean"] is not None else None,
                "pass_rate_variance": round(pass_stats["variance"], 6) if pass_stats["variance"] is not None else None,
                "pass_rate_std": round(pass_stats["std"], 4) if pass_stats["std"] is not None else None,
                "pass_rate_min": round(pass_stats["min"], 4) if pass_stats["min"] is not None else None,
                "pass_rate_max": round(pass_stats["max"], 4) if pass_stats["max"] is not None else None,
                "pass_rate_range": round(pass_stats["range"], 4) if pass_stats["range"] is not None else None,
            }
        )
    output.sort(key=lambda row: (scope_order.get(str(row.get("scope")), 99), row.get("series") or "", row.get("model") or ""))
    return output


def compute_metric_distribution_stats(
    score_pivot: list[dict[str, Any]],
    low_rate_pivot: list[dict[str, Any]],
    model_order: list[str],
) -> list[dict[str, Any]]:
    low_lookup = {
        (row.get("level"), row.get("metric")): row
        for row in low_rate_pivot
    }
    rows: list[dict[str, Any]] = []
    score_stats = distribution_stats_for_pivot(
        score_pivot,
        model_order,
        ["level", "metric"],
        "平均分",
        lower_is_better=False,
    )
    low_stats = distribution_stats_for_pivot(
        [low_lookup.get((row.get("level"), row.get("metric")), {}) for row in score_pivot],
        model_order,
        ["level", "metric"],
        "低分率",
        lower_is_better=True,
    )
    low_by_key = {
        (row.get("level"), row.get("metric")): row
        for row in low_stats
    }
    pass_rate_pivot = complement_rate_pivot(low_rate_pivot, model_order, ["level", "metric"])
    pass_stats = distribution_stats_for_pivot(
        [next((row for row in pass_rate_pivot if row.get("level") == score_row.get("level") and row.get("metric") == score_row.get("metric")), {}) for score_row in score_pivot],
        model_order,
        ["level", "metric"],
        "通过率",
        lower_is_better=False,
    )
    pass_by_key = {
        (row.get("level"), row.get("metric")): row
        for row in pass_stats
    }
    for score_row in score_stats:
        low_row = low_by_key.get((score_row.get("level"), score_row.get("metric")), {})
        pass_row = pass_by_key.get((score_row.get("level"), score_row.get("metric")), {})
        rows.append(
            {
                "level": score_row.get("level"),
                "metric": score_row.get("metric"),
                "score_mean": score_row.get("mean"),
                "score_median": score_row.get("median"),
                "score_variance": score_row.get("variance"),
                "score_std": score_row.get("std"),
                "score_range": score_row.get("range"),
                "score_iqr": score_row.get("iqr"),
                "score_best_model": score_row.get("best_model"),
                "score_best": score_row.get("best_value"),
                "score_worst_model": score_row.get("worst_model"),
                "score_worst": score_row.get("worst_value"),
                "low_rate_mean": low_row.get("mean"),
                "low_rate_median": low_row.get("median"),
                "low_rate_variance": low_row.get("variance"),
                "low_rate_std": low_row.get("std"),
                "low_rate_range": low_row.get("range"),
                "low_rate_iqr": low_row.get("iqr"),
                "lowest_low_rate_model": low_row.get("best_model"),
                "lowest_low_rate": low_row.get("best_value"),
                "highest_low_rate_model": low_row.get("worst_model"),
                "highest_low_rate": low_row.get("worst_value"),
                "pass_rate_mean": pass_row.get("mean"),
                "pass_rate_median": pass_row.get("median"),
                "pass_rate_variance": pass_row.get("variance"),
                "pass_rate_std": pass_row.get("std"),
                "pass_rate_range": pass_row.get("range"),
                "pass_rate_iqr": pass_row.get("iqr"),
                "highest_pass_rate_model": pass_row.get("best_model"),
                "highest_pass_rate": pass_row.get("best_value"),
                "lowest_pass_rate_model": pass_row.get("worst_model"),
                "lowest_pass_rate": pass_row.get("worst_value"),
            }
        )
    rows.sort(key=lambda row: (row.get("level") != "Turn-Level", row.get("metric") or ""))
    return rows


def metric_pivot_rows(
    metric_rows: list[dict[str, Any]],
    model_order: list[str],
    value_key: str = "avg_score",
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], dict[str, Any]] = {}
    for row in metric_rows:
        key = (row.get("level", ""), row.get("metric", ""))
        grouped.setdefault(key, {"level": key[0], "metric": key[1]})
        grouped[key][row["model"]] = row.get(value_key)

    level_order = {"Turn-Level": 0, "Session-Level": 1}
    rows = list(grouped.values())
    rows.sort(key=lambda row: (level_order.get(row["level"], 99), row["metric"]))
    return [{**{"level": row["level"], "metric": row["metric"]}, **{m: row.get(m) for m in model_order}} for row in rows]


def split_environment_label(label: str | None) -> tuple[str, str]:
    text = str(label or "unknown").strip() or "unknown"
    if "-" not in text:
        return text, "unknown"
    major, detail = text.split("-", 1)
    return major or "unknown", detail or "unknown"


def conversation_primary_category(conversation: dict[str, Any], category_key: str) -> str | None:
    payload = conversation.get(category_key)
    if isinstance(payload, dict):
        category = payload.get("primary_category")
        return str(category) if category else None
    return None


def load_scene_environment_analysis(project_root: Path) -> dict[str, Any]:
    dataset_specs = [
        {
            "task": "image",
            "path": project_root / "omnibench_dataset" / "image_final_with_mimt_category.json",
            "conversation_key": "image_conversation",
            "category_key": "mimt_category",
        },
        {
            "task": "video",
            "path": project_root / "omnibench_dataset" / "video_final_with_vqa_category.json",
            "conversation_key": "stream_conversation",
            "category_key": "vqa_category",
        },
    ]
    dataset_rows: list[dict[str, Any]] = []
    major_counts: dict[str, dict[str, int]] = defaultdict(lambda: {"image": 0, "video": 0, "total": 0})
    detail_counts: dict[tuple[str, str], dict[str, int]] = defaultdict(
        lambda: {"image": 0, "video": 0, "total": 0}
    )
    primary_counts: dict[str, dict[str, int]] = defaultdict(lambda: {"image": 0, "video": 0, "total": 0})
    matrix_counts: dict[tuple[str, str], dict[str, int]] = defaultdict(
        lambda: {"image": 0, "video": 0, "total": 0}
    )

    for spec in dataset_specs:
        path = spec["path"]
        task = spec["task"]
        if not path.exists():
            dataset_rows.append(
                {
                    "task": task,
                    "path": str(path),
                    "session_count": 0,
                    "categorized_turn_count": 0,
                    "major_environment_count": 0,
                    "detail_environment_count": 0,
                    "primary_category_count": 0,
                    "missing": True,
                }
            )
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        task_majors: set[str] = set()
        task_details: set[tuple[str, str]] = set()
        task_primaries: set[str] = set()
        categorized_turns = 0

        for item in data:
            major, detail = split_environment_label(item.get("environment"))
            task_majors.add(major)
            task_details.add((major, detail))
            major_counts[major][task] += 1
            major_counts[major]["total"] += 1
            detail_counts[(major, detail)][task] += 1
            detail_counts[(major, detail)]["total"] += 1

            conversations = (
                item.get(spec["conversation_key"], {}).get("conversations", [])
                if isinstance(item.get(spec["conversation_key"]), dict)
                else []
            )
            for conversation in conversations:
                category = conversation_primary_category(conversation, spec["category_key"])
                if not category:
                    continue
                categorized_turns += 1
                task_primaries.add(category)
                primary_counts[category][task] += 1
                primary_counts[category]["total"] += 1
                matrix_counts[(major, category)][task] += 1
                matrix_counts[(major, category)]["total"] += 1

        dataset_rows.append(
            {
                "task": task,
                "path": str(path),
                "session_count": len(data),
                "categorized_turn_count": categorized_turns,
                "major_environment_count": len(task_majors),
                "detail_environment_count": len(task_details),
                "primary_category_count": len(task_primaries),
                "missing": False,
            }
        )

    major_rows = [
        {"environment_major": major, **counts}
        for major, counts in major_counts.items()
    ]
    major_rows.sort(key=lambda row: row["total"], reverse=True)
    detail_rows = [
        {"environment_major": major, "environment_detail": detail, **counts}
        for (major, detail), counts in detail_counts.items()
    ]
    detail_rows.sort(key=lambda row: row["total"], reverse=True)
    primary_rows = [
        {"primary_category": category, **counts}
        for category, counts in primary_counts.items()
    ]
    primary_rows.sort(key=lambda row: row["total"], reverse=True)
    matrix_rows = [
        {
            "environment_major": major,
            "primary_category": category,
            **counts,
        }
        for (major, category), counts in matrix_counts.items()
    ]
    matrix_rows.sort(key=lambda row: (row["environment_major"], -row["total"], row["primary_category"]))
    return {
        "dataset_rows": dataset_rows,
        "major_rows": major_rows,
        "detail_rows": detail_rows,
        "primary_rows": primary_rows,
        "matrix_rows": matrix_rows,
    }


def threshold_label(threshold_key: str) -> str:
    for key, _, label in LOW_SCORE_THRESHOLDS:
        if key == threshold_key:
            return label
    return threshold_key


def pivot_rows_by_label(
    rows: list[dict[str, Any]],
    model_order: list[str],
    label_key: str,
    value_key: str,
    preferred_order: list[str] | None = None,
) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    for row in rows:
        label = str(row.get(label_key) or "")
        model = str(row.get("model") or "")
        if not label or not model:
            continue
        grouped.setdefault(label, {label_key: label})
        grouped[label][model] = row.get(value_key)

    ordered_labels = ordered_unique(list(grouped), preferred_order)
    return [{**grouped[label], **{model: grouped[label].get(model) for model in model_order}} for label in ordered_labels]


def aggregate_mean_threshold_rows(
    rows: list[dict[str, Any]],
    group_key: str,
    preferred_order: list[str] | None = None,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        label = row.get(group_key)
        if label:
            grouped[str(label)].append(row)

    ordered_labels = ordered_unique(list(grouped), preferred_order)
    aggregated: list[dict[str, Any]] = []
    for label in ordered_labels:
        label_rows = grouped.get(label, [])
        bucket: dict[str, Any] = {group_key: label}
        for threshold_key, _, _ in LOW_SCORE_THRESHOLDS:
            values = finite_values([maybe_float(row.get(f"{threshold_key}_rate")) for row in label_rows])
            bucket[threshold_key] = round(mean(values), 4) if values else None
        avg_scores = finite_values([maybe_float(row.get("avg_score")) for row in label_rows])
        bucket["avg_score_mean"] = round(mean(avg_scores), 4) if avg_scores else None
        bucket["model_count"] = len(label_rows)
        aggregated.append(bucket)
    return aggregated


def matrix_rows_from_nested_counts(
    nested_counts: dict[str, dict[str, int]],
    row_key: str,
    row_order: list[str] | None = None,
    col_order: list[str] | None = None,
) -> tuple[list[dict[str, Any]], list[str]]:
    total_by_col: dict[str, int] = defaultdict(int)
    for counts in nested_counts.values():
        for column, value in counts.items():
            total_by_col[column] += int(value or 0)
    ordered_cols = (
        [column for column in col_order if column in total_by_col]
        if col_order is not None
        else [column for column, _ in sorted(total_by_col.items(), key=lambda item: (-item[1], item[0]))]
    )

    row_labels = row_order or ordered_unique(list(nested_counts))
    rows: list[dict[str, Any]] = []
    for label in row_labels:
        counts = nested_counts.get(label, {})
        total = sum(int(value or 0) for value in counts.values())
        row: dict[str, Any] = {row_key: label, "total_count": total}
        for column in ordered_cols:
            value = int(counts.get(column, 0))
            row[column] = (value / total) if total else None
            row[f"{column}__count"] = value
        rows.append(row)
    return rows, ordered_cols


def collect_multistage_phase_analysis(
    root: Path,
    reports: list[ModelInfo],
    model_summary: list[dict[str, Any]],
    scene_stats: dict[str, Any],
    gemini_text_only: bool = False,
) -> dict[str, Any]:
    model_order = [row["model"] for row in model_summary]
    summary_lookup = {row["model"]: row for row in model_summary}
    major_order = [str(row.get("environment_major")) for row in scene_stats.get("major_rows", [])]
    detail_order = [
        f"{row.get('environment_major')}-{row.get('environment_detail')}"
        for row in scene_stats.get("detail_rows", [])
    ]

    generation_rows: list[dict[str, Any]] = []
    metric_threshold_counts: dict[tuple[str, str, str, str], dict[str, int]] = defaultdict(
        lambda: {"low_count": 0, "total_count": 0}
    )
    level_pass_counts: dict[tuple[str, str], dict[str, int]] = defaultdict(
        lambda: {"pass_count": 0, "total_count": 0}
    )
    task_turn_buckets: dict[tuple[str, str], dict[str, Any]] = defaultdict(
        lambda: {"score_sum": 0.0, "count": 0, "lt4": 0, "lt3": 0, "lt2": 0}
    )
    primary_ability_pass_buckets: dict[tuple[str, str], dict[str, Any]] = defaultdict(
        lambda: {"count": 0, "pass_count": 0}
    )
    scene_major_buckets: dict[tuple[str, str], dict[str, Any]] = defaultdict(
        lambda: {"score_sum": 0.0, "count": 0, "lt4": 0, "lt3": 0, "lt2": 0}
    )
    scene_detail_buckets: dict[tuple[str, str], dict[str, Any]] = defaultdict(
        lambda: {"score_sum": 0.0, "count": 0, "lt4": 0, "lt3": 0, "lt2": 0}
    )
    scene_primary_buckets: dict[tuple[str, str], dict[str, Any]] = defaultdict(
        lambda: {"score_sum": 0.0, "count": 0, "lt4": 0, "lt3": 0, "lt2": 0}
    )
    phase4_model_rows: list[dict[str, Any]] = []
    phase4_turn_primary_counts_by_model: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    phase4_session_primary_counts_by_model: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    phase4_turn_secondary_counts_by_model: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    phase4_session_secondary_counts_by_model: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    phase4_turn_primary_overall: dict[str, int] = defaultdict(int)
    phase4_session_primary_overall: dict[str, int] = defaultdict(int)
    phase4_turn_secondary_overall: dict[str, int] = defaultdict(int)
    phase4_session_secondary_overall: dict[str, int] = defaultdict(int)
    phase4_turn_metric_primary_nested: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    phase4_session_metric_primary_nested: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    phase4_turn_metric_secondary_nested: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    phase4_session_metric_secondary_nested: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    phase4_turn_task_primary_nested: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    phase4_session_task_primary_nested: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for info in reports:
        model = info.model_name
        model_root = root / info.model_dir
        series = info.series
        phase1_summary_files = sorted((model_root / "phase1").glob("*_run_summary.json"))
        phase1_summary = safe_read_json(phase1_summary_files[0]) if phase1_summary_files else {}
        phase2_rounds_path = model_root / "phase2" / "rounds.jsonl"
        phase3_turn_path = model_root / "phase3" / "turn_judgements.jsonl"
        phase3_session_path = model_root / "phase3" / "session_judgements.jsonl"
        phase4_turn_summary_path = model_root / "phase4" / "turn_attribution_summary.json"
        phase4_session_summary_path = model_root / "phase4" / "session_attribution_summary.json"
        phase4_metric_failure_path = model_root / "phase4" / "metric_failure_summary.json"
        phase4_task_error_path = model_root / "phase4" / "error_reason_by_task_summary.json"
        phase4_turn_attributions_path = model_root / "phase4" / "turn_error_attributions.jsonl"
        phase4_session_attributions_path = model_root / "phase4" / "session_error_attributions.jsonl"
        phase4_low_turns_path = model_root / "phase4" / "low_score_turns.jsonl"
        phase4_low_sessions_path = model_root / "phase4" / "low_score_sessions.jsonl"

        round_meta: dict[str, dict[str, Any]] = {}
        dialogue_meta: dict[str, dict[str, Any]] = {}
        round_total = 0
        empty_predictions = 0
        error_rounds = 0
        latency_total = 0.0
        latency_count = 0
        phase1_tasks = phase1_summary.get("tasks") or []
        phase1_tasks = [
            task
            for task in phase1_tasks
            if include_task_for_variant(info, str(task.get("task_name") or ""), gemini_text_only)
        ]
        phase1_task_count = len(phase1_tasks)
        phase1_declared_dialogues = 0
        for task in phase1_tasks:
            declared = maybe_int(task.get("dialogue_count_total")) or 0
            attempted = maybe_int(task.get("attempted_dialogues")) or 0
            completed = maybe_int(task.get("completed_dialogues")) or 0
            skipped = maybe_int(task.get("skipped_dialogues")) or 0
            phase1_declared_dialogues += max(declared, attempted + skipped, completed + skipped)
        phase1_skipped_dialogues = sum(maybe_int(task.get("skipped_dialogues")) or 0 for task in phase1_tasks)
        phase1_failed_rounds = sum(maybe_int(task.get("failed_rounds")) or 0 for task in phase1_tasks)

        for obj in iter_jsonl(phase2_rounds_path) or []:
            if not include_task_for_variant(info, task_name_from_record(obj), gemini_text_only):
                continue
            round_total += 1
            if obj.get("prediction_is_empty"):
                empty_predictions += 1
            if obj.get("has_error"):
                error_rounds += 1
            latency = maybe_float(obj.get("latency_seconds"))
            if latency is not None:
                latency_total += latency
                latency_count += 1
            round_id = str(obj.get("round_id") or "")
            dialogue_id = str(obj.get("dialogue_id") or "")
            environment_major, environment_detail = split_environment_label(obj.get("environment"))
            meta = {
                "environment_major": environment_major,
                "environment_detail": environment_detail,
                "primary_category": obj.get("primary_category"),
                "task_name": obj.get("task_name"),
            }
            if round_id:
                round_meta[round_id] = meta
            if dialogue_id and dialogue_id not in dialogue_meta:
                dialogue_meta[dialogue_id] = meta

        generation_rows.append(
            {
                "model": model,
                "series": series,
                "phase1_task_count": phase1_task_count,
                "phase1_declared_dialogues": phase1_declared_dialogues,
                "phase1_skipped_dialogues": phase1_skipped_dialogues,
                "phase1_failed_rounds": phase1_failed_rounds,
                "round_count": round_total,
                "empty_prediction_rate": (empty_predictions / round_total) if round_total else None,
                "error_rate": (error_rounds / round_total) if round_total else None,
                "avg_latency_seconds": (latency_total / latency_count) if latency_count else None,
            }
        )

        for obj in iter_jsonl(phase3_turn_path) or []:
            if obj.get("status") != "success":
                continue
            task_name = str(obj.get("task_name") or "")
            if not include_task_for_variant(info, task_name, gemini_text_only):
                continue
            level_avg_score = metric_vector_mean(obj, TURN_METRICS)
            if level_avg_score is not None:
                level_bucket = level_pass_counts[(model, "Turn-Level")]
                level_bucket["total_count"] += 1
                if level_avg_score >= 4.0:
                    level_bucket["pass_count"] += 1
            round_id = str(obj.get("round_id") or "")
            meta = round_meta.get(round_id, {})
            environment_major = str(meta.get("environment_major") or "unknown")
            environment_detail = str(meta.get("environment_detail") or "unknown")
            environment_detail_label = f"{environment_major}-{environment_detail}"
            primary_category = str(meta.get("primary_category") or obj.get("primary_category") or "unknown")
            avg_score = maybe_float(obj.get("avg_score"))
            if avg_score is not None:
                task_bucket = task_turn_buckets[(model, task_name)]
                task_bucket["score_sum"] += avg_score
                task_bucket["count"] += 1
                if primary_category and primary_category != "unknown":
                    ability_bucket = primary_ability_pass_buckets[(model, primary_category)]
                    ability_bucket["count"] += 1
                    if avg_score >= 4.0:
                        ability_bucket["pass_count"] += 1
                scene_bucket = scene_major_buckets[(model, environment_major)]
                scene_bucket["score_sum"] += avg_score
                scene_bucket["count"] += 1
                detail_bucket = scene_detail_buckets[(model, environment_detail_label)]
                detail_bucket["score_sum"] += avg_score
                detail_bucket["count"] += 1
                pair_bucket = scene_primary_buckets[(environment_major, primary_category)]
                pair_bucket["score_sum"] += avg_score
                pair_bucket["count"] += 1
                for threshold_key, upper, _ in LOW_SCORE_THRESHOLDS:
                    if avg_score < upper:
                        task_bucket[threshold_key] += 1
                        scene_bucket[threshold_key] += 1
                        detail_bucket[threshold_key] += 1
                        pair_bucket[threshold_key] += 1

            for metric, value in (obj.get("score_vector") or {}).items():
                score = maybe_float(value)
                if score is None:
                    continue
                for threshold_key, upper, _ in LOW_SCORE_THRESHOLDS:
                    bucket = metric_threshold_counts[(model, "Turn-Level", str(metric), threshold_key)]
                    bucket["total_count"] += 1
                    if score < upper:
                        bucket["low_count"] += 1

        for obj in iter_jsonl(phase3_session_path) or []:
            if obj.get("status") != "success":
                continue
            if not include_task_for_variant(info, task_name_from_record(obj), gemini_text_only):
                continue
            level_avg_score = metric_vector_mean(obj, SESSION_METRICS)
            if level_avg_score is not None:
                level_bucket = level_pass_counts[(model, "Session-Level")]
                level_bucket["total_count"] += 1
                if level_avg_score >= 4.0:
                    level_bucket["pass_count"] += 1
            for metric, value in (obj.get("score_vector") or {}).items():
                score = maybe_float(value)
                if score is None:
                    continue
                for threshold_key, upper, _ in LOW_SCORE_THRESHOLDS:
                    bucket = metric_threshold_counts[(model, "Session-Level", str(metric), threshold_key)]
                    bucket["total_count"] += 1
                    if score < upper:
                        bucket["low_count"] += 1

        turn_summary = safe_read_json(phase4_turn_summary_path)
        session_summary = safe_read_json(phase4_session_summary_path)
        metric_failure = safe_read_json(phase4_metric_failure_path)
        task_error_summary = safe_read_json(phase4_task_error_path)

        turn_low_count = maybe_int(turn_summary.get("record_count")) or 0
        session_low_count = maybe_int(session_summary.get("record_count")) or 0
        turn_critical = maybe_int(
            ((((task_error_summary.get("overall") or {}).get("turn") or {}).get("severity_counts") or {}).get("critical"))
        ) or 0
        session_critical = maybe_int(
            ((((task_error_summary.get("overall") or {}).get("session") or {}).get("severity_counts") or {}).get("critical"))
        ) or 0
        model_turn_total = maybe_int(summary_lookup.get(model, {}).get("turn_sample_count_est")) or 0
        model_session_total = maybe_int(summary_lookup.get(model, {}).get("session_sample_count_est")) or 0
        if gemini_text_only and series == "Gemini":
            turn_low_records = [
                obj
                for obj in (iter_jsonl(phase4_low_turns_path) or [])
                if include_task_for_variant(info, task_name_from_record(obj), gemini_text_only)
            ]
            session_low_records = [
                obj
                for obj in (iter_jsonl(phase4_low_sessions_path) or [])
                if include_task_for_variant(info, task_name_from_record(obj), gemini_text_only)
            ]
            turn_low_count = len(turn_low_records)
            session_low_count = len(session_low_records)
            turn_critical = sum(
                1 for obj in turn_low_records if str(obj.get("severity") or "") == "critical"
            )
            session_critical = sum(
                1 for obj in session_low_records if str(obj.get("severity") or "") == "critical"
            )
            phase4_model_rows.append(
                {
                    "model": model,
                    "series": series,
                    "turn_low_count": turn_low_count,
                    "turn_low_rate": (turn_low_count / model_turn_total) if model_turn_total else None,
                    "turn_critical_share": (turn_critical / turn_low_count) if turn_low_count else None,
                    "session_low_count": session_low_count,
                    "session_low_rate": (
                        session_low_count / model_session_total
                    )
                    if model_session_total
                    else None,
                    "session_critical_share": (
                        session_critical / session_low_count
                    )
                    if session_low_count
                    else None,
                }
            )

            for obj in iter_jsonl(phase4_turn_attributions_path) or []:
                if obj.get("status") != "success":
                    continue
                task_name = task_name_from_record(obj)
                if not include_task_for_variant(info, task_name, gemini_text_only):
                    continue
                primary = str(obj.get("primary_error_category") or "").strip()
                if primary:
                    phase4_turn_primary_counts_by_model[model][primary] += 1
                    phase4_turn_primary_overall[primary] += 1
                    if task_name:
                        phase4_turn_task_primary_nested[task_name][primary] += 1
                secondary_categories = [
                    str(category or "").strip()
                    for category in coerce_list(obj.get("secondary_error_categories"))
                    if str(category or "").strip()
                ]
                for category in secondary_categories:
                    phase4_turn_secondary_counts_by_model[model][category] += 1
                    phase4_turn_secondary_overall[category] += 1
                for metric in coerce_list(obj.get("affected_metrics")):
                    metric_name = str(metric or "").strip()
                    if not metric_name:
                        continue
                    if primary:
                        phase4_turn_metric_primary_nested[metric_name][primary] += 1
                    for category in secondary_categories:
                        phase4_turn_metric_secondary_nested[metric_name][category] += 1

            for obj in iter_jsonl(phase4_session_attributions_path) or []:
                if obj.get("status") != "success":
                    continue
                task_name = task_name_from_record(obj)
                if not include_task_for_variant(info, task_name, gemini_text_only):
                    continue
                primary = str(obj.get("primary_error_category") or "").strip()
                if primary:
                    phase4_session_primary_counts_by_model[model][primary] += 1
                    phase4_session_primary_overall[primary] += 1
                    if task_name:
                        phase4_session_task_primary_nested[task_name][primary] += 1
                secondary_categories = [
                    str(category or "").strip()
                    for category in coerce_list(obj.get("secondary_error_categories"))
                    if str(category or "").strip()
                ]
                for category in secondary_categories:
                    phase4_session_secondary_counts_by_model[model][category] += 1
                    phase4_session_secondary_overall[category] += 1
                for metric in coerce_list(obj.get("affected_metrics")):
                    metric_name = str(metric or "").strip()
                    if not metric_name:
                        continue
                    if primary:
                        phase4_session_metric_primary_nested[metric_name][primary] += 1
                    for category in secondary_categories:
                        phase4_session_metric_secondary_nested[metric_name][category] += 1

            continue
        phase4_model_rows.append(
            {
                "model": model,
                "series": series,
                "turn_low_count": turn_low_count,
                "turn_low_rate": (turn_low_count / model_turn_total) if model_turn_total else None,
                "turn_critical_share": (turn_critical / turn_low_count) if turn_low_count else None,
                "session_low_count": session_low_count,
                "session_low_rate": (session_low_count / model_session_total) if model_session_total else None,
                "session_critical_share": (session_critical / session_low_count) if session_low_count else None,
            }
        )

        for category, count in (turn_summary.get("primary_error_category_counts") or {}).items():
            count_int = maybe_int(count) or 0
            phase4_turn_primary_counts_by_model[model][str(category)] += count_int
            phase4_turn_primary_overall[str(category)] += count_int
        for category, count in (session_summary.get("primary_error_category_counts") or {}).items():
            count_int = maybe_int(count) or 0
            phase4_session_primary_counts_by_model[model][str(category)] += count_int
            phase4_session_primary_overall[str(category)] += count_int
        for category, count in (turn_summary.get("secondary_error_category_counts") or {}).items():
            count_int = maybe_int(count) or 0
            phase4_turn_secondary_counts_by_model[model][str(category)] += count_int
            phase4_turn_secondary_overall[str(category)] += count_int
        for category, count in (session_summary.get("secondary_error_category_counts") or {}).items():
            count_int = maybe_int(count) or 0
            phase4_session_secondary_counts_by_model[model][str(category)] += count_int
            phase4_session_secondary_overall[str(category)] += count_int

        for metric, counts in (metric_failure.get("turn_metric_to_primary_error_counts") or {}).items():
            for category, count in (counts or {}).items():
                phase4_turn_metric_primary_nested[str(metric)][str(category)] += maybe_int(count) or 0
        for metric, counts in (metric_failure.get("session_metric_to_primary_error_counts") or {}).items():
            for category, count in (counts or {}).items():
                phase4_session_metric_primary_nested[str(metric)][str(category)] += maybe_int(count) or 0

        for obj in iter_jsonl(phase4_turn_attributions_path) or []:
            if obj.get("status") != "success":
                continue
            for metric in list(obj.get("affected_metrics") or []):
                metric_name = str(metric or "").strip()
                if not metric_name:
                    continue
                for category in list(obj.get("secondary_error_categories") or []):
                    category_name = str(category or "").strip()
                    if category_name:
                        phase4_turn_metric_secondary_nested[metric_name][category_name] += 1

        for obj in iter_jsonl(phase4_session_attributions_path) or []:
            if obj.get("status") != "success":
                continue
            for metric in list(obj.get("affected_metrics") or []):
                metric_name = str(metric or "").strip()
                if not metric_name:
                    continue
                for category in list(obj.get("secondary_error_categories") or []):
                    category_name = str(category or "").strip()
                    if category_name:
                        phase4_session_metric_secondary_nested[metric_name][category_name] += 1

        for task_name, payload in (task_error_summary.get("by_task") or {}).items():
            turn_counts = (((payload or {}).get("turn") or {}).get("primary_error_category_counts") or {})
            session_counts = (((payload or {}).get("session") or {}).get("primary_error_category_counts") or {})
            for category, count in turn_counts.items():
                phase4_turn_task_primary_nested[str(task_name)][str(category)] += maybe_int(count) or 0
            for category, count in session_counts.items():
                phase4_session_task_primary_nested[str(task_name)][str(category)] += maybe_int(count) or 0

    metric_threshold_rows: list[dict[str, Any]] = []
    for (model, level, metric, threshold_key), bucket in metric_threshold_counts.items():
        total_count = int(bucket["total_count"])
        low_count = int(bucket["low_count"])
        metric_threshold_rows.append(
            {
                "model": model,
                "series": summary_lookup.get(model, {}).get("series", ""),
                "level": level,
                "metric": metric,
                "threshold": threshold_key,
                "threshold_label": threshold_label(threshold_key),
                "low_count": low_count,
                "total_count": total_count,
                "low_rate": (low_count / total_count) if total_count else None,
            }
        )

    level_pass_rate_rows: list[dict[str, Any]] = []
    for (model, level), bucket in level_pass_counts.items():
        total_count = int(bucket["total_count"])
        pass_count = int(bucket["pass_count"])
        level_pass_rate_rows.append(
            {
                "model": model,
                "series": summary_lookup.get(model, {}).get("series", ""),
                "level": level,
                "pass_count": pass_count,
                "total_count": total_count,
                "pass_rate": (pass_count / total_count) if total_count else None,
            }
        )

    level_pass_rate_pivot = pivot_rows_by_label(
        level_pass_rate_rows,
        model_order,
        "level",
        "pass_rate",
        preferred_order=["Turn-Level", "Session-Level"],
    )

    metric_threshold_pivots: dict[str, list[dict[str, Any]]] = {}
    metric_threshold_overall_rows: list[dict[str, Any]] = []
    for level in ("Turn-Level", "Session-Level"):
        metrics = ordered_unique(
            [str(row.get("metric")) for row in metric_threshold_rows if row.get("level") == level]
        )
        for threshold_key, _, _ in LOW_SCORE_THRESHOLDS:
            filtered = [
                row
                for row in metric_threshold_rows
                if row.get("level") == level and row.get("threshold") == threshold_key
            ]
            pivot = metric_pivot_rows(filtered, model_order, value_key="low_rate")
            metric_threshold_pivots[f"{level}::{threshold_key}"] = pivot
        for metric in metrics:
            base = {"level": level, "metric": metric}
            for threshold_key, _, _ in LOW_SCORE_THRESHOLDS:
                values = finite_values(
                    [
                        maybe_float(row.get("low_rate"))
                        for row in metric_threshold_rows
                        if row.get("level") == level
                        and row.get("metric") == metric
                        and row.get("threshold") == threshold_key
                    ]
                )
                base[threshold_key] = round(mean(values), 4) if values else None
            metric_threshold_overall_rows.append(base)

    task_turn_low_rows: list[dict[str, Any]] = []
    for (model, task_name), bucket in task_turn_buckets.items():
        count = int(bucket["count"])
        if not count:
            continue
        row = {
            "model": model,
            "series": summary_lookup.get(model, {}).get("series", ""),
            "task": task_name,
            "avg_score": round(bucket["score_sum"] / count, 4),
            "sample_count": count,
        }
        for threshold_key, _, _ in LOW_SCORE_THRESHOLDS:
            low_count = int(bucket[threshold_key])
            row[f"{threshold_key}_count"] = low_count
            row[f"{threshold_key}_rate"] = low_count / count
        task_turn_low_rows.append(row)

    task_turn_threshold_pivots = {
        threshold_key: pivot_rows_by_label(
            task_turn_low_rows,
            model_order,
            "task",
            f"{threshold_key}_rate",
            preferred_order=TASK_ORDER,
        )
        for threshold_key, _, _ in LOW_SCORE_THRESHOLDS
    }
    task_turn_threshold_overall_rows = aggregate_mean_threshold_rows(
        task_turn_low_rows,
        "task",
        preferred_order=TASK_ORDER,
    )

    primary_ability_pass_rows: list[dict[str, Any]] = []
    for (model, ability), bucket in primary_ability_pass_buckets.items():
        count = int(bucket["count"])
        if not count:
            continue
        pass_count = int(bucket["pass_count"])
        primary_ability_pass_rows.append(
            {
                "model": model,
                "series": summary_lookup.get(model, {}).get("series", ""),
                "ability": ability,
                "sample_count": count,
                "pass_count": pass_count,
                "pass_rate": pass_count / count,
            }
        )

    scene_major_model_rows: list[dict[str, Any]] = []
    for (model, environment_major), bucket in scene_major_buckets.items():
        count = int(bucket["count"])
        if not count:
            continue
        row = {
            "model": model,
            "series": summary_lookup.get(model, {}).get("series", ""),
            "environment_major": environment_major,
            "avg_score": round(bucket["score_sum"] / count, 4),
            "sample_count": count,
        }
        for threshold_key, _, _ in LOW_SCORE_THRESHOLDS:
            low_count = int(bucket[threshold_key])
            row[f"{threshold_key}_count"] = low_count
            row[f"{threshold_key}_rate"] = low_count / count
        scene_major_model_rows.append(row)

    scene_major_score_pivot = pivot_rows_by_label(
        scene_major_model_rows,
        model_order,
        "environment_major",
        "avg_score",
        preferred_order=major_order,
    )
    scene_major_low_rate_pivots = {
        threshold_key: pivot_rows_by_label(
            scene_major_model_rows,
            model_order,
            "environment_major",
            f"{threshold_key}_rate",
            preferred_order=major_order,
        )
        for threshold_key, _, _ in LOW_SCORE_THRESHOLDS
    }
    scene_major_pass_rate_pivot = complement_rate_pivot(
        scene_major_low_rate_pivots.get("lt4", []),
        model_order,
        ["environment_major"],
    )
    scene_major_overall_rows = aggregate_mean_threshold_rows(
        scene_major_model_rows,
        "environment_major",
        preferred_order=major_order,
    )
    scene_major_score_stats = distribution_stats_for_pivot(
        scene_major_score_pivot,
        model_order,
        ["environment_major"],
        "场景均分",
        lower_is_better=False,
    )
    scene_major_lt4_stats = distribution_stats_for_pivot(
        scene_major_low_rate_pivots.get("lt4", []),
        model_order,
        ["environment_major"],
        "场景<4低分率",
        lower_is_better=True,
    )
    scene_major_pass_stats = distribution_stats_for_pivot(
        scene_major_pass_rate_pivot,
        model_order,
        ["environment_major"],
        "场景通过率",
        lower_is_better=False,
    )

    scene_detail_model_rows: list[dict[str, Any]] = []
    for (model, environment_detail), bucket in scene_detail_buckets.items():
        count = int(bucket["count"])
        if not count:
            continue
        major, detail = split_environment_label(environment_detail)
        row = {
            "model": model,
            "series": summary_lookup.get(model, {}).get("series", ""),
            "environment_detail": environment_detail,
            "environment_major": major,
            "detail": detail,
            "avg_score": round(bucket["score_sum"] / count, 4),
            "sample_count": count,
        }
        for threshold_key, _, _ in LOW_SCORE_THRESHOLDS:
            low_count = int(bucket[threshold_key])
            row[f"{threshold_key}_count"] = low_count
            row[f"{threshold_key}_rate"] = low_count / count
        scene_detail_model_rows.append(row)

    scene_detail_score_pivot = pivot_rows_by_label(
        scene_detail_model_rows,
        model_order,
        "environment_detail",
        "avg_score",
        preferred_order=detail_order,
    )
    scene_detail_low_rate_pivots = {
        threshold_key: pivot_rows_by_label(
            scene_detail_model_rows,
            model_order,
            "environment_detail",
            f"{threshold_key}_rate",
            preferred_order=detail_order,
        )
        for threshold_key, _, _ in LOW_SCORE_THRESHOLDS
    }
    scene_detail_pass_rate_pivot = complement_rate_pivot(
        scene_detail_low_rate_pivots.get("lt4", []),
        model_order,
        ["environment_detail"],
    )
    scene_detail_overall_rows = aggregate_mean_threshold_rows(
        scene_detail_model_rows,
        "environment_detail",
        preferred_order=detail_order,
    )
    scene_detail_score_stats = distribution_stats_for_pivot(
        scene_detail_score_pivot,
        model_order,
        ["environment_detail"],
        "Scene detail score",
        lower_is_better=False,
    )
    scene_detail_lt4_stats = distribution_stats_for_pivot(
        scene_detail_low_rate_pivots.get("lt4", []),
        model_order,
        ["environment_detail"],
        "Scene detail <4 low rate",
        lower_is_better=True,
    )
    scene_detail_pass_stats = distribution_stats_for_pivot(
        scene_detail_pass_rate_pivot,
        model_order,
        ["environment_detail"],
        "Scene detail pass rate",
        lower_is_better=False,
    )

    primary_count_totals: dict[str, int] = defaultdict(int)
    for (_, primary_category), bucket in scene_primary_buckets.items():
        primary_count_totals[primary_category] += int(bucket["count"])
    top_scene_primary_categories = [
        category
        for category, _ in sorted(primary_count_totals.items(), key=lambda item: (-item[1], item[0]))[:18]
    ]
    scene_primary_score_rows: list[dict[str, Any]] = []
    scene_primary_low_rate_rows: list[dict[str, Any]] = []
    scene_primary_pass_rate_rows: list[dict[str, Any]] = []
    for environment_major in major_order:
        score_row: dict[str, Any] = {"environment_major": environment_major}
        low_row: dict[str, Any] = {"environment_major": environment_major}
        pass_row: dict[str, Any] = {"environment_major": environment_major}
        for primary_category in top_scene_primary_categories:
            bucket = scene_primary_buckets.get((environment_major, primary_category), {})
            count = int(bucket.get("count") or 0)
            low_count = int(bucket.get("lt4") or 0)
            score_row[primary_category] = (
                round(float(bucket.get("score_sum") or 0.0) / count, 4) if count else None
            )
            low_row[primary_category] = (
                round(low_count / count, 4) if count else None
            )
            pass_row[primary_category] = (
                round(1.0 - (low_count / count), 4) if count else None
            )
            score_row[f"{primary_category}__count"] = count
            low_row[f"{primary_category}__count"] = count
            pass_row[f"{primary_category}__count"] = count
        scene_primary_score_rows.append(score_row)
        scene_primary_low_rate_rows.append(low_row)
        scene_primary_pass_rate_rows.append(pass_row)

    turn_primary_categories = [
        category
        for category, _ in sorted(phase4_turn_primary_overall.items(), key=lambda item: (-item[1], item[0]))
    ]
    session_primary_categories = [
        category
        for category, _ in sorted(phase4_session_primary_overall.items(), key=lambda item: (-item[1], item[0]))
    ]
    phase4_turn_primary_pivot: list[dict[str, Any]] = []
    phase4_turn_primary_count_pivot: list[dict[str, Any]] = []
    for category in turn_primary_categories:
        row = {"error_category": category}
        count_row = {"error_category": category}
        for model in model_order:
            total = sum(phase4_turn_primary_counts_by_model.get(model, {}).values())
            count = int(phase4_turn_primary_counts_by_model.get(model, {}).get(category, 0))
            row[model] = (count / total) if total else None
            count_row[model] = count
        phase4_turn_primary_pivot.append(row)
        phase4_turn_primary_count_pivot.append(count_row)
    phase4_session_primary_pivot: list[dict[str, Any]] = []
    phase4_session_primary_count_pivot: list[dict[str, Any]] = []
    for category in session_primary_categories:
        row = {"error_category": category}
        count_row = {"error_category": category}
        for model in model_order:
            total = sum(phase4_session_primary_counts_by_model.get(model, {}).values())
            count = int(phase4_session_primary_counts_by_model.get(model, {}).get(category, 0))
            row[model] = (count / total) if total else None
            count_row[model] = count
        phase4_session_primary_pivot.append(row)
        phase4_session_primary_count_pivot.append(count_row)
    phase4_turn_primary_count_pivot = prepend_model_total_row(
        phase4_turn_primary_count_pivot,
        model_order,
        "error_category",
    )
    phase4_session_primary_count_pivot = prepend_model_total_row(
        phase4_session_primary_count_pivot,
        model_order,
        "error_category",
    )

    phase4_turn_primary_overall_rows = []
    turn_primary_total = sum(phase4_turn_primary_overall.values())
    for category in turn_primary_categories:
        count = int(phase4_turn_primary_overall[category])
        phase4_turn_primary_overall_rows.append(
            {
                "error_category": category,
                "count": count,
                "share": (count / turn_primary_total) if turn_primary_total else None,
            }
        )
    phase4_session_primary_overall_rows = []
    session_primary_total = sum(phase4_session_primary_overall.values())
    for category in session_primary_categories:
        count = int(phase4_session_primary_overall[category])
        phase4_session_primary_overall_rows.append(
            {
                "error_category": category,
                "count": count,
                "share": (count / session_primary_total) if session_primary_total else None,
            }
        )
    phase4_turn_secondary_overall_rows = [
        {
            "error_category": category,
            "count": int(count),
            "share": (int(count) / sum(phase4_turn_secondary_overall.values()))
            if phase4_turn_secondary_overall
            else None,
        }
        for category, count in sorted(phase4_turn_secondary_overall.items(), key=lambda item: (-item[1], item[0]))
    ]
    phase4_session_secondary_overall_rows = [
        {
            "error_category": category,
            "count": int(count),
            "share": (int(count) / sum(phase4_session_secondary_overall.values()))
            if phase4_session_secondary_overall
            else None,
        }
        for category, count in sorted(phase4_session_secondary_overall.items(), key=lambda item: (-item[1], item[0]))
    ]
    turn_secondary_categories = [
        str(row.get("error_category") or "") for row in phase4_turn_secondary_overall_rows
    ]
    session_secondary_categories = [
        str(row.get("error_category") or "") for row in phase4_session_secondary_overall_rows
    ]
    phase4_turn_secondary_pivot: list[dict[str, Any]] = []
    phase4_turn_secondary_count_pivot: list[dict[str, Any]] = []
    for category in turn_secondary_categories:
        row = {"error_category": category}
        count_row = {"error_category": category}
        for model in model_order:
            total = sum(phase4_turn_secondary_counts_by_model.get(model, {}).values())
            count = int(phase4_turn_secondary_counts_by_model.get(model, {}).get(category, 0))
            row[model] = (count / total) if total else None
            count_row[model] = count
        phase4_turn_secondary_pivot.append(row)
        phase4_turn_secondary_count_pivot.append(count_row)
    phase4_session_secondary_pivot: list[dict[str, Any]] = []
    phase4_session_secondary_count_pivot: list[dict[str, Any]] = []
    for category in session_secondary_categories:
        row = {"error_category": category}
        count_row = {"error_category": category}
        for model in model_order:
            total = sum(phase4_session_secondary_counts_by_model.get(model, {}).values())
            count = int(phase4_session_secondary_counts_by_model.get(model, {}).get(category, 0))
            row[model] = (count / total) if total else None
            count_row[model] = count
        phase4_session_secondary_pivot.append(row)
        phase4_session_secondary_count_pivot.append(count_row)
    phase4_turn_secondary_count_pivot = prepend_model_total_row(
        phase4_turn_secondary_count_pivot,
        model_order,
        "error_category",
    )
    phase4_session_secondary_count_pivot = prepend_model_total_row(
        phase4_session_secondary_count_pivot,
        model_order,
        "error_category",
    )

    phase4_turn_metric_primary_rows, phase4_turn_metric_categories = matrix_rows_from_nested_counts(
        phase4_turn_metric_primary_nested,
        "metric",
    )
    phase4_session_metric_primary_rows, phase4_session_metric_categories = matrix_rows_from_nested_counts(
        phase4_session_metric_primary_nested,
        "metric",
    )
    phase4_turn_metric_secondary_rows, phase4_turn_metric_secondary_categories = matrix_rows_from_nested_counts(
        phase4_turn_metric_secondary_nested,
        "metric",
    )
    phase4_session_metric_secondary_rows, phase4_session_metric_secondary_categories = matrix_rows_from_nested_counts(
        phase4_session_metric_secondary_nested,
        "metric",
    )
    phase4_turn_task_primary_rows, phase4_turn_task_categories = matrix_rows_from_nested_counts(
        phase4_turn_task_primary_nested,
        "task",
        row_order=TASK_ORDER,
    )
    phase4_session_task_primary_rows, phase4_session_task_categories = matrix_rows_from_nested_counts(
        phase4_session_task_primary_nested,
        "task",
        row_order=TASK_ORDER,
    )

    return {
        "generation_rows": generation_rows,
        "metric_threshold_rows": metric_threshold_rows,
        "metric_threshold_pivots": metric_threshold_pivots,
        "metric_threshold_overall_rows": metric_threshold_overall_rows,
        "level_pass_rate_rows": level_pass_rate_rows,
        "level_pass_rate_pivot": level_pass_rate_pivot,
        "task_turn_low_rows": task_turn_low_rows,
        "task_turn_threshold_pivots": task_turn_threshold_pivots,
        "task_turn_threshold_overall_rows": task_turn_threshold_overall_rows,
        "primary_ability_pass_rows": primary_ability_pass_rows,
        "scene_major_model_rows": scene_major_model_rows,
        "scene_major_score_pivot": scene_major_score_pivot,
        "scene_major_low_rate_pivots": scene_major_low_rate_pivots,
        "scene_major_pass_rate_pivot": scene_major_pass_rate_pivot,
        "scene_major_overall_rows": scene_major_overall_rows,
        "scene_major_score_stats": scene_major_score_stats,
        "scene_major_lt4_stats": scene_major_lt4_stats,
        "scene_major_pass_stats": scene_major_pass_stats,
        "scene_detail_model_rows": scene_detail_model_rows,
        "scene_detail_score_pivot": scene_detail_score_pivot,
        "scene_detail_low_rate_pivots": scene_detail_low_rate_pivots,
        "scene_detail_pass_rate_pivot": scene_detail_pass_rate_pivot,
        "scene_detail_overall_rows": scene_detail_overall_rows,
        "scene_detail_score_stats": scene_detail_score_stats,
        "scene_detail_lt4_stats": scene_detail_lt4_stats,
        "scene_detail_pass_stats": scene_detail_pass_stats,
        "scene_primary_top_categories": top_scene_primary_categories,
        "scene_primary_score_rows": scene_primary_score_rows,
        "scene_primary_low_rate_rows": scene_primary_low_rate_rows,
        "scene_primary_pass_rate_rows": scene_primary_pass_rate_rows,
        "phase4_model_rows": phase4_model_rows,
        "phase4_turn_primary_pivot": phase4_turn_primary_pivot,
        "phase4_session_primary_pivot": phase4_session_primary_pivot,
        "phase4_turn_primary_count_pivot": phase4_turn_primary_count_pivot,
        "phase4_session_primary_count_pivot": phase4_session_primary_count_pivot,
        "phase4_turn_primary_overall_rows": phase4_turn_primary_overall_rows,
        "phase4_session_primary_overall_rows": phase4_session_primary_overall_rows,
        "phase4_turn_secondary_overall_rows": phase4_turn_secondary_overall_rows,
        "phase4_session_secondary_overall_rows": phase4_session_secondary_overall_rows,
        "phase4_turn_secondary_pivot": phase4_turn_secondary_pivot,
        "phase4_session_secondary_pivot": phase4_session_secondary_pivot,
        "phase4_turn_secondary_count_pivot": phase4_turn_secondary_count_pivot,
        "phase4_session_secondary_count_pivot": phase4_session_secondary_count_pivot,
        "phase4_turn_metric_primary_rows": phase4_turn_metric_primary_rows,
        "phase4_session_metric_primary_rows": phase4_session_metric_primary_rows,
        "phase4_turn_metric_secondary_rows": phase4_turn_metric_secondary_rows,
        "phase4_session_metric_secondary_rows": phase4_session_metric_secondary_rows,
        "phase4_turn_metric_categories": phase4_turn_metric_categories,
        "phase4_session_metric_categories": phase4_session_metric_categories,
        "phase4_turn_metric_secondary_categories": phase4_turn_metric_secondary_categories,
        "phase4_session_metric_secondary_categories": phase4_session_metric_secondary_categories,
        "phase4_turn_task_primary_rows": phase4_turn_task_primary_rows,
        "phase4_session_task_primary_rows": phase4_session_task_primary_rows,
        "phase4_turn_task_categories": phase4_turn_task_categories,
        "phase4_session_task_categories": phase4_session_task_categories,
    }


def ordered_unique(values: list[str], preferred: list[str] | None = None) -> list[str]:
    preferred = preferred or []
    seen = set()
    ordered: list[str] = []
    for value in preferred:
        if value in values and value not in seen:
            ordered.append(value)
            seen.add(value)
    for value in sorted(values):
        if value not in seen:
            ordered.append(value)
            seen.add(value)
    return ordered


def unique_task_rows(task_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for row in task_rows:
        model = row.get("model")
        task = row.get("task")
        score = row.get("avg_score")
        if not model or not task or score is None:
            continue
        key = (model, task)
        existing = by_key.get(key)
        if existing is None or str(row.get("source_table", "")).startswith("最强"):
            by_key[key] = row
    return list(by_key.values())


def rankings_by_group(
    rows: list[dict[str, Any]],
    group_key: str,
    value_key: str,
    descending: bool = True,
) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        group = row.get(group_key)
        value = row.get(value_key)
        if group and value is not None:
            grouped[str(group)].append(row)

    ranked: dict[str, list[dict[str, Any]]] = {}
    for group, items in grouped.items():
        sorted_items = sorted(
            items,
            key=lambda row: row.get(value_key) if row.get(value_key) is not None else -math.inf,
            reverse=descending,
        )
        result: list[dict[str, Any]] = []
        last_value: Any = None
        last_rank = 0
        for index, item in enumerate(sorted_items, start=1):
            value = item.get(value_key)
            rank = last_rank if value == last_value else index
            last_value = value
            last_rank = rank
            result.append({**item, "rank": rank})
        ranked[group] = result
    return ranked


def rank_lookup_for_key(
    rows: list[dict[str, Any]],
    value_key: str,
    id_key: str = "model",
    descending: bool = True,
) -> dict[str, int]:
    scored: list[tuple[str, float]] = []
    for row in rows:
        value = maybe_float(row.get(value_key))
        model = row.get(id_key)
        if value is None or model is None:
            continue
        scored.append((str(model), value))
    scored.sort(key=lambda item: item[1], reverse=descending)

    ranks: dict[str, int] = {}
    last_value: float | None = None
    last_rank = 0
    for index, (model, value) in enumerate(scored, start=1):
        rank = last_rank if value == last_value else index
        ranks[model] = rank
        last_value = value
        last_rank = rank
    return ranks


def rank_lookup_for_row_values(
    row: dict[str, Any],
    model_order: list[str],
    descending: bool = True,
) -> dict[str, int]:
    scored: list[tuple[str, float]] = []
    for model in model_order:
        value = maybe_float(row.get(model))
        if value is not None:
            scored.append((model, value))
    scored.sort(key=lambda item: item[1], reverse=descending)

    ranks: dict[str, int] = {}
    last_value: float | None = None
    last_rank = 0
    for index, (model, value) in enumerate(scored, start=1):
        rank = last_rank if value == last_value else index
        ranks[model] = rank
        last_value = value
        last_rank = rank
    return ranks


def task_pivot_rows(
    task_rows: list[dict[str, Any]],
    model_summary: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[str], dict[str, list[dict[str, Any]]]]:
    unique_rows = unique_task_rows(task_rows)
    model_order = [row["model"] for row in model_summary]
    tasks = ordered_unique([row["task"] for row in unique_rows], TASK_ORDER)
    ranks = rankings_by_group(unique_rows, "task", "avg_score", descending=True)
    rank_lookup = {
        (row["model"], task): row["rank"]
        for task, rows in ranks.items()
        for row in rows
    }
    scores = {(row["model"], row["task"]): row for row in unique_rows}

    matrix: list[dict[str, Any]] = []
    summary_by_model = {row["model"]: row for row in model_summary}
    for model in model_order:
        row = {
            "model": model,
            "series": summary_by_model.get(model, {}).get("series", ""),
        }
        for task in tasks:
            task_row = scores.get((model, task), {})
            row[task] = task_row.get("avg_score")
            row[f"{task}__rank"] = rank_lookup.get((model, task))
            row[f"{task}__sample_count"] = task_row.get("sample_count")
        matrix.append(row)
    return matrix, tasks, ranks


def metric_rankings(metric_rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    rows = [
        {
            **row,
            "metric_label": f"{row.get('level', '')} / {row.get('metric', '')}",
        }
        for row in metric_rows
        if row.get("avg_score") is not None
    ]
    return rankings_by_group(rows, "metric_label", "avg_score", descending=True)


def ability_rankings(ability_rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    return rankings_by_group(
        [row for row in ability_rows if row.get("overall_score") is not None],
        "ability",
        "overall_score",
        descending=True,
    )


def score_color(value: float | None, lower: float = 1.0, upper: float = 5.0) -> str:
    if value is None:
        return "#f1f5f9"
    ratio = max(0.0, min(1.0, (value - lower) / (upper - lower)))
    if ratio < 0.5:
        start = (190, 70, 52)
        end = (243, 198, 91)
        t = ratio / 0.5
    else:
        start = (243, 198, 91)
        end = (59, 151, 105)
        t = (ratio - 0.5) / 0.5
    rgb = tuple(round(start[i] + (end[i] - start[i]) * t) for i in range(3))
    return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"


def text_color_for_background(hex_color: str) -> str:
    hex_color = hex_color.lstrip("#")
    red = int(hex_color[0:2], 16)
    green = int(hex_color[2:4], 16)
    blue = int(hex_color[4:6], 16)
    luminance = (0.299 * red + 0.587 * green + 0.114 * blue) / 255
    return "#111827" if luminance > 0.58 else "#ffffff"


def fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def percent(value: Any) -> str:
    if value is None:
        return ""
    return f"{float(value) * 100:.1f}%"


def bar_cell(value: float | None, max_value: float = 5.0, rank: Any = None) -> str:
    if value is None:
        return '<span class="muted">n/a</span>'
    width = max(0.0, min(100.0, value / max_value * 100.0))
    color = score_color(value)
    rank_text = f"<small>#{rank}</small>" if rank else ""
    return (
        '<div class="bar-cell">'
        f'<div class="bar" style="width:{width:.1f}%;background:{color}"></div>'
        f"<span>{value:.3f}{rank_text}</span>"
        "</div>"
    )


def heat_cell(
    value: Any,
    lower: float = 1.0,
    upper: float = 5.0,
    reverse: bool = False,
    as_percent: bool = False,
    rank: Any = None,
    digits: int = 3,
) -> str:
    number = maybe_float(value)
    if number is None:
        return '<td class="blank"></td>'
    color_value = (upper + lower - number) if reverse else number
    color = score_color(color_value, lower=lower, upper=upper)
    text_color = text_color_for_background(color)
    text = f"{number * 100:.1f}%" if as_percent else f"{number:.{digits}f}"
    rank_text = f"<small>#{rank}</small>" if rank else ""
    class_name = "heat rank-heat" if rank else "heat"
    content = f"<span>{text}</span>{rank_text}" if rank else text
    return (
        f'<td class="{class_name}" style="background:{color};color:{text_color}">'
        f"{content}</td>"
    )


def render_leaderboard_rows(rows: list[dict[str, Any]]) -> str:
    rank_maps = {
        key: rank_lookup_for_key(rows, key)
        for key in (
            "turn_metric_avg",
            "session_metric_avg",
            "text_task_weighted_avg",
            "tts_task_weighted_avg",
            "all_task_weighted_avg",
        )
    }
    html_rows = []
    for row in rows:
        model = str(row.get("model", ""))
        html_rows.append(
            "<tr>"
            f"<td>{escape(model)}</td>"
            f"<td>{escape(str(row.get('series', '')))}</td>"
            f"<td>{escape(str(row.get('task_coverage') or ''))}</td>"
            f"<td>{fmt(row.get('turn_sample_count_est'), 0)}</td>"
            f"<td>{fmt(row.get('session_sample_count_est'), 0)}</td>"
            f"<td>{bar_cell(row.get('turn_metric_avg'), rank=rank_maps['turn_metric_avg'].get(model))}</td>"
            f"<td>{bar_cell(row.get('session_metric_avg'), rank=rank_maps['session_metric_avg'].get(model))}</td>"
            f"<td>{bar_cell(row.get('text_task_weighted_avg'), rank=rank_maps['text_task_weighted_avg'].get(model))}</td>"
            f"<td>{bar_cell(row.get('tts_task_weighted_avg'), rank=rank_maps['tts_task_weighted_avg'].get(model))}</td>"
            f"<td>{bar_cell(row.get('all_task_weighted_avg'), rank=rank_maps['all_task_weighted_avg'].get(model))}</td>"
            f"<td>{escape(str(row.get('covered_tasks') or ''))}</td>"
            "</tr>"
        )
    return "\n".join(html_rows)


def heat_rank_cell(score: Any, rank: Any = None) -> str:
    return heat_cell(score, rank=rank)


def render_task_matrix(
    rows: list[dict[str, Any]],
    tasks: list[str],
    model_order: list[str] | None = None,
) -> str:
    selected = rows
    local_ranks: dict[tuple[str, str], int] = {}
    if model_order is not None:
        allowed = set(model_order)
        selected = [row for row in rows if row.get("model") in allowed]
        for task in tasks:
            scored = sorted(
                [row for row in selected if row.get(task) is not None],
                key=lambda row: row.get(task) or -math.inf,
                reverse=True,
            )
            for index, row in enumerate(scored, start=1):
                local_ranks[(row["model"], task)] = index
    header = "".join(f"<th>{escape(task)}</th>" for task in tasks)
    body = []
    for row in selected:
        cells = "".join(
            heat_rank_cell(
                row.get(task),
                local_ranks.get((row["model"], task), row.get(f"{task}__rank")),
            )
            for task in tasks
        )
        body.append(
            "<tr>"
            f"<th>{escape(str(row.get('model', '')))}</th>"
            f"<td>{escape(str(row.get('series', '')))}</td>"
            f"{cells}</tr>"
        )
    return (
        '<div class="table-scroll"><table class="heatmap task-matrix" data-row-sortable="true">'
        f"<thead><tr><th>Model</th><th>Series</th>{header}</tr></thead>"
        f"<tbody>{''.join(body)}</tbody></table></div>"
    )


def render_task_rankings(
    rankings: dict[str, list[dict[str, Any]]],
    tasks: list[str],
    model_order: list[str] | None = None,
) -> str:
    allowed = set(model_order) if model_order else None
    cards = []
    for task in tasks:
        rows = rankings.get(task, [])
        if allowed is not None:
            rows = [row for row in rows if row.get("model") in allowed]
            rows = sorted(rows, key=lambda row: row.get("avg_score") or -math.inf, reverse=True)
            rows = [{**row, "rank": index} for index, row in enumerate(rows, start=1)]
        body = "".join(
            "<tr>"
            f"<td>{row.get('rank')}</td>"
            f"<td>{escape(str(row.get('model', '')))}</td>"
            f"<td>{escape(str(row.get('series', '')))}</td>"
            f"<td>{fmt(row.get('avg_score'))}</td>"
            f"<td>{fmt(row.get('sample_count'), 0)}</td>"
            "</tr>"
            for row in rows
        )
        cards.append(
            '<details class="rank-detail">'
            f"<summary>{escape(task)}</summary>"
            '<table class="mini-table"><thead><tr><th>#</th><th>Model</th><th>Series</th>'
            "<th>均分</th><th>样本数</th></tr></thead>"
            f"<tbody>{body}</tbody></table></details>"
        )
    return '<div class="rank-grid">' + "\n".join(cards) + "</div>"


def short_task_label(task: str) -> str:
    return (
        task.replace("omnibench_", "")
        .replace("image_multi_", "img_")
        .replace("video_stream_", "vid_")
    )


def short_metric_label(label: str) -> str:
    return (
        label.replace("Turn-Level / ", "T/")
        .replace("Session-Level / ", "S/")
        .replace("overall_helpfulness_trustworthiness", "helpfulness")
        .replace("proactiveness_helpfulness", "proactive")
        .replace("intent_understanding_depth", "intent_depth")
        .replace("user_state_adaptation", "state_adapt")
        .replace("session_consistency", "session_cons")
        .replace("intent_fulfillment", "intent_fulfill")
        .replace("persona_adaptation", "persona")
    )


def metric_pass_row_label(row: dict[str, Any]) -> str:
    return short_metric_label(f"{row.get('level')} / {row.get('metric')}")


def ability_row_label(row: dict[str, Any]) -> str:
    return str(row.get("ability") or "")


def environment_major_row_label(row: dict[str, Any]) -> str:
    return str(row.get("environment_major") or "")


def environment_detail_row_label(row: dict[str, Any]) -> str:
    return str(row.get("environment_detail") or "")


def svg_legend(series_names: list[str], x: int, y: int) -> str:
    items = []
    cursor = x
    for series in series_names:
        color = SERIES_COLORS.get(series, SERIES_COLORS["Other"])
        items.append(
            f'<circle cx="{cursor}" cy="{y}" r="5" fill="{color}" />'
            f'<text x="{cursor + 9}" y="{y + 4}" class="legend">{escape(series)}</text>'
        )
        cursor += 86
    return "".join(items)


def value_to_y(value: float, top: float, bottom: float, lower: float = 1.0, upper: float = 5.0) -> float:
    ratio = (value - lower) / (upper - lower)
    ratio = max(0.0, min(1.0, ratio))
    return bottom - ratio * (bottom - top)


def render_task_line_chart(task_matrix: list[dict[str, Any]], tasks: list[str]) -> str:
    width, height = 1080, 360
    left, right, top, bottom = 88, 40, 34, 272
    if not tasks:
        return ""
    step = (width - left - right) / max(1, len(tasks) - 1)
    x_for = {task: left + index * step for index, task in enumerate(tasks)}
    series_names = ordered_unique([str(row.get("series", "")) for row in task_matrix])

    grid = []
    for score in [1, 2, 3, 4, 5]:
        y = value_to_y(score, top, bottom)
        grid.append(
            f'<line x1="{left}" y1="{y:.1f}" x2="{width - right}" y2="{y:.1f}" class="grid" />'
            f'<text x="{left - 12}" y="{y + 4:.1f}" text-anchor="end" class="axis">{score}</text>'
        )
    labels = []
    for task in tasks:
        x = x_for[task]
        labels.append(
            f'<text x="{x:.1f}" y="{bottom + 28}" text-anchor="middle" class="axis">{escape(short_task_label(task))}</text>'
        )

    lines = []
    for row in task_matrix:
        points = []
        for task in tasks:
            score = row.get(task)
            if score is None:
                continue
            points.append((x_for[task], value_to_y(float(score), top, bottom), task, float(score)))
        if len(points) < 2:
            continue
        color = SERIES_COLORS.get(str(row.get("series")), SERIES_COLORS["Other"])
        point_text = " ".join(f"{x:.1f},{y:.1f}" for x, y, _, _ in points)
        model = str(row.get("model"))
        key = escape(model, quote=True)
        circles = "".join(
            f'<circle class="chart-data" data-key="{key}" data-tooltip="{escape(f"{model} / {task}: {score:.3f}", quote=True)}" '
            f'cx="{x:.1f}" cy="{y:.1f}" r="3.2" fill="{color}" />'
            for x, y, task, score in points
        )
        lines.append(
            f'<polyline class="chart-data" data-key="{key}" data-tooltip="{key}" points="{point_text}" '
            f'fill="none" stroke="{color}" stroke-width="1.8" opacity="0.42" />{circles}'
        )

    return f"""
<div class="chart-card">
  <h3>任务走势折线图</h3>
  <p class="section-note">每条线是一个模型，横轴是四个任务；缺失的 TTS 任务不会补值。它能直接看出同一模型从 image 到 video、从 text 到 TTS 的掉分。</p>
  <svg class="chart" viewBox="0 0 {width} {height}" role="img">
    <style>
      .grid {{ stroke:#e2e8f0; stroke-width:1; }}
      .axis {{ fill:#64748b; font-size:12px; }}
      .legend {{ fill:#475569; font-size:12px; }}
    </style>
    {''.join(grid)}
    <line x1="{left}" y1="{bottom}" x2="{width - right}" y2="{bottom}" class="grid" />
    {''.join(labels)}
    {''.join(lines)}
    {svg_legend(series_names, left, height - 26)}
  </svg>
</div>
"""


def render_task_bar_charts(rankings: dict[str, list[dict[str, Any]]], tasks: list[str], limit: int = 12) -> str:
    cards = []
    for task in tasks:
        rows = rankings.get(task, [])[:limit]
        if not rows:
            continue
        width, height = 460, 34 + len(rows) * 25
        left, right, top = 148, 24, 20
        max_score = max(float(row.get("avg_score") or 0) for row in rows) or 5.0
        min_score = min(float(row.get("avg_score") or 0) for row in rows)
        span = max(0.5, max_score - min_score)
        bars = []
        for index, row in enumerate(rows):
            score = float(row.get("avg_score") or 0)
            y = top + index * 25
            bar_width = (score - max(1.0, min_score - 0.1)) / (span + 0.1) * (width - left - right)
            bar_width = max(4, bar_width)
            color = SERIES_COLORS.get(str(row.get("series")), SERIES_COLORS["Other"])
            model = str(row.get("model"))
            tooltip = f"{model} / {task}: {score:.3f}"
            bars.append(
                f'<text x="{left - 8}" y="{y + 14}" text-anchor="end" class="axis">#{row.get("rank")} {escape(model)}</text>'
                f'<rect class="chart-data" data-key="{escape(model, quote=True)}" data-tooltip="{escape(tooltip, quote=True)}" '
                f'x="{left}" y="{y}" width="{bar_width:.1f}" height="17" rx="3" fill="{color}" opacity="0.82" />'
                f'<text x="{left + bar_width + 6:.1f}" y="{y + 14}" class="axis">{score:.3f}</text>'
            )
        cards.append(
            '<div class="chart-card small-chart">'
            f'<h3>{escape(task)}</h3>'
            f'<svg class="chart" viewBox="0 0 {width} {height}"><style>.axis{{fill:#475569;font-size:11px;}}</style>{"".join(bars)}</svg>'
            "</div>"
        )
    return '<div class="chart-grid">' + "\n".join(cards) + "</div>"


def render_turn_session_scatter(model_summary: list[dict[str, Any]]) -> str:
    width, height = 680, 440
    left, right, top, bottom = 64, 36, 28, 360
    points = []
    series_names = ordered_unique([str(row.get("series", "")) for row in model_summary])
    for row in model_summary:
        turn = row.get("turn_metric_avg")
        session = row.get("session_metric_avg")
        if turn is None or session is None:
            continue
        x = left + (float(turn) - 1.0) / 4.0 * (width - left - right)
        y = value_to_y(float(session), top, bottom)
        color = SERIES_COLORS.get(str(row.get("series")), SERIES_COLORS["Other"])
        model = str(row.get("model"))
        tooltip = f"{model} Turn={float(turn):.3f}, Session={float(session):.3f}"
        points.append(
            f'<circle class="chart-data" data-key="{escape(model, quote=True)}" data-tooltip="{escape(tooltip, quote=True)}" '
            f'cx="{x:.1f}" cy="{y:.1f}" r="6" fill="{color}" opacity="0.82" />'
        )

    grid = []
    for score in [1, 2, 3, 4, 5]:
        y = value_to_y(score, top, bottom)
        x = left + (score - 1) / 4 * (width - left - right)
        grid.append(
            f'<line x1="{left}" y1="{y:.1f}" x2="{width - right}" y2="{y:.1f}" class="grid" />'
            f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{bottom}" class="grid" />'
            f'<text x="{left - 10}" y="{y + 4:.1f}" text-anchor="end" class="axis">{score}</text>'
            f'<text x="{x:.1f}" y="{bottom + 20}" text-anchor="middle" class="axis">{score}</text>'
        )
    return f"""
<div class="chart-card">
  <h3>Turn 均值 vs Session 均值</h3>
  <p class="section-note">右上角代表单轮和整段都强；靠右但偏下的点说明单轮表现不错，但多轮会话拖后腿。</p>
  <svg class="chart" viewBox="0 0 {width} {height}">
    <style>.grid{{stroke:#e2e8f0;stroke-width:1;}}.axis{{fill:#64748b;font-size:12px;}}.legend{{fill:#475569;font-size:12px;}}</style>
    {''.join(grid)}
    <text x="{width / 2:.1f}" y="{height - 16}" text-anchor="middle" class="axis">Turn-Level 宏平均</text>
    <text x="16" y="{(top + bottom) / 2:.1f}" class="axis" transform="rotate(-90 16 {(top + bottom) / 2:.1f})">Session-Level 宏平均</text>
    {''.join(points)}
    {svg_legend(series_names, left, height - 52)}
  </svg>
</div>
"""


def render_metric_profile_chart(metric_score_pivot: list[dict[str, Any]], model_summary: list[dict[str, Any]]) -> str:
    width, height = 1160, 410
    left, right, top, bottom = 76, 30, 28, 295
    labels = [f"{row.get('level')} / {row.get('metric')}" for row in metric_score_pivot]
    if not labels:
        return ""
    step = (width - left - right) / max(1, len(labels) - 1)
    series_names = ordered_unique([str(row.get("series", "")) for row in model_summary])
    model_to_series = {row["model"]: row["series"] for row in model_summary}

    grid = []
    for score in [1, 2, 3, 4, 5]:
        y = value_to_y(score, top, bottom)
        grid.append(
            f'<line x1="{left}" y1="{y:.1f}" x2="{width - right}" y2="{y:.1f}" class="grid" />'
            f'<text x="{left - 10}" y="{y + 4:.1f}" text-anchor="end" class="axis">{score}</text>'
        )
    xlabels = []
    for index, label in enumerate(labels):
        x = left + index * step
        xlabels.append(
            f'<text x="{x:.1f}" y="{bottom + 22}" text-anchor="end" class="axis" transform="rotate(-35 {x:.1f} {bottom + 22})">{escape(short_metric_label(label))}</text>'
        )

    lines = []
    for model in [row["model"] for row in model_summary]:
        points = []
        for index, row in enumerate(metric_score_pivot):
            score = row.get(model)
            if score is None:
                continue
            points.append((left + index * step, value_to_y(float(score), top, bottom)))
        if len(points) < 2:
            continue
        color = SERIES_COLORS.get(str(model_to_series.get(model)), SERIES_COLORS["Other"])
        point_text = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
        lines.append(
            f'<polyline class="chart-data" data-key="{escape(model, quote=True)}" data-tooltip="{escape(model, quote=True)}" '
            f'points="{point_text}" fill="none" stroke="{color}" stroke-width="1.5" opacity="0.34" />'
        )
    return f"""
<div class="chart-card">
  <h3>12 个整体指标的模型画像</h3>
  <p class="section-note">每条线是一个模型。线条在某些指标处明显下探，通常就是该模型相对短板。</p>
  <svg class="chart" viewBox="0 0 {width} {height}">
    <style>.grid{{stroke:#e2e8f0;stroke-width:1;}}.axis{{fill:#64748b;font-size:11px;}}.legend{{fill:#475569;font-size:12px;}}</style>
    {''.join(grid)}
    {''.join(xlabels)}
    {''.join(lines)}
    {svg_legend(series_names, left, height - 24)}
  </svg>
</div>
"""


def render_metric_rank_bar_charts(rankings: dict[str, list[dict[str, Any]]], limit: int = 8) -> str:
    cards = []
    labels = sorted(rankings)
    for label in labels:
        rows = rankings[label][:limit]
        width, height = 430, 30 + len(rows) * 24
        left, right, top = 142, 20, 18
        min_score = min(float(row.get("avg_score") or 0) for row in rows)
        max_score = max(float(row.get("avg_score") or 0) for row in rows)
        span = max(0.35, max_score - min_score)
        bars = []
        for index, row in enumerate(rows):
            score = float(row.get("avg_score") or 0)
            y = top + index * 24
            baseline = max(1.0, min_score - 0.1)
            bar_width = (score - baseline) / (span + 0.1) * (width - left - right)
            bar_width = max(4, bar_width)
            color = SERIES_COLORS.get(str(row.get("series")), SERIES_COLORS["Other"])
            model = str(row.get("model"))
            tooltip = f"{model} / {label}: {score:.3f}, low_rate={percent(row.get('low_rate'))}"
            bars.append(
                f'<text x="{left - 8}" y="{y + 13}" text-anchor="end" class="axis">#{row.get("rank")} {escape(model)}</text>'
                f'<rect class="chart-data" data-key="{escape(model, quote=True)}" data-tooltip="{escape(tooltip, quote=True)}" '
                f'x="{left}" y="{y}" width="{bar_width:.1f}" height="16" rx="3" fill="{color}" opacity="0.82" />'
                f'<text x="{left + bar_width + 5:.1f}" y="{y + 13}" class="axis">{score:.3f}</text>'
            )
        cards.append(
            '<details class="rank-detail chart-detail">'
            f'<summary>{escape(short_metric_label(label))}</summary>'
            f'<svg class="chart" viewBox="0 0 {width} {height}"><style>.axis{{fill:#475569;font-size:11px;}}</style>{"".join(bars)}</svg>'
            "</details>"
        )
    return '<div class="rank-grid">' + "\n".join(cards) + "</div>"


def quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    position = (len(values) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return values[int(position)]
    return values[lower] * (upper - position) + values[upper] * (position - lower)


def render_ability_boxplot(
    ability_rows: list[dict[str, Any]],
    model_summary: list[dict[str, Any]],
    title: str,
) -> str:
    values_by_model: dict[str, list[float]] = defaultdict(list)
    for row in ability_rows:
        score = row.get("overall_score")
        if score is not None:
            values_by_model[row["model"]].append(float(score))

    models = [row["model"] for row in model_summary if values_by_model.get(row["model"])]
    width = max(1180, 54 + len(models) * 42)
    height = 430
    left, top, bottom = 58, 28, 310
    step = (width - left - 24) / max(1, len(models))
    series_lookup = {row["model"]: row["series"] for row in model_summary}
    grid = []
    for score in [1, 2, 3, 4, 5]:
        y = value_to_y(score, top, bottom)
        grid.append(
            f'<line x1="{left}" y1="{y:.1f}" x2="{width - 20}" y2="{y:.1f}" class="grid" />'
            f'<text x="{left - 10}" y="{y + 4:.1f}" text-anchor="end" class="axis">{score}</text>'
        )

    boxes = []
    for index, model in enumerate(models):
        values = sorted(values_by_model[model])
        q1 = quantile(values, 0.25)
        med = quantile(values, 0.5)
        q3 = quantile(values, 0.75)
        low = values[0]
        high = values[-1]
        x = left + index * step + step / 2
        color = SERIES_COLORS.get(str(series_lookup.get(model)), SERIES_COLORS["Other"])
        y_low = value_to_y(low, top, bottom)
        y_high = value_to_y(high, top, bottom)
        y_q1 = value_to_y(q1, top, bottom)
        y_q3 = value_to_y(q3, top, bottom)
        y_med = value_to_y(med, top, bottom)
        box_h = max(2, y_q1 - y_q3)
        tooltip = f"{model} min={low:.3f}, q1={q1:.3f}, median={med:.3f}, q3={q3:.3f}, max={high:.3f}"
        boxes.append(
            f'<g class="chart-data" data-key="{escape(model, quote=True)}" data-tooltip="{escape(tooltip, quote=True)}">'
            f'<line x1="{x:.1f}" y1="{y_high:.1f}" x2="{x:.1f}" y2="{y_low:.1f}" stroke="{color}" stroke-width="1.5" opacity="0.85" />'
            f'<rect x="{x - 9:.1f}" y="{y_q3:.1f}" width="18" height="{box_h:.1f}" fill="{color}" opacity="0.28" stroke="{color}" />'
            f'<line x1="{x - 11:.1f}" y1="{y_med:.1f}" x2="{x + 11:.1f}" y2="{y_med:.1f}" stroke="{color}" stroke-width="2.2" />'
            f'<text x="{x:.1f}" y="{bottom + 18}" text-anchor="end" class="axis" transform="rotate(-48 {x:.1f} {bottom + 18})">{escape(model)}</text>'
            "</g>"
        )

    return f"""
<div class="chart-card">
  <h3>{escape(title)}</h3>
  <p class="section-note">箱体显示模型在全部能力上的分布：中位数越高越好，箱体越短说明能力更均衡，下须很低说明存在明显短板。</p>
  <svg class="chart" viewBox="0 0 {width} {height}">
    <style>.grid{{stroke:#e2e8f0;stroke-width:1;}}.axis{{fill:#64748b;font-size:11px;}}</style>
    {''.join(grid)}
    {''.join(boxes)}
  </svg>
</div>
"""


def render_ability_stat_bars(
    rows: list[dict[str, Any]],
    value_key: str,
    title: str,
    limit: int = 16,
    ascending: bool = True,
    lower: float = 1.0,
    upper: float = 5.0,
) -> str:
    selected = [
        row for row in rows if row.get(value_key) is not None and row.get("ability")
    ]
    selected.sort(key=lambda row: row.get(value_key) or 0, reverse=not ascending)
    selected = selected[:limit]
    if not selected:
        return ""
    width, height = 760, 42 + len(selected) * 25
    left, right, top = 210, 40, 24
    bars = []
    max_value = max(float(row.get(value_key) or 0) for row in selected)
    min_value = min(float(row.get(value_key) or 0) for row in selected)
    chart_lower = min(lower, min_value)
    chart_upper = max(upper, max_value)
    span = max(0.1, chart_upper - chart_lower)
    for index, row in enumerate(selected):
        value = float(row.get(value_key) or 0)
        y = top + index * 25
        bar_width = (value - chart_lower) / span * (width - left - right)
        color = score_color(value, lower=chart_lower, upper=chart_upper)
        ability = str(row.get("ability"))
        tooltip = f"{ability}: {value:.3f}"
        bars.append(
            f'<text x="{left - 8}" y="{y + 14}" text-anchor="end" class="axis">{escape(ability)}</text>'
            f'<rect class="chart-data" data-key="{escape(ability, quote=True)}" data-tooltip="{escape(tooltip, quote=True)}" '
            f'x="{left}" y="{y}" width="{bar_width:.1f}" height="17" rx="3" fill="{color}" opacity="0.88" />'
            f'<text x="{left + bar_width + 6:.1f}" y="{y + 14}" class="axis">{value:.3f}</text>'
        )
    return f"""
<div class="chart-card">
  <h3>{escape(title)}</h3>
  <svg class="chart" viewBox="0 0 {width} {height}">
    <style>.axis{{fill:#475569;font-size:12px;}}</style>
    {''.join(bars)}
  </svg>
</div>
"""


def render_distribution_stats_table(
    rows: list[dict[str, Any]],
    label_columns: list[tuple[str, str]],
    title: str,
    limit: int | None = None,
) -> str:
    selected = rows[:limit] if limit else rows
    headers = "".join(f"<th>{escape(label)}</th>" for label, _ in label_columns)
    body = []
    for row in selected:
        label_cells = "".join(f"<td>{escape(str(row.get(key, '')))}</td>" for _, key in label_columns)
        body.append(
            "<tr>"
            f"{label_cells}"
            f"<td>{fmt(row.get('score_mean') if 'score_mean' in row else row.get('mean'))}</td>"
            f"<td>{fmt(row.get('score_variance') if 'score_variance' in row else row.get('variance'), 6)}</td>"
            f"<td>{fmt(row.get('score_std') if 'score_std' in row else row.get('std'))}</td>"
            f"<td>{fmt(row.get('score_range') if 'score_range' in row else row.get('range'))}</td>"
            f"<td>{fmt(row.get('score_iqr') if 'score_iqr' in row else row.get('iqr'))}</td>"
            f"<td>{escape(str(row.get('score_best_model') or row.get('best_model') or ''))}</td>"
            f"<td>{fmt(row.get('score_best') if 'score_best' in row else row.get('best_value'))}</td>"
            f"<td>{escape(str(row.get('score_worst_model') or row.get('worst_model') or ''))}</td>"
            f"<td>{fmt(row.get('score_worst') if 'score_worst' in row else row.get('worst_value'))}</td>"
            "</tr>"
        )
    return (
        f"<h3>{escape(title)}</h3>"
        '<div class="table-scroll">'
        '<table class="mini-table stats-table">'
        f"<thead><tr>{headers}<th>均值</th><th>方差</th><th>标准差</th><th>极差</th><th>IQR</th>"
        "<th>最高模型</th><th>最高值</th><th>最低模型</th><th>最低值</th></tr></thead>"
        f"<tbody>{''.join(body)}</tbody></table></div>"
    )


def render_metric_distribution_stats(stats_rows: list[dict[str, Any]]) -> str:
    body = []
    for row in stats_rows:
        body.append(
            "<tr>"
            f"<td>{escape(str(row.get('level', '')))}</td>"
            f"<td>{escape(str(row.get('metric', '')))}</td>"
            f"<td>{fmt(row.get('score_mean'))}</td>"
            f"<td>{fmt(row.get('score_variance'), 6)}</td>"
            f"<td>{fmt(row.get('score_std'))}</td>"
            f"<td>{fmt(row.get('score_range'))}</td>"
            f"<td>{fmt(row.get('score_iqr'))}</td>"
            f"<td>{escape(str(row.get('score_best_model') or ''))}</td>"
            f"<td>{fmt(row.get('score_best'))}</td>"
            f"<td>{escape(str(row.get('score_worst_model') or ''))}</td>"
            f"<td>{fmt(row.get('score_worst'))}</td>"
            f"<td>{percent(row.get('low_rate_mean'))}</td>"
            f"<td>{fmt(row.get('low_rate_variance'), 6)}</td>"
            f"<td>{fmt(row.get('low_rate_std'))}</td>"
            f"<td>{fmt(row.get('low_rate_range'))}</td>"
            f"<td>{escape(str(row.get('lowest_low_rate_model') or ''))}</td>"
            f"<td>{percent(row.get('lowest_low_rate'))}</td>"
            f"<td>{escape(str(row.get('highest_low_rate_model') or ''))}</td>"
            f"<td>{percent(row.get('highest_low_rate'))}</td>"
            f"<td>{percent(row.get('pass_rate_mean'))}</td>"
            f"<td>{fmt(row.get('pass_rate_variance'), 6)}</td>"
            f"<td>{fmt(row.get('pass_rate_std'))}</td>"
            f"<td>{fmt(row.get('pass_rate_range'))}</td>"
            f"<td>{escape(str(row.get('highest_pass_rate_model') or ''))}</td>"
            f"<td>{percent(row.get('highest_pass_rate'))}</td>"
            f"<td>{escape(str(row.get('lowest_pass_rate_model') or ''))}</td>"
            f"<td>{percent(row.get('lowest_pass_rate'))}</td>"
            "</tr>"
        )
    return (
        '<div class="stat-note">'
        "<strong>统计学含义与低分标准：</strong>低分率统一按 <code>&lt;4</code> 计算；由于单项指标为 1-5 整数分，"
        "它等价于该指标得分 <code>&lt;=3</code> 的样本数 / 有效样本数。通过率 = <code>1 - 低分率</code>，"
        "也就是该指标得分 <code>&gt;=4</code> 的样本占比。均值表示该指标的整体难度；方差/标准差越大，说明该指标越能区分模型；"
        "极差反映最好与最差模型的跨度；IQR 反映中间 50% 模型的稳定区间。"
        "</div>"
        '<div class="table-scroll">'
        '<table class="mini-table stats-table metric-stats-table">'
        "<thead><tr><th>Level</th><th>Metric</th><th>平均分均值</th><th>平均分方差</th><th>平均分标准差</th>"
        "<th>平均分极差</th><th>平均分IQR</th><th>最高模型</th><th>最高分</th><th>最低模型</th><th>最低分</th>"
        "<th>低分率均值</th><th>低分率方差</th><th>低分率标准差</th><th>低分率极差</th>"
        "<th>最低低分率模型</th><th>最低低分率</th><th>最高低分率模型</th><th>最高低分率</th>"
        "<th>通过率均值</th><th>通过率方差</th><th>通过率标准差</th><th>通过率极差</th>"
        "<th>最高通过率模型</th><th>最高通过率</th><th>最低通过率模型</th><th>最低通过率</th></tr></thead>"
        f"<tbody>{''.join(body)}</tbody></table></div>"
    )


def render_model_metric_self_stats(rows: list[dict[str, Any]], scope: str, title: str) -> str:
    selected = [row for row in rows if row.get("scope") == scope]
    if not selected:
        return ""
    body = []
    for row in selected:
        body.append(
            "<tr>"
            f"<td>{escape(str(row.get('model', '')))}</td>"
            f"<td>{escape(str(row.get('series', '')))}</td>"
            f"<td>{fmt(row.get('metric_count'), 0)}</td>"
            f"<td>{fmt(row.get('score_mean'))}</td>"
            f"<td>{fmt(row.get('score_variance'), 6)}</td>"
            f"<td>{fmt(row.get('score_std'))}</td>"
            f"<td>{fmt(row.get('score_range'))}</td>"
            f"<td>{fmt(row.get('score_iqr'))}</td>"
            f"<td>{percent(row.get('low_rate_mean'))}</td>"
            f"<td>{fmt(row.get('low_rate_variance'), 6)}</td>"
            f"<td>{fmt(row.get('low_rate_std'))}</td>"
            f"<td>{fmt(row.get('low_rate_range'))}</td>"
            f"<td>{percent(row.get('pass_rate_mean'))}</td>"
            f"<td>{fmt(row.get('pass_rate_variance'), 6)}</td>"
            f"<td>{fmt(row.get('pass_rate_std'))}</td>"
            f"<td>{fmt(row.get('pass_rate_range'))}</td>"
            "</tr>"
        )
    return (
        f"<h3>{escape(title)}</h3>"
        '<div class="table-scroll">'
        '<table class="mini-table stats-table" data-row-sortable="true">'
        "<thead><tr><th>Model</th><th>Series</th><th>维度数</th>"
        "<th>平均分均值</th><th>平均分方差</th><th>平均分标准差</th><th>平均分极差</th><th>平均分IQR</th>"
        "<th>低分率均值</th><th>低分率方差</th><th>低分率标准差</th><th>低分率极差</th>"
        "<th>通过率均值</th><th>通过率方差</th><th>通过率标准差</th><th>通过率极差</th></tr></thead>"
        f"<tbody>{''.join(body)}</tbody></table></div>"
    )


def render_generation_phase_table(rows: list[dict[str, Any]]) -> str:
    empty_ranks = rank_lookup_for_key(rows, "empty_prediction_rate", descending=False)
    error_ranks = rank_lookup_for_key(rows, "error_rate", descending=False)
    latency_ranks = rank_lookup_for_key(rows, "avg_latency_seconds", descending=False)
    body = []
    for row in rows:
        model = str(row.get("model", ""))
        body.append(
            "<tr>"
            f"<td>{escape(model)}</td>"
            f"<td>{escape(str(row.get('series', '')))}</td>"
            f"<td>{fmt(row.get('round_count'), 0)}</td>"
            f"<td>{percent(row.get('empty_prediction_rate'))} <small>#{empty_ranks.get(model, '')}</small></td>"
            f"<td>{percent(row.get('error_rate'))} <small>#{error_ranks.get(model, '')}</small></td>"
            f"<td>{fmt(row.get('avg_latency_seconds'))} <small>#{latency_ranks.get(model, '')}</small></td>"
            "</tr>"
        )
    return (
        '<div class="table-scroll">'
        '<table class="mini-table" data-row-sortable="true">'
        "<thead><tr><th>Model</th><th>Series</th><th>标准化轮次数</th>"
        "<th>空回答率</th><th>请求错误率</th><th>平均延迟 (s)</th></tr></thead>"
        f"<tbody>{''.join(body)}</tbody></table></div>"
    )


def render_matrix_heatmap(
    rows: list[dict[str, Any]],
    column_order: list[str],
    row_key: str,
    row_header: str,
    *,
    as_percent: bool = False,
    reverse: bool = False,
    count_key: str | None = None,
    count_header: str | None = None,
    lower: float | None = None,
    upper: float | None = None,
    digits: int = 3,
) -> str:
    if not rows:
        return ""
    header = "".join(f"<th>{escape(column)}</th>" for column in column_order)
    lead_header = f"<th>{escape(row_header)}</th>"
    if count_key and count_header:
        lead_header += f"<th>{escape(count_header)}</th>"
    body_rows = []
    for row in rows:
        ranks = rank_lookup_for_row_values(row, column_order, descending=not reverse)
        cells = "".join(
            heat_cell(
                row.get(column),
                lower=0.0 if lower is None and as_percent else (1.0 if lower is None else lower),
                upper=1.0 if upper is None and as_percent else (5.0 if upper is None else upper),
                reverse=reverse,
                as_percent=as_percent,
                rank=ranks.get(column),
                digits=digits,
            )
            for column in column_order
        )
        row_head = f"<th>{escape(str(row.get(row_key, '')))}</th>"
        if count_key and count_header:
            row_head += f"<td>{fmt(row.get(count_key), 0)}</td>"
        body_rows.append(f"<tr>{row_head}{cells}</tr>")
    frozen_cols = 2 if count_key and count_header else 1
    return (
        '<div class="table-scroll"><table class="heatmap custom-heatmap" '
        f'data-col-sortable="true" data-frozen-cols="{frozen_cols}">'
        f"<thead><tr>{lead_header}{header}</tr></thead>"
        f"<tbody>{''.join(body_rows)}</tbody></table></div>"
    )


def render_threshold_line_chart(
    rows: list[dict[str, Any]],
    label_key: str,
    title: str,
    label_formatter: Any,
) -> str:
    if not rows:
        return ""
    width = max(720, 90 + len(rows) * 82)
    height = 360
    left, right, top, bottom = 72, 30, 26, 252
    step = (width - left - right) / max(1, len(rows) - 1)
    grid = []
    for value in [0.0, 0.25, 0.5, 0.75, 1.0]:
        y = value_to_y(value, top, bottom, lower=0.0, upper=1.0)
        grid.append(
            f'<line x1="{left}" y1="{y:.1f}" x2="{width - right}" y2="{y:.1f}" class="grid" />'
            f'<text x="{left - 10}" y="{y + 4:.1f}" text-anchor="end" class="axis">{value * 100:.0f}%</text>'
        )
    xlabels = "".join(
        f'<text x="{left + index * step:.1f}" y="{bottom + 20}" text-anchor="end" class="axis" '
        f'transform="rotate(-35 {left + index * step:.1f} {bottom + 20})">{escape(str(label_formatter(row.get(label_key))))}</text>'
        for index, row in enumerate(rows)
    )
    threshold_styles = {
        "lt4": ("<4分", "#2563eb"),
        "lt3": ("<3分", "#d97706"),
        "lt2": ("<2分", "#dc2626"),
    }
    polylines = []
    legend_bits = []
    legend_x = left
    for threshold_key, (legend_label, color) in threshold_styles.items():
        points = []
        circles = []
        for index, row in enumerate(rows):
            value = maybe_float(row.get(threshold_key))
            if value is None:
                continue
            x = left + index * step
            y = value_to_y(value, top, bottom, lower=0.0, upper=1.0)
            points.append((x, y))
            tooltip = f"{legend_label} / {row.get(label_key)}: {percent(value)}"
            circles.append(
                f'<circle class="chart-data" data-key="{threshold_key}" data-tooltip="{escape(tooltip, quote=True)}" '
                f'cx="{x:.1f}" cy="{y:.1f}" r="3.2" fill="{color}" />'
            )
        if len(points) >= 2:
            point_text = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
            polylines.append(
                f'<polyline class="chart-data" data-key="{threshold_key}" data-tooltip="{legend_label}" '
                f'points="{point_text}" fill="none" stroke="{color}" stroke-width="2.1" opacity="0.86" />'
                + "".join(circles)
            )
        legend_bits.append(
            f'<circle cx="{legend_x}" cy="{height - 24}" r="5" fill="{color}" />'
            f'<text x="{legend_x + 10}" y="{height - 20}" class="legend">{legend_label}</text>'
        )
        legend_x += 74
    return f"""
<div class="chart-card">
  <h3>{escape(title)}</h3>
  <svg class="chart" viewBox="0 0 {width} {height}">
    <style>.grid{{stroke:#e2e8f0;stroke-width:1;}}.axis{{fill:#64748b;font-size:11px;}}.legend{{fill:#475569;font-size:12px;}}</style>
    {''.join(grid)}
    {xlabels}
    {''.join(polylines)}
    {''.join(legend_bits)}
  </svg>
</div>
"""


def render_pivot_pass_rate_line_chart(
    rows: list[dict[str, Any]],
    model_summary: list[dict[str, Any]],
    title: str,
    label_getter: Any,
) -> str:
    if not rows:
        return ""
    model_order = [row["model"] for row in model_summary]
    series_by_model = {row["model"]: row.get("series", "Other") for row in model_summary}
    labels = [str(label_getter(row)) for row in rows]
    width = max(980, 92 + len(labels) * 58)
    height = 430
    left, right, top, bottom = 72, 30, 28, 300
    step = (width - left - right) / max(1, len(labels) - 1)
    grid = []
    for value in [0.0, 0.25, 0.5, 0.75, 1.0]:
        y = value_to_y(value, top, bottom, lower=0.0, upper=1.0)
        grid.append(
            f'<line x1="{left}" y1="{y:.1f}" x2="{width - right}" y2="{y:.1f}" class="grid" />'
            f'<text x="{left - 10}" y="{y + 4:.1f}" text-anchor="end" class="axis">{value * 100:.0f}%</text>'
        )
    xlabels = "".join(
        f'<text x="{left + index * step:.1f}" y="{bottom + 20}" text-anchor="end" class="axis" '
        f'transform="rotate(-45 {left + index * step:.1f} {bottom + 20})">{escape(label)}</text>'
        for index, label in enumerate(labels)
    )
    lines = []
    for model in model_order:
        points = []
        markers = []
        for index, row in enumerate(rows):
            value = maybe_float(row.get(model))
            if value is None:
                continue
            x = left + index * step
            y = value_to_y(value, top, bottom, lower=0.0, upper=1.0)
            label = labels[index]
            points.append((x, y))
            tooltip = f"{model} / {label}: {percent(value)}"
            key = escape(model, quote=True)
            color = SERIES_COLORS.get(str(series_by_model.get(model)), SERIES_COLORS["Other"])
            markers.append(
                f'<circle class="chart-data" data-key="{key}" data-tooltip="{escape(tooltip, quote=True)}" '
                f'cx="{x:.1f}" cy="{y:.1f}" r="2.7" fill="{color}" />'
            )
        if len(points) < 2:
            continue
        color = SERIES_COLORS.get(str(series_by_model.get(model)), SERIES_COLORS["Other"])
        key = escape(model, quote=True)
        point_text = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
        lines.append(
            f'<polyline class="chart-data" data-key="{key}" data-tooltip="{key}" points="{point_text}" '
            f'fill="none" stroke="{color}" stroke-width="1.8" opacity="0.52" />'
            + "".join(markers)
        )
    legend_series = ordered_unique([str(row.get("series", "Other")) for row in model_summary])
    return f"""
<div class="chart-card">
  <h3>{escape(title)}</h3>
  <div class="chart-viewport">
    <svg class="chart" viewBox="0 0 {width} {height}" style="min-width:{width}px">
      <style>.grid{{stroke:#e2e8f0;stroke-width:1;}}.axis{{fill:#64748b;font-size:11px;}}.legend{{fill:#475569;font-size:12px;}}</style>
      {''.join(grid)}
      {xlabels}
      {''.join(lines)}
      {svg_legend(legend_series, left, height - 24)}
    </svg>
  </div>
</div>
"""


def is_total_count_row(row: dict[str, Any], row_key: str) -> bool:
    return str(row.get(row_key) or "").strip().lower().startswith("total")


def failure_count_color(value: float, upper: float) -> str:
    if upper <= 0:
        return "#f1f5f9"
    return score_color(upper - value, lower=0.0, upper=upper)


def render_failure_total_bar_chart(
    rows: list[dict[str, Any]],
    model_order: list[str],
    row_key: str,
    title: str,
) -> str:
    if not rows:
        return ""
    total_row = next((row for row in rows if is_total_count_row(row, row_key)), None)
    if total_row is None:
        total_row = {
            row_key: "Total low cases",
            **{
                model: sum(int(maybe_float(row.get(model)) or 0) for row in rows)
                for model in model_order
            },
        }
    values = [
        {"model": model, "count": int(maybe_float(total_row.get(model)) or 0)}
        for model in model_order
    ]
    values.sort(key=lambda row: row["count"], reverse=True)
    max_value = max((row["count"] for row in values), default=0) or 1
    width, height = 820, 42 + len(values) * 24
    left, right, top = 210, 72, 22
    bars = []
    for index, row in enumerate(values):
        value = row["count"]
        y = top + index * 24
        bar_width = value / max_value * (width - left - right)
        color = failure_count_color(value, max_value)
        tooltip = f"{row['model']}: {value}"
        bars.append(
            f'<text x="{left - 8}" y="{y + 14}" text-anchor="end" class="axis">{escape(row["model"])}</text>'
            f'<rect class="chart-data" data-key="{escape(row["model"], quote=True)}" data-tooltip="{escape(tooltip, quote=True)}" '
            f'x="{left}" y="{y}" width="{bar_width:.1f}" height="17" rx="3" fill="{color}" opacity="0.9" />'
            f'<text x="{left + bar_width + 6:.1f}" y="{y + 14}" class="axis">{value}</text>'
        )
    return f"""
<div class="chart-card">
  <h3>{escape(title)}</h3>
  <svg class="chart" viewBox="0 0 {width} {height}">
    <style>.axis{{fill:#475569;font-size:11px;}}</style>
    {''.join(bars)}
  </svg>
</div>
"""


def render_failure_count_line_chart(
    rows: list[dict[str, Any]],
    model_order: list[str],
    row_key: str,
    title: str,
) -> str:
    category_rows = [row for row in rows if not is_total_count_row(row, row_key)]
    if not category_rows:
        return ""
    max_value = max(
        finite_values([maybe_float(row.get(model)) for row in category_rows for model in model_order])
        or [0.0]
    )
    upper = max(max_value, 1.0)
    width = max(1200, 110 + len(model_order) * 72)
    height = 470
    left, right, top, bottom = 78, 34, 28, 305
    step = (width - left - right) / max(1, len(model_order) - 1)
    grid = []
    for ratio in [0.0, 0.25, 0.5, 0.75, 1.0]:
        value = upper * ratio
        y = value_to_y(value, top, bottom, lower=0.0, upper=upper)
        grid.append(
            f'<line x1="{left}" y1="{y:.1f}" x2="{width - right}" y2="{y:.1f}" class="grid" />'
            f'<text x="{left - 10}" y="{y + 4:.1f}" text-anchor="end" class="axis">{value:.0f}</text>'
        )
    xlabels = "".join(
        f'<text x="{left + index * step:.1f}" y="{bottom + 20}" text-anchor="end" class="axis" '
        f'transform="rotate(-45 {left + index * step:.1f} {bottom + 20})">{escape(model)}</text>'
        for index, model in enumerate(model_order)
    )
    palette = [
        "#dc2626",
        "#d97706",
        "#2563eb",
        "#059669",
        "#7c3aed",
        "#0891b2",
        "#be123c",
        "#64748b",
    ]
    lines = []
    legend = []
    legend_x = left
    legend_y = height - 30
    for row_index, row in enumerate(category_rows):
        category = str(row.get(row_key) or "")
        color = palette[row_index % len(palette)]
        points = []
        markers = []
        for index, model in enumerate(model_order):
            value = maybe_float(row.get(model))
            if value is None:
                continue
            x = left + index * step
            y = value_to_y(value, top, bottom, lower=0.0, upper=upper)
            points.append((x, y))
            tooltip = f"{category} / {model}: {int(value)}"
            key = escape(category, quote=True)
            markers.append(
                f'<circle class="chart-data" data-key="{key}" data-tooltip="{escape(tooltip, quote=True)}" '
                f'cx="{x:.1f}" cy="{y:.1f}" r="3" fill="{color}" />'
            )
        if len(points) >= 2:
            point_text = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
            lines.append(
                f'<polyline class="chart-data" data-key="{escape(category, quote=True)}" '
                f'data-tooltip="{escape(category, quote=True)}" points="{point_text}" '
                f'fill="none" stroke="{color}" stroke-width="2.1" opacity="0.82" />'
                + "".join(markers)
            )
        legend.append(
            f'<circle cx="{legend_x}" cy="{legend_y}" r="5" fill="{color}" />'
            f'<text x="{legend_x + 9}" y="{legend_y + 4}" class="legend">{escape(category)}</text>'
        )
        legend_x += min(210, max(92, len(category) * 12))
        if legend_x > width - 220:
            legend_x = left
            legend_y += 18
    return f"""
<div class="chart-card">
  <h3>{escape(title)}</h3>
  <div class="chart-viewport">
    <svg class="chart" viewBox="0 0 {width} {height}" style="min-width:{width}px">
      <style>.grid{{stroke:#e2e8f0;stroke-width:1;}}.axis{{fill:#64748b;font-size:11px;}}.legend{{fill:#475569;font-size:12px;}}</style>
      {''.join(grid)}
      {xlabels}
      {''.join(lines)}
      {''.join(legend)}
    </svg>
  </div>
</div>
"""


def render_simple_bar_chart(
    rows: list[dict[str, Any]],
    label_key: str,
    value_key: str,
    title: str,
    *,
    limit: int = 12,
    ascending: bool = False,
    as_percent: bool = False,
) -> str:
    selected = [row for row in rows if maybe_float(row.get(value_key)) is not None][:] 
    selected.sort(key=lambda row: maybe_float(row.get(value_key)) or 0, reverse=not ascending)
    selected = selected[:limit]
    if not selected:
        return ""
    width, height = 620, 40 + len(selected) * 25
    left, right, top = 210, 34, 22
    max_value = max(maybe_float(row.get(value_key)) or 0 for row in selected) or 1.0
    bars = []
    for index, row in enumerate(selected):
        value = maybe_float(row.get(value_key)) or 0.0
        y = top + index * 25
        bar_width = value / max_value * (width - left - right)
        color = score_color(value if not as_percent else (1 + value * 4), lower=1.0, upper=5.0)
        label = str(row.get(label_key, ""))
        text = percent(value) if as_percent else fmt(value)
        tooltip = f"{label}: {text}"
        bars.append(
            f'<text x="{left - 8}" y="{y + 14}" text-anchor="end" class="axis">{escape(label)}</text>'
            f'<rect class="chart-data" data-key="{escape(label, quote=True)}" data-tooltip="{escape(tooltip, quote=True)}" '
            f'x="{left}" y="{y}" width="{bar_width:.1f}" height="17" rx="3" fill="{color}" opacity="0.86" />'
            f'<text x="{left + bar_width + 6:.1f}" y="{y + 14}" class="axis">{escape(text)}</text>'
        )
    return (
        '<div class="chart-card">'
        f"<h3>{escape(title)}</h3>"
        f'<svg class="chart" viewBox="0 0 {width} {height}"><style>.axis{{fill:#475569;font-size:11px;}}</style>{"".join(bars)}</svg>'
        "</div>"
    )


def render_phase4_model_summary_table(rows: list[dict[str, Any]]) -> str:
    turn_rate_ranks = rank_lookup_for_key(rows, "turn_low_rate", descending=False)
    session_rate_ranks = rank_lookup_for_key(rows, "session_low_rate", descending=False)
    body = []
    for row in rows:
        model = str(row.get("model", ""))
        body.append(
            "<tr>"
            f"<td>{escape(model)}</td>"
            f"<td>{escape(str(row.get('series', '')))}</td>"
            f"<td>{fmt(row.get('turn_low_count'), 0)}</td>"
            f"<td>{percent(row.get('turn_low_rate'))} <small>#{turn_rate_ranks.get(model, '')}</small></td>"
            f"<td>{percent(row.get('turn_critical_share'))}</td>"
            f"<td>{fmt(row.get('session_low_count'), 0)}</td>"
            f"<td>{percent(row.get('session_low_rate'))} <small>#{session_rate_ranks.get(model, '')}</small></td>"
            f"<td>{percent(row.get('session_critical_share'))}</td>"
            "</tr>"
        )
    return (
        '<div class="table-scroll">'
        '<table class="mini-table" data-row-sortable="true">'
        "<thead><tr><th>Model</th><th>Series</th><th>Turn 低分样本数</th><th>Turn 低分样本率</th>"
        "<th>Turn Critical 占比</th><th>Session 低分样本数</th><th>Session 低分样本率</th><th>Session Critical 占比</th></tr></thead>"
        f"<tbody>{''.join(body)}</tbody></table></div>"
    )


def render_generation_phase_table(rows: list[dict[str, Any]]) -> str:
    empty_ranks = rank_lookup_for_key(rows, "empty_prediction_rate", descending=False)
    error_ranks = rank_lookup_for_key(rows, "error_rate", descending=False)
    latency_ranks = rank_lookup_for_key(rows, "avg_latency_seconds", descending=False)
    body = []
    for row in rows:
        model = str(row.get("model", ""))
        body.append(
            "<tr>"
            f"<td>{escape(model)}</td>"
            f"<td>{escape(str(row.get('series', '')))}</td>"
            f"<td>{fmt(row.get('phase1_task_count'), 0)}</td>"
            f"<td>{fmt(row.get('phase1_declared_dialogues'), 0)}</td>"
            f"<td>{fmt(row.get('round_count'), 0)}</td>"
            f"<td>{percent(row.get('empty_prediction_rate'))} <small>#{empty_ranks.get(model, '')}</small></td>"
            f"<td>{percent(row.get('error_rate'))} <small>#{error_ranks.get(model, '')}</small></td>"
            f"<td>{fmt(row.get('avg_latency_seconds'))} <small>#{latency_ranks.get(model, '')}</small></td>"
            "</tr>"
        )
    return (
        '<div class="table-scroll">'
        '<table class="mini-table" data-row-sortable="true">'
        "<thead><tr><th>Model</th><th>Series</th><th>Phase1 任务数</th><th>Phase1 对话组数</th>"
        "<th>Phase2 标准化轮次数</th><th>空回答率</th><th>请求错误率</th><th>平均延迟 (s)</th></tr></thead>"
        f"<tbody>{''.join(body)}</tbody></table></div>"
    )


def render_error_taxonomy_relation_section() -> str:
    def _table(title: str, taxonomy: dict[str, dict[str, str]]) -> str:
        if not taxonomy:
            return ""
        body = []
        for primary, secondary_map in taxonomy.items():
            body.append(
                "<tr>"
                f"<th>{escape(str(primary))}</th>"
                f"<td>{escape('、'.join(str(item) for item in secondary_map))}</td>"
                "</tr>"
            )
        return (
            f"<h3>{escape(title)}</h3>"
            '<div class="table-scroll"><table class="mini-table">'
            "<thead><tr><th>一级失败原因</th><th>从属二级失败原因</th></tr></thead>"
            f"<tbody>{''.join(body)}</tbody></table></div>"
        )

    return (
        '<div class="subsection-block">'
        '<p class="section-note">一级失败原因与二级失败原因的从属关系来自 '
        '<code>tools/RUBRIC-MME/error_taxonomy.py</code>。二级原因是对一级原因的细粒度拆分，'
        '同一低分样本可以被标注 1-3 个二级原因，所以二级原因的绝对数量是标签分配次数。</p>'
        f'{_table("Turn 一级/二级失败原因从属关系", TURN_ERROR_CATEGORIES_CN)}'
        f'{_table("Session 一级/二级失败原因从属关系", SESSION_ERROR_CATEGORIES_CN)}'
        '</div>'
    )


def render_phase4_failure_section(phase_analysis: dict[str, Any], model_order: list[str]) -> str:
    turn_primary_overall_rows = phase_analysis.get("phase4_turn_primary_overall_rows", [])
    session_primary_overall_rows = phase_analysis.get("phase4_session_primary_overall_rows", [])
    turn_secondary_overall_rows = phase_analysis.get("phase4_turn_secondary_overall_rows", [])
    session_secondary_overall_rows = phase_analysis.get("phase4_session_secondary_overall_rows", [])
    turn_primary_pivot = phase_analysis.get("phase4_turn_primary_pivot", [])
    session_primary_pivot = phase_analysis.get("phase4_session_primary_pivot", [])
    turn_primary_count_pivot = phase_analysis.get("phase4_turn_primary_count_pivot", [])
    session_primary_count_pivot = phase_analysis.get("phase4_session_primary_count_pivot", [])
    turn_secondary_pivot = phase_analysis.get("phase4_turn_secondary_pivot", [])
    session_secondary_pivot = phase_analysis.get("phase4_session_secondary_pivot", [])
    turn_secondary_count_pivot = phase_analysis.get("phase4_turn_secondary_count_pivot", [])
    session_secondary_count_pivot = phase_analysis.get("phase4_session_secondary_count_pivot", [])
    turn_metric_rows = phase_analysis.get("phase4_turn_metric_primary_rows", [])
    session_metric_rows = phase_analysis.get("phase4_session_metric_primary_rows", [])
    turn_metric_categories = phase_analysis.get("phase4_turn_metric_categories", [])
    session_metric_categories = phase_analysis.get("phase4_session_metric_categories", [])
    turn_metric_secondary_rows = phase_analysis.get("phase4_turn_metric_secondary_rows", [])
    session_metric_secondary_rows = phase_analysis.get("phase4_session_metric_secondary_rows", [])
    turn_metric_secondary_categories = phase_analysis.get("phase4_turn_metric_secondary_categories", [])
    session_metric_secondary_categories = phase_analysis.get("phase4_session_metric_secondary_categories", [])
    turn_task_rows = phase_analysis.get("phase4_turn_task_primary_rows", [])
    session_task_rows = phase_analysis.get("phase4_session_task_primary_rows", [])
    turn_task_categories = phase_analysis.get("phase4_turn_task_categories", [])
    session_task_categories = phase_analysis.get("phase4_session_task_categories", [])
    turn_metric_primary_count_rows = count_matrix_from_ratio_rows(turn_metric_rows, turn_metric_categories, "metric")
    session_metric_primary_count_rows = count_matrix_from_ratio_rows(session_metric_rows, session_metric_categories, "metric")
    turn_metric_secondary_count_rows = count_matrix_from_ratio_rows(
        turn_metric_secondary_rows,
        turn_metric_secondary_categories,
        "metric",
    )
    session_metric_secondary_count_rows = count_matrix_from_ratio_rows(
        session_metric_secondary_rows,
        session_metric_secondary_categories,
        "metric",
    )
    if not any(
        [
            phase_analysis.get("phase4_model_rows"),
            turn_primary_overall_rows,
            session_primary_overall_rows,
        ]
    ):
        return ""
    return f"""
<section id="phase4-section">
  <h2>低分归因与失败结构</h2>
  <p class="section-note">
    这一部分直接使用每个模型的低分归因结果。Turn / Session 是分开抽取、分开归因的：
    Turn 对应低分轮次，Session 对应低分整段会话。这里展示的不是“所有样本的错误率”，而是“进入低分池之后，
    失败主要集中在哪些原因、哪些任务、哪些指标”，用来解释模型为什么会在某些维度上形成系统性短板。
    一级/二级失败原因的层级定义来自 <code>tools/RUBRIC-MME/error_taxonomy.py</code>，二级原因是一类失败原因下的更细小类。
  </p>
  <h3>各模型进入低分归因池的规模</h3>
  {render_error_taxonomy_relation_section()}
  {render_phase4_model_summary_table(phase_analysis.get("phase4_model_rows", []))}
  <div class="chart-grid">
    {render_simple_bar_chart(turn_primary_overall_rows, "error_category", "share", "Turn 一级失败原因占比", limit=8, as_percent=True)}
    {render_simple_bar_chart(session_primary_overall_rows, "error_category", "share", "Session 一级失败原因占比", limit=8, as_percent=True)}
    {render_simple_bar_chart(turn_secondary_overall_rows, "error_category", "share", "Turn 二级失败原因全量占比", limit=max(1, len(turn_secondary_overall_rows)), as_percent=True)}
    {render_simple_bar_chart(session_secondary_overall_rows, "error_category", "share", "Session 二级失败原因全量占比", limit=max(1, len(session_secondary_overall_rows)), as_percent=True)}
  </div>
  <h3>Turn 一级失败原因在不同模型中的占比</h3>
  {render_matrix_heatmap(turn_primary_pivot, model_order, "error_category", "Turn 一级原因", as_percent=True, reverse=True)}
  <h3>Turn 一级失败原因在不同模型中的绝对数量</h3>
  {render_matrix_heatmap(turn_primary_count_pivot, model_order, "error_category", "Turn 一级原因", lower=0.0, upper=max_matrix_value(turn_primary_count_pivot, model_order), digits=0, reverse=True)}
  <div class="chart-grid">
    {render_failure_total_bar_chart(turn_primary_count_pivot, model_order, "error_category", "Turn 一级失败原因总数柱状图")}
    {render_failure_count_line_chart(turn_primary_count_pivot, model_order, "error_category", "Turn 一级失败原因绝对数量折线图")}
  </div>
  <h3>Session 一级失败原因在不同模型中的占比</h3>
  {render_matrix_heatmap(session_primary_pivot, model_order, "error_category", "Session 一级原因", as_percent=True, reverse=True)}
  <h3>Session 一级失败原因在不同模型中的绝对数量</h3>
  {render_matrix_heatmap(session_primary_count_pivot, model_order, "error_category", "Session 一级原因", lower=0.0, upper=max_matrix_value(session_primary_count_pivot, model_order), digits=0, reverse=True)}
  <div class="chart-grid">
    {render_failure_total_bar_chart(session_primary_count_pivot, model_order, "error_category", "Session 一级失败原因总数柱状图")}
    {render_failure_count_line_chart(session_primary_count_pivot, model_order, "error_category", "Session 一级失败原因绝对数量折线图")}
  </div>
  <h3>Turn 二级失败原因在不同模型中的占比</h3>
  {render_matrix_heatmap(turn_secondary_pivot, model_order, "error_category", "Turn Secondary Reason", as_percent=True, reverse=True)}
  <h3>Turn 二级失败原因在不同模型中的绝对数量</h3>
  {render_matrix_heatmap(turn_secondary_count_pivot, model_order, "error_category", "Turn Secondary Reason", lower=0.0, upper=max_matrix_value(turn_secondary_count_pivot, model_order), digits=0, reverse=True)}
  <h3>Session 二级失败原因在不同模型中的占比</h3>
  {render_matrix_heatmap(session_secondary_pivot, model_order, "error_category", "Session Secondary Reason", as_percent=True, reverse=True)}
  <h3>Session 二级失败原因在不同模型中的绝对数量</h3>
  {render_matrix_heatmap(session_secondary_count_pivot, model_order, "error_category", "Session Secondary Reason", lower=0.0, upper=max_matrix_value(session_secondary_count_pivot, model_order), digits=0, reverse=True)}
  <p class="section-note">
    下面的“指标 × 失败原因”来自低分归因记录中的 <code>affected_metrics</code> 与原因标签。
    对某个指标来说，<code>Low Cases</code> 是该指标被归因模型标为受影响指标的低分归因记录数；
    一级原因比例 = 该指标下某一级原因计数 / 该指标所有一级原因计数之和。二级原因是一级原因下的小类，
    一个低分记录可被打上 1-3 个二级原因，因此二级原因的 <code>Low Cases</code> 是“二级标签分配次数”，可能大于唯一低分记录数。
    绝对数量热力图显示真实计数，占比热力图显示行内归一化后的比例。
  </p>
  <h3>Turn 指标 × 一级失败原因</h3>
  {render_matrix_heatmap(turn_metric_rows, turn_metric_categories, "metric", "Metric", as_percent=True, count_key="total_count", count_header="Low Cases", reverse=True)}
  <h3>Turn 指标 × 一级失败原因：绝对数量</h3>
  {render_matrix_heatmap(turn_metric_primary_count_rows, turn_metric_categories, "metric", "Metric", count_key="total_count", count_header="Low Cases", lower=0.0, upper=max_matrix_value(turn_metric_primary_count_rows, turn_metric_categories), digits=0, reverse=True)}
  <h3>Session 指标 × 一级失败原因</h3>
  {render_matrix_heatmap(session_metric_rows, session_metric_categories, "metric", "Metric", as_percent=True, count_key="total_count", count_header="Low Cases", reverse=True)}
  <h3>Session 指标 × 一级失败原因：绝对数量</h3>
  {render_matrix_heatmap(session_metric_primary_count_rows, session_metric_categories, "metric", "Metric", count_key="total_count", count_header="Low Cases", lower=0.0, upper=max_matrix_value(session_metric_primary_count_rows, session_metric_categories), digits=0, reverse=True)}
  <h3>Turn 指标 × 二级失败原因</h3>
  {render_matrix_heatmap(turn_metric_secondary_rows, turn_metric_secondary_categories, "metric", "Metric", as_percent=True, count_key="total_count", count_header="Low Cases", reverse=True)}
  <h3>Turn 指标 × 二级失败原因：绝对数量</h3>
  {render_matrix_heatmap(turn_metric_secondary_count_rows, turn_metric_secondary_categories, "metric", "Metric", count_key="total_count", count_header="Low Cases", lower=0.0, upper=max_matrix_value(turn_metric_secondary_count_rows, turn_metric_secondary_categories), digits=0, reverse=True)}
  <h3>Session 指标 × 二级失败原因</h3>
  {render_matrix_heatmap(session_metric_secondary_rows, session_metric_secondary_categories, "metric", "Metric", as_percent=True, count_key="total_count", count_header="Low Cases", reverse=True)}
  <h3>Session 指标 × 二级失败原因：绝对数量</h3>
  {render_matrix_heatmap(session_metric_secondary_count_rows, session_metric_secondary_categories, "metric", "Metric", count_key="total_count", count_header="Low Cases", lower=0.0, upper=max_matrix_value(session_metric_secondary_count_rows, session_metric_secondary_categories), digits=0, reverse=True)}
  <h3>Turn 任务 × 一级失败原因</h3>
  {render_matrix_heatmap(turn_task_rows, turn_task_categories, "task", "Task", as_percent=True, count_key="total_count", count_header="Low Cases", reverse=True)}
  <h3>Session 任务 × 一级失败原因</h3>
  {render_matrix_heatmap(session_task_rows, session_task_categories, "task", "Task", as_percent=True, count_key="total_count", count_header="Low Cases", reverse=True)}
</section>
"""


def render_scene_score_section(phase_analysis: dict[str, Any], model_order: list[str]) -> str:
    major_score_pivot = phase_analysis.get("scene_major_score_pivot", [])
    major_low_pivots = phase_analysis.get("scene_major_low_rate_pivots", {})
    major_pass_pivot = phase_analysis.get("scene_major_pass_rate_pivot", [])
    major_overall_rows = phase_analysis.get("scene_major_overall_rows", [])
    major_score_stats = phase_analysis.get("scene_major_score_stats", [])
    major_lt4_stats = phase_analysis.get("scene_major_lt4_stats", [])
    major_pass_stats = phase_analysis.get("scene_major_pass_stats", [])
    detail_score_pivot = phase_analysis.get("scene_detail_score_pivot", [])
    detail_low_pivots = phase_analysis.get("scene_detail_low_rate_pivots", {})
    detail_pass_pivot = phase_analysis.get("scene_detail_pass_rate_pivot", [])
    detail_overall_rows = phase_analysis.get("scene_detail_overall_rows", [])
    detail_score_stats = phase_analysis.get("scene_detail_score_stats", [])
    detail_lt4_stats = phase_analysis.get("scene_detail_lt4_stats", [])
    detail_pass_stats = phase_analysis.get("scene_detail_pass_stats", [])
    primary_top_categories = phase_analysis.get("scene_primary_top_categories", [])
    primary_score_rows = phase_analysis.get("scene_primary_score_rows", [])
    primary_low_rows = phase_analysis.get("scene_primary_low_rate_rows", [])
    primary_pass_rows = phase_analysis.get("scene_primary_pass_rate_rows", [])
    scene_model_summary = []
    seen_scene_models = set()
    for row in phase_analysis.get("scene_major_model_rows", []):
        model = str(row.get("model") or "")
        if model and model not in seen_scene_models:
            seen_scene_models.add(model)
            scene_model_summary.append({"model": model, "series": row.get("series", "Other")})
    if not major_score_pivot:
        return ""
    return f"""
<section id="scene-score-section">
  <h2>场景与分数关系</h2>
  <p class="section-note">
    这一部分把 <code>phase2/rounds.jsonl</code> 中的 <code>environment</code> 与
    <code>phase3/turn_judgements.jsonl</code> 中的 Turn 均分对齐，观察不同场景是否更容易触发失分。
    场景大类来自 <code>environment</code> 中第一个 <code>-</code> 之前的字段，因此这里看的是“场景大类”
    与得分 / 低分风险之间的关系。
  </p>
  <p class="section-note">
    场景大类平均分的公式是：对同一模型、同一场景大类下所有成功 turn 的 <code>avg_score</code> 求平均；
    其中 <code>avg_score</code> 是该 turn 的 8 个 Turn-Level 指标均值。场景大类 <code>&lt;4</code> 低分率 =
    该场景下 <code>avg_score &lt; 4</code> 的成功 turn 数 / 该场景成功 turn 总数；通过率 =
    <code>1 - 低分率</code>，即 <code>avg_score &gt;= 4</code> 的成功 turn 占比。
  </p>
  <div class="chart-grid">
    {render_threshold_line_chart(major_overall_rows, "environment_major", "跨模型平均的场景低分梯度", lambda value: value)}
    {render_simple_bar_chart(major_score_stats, "environment_major", "mean", "场景平均分最低 Top 10", limit=10, ascending=True)}
    {render_simple_bar_chart(major_score_stats, "environment_major", "range", "模型分化最大的场景 Top 10", limit=10)}
    {render_simple_bar_chart(major_lt4_stats, "environment_major", "mean", "场景 <4 低分率最高 Top 10", limit=10, as_percent=True)}
  </div>
  <h3>场景大类平均分热力图</h3>
  {render_matrix_heatmap(major_score_pivot, model_order, "environment_major", "Environment Major")}
<h3>场景大类低分率热力图：&lt;4 分</h3>
  {render_matrix_heatmap(major_low_pivots.get("lt4", []), model_order, "environment_major", "Environment Major", as_percent=True, reverse=True)}
  <h3>场景大类通过率热力图：&gt;=4 分</h3>
  {render_matrix_heatmap(major_pass_pivot, model_order, "environment_major", "Environment Major", as_percent=True)}
  {render_pivot_pass_rate_line_chart(major_pass_pivot, scene_model_summary, "场景大类通过率折线图（>=4 分）", environment_major_row_label)}
  <details class="rank-detail">
    <summary>展开查看场景大类更严重低分率：&lt;3 分</summary>
    {render_matrix_heatmap(major_low_pivots.get("lt3", []), model_order, "environment_major", "Environment Major", as_percent=True, reverse=True)}
  </details>
  {render_distribution_stats_table(major_score_stats, [("Environment Major", "environment_major")], "场景大类平均分统计学诊断")}
  {render_distribution_stats_table(major_lt4_stats, [("Environment Major", "environment_major")], "场景大类 <4 低分率统计学诊断")}
  {render_distribution_stats_table(major_pass_stats, [("Environment Major", "environment_major")], "场景大类通过率统计学诊断")}
  <h3>场景小类平均分热力图</h3>
  <p class="section-note">
    场景小类使用完整 <code>environment</code> 标签，即 <code>场景大类-细分场景</code>。计算公式与大类一致：同一模型、同一小类下所有成功 turn 的
    <code>avg_score</code> 求均值；<code>&lt;4</code> 低分率为 <code>avg_score &lt; 4</code> 的比例；通过率为 <code>avg_score &gt;= 4</code> 的比例。
  </p>
  <div class="chart-grid">
    {render_threshold_line_chart(detail_overall_rows, "environment_detail", "跨模型平均的场景小类低分梯度", lambda value: value)}
    {render_simple_bar_chart(detail_score_stats, "environment_detail", "mean", "场景小类平均分最低 Top 12", limit=12, ascending=True)}
    {render_simple_bar_chart(detail_score_stats, "environment_detail", "range", "模型分化最大的场景小类 Top 12", limit=12)}
    {render_simple_bar_chart(detail_lt4_stats, "environment_detail", "mean", "场景小类 <4 低分率最高 Top 12", limit=12, as_percent=True)}
  </div>
  {render_matrix_heatmap(detail_score_pivot, model_order, "environment_detail", "Environment Detail")}
  <h3>场景小类低分率热力图（&lt;4 分）</h3>
  {render_matrix_heatmap(detail_low_pivots.get("lt4", []), model_order, "environment_detail", "Environment Detail", as_percent=True, reverse=True)}
  <h3>场景小类通过率热力图（&gt;=4 分）</h3>
  {render_matrix_heatmap(detail_pass_pivot, model_order, "environment_detail", "Environment Detail", as_percent=True)}
  {render_pivot_pass_rate_line_chart(detail_pass_pivot, scene_model_summary, "场景小类通过率折线图（>=4 分）", environment_detail_row_label)}
  <details class="rank-detail">
    <summary>展开查看场景小类更严重低分率（&lt;3 分）</summary>
    {render_matrix_heatmap(detail_low_pivots.get("lt3", []), model_order, "environment_detail", "Environment Detail", as_percent=True, reverse=True)}
  </details>
  {render_distribution_stats_table(detail_score_stats, [("Environment Detail", "environment_detail")], "场景小类平均分统计学诊断")}
  {render_distribution_stats_table(detail_lt4_stats, [("Environment Detail", "environment_detail")], "场景小类 <4 低分率统计学诊断")}
  {render_distribution_stats_table(detail_pass_stats, [("Environment Detail", "environment_detail")], "场景小类通过率统计学诊断")}
  <p class="section-note">
    场景大类 × Primary Category 的矩阵不再区分模型，而是把所有模型的成功 turn 合并后统计。
    每个格子的平均分 = 同时落在该场景大类和该 Primary Category 下的 turn <code>avg_score</code> 均值；
    <code>&lt;4</code> 低分率 = 这些 turn 中 <code>avg_score &lt; 4</code> 的比例；通过率 =
    <code>avg_score &gt;= 4</code> 的比例。格子背后的样本数会随鼠标悬停或表格导出 CSV 一起保留。
  </p>
  <h3>场景大类 × Primary Category 平均分</h3>
  {render_matrix_heatmap(primary_score_rows, primary_top_categories, "environment_major", "Environment Major")}
  <h3>场景大类 × Primary Category &lt;4 低分率</h3>
  {render_matrix_heatmap(primary_low_rows, primary_top_categories, "environment_major", "Environment Major", as_percent=True, reverse=True)}
  <h3>场景大类 × Primary Category 通过率</h3>
  {render_matrix_heatmap(primary_pass_rows, primary_top_categories, "environment_major", "Environment Major", as_percent=True)}
</section>
"""


def render_task_low_score_section(phase_analysis: dict[str, Any], model_order: list[str]) -> str:
    overall_rows = phase_analysis.get("task_turn_threshold_overall_rows", [])
    pivots = phase_analysis.get("task_turn_threshold_pivots", {})
    return f"""
<h3>任务维度 Turn 低分率</h3>
<p class="section-note">
  这里的低分标准直接来自 phase3 的 turn-level 原始均分：<code>&lt;4</code> 表示进入低分区，<code>&lt;3</code> 表示更明显失分，<code>&lt;2</code> 表示极端低分。
  由于 turn 平均分本身是 1-5 的离散值，所以 <code>&lt;4</code> 等价于 “3 分及以下”，<code>&lt;3</code> 等价于 “2 分及以下”。
</p>
<div class="chart-grid">
  {render_threshold_line_chart(overall_rows, "task", "跨模型平均的任务低分梯度", short_task_label)}
</div>
<h3>任务 Turn 低分率热力图：&lt;4分</h3>
{render_matrix_heatmap(pivots.get('lt4', []), model_order, 'task', 'Task', as_percent=True, reverse=True)}
<details class="rank-detail">
  <summary>展开查看任务 Turn 更严重低分率：&lt;3分</summary>
  {render_matrix_heatmap(pivots.get('lt3', []), model_order, 'task', 'Task', as_percent=True, reverse=True)}
</details>
<details class="rank-detail">
  <summary>展开查看任务 Turn 极端低分率：&lt;2分</summary>
  {render_matrix_heatmap(pivots.get('lt2', []), model_order, 'task', 'Task', as_percent=True, reverse=True)}
</details>
"""


def render_metric_threshold_section(phase_analysis: dict[str, Any], model_order: list[str]) -> str:
    overall_rows = phase_analysis.get("metric_threshold_overall_rows", [])
    turn_rows = [row for row in overall_rows if row.get("level") == "Turn-Level"]
    session_rows = [row for row in overall_rows if row.get("level") == "Session-Level"]
    pivots = phase_analysis.get("metric_threshold_pivots", {})
    lt4_rows = pivots.get('Turn-Level::lt4', []) + pivots.get('Session-Level::lt4', [])
    pass_rows = complement_rate_pivot(lt4_rows, model_order, ["level", "metric"])
    return f"""
<h3>多阈值低分率扩展</h3>
<p class="section-note">
  这里直接使用 <code>phase3/turn_judgements.jsonl</code> 与 <code>phase3/session_judgements.jsonl</code> 中的逐样本
  <code>score_vector</code>。对每个模型、每个指标分别统计：
  <code>&lt;4</code> 低分率 = 该指标分数 <code>&lt;4</code> 的样本数 / 该指标有效样本数。
  由于指标分数是 1-5 整数，<code>&lt;4</code> 等价于 <code>&lt;=3</code>；通过率 = <code>1 - &lt;4 低分率</code>，
  等价于该指标分数 <code>&gt;=4</code> 的样本占比。<code>&lt;3</code> 与 <code>&lt;2</code> 用于观察更严重的尾部风险。
</p>
<div class="chart-grid">
  {render_threshold_line_chart(turn_rows, "metric", "Turn-Level 指标低分梯度", short_metric_label)}
  {render_threshold_line_chart(session_rows, "metric", "Session-Level 指标低分梯度", short_metric_label)}
</div>
<h3>低分率热力图：&lt;4分</h3>
{render_metric_heatmap(lt4_rows, model_order, as_percent=True, reverse=True)}
<h3>通过率热力图：&gt;=4分</h3>
{render_metric_heatmap(pass_rows, model_order, as_percent=True)}
<details class="rank-detail">
  <summary>展开查看更严重低分率：&lt;3分</summary>
  {render_metric_heatmap(pivots.get('Turn-Level::lt3', []) + pivots.get('Session-Level::lt3', []), model_order, as_percent=True, reverse=True)}
</details>
<details class="rank-detail">
  <summary>展开查看极端低分率：&lt;2分</summary>
  {render_metric_heatmap(pivots.get('Turn-Level::lt2', []) + pivots.get('Session-Level::lt2', []), model_order, as_percent=True, reverse=True)}
</details>
"""


def render_ability_profile_line_chart(
    ability_pivot: list[dict[str, Any]],
    model_summary: list[dict[str, Any]],
    title: str,
) -> str:
    if not ability_pivot:
        return ""
    model_order = [row["model"] for row in model_summary]
    series_by_model = {row["model"]: row.get("series", "Other") for row in model_summary}
    labels = [str(row.get("ability", "")) for row in ability_pivot]
    width = max(1350, 96 + len(labels) * 46)
    height = 520
    left, right, top, bottom = 76, 34, 28, 365
    step = (width - left - right) / max(1, len(labels) - 1)
    grid = []
    for score in [1, 2, 3, 4, 5]:
        y = value_to_y(score, top, bottom)
        grid.append(
            f'<line x1="{left}" y1="{y:.1f}" x2="{width - right}" y2="{y:.1f}" class="grid" />'
            f'<text x="{left - 10}" y="{y + 4:.1f}" text-anchor="end" class="axis">{score}</text>'
        )
    xlabels = "".join(
        f'<text x="{left + i * step:.1f}" y="{bottom + 20}" text-anchor="end" class="axis" '
        f'transform="rotate(-55 {left + i * step:.1f} {bottom + 20})">{escape(label)}</text>'
        for i, label in enumerate(labels)
    )
    lines = []
    for model in model_order:
        points = []
        markers = []
        for index, row in enumerate(ability_pivot):
            value = maybe_float(row.get(model))
            if value is None:
                continue
            x = left + index * step
            y = value_to_y(value, top, bottom)
            points.append((x, y))
            tooltip = f"{model} / {row.get('ability')}: {value:.3f}"
            key = escape(model, quote=True)
            markers.append(
                f'<circle class="chart-data" data-key="{key}" data-tooltip="{escape(tooltip, quote=True)}" '
                f'cx="{x:.1f}" cy="{y:.1f}" r="2.6" fill="{SERIES_COLORS.get(series_by_model.get(model, "Other"), SERIES_COLORS["Other"])}" />'
            )
        if len(points) < 2:
            continue
        color = SERIES_COLORS.get(series_by_model.get(model, "Other"), SERIES_COLORS["Other"])
        key = escape(model, quote=True)
        point_text = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
        lines.append(
            f'<polyline class="chart-data" data-key="{key}" data-tooltip="{key}" points="{point_text}" '
            f'fill="none" stroke="{color}" stroke-width="1.8" opacity="0.58" />'
            + "".join(markers)
        )
    legend_series = ordered_unique([str(row.get("series", "Other")) for row in model_summary])
    return f"""
<div class="chart-card">
  <h3>{escape(title)}</h3>
  <div class="chart-viewport">
    <svg class="chart" viewBox="0 0 {width} {height}" style="min-width:{width}px">
      <style>.grid{{stroke:#e2e8f0;stroke-width:1;}}.axis{{fill:#64748b;font-size:11px;}}.legend{{fill:#475569;font-size:12px;}}</style>
      {''.join(grid)}
      {xlabels}
      {''.join(lines)}
      {svg_legend(legend_series, left, height - 24)}
    </svg>
  </div>
</div>
"""


def render_scene_distribution_chart(major_rows: list[dict[str, Any]]) -> str:
    if not major_rows:
        return ""
    width, height = 980, 42 + len(major_rows) * 30
    left, right, top = 230, 70, 26
    max_total = max(row.get("total", 0) for row in major_rows) or 1
    bars = []
    for index, row in enumerate(major_rows):
        y = top + index * 30
        image_count = int(row.get("image") or 0)
        video_count = int(row.get("video") or 0)
        total = int(row.get("total") or 0)
        image_w = image_count / max_total * (width - left - right)
        video_w = video_count / max_total * (width - left - right)
        label = str(row.get("environment_major"))
        bars.append(
            f'<text x="{left - 8}" y="{y + 15}" text-anchor="end" class="axis">{escape(label)}</text>'
            f'<rect x="{left}" y="{y}" width="{image_w:.1f}" height="18" rx="3" fill="#2563eb" opacity="0.82" />'
            f'<rect x="{left + image_w:.1f}" y="{y}" width="{video_w:.1f}" height="18" rx="3" fill="#d97706" opacity="0.82" />'
            f'<text x="{left + image_w + video_w + 6:.1f}" y="{y + 15}" class="axis">{total}</text>'
        )
    return f"""
<div class="chart-card">
  <h3>场景大类分布：Image / Video / Total</h3>
  <svg class="chart" viewBox="0 0 {width} {height}">
    <style>.axis{{fill:#475569;font-size:12px;}}.legend{{fill:#475569;font-size:12px;}}</style>
    {''.join(bars)}
    <rect x="{left}" y="{height - 24}" width="12" height="12" fill="#2563eb" opacity="0.82" /><text x="{left + 18}" y="{height - 14}" class="legend">image</text>
    <rect x="{left + 80}" y="{height - 24}" width="12" height="12" fill="#d97706" opacity="0.82" /><text x="{left + 98}" y="{height - 14}" class="legend">video</text>
  </svg>
</div>
"""


def render_scene_primary_heatmap(scene_stats: dict[str, Any], top_primary: int = 18) -> str:
    major_rows = scene_stats.get("major_rows", [])
    primary_rows = scene_stats.get("primary_rows", [])[:top_primary]
    matrix_rows = scene_stats.get("matrix_rows", [])
    if not major_rows or not primary_rows:
        return ""
    majors = [row["environment_major"] for row in major_rows]
    primaries = [row["primary_category"] for row in primary_rows]
    lookup = {
        (row["environment_major"], row["primary_category"]): int(row.get("total") or 0)
        for row in matrix_rows
    }
    max_count = max(lookup.values()) if lookup else 1
    header = "".join(f"<th>{escape(primary)}</th>" for primary in primaries)
    body = []
    for major in majors:
        cells = []
        for primary in primaries:
            value = lookup.get((major, primary), 0)
            if value <= 0:
                cells.append('<td class="blank">0</td>')
                continue
            intensity = value / max_count
            color = score_color(1 + intensity * 4)
            text_color = text_color_for_background(color)
            cells.append(
                f'<td class="heat" style="background:{color};color:{text_color}">{value}</td>'
            )
        body.append(f"<tr><th>{escape(major)}</th>{''.join(cells)}</tr>")
    return (
        '<div class="table-scroll">'
        '<table class="heatmap scene-matrix">'
        f"<thead><tr><th>Environment Major</th>{header}</tr></thead>"
        f"<tbody>{''.join(body)}</tbody></table></div>"
    )


def render_scene_environment_section(scene_stats: dict[str, Any]) -> str:
    if not scene_stats:
        return "<p class=\"section-note\">未找到 omnibench_dataset 场景分类文件。</p>"
    dataset_rows = scene_stats.get("dataset_rows", [])
    major_rows = scene_stats.get("major_rows", [])
    detail_rows = scene_stats.get("detail_rows", [])
    primary_rows = scene_stats.get("primary_rows", [])

    dataset_body = "".join(
        "<tr>"
        f"<td>{escape(str(row.get('task')))}</td>"
        f"<td>{fmt(row.get('session_count'), 0)}</td>"
        f"<td>{fmt(row.get('categorized_turn_count'), 0)}</td>"
        f"<td>{fmt(row.get('major_environment_count'), 0)}</td>"
        f"<td>{fmt(row.get('detail_environment_count'), 0)}</td>"
        f"<td>{fmt(row.get('primary_category_count'), 0)}</td>"
        "</tr>"
        for row in dataset_rows
    )
    major_body = "".join(
        "<tr>"
        f"<td>{escape(str(row.get('environment_major')))}</td>"
        f"<td>{fmt(row.get('image'), 0)}</td>"
        f"<td>{fmt(row.get('video'), 0)}</td>"
        f"<td>{fmt(row.get('total'), 0)}</td>"
        "</tr>"
        for row in major_rows
    )
    detail_body = "".join(
        "<tr>"
        f"<td>{escape(str(row.get('environment_major')))}</td>"
        f"<td>{escape(str(row.get('environment_detail')))}</td>"
        f"<td>{fmt(row.get('image'), 0)}</td>"
        f"<td>{fmt(row.get('video'), 0)}</td>"
        f"<td>{fmt(row.get('total'), 0)}</td>"
        "</tr>"
        for row in detail_rows
    )
    primary_body = "".join(
        "<tr>"
        f"<td>{escape(str(row.get('primary_category')))}</td>"
        f"<td>{fmt(row.get('image'), 0)}</td>"
        f"<td>{fmt(row.get('video'), 0)}</td>"
        f"<td>{fmt(row.get('total'), 0)}</td>"
        "</tr>"
        for row in primary_rows[:30]
    )
    return f"""
<section id="scene-section">
  <h2>场景分类下的实验结果</h2>
  <p class="section-note">
    数据来源为 <code>omnibench_dataset/image_final_with_mimt_category.json</code> 与
    <code>omnibench_dataset/video_final_with_vqa_category.json</code>。<code>environment</code>
    按第一个 <code>-</code> 切分：前半部分为场景大类，后半部分为细分场景；二维热力图统计的是“场景大类 × 对话轮次 Primary Category”的出现次数。
  </p>
  <div class="table-scroll">
    <table class="mini-table">
      <thead><tr><th>Task</th><th>样本组数</th><th>带类别轮次数</th><th>场景大类数</th><th>细分场景数</th><th>Primary 数</th></tr></thead>
      <tbody>{dataset_body}</tbody>
    </table>
  </div>
  {render_scene_distribution_chart(major_rows)}
  <h3>场景大类分布</h3>
  <div class="table-scroll"><table class="mini-table"><thead><tr><th>场景大类</th><th>Image</th><th>Video</th><th>Total</th></tr></thead><tbody>{major_body}</tbody></table></div>
  <h3>场景大类与场景小类从属关系（全量）</h3>
  <p class="section-note">每一行表示一个 <code>environment</code> 标签拆分后的从属关系：左侧是场景大类，右侧是该大类下的细分场景，并保留 image / video / total 的数据组数量。</p>
  <div class="table-scroll"><table class="mini-table"><thead><tr><th>场景大类</th><th>细分场景</th><th>Image</th><th>Video</th><th>Total</th></tr></thead><tbody>{detail_body}</tbody></table></div>
  <h3>Primary Category 轮次分布 Top 30</h3>
  <div class="table-scroll"><table class="mini-table"><thead><tr><th>Primary Category</th><th>Image</th><th>Video</th><th>Total</th></tr></thead><tbody>{primary_body}</tbody></table></div>
  <h3>场景大类 × Primary Category 二维分布 Top 18 Primary</h3>
  {render_scene_primary_heatmap(scene_stats)}
</section>
"""


def render_series_task_chart(task_matrix: list[dict[str, Any]], tasks: list[str]) -> str:
    by_series: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in task_matrix:
        by_series[str(row.get("series"))].append(row)
    width, height = 900, 370
    left, right, top, bottom = 80, 36, 28, 275
    step = (width - left - right) / max(1, len(tasks) - 1)
    lines = []
    for series in sorted(by_series, key=lambda item: SERIES_ORDER.get(item, 999)):
        points = []
        for index, task in enumerate(tasks):
            vals = finite_values([row.get(task) for row in by_series[series]])
            if not vals:
                continue
            points.append((left + index * step, value_to_y(mean(vals), top, bottom), mean(vals), task))
        if len(points) < 2:
            continue
        color = SERIES_COLORS.get(series, SERIES_COLORS["Other"])
        point_text = " ".join(f"{x:.1f},{y:.1f}" for x, y, _, _ in points)
        key = escape(series, quote=True)
        labels = "".join(
            f'<circle class="chart-data" data-key="{key}" data-tooltip="{escape(f"{series} {task}: {val:.3f}", quote=True)}" '
            f'cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{color}" />'
            for x, y, val, task in points
        )
        lines.append(
            f'<polyline class="chart-data" data-key="{key}" data-tooltip="{key}" points="{point_text}" '
            f'fill="none" stroke="{color}" stroke-width="2.4" opacity="0.85" />{labels}'
        )
    grid = []
    for score in [1, 2, 3, 4, 5]:
        y = value_to_y(score, top, bottom)
        grid.append(
            f'<line x1="{left}" y1="{y:.1f}" x2="{width - right}" y2="{y:.1f}" class="grid" />'
            f'<text x="{left - 10}" y="{y + 4:.1f}" text-anchor="end" class="axis">{score}</text>'
        )
    labels = "".join(
        f'<text x="{left + i * step:.1f}" y="{bottom + 24}" text-anchor="middle" class="axis">{escape(short_task_label(task))}</text>'
        for i, task in enumerate(tasks)
    )
    return f"""
<div class="chart-card">
  <h3>系列平均任务走势</h3>
  <svg class="chart" viewBox="0 0 {width} {height}">
    <style>.grid{{stroke:#e2e8f0;stroke-width:1;}}.axis{{fill:#64748b;font-size:12px;}}.legend{{fill:#475569;font-size:12px;}}</style>
    {''.join(grid)}
    {labels}
    {''.join(lines)}
    {svg_legend(sorted(by_series, key=lambda item: SERIES_ORDER.get(item, 999)), left, height - 26)}
  </svg>
</div>
"""


def render_series_metric_profile(metric_score_pivot: list[dict[str, Any]], model_summary: list[dict[str, Any]]) -> str:
    by_series: dict[str, list[str]] = defaultdict(list)
    for row in model_summary:
        by_series[str(row.get("series"))].append(row["model"])
    width, height = 1100, 400
    left, right, top, bottom = 76, 30, 28, 285
    labels = [f"{row.get('level')} / {row.get('metric')}" for row in metric_score_pivot]
    step = (width - left - right) / max(1, len(labels) - 1)
    lines = []
    for series in sorted(by_series, key=lambda item: SERIES_ORDER.get(item, 999)):
        points = []
        for index, row in enumerate(metric_score_pivot):
            vals = finite_values([row.get(model) for model in by_series[series]])
            if not vals:
                continue
            points.append((left + index * step, value_to_y(mean(vals), top, bottom)))
        if len(points) < 2:
            continue
        color = SERIES_COLORS.get(series, SERIES_COLORS["Other"])
        key = escape(series, quote=True)
        lines.append(
            f'<polyline class="chart-data" data-key="{key}" data-tooltip="{key}" '
            f'points="{" ".join(f"{x:.1f},{y:.1f}" for x, y in points)}" fill="none" '
            f'stroke="{color}" stroke-width="2.4" opacity="0.9" />'
        )
    grid = []
    for score in [1, 2, 3, 4, 5]:
        y = value_to_y(score, top, bottom)
        grid.append(
            f'<line x1="{left}" y1="{y:.1f}" x2="{width - right}" y2="{y:.1f}" class="grid" />'
            f'<text x="{left - 10}" y="{y + 4:.1f}" text-anchor="end" class="axis">{score}</text>'
        )
    xlabels = "".join(
        f'<text x="{left + i * step:.1f}" y="{bottom + 22}" text-anchor="end" class="axis" transform="rotate(-35 {left + i * step:.1f} {bottom + 22})">{escape(short_metric_label(label))}</text>'
        for i, label in enumerate(labels)
    )
    return f"""
<div class="chart-card">
  <h3>系列平均整体指标画像</h3>
  <svg class="chart" viewBox="0 0 {width} {height}">
    <style>.grid{{stroke:#e2e8f0;stroke-width:1;}}.axis{{fill:#64748b;font-size:11px;}}.legend{{fill:#475569;font-size:12px;}}</style>
    {''.join(grid)}
    {xlabels}
    {''.join(lines)}
    {svg_legend(sorted(by_series, key=lambda item: SERIES_ORDER.get(item, 999)), left, height - 24)}
  </svg>
</div>
"""


def render_metric_heatmap(
    rows: list[dict[str, Any]],
    model_order: list[str],
    as_percent: bool = False,
    reverse: bool = False,
) -> str:
    header = "".join(f"<th>{escape(model)}</th>" for model in model_order)
    body_rows = []
    for row in rows:
        ranks = rank_lookup_for_row_values(row, model_order, descending=not reverse)
        cells = "".join(
            heat_cell(
                row.get(model),
                lower=0.0 if as_percent else 1.0,
                upper=1.0 if as_percent else 5.0,
                reverse=reverse,
                as_percent=as_percent,
                rank=ranks.get(model),
            )
            for model in model_order
        )
        body_rows.append(
            "<tr>"
            f"<th>{escape(row.get('level', ''))}</th>"
            f"<th>{escape(row.get('metric', ''))}</th>"
            f"{cells}</tr>"
        )
    return (
        '<div class="table-scroll"><table class="heatmap metric-table" data-col-sortable="true" data-frozen-cols="2">'
        f"<thead><tr><th>Level</th><th>Metric</th>{header}</tr></thead>"
        f"<tbody>{''.join(body_rows)}</tbody></table></div>"
    )


def render_ability_heatmap(
    rows: list[dict[str, Any]],
    model_order: list[str],
    limit: int | None = None,
    *,
    as_percent: bool = False,
    reverse: bool = False,
) -> str:
    selected = rows[:limit] if limit else rows
    header = "".join(f"<th>{escape(model)}</th>" for model in model_order)
    body_rows = []
    for row in selected:
        ranks = rank_lookup_for_row_values(row, model_order, descending=not reverse)
        cells = "".join(
            heat_cell(
                row.get(model),
                lower=0.0 if as_percent else 1.0,
                upper=1.0 if as_percent else 5.0,
                reverse=reverse,
                as_percent=as_percent,
                rank=ranks.get(model),
            )
            for model in model_order
        )
        body_rows.append(
            "<tr>"
            f"<th>{escape(row.get('ability', ''))}</th>"
            f"<td>{fmt(row.get('model_count'), 0)}</td>"
            f"{cells}</tr>"
        )
    return (
        '<div class="table-scroll"><table class="heatmap ability-table" data-col-sortable="true" data-frozen-cols="2">'
        "<thead><tr><th>Ability</th><th>覆盖模型数</th>"
        f"{header}</tr></thead><tbody>{''.join(body_rows)}</tbody></table></div>"
    )


def render_metric_rankings(rankings: dict[str, list[dict[str, Any]]]) -> str:
    level_order = {"Turn-Level": 0, "Session-Level": 1}
    labels = sorted(
        rankings,
        key=lambda label: (
            level_order.get(label.split(" / ", 1)[0], 99),
            label.split(" / ", 1)[-1],
        ),
    )
    cards = []
    for label in labels:
        rows = rankings[label]
        body = "".join(
            "<tr>"
            f"<td>{row.get('rank')}</td>"
            f"<td>{escape(str(row.get('model', '')))}</td>"
            f"<td>{escape(str(row.get('series', '')))}</td>"
            f"<td>{fmt(row.get('avg_score'))}</td>"
            f"<td>{percent(row.get('low_rate'))}</td>"
            "</tr>"
            for row in rows
        )
        cards.append(
            '<details class="rank-detail">'
            f"<summary>{escape(label)}</summary>"
            '<div class="rank-table-scroll">'
            '<table class="mini-table"><thead><tr><th>#</th><th>Model</th><th>Series</th>'
            "<th>平均分</th><th>低分率</th></tr></thead>"
            f"<tbody>{body}</tbody></table></div></details>"
        )
    return '<div class="rank-grid metric-ranks">' + "\n".join(cards) + "</div>"


def render_ability_rankings(
    rankings: dict[str, list[dict[str, Any]]],
    model_order: list[str] | None = None,
) -> str:
    allowed = set(model_order) if model_order else None
    details = []
    for ability in sorted(rankings):
        rows = rankings[ability]
        if allowed is not None:
            rows = [row for row in rows if row.get("model") in allowed]
            rows = sorted(rows, key=lambda row: row.get("overall_score") or -math.inf, reverse=True)
            rows = [{**row, "rank": index} for index, row in enumerate(rows, start=1)]
        body = "".join(
            "<tr>"
            f"<td>{row.get('rank')}</td>"
            f"<td>{escape(str(row.get('model', '')))}</td>"
            f"<td>{escape(str(row.get('series', '')))}</td>"
            f"<td>{fmt(row.get('overall_score'))}</td>"
            f"<td>{fmt(row.get('sample_count'), 0)}</td>"
            f"<td>{fmt(row.get('accuracy'))}</td>"
            f"<td>{fmt(row.get('proactiveness'))}</td>"
            f"<td>{fmt(row.get('intent_depth'))}</td>"
            "</tr>"
            for row in rows
        )
        details.append(
            '<details class="rank-detail ability-rank" data-ability="'
            f'{escape(ability)}">'
            f"<summary>{escape(ability)}</summary>"
            '<table class="mini-table"><thead><tr><th>#</th><th>Model</th><th>Series</th>'
            "<th>整体均分</th><th>样本数</th><th>Accuracy</th>"
            "<th>Proactiveness</th><th>Intent Depth</th></tr></thead>"
            f"<tbody>{body}</tbody></table></details>"
        )
    return '<div class="rank-list">' + "\n".join(details) + "</div>"


def render_series_sections(
    model_summary: list[dict[str, Any]],
    task_matrix: list[dict[str, Any]],
    tasks: list[str],
    task_rankings: dict[str, list[dict[str, Any]]],
    metric_score_pivot: list[dict[str, Any]],
    primary_pivot: list[dict[str, Any]],
    secondary_pivot: list[dict[str, Any]],
) -> str:
    by_series: dict[str, list[str]] = defaultdict(list)
    summary_by_model = {row["model"]: row for row in model_summary}
    for row in model_summary:
        by_series[row["series"]].append(row["model"])

    sections = []
    for series in sorted(by_series, key=lambda item: SERIES_ORDER.get(item, 999)):
        models = by_series[series]
        rows = sorted(
            [summary_by_model[model] for model in models],
            key=lambda row: row.get("display_order") or 999,
        )
        summary_rank_maps = {
            key: rank_lookup_for_key(rows, key)
            for key in (
                "text_task_weighted_avg",
                "all_task_weighted_avg",
                "turn_metric_avg",
                "session_metric_avg",
            )
        }
        summary_body = "".join(
            "<tr>"
            f"<td>{escape(str(row.get('model', '')))}</td>"
            f"<td>{escape(str(row.get('task_coverage') or ''))}</td>"
            f"<td>{bar_cell(row.get('text_task_weighted_avg'), rank=summary_rank_maps['text_task_weighted_avg'].get(str(row.get('model', ''))))}</td>"
            f"<td>{bar_cell(row.get('all_task_weighted_avg'), rank=summary_rank_maps['all_task_weighted_avg'].get(str(row.get('model', ''))))}</td>"
            f"<td>{bar_cell(row.get('turn_metric_avg'), rank=summary_rank_maps['turn_metric_avg'].get(str(row.get('model', ''))))}</td>"
            f"<td>{bar_cell(row.get('session_metric_avg'), rank=summary_rank_maps['session_metric_avg'].get(str(row.get('model', ''))))}</td>"
            "</tr>"
            for row in rows
        )
        sections.append(
            '<details class="series-detail">'
            f"<summary>{escape(series)} 系列内比较</summary>"
            '<h3>系列内覆盖与宏均值概览</h3>'
            '<table class="mini-table"><thead><tr><th>Model</th><th>任务覆盖</th>'
            "<th>文本任务加权均分</th><th>全部任务加权均分</th>"
            f"<th>Turn 指标加权均值</th><th>Session 指标加权均值</th></tr></thead><tbody>{summary_body}</tbody></table>"
            "<h3>任务均分与系列内排名</h3>"
            f"{render_task_matrix(task_matrix, tasks, models)}"
            f"{render_task_rankings(task_rankings, tasks, models)}"
            "<h3>整体指标平均分热力图</h3>"
            f"{render_metric_heatmap(metric_score_pivot, models)}"
            "<h3>Primary Category 整体均分热力图</h3>"
            f"{render_ability_heatmap(primary_pivot, models)}"
            "<h3>Secondary Category 整体均分热力图</h3>"
            f"{render_ability_heatmap(secondary_pivot, models)}"
            "</details>"
        )
    return "\n".join(sections)


def render_series_cards(series_summary: list[dict[str, Any]]) -> str:
    by_series: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in series_summary:
        by_series[row["series"]].append(row)

    cards = []
    for series, rows in by_series.items():
        session_ranks = rank_lookup_for_key(rows, "session_metric_avg")
        rows_html = []
        for row in rows:
            model = str(row.get("model", ""))
            rows_html.append(
                "<tr>"
                f"<td>{row.get('series_rank')}</td>"
                f"<td>{escape(model)}</td>"
                f"<td>{bar_cell(row.get('all_metric_avg'))}</td>"
                f"<td>{bar_cell(row.get('session_metric_avg'), rank=session_ranks.get(model))}</td>"
                f"<td>{fmt(row.get('gap_to_series_best'))}</td>"
                "</tr>"
            )
        cards.append(
            '<section class="series-card">'
            f"<h3>{escape(series)}</h3>"
            '<table><thead><tr><th>#</th><th>Model</th><th>Overall</th>'
            "<th>Session</th><th>Gap</th></tr></thead>"
            f"<tbody>{''.join(rows_html)}</tbody></table></section>"
        )
    return "\n".join(cards)


def render_takeaways(key_takeaways: list[dict[str, Any]]) -> str:
    by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in key_takeaways:
        by_model[row["model"]].append(row)
    chunks = []
    for model, rows in sorted(by_model.items()):
        bullets = "".join(
            f"<li><strong>{escape(str(row.get('label', '')))}</strong>"
            f"{'：' if row.get('label') else ''}{escape(str(row.get('text', '')))}</li>"
            for row in rows
        )
        chunks.append(f"<details><summary>{escape(model)}</summary><ul>{bullets}</ul></details>")
    return "\n".join(chunks)


def render_html(
    out_path: Path,
    root: Path,
    model_summary: list[dict[str, Any]],
    task_matrix: list[dict[str, Any]],
    tasks: list[str],
    task_rankings_map: dict[str, list[dict[str, Any]]],
    metric_score_pivot: list[dict[str, Any]],
    metric_low_rate_pivot: list[dict[str, Any]],
    metric_pass_rate_pivot: list[dict[str, Any]],
    metric_distribution_stats: list[dict[str, Any]],
    model_metric_self_stats: list[dict[str, Any]],
    metric_rankings_map: dict[str, list[dict[str, Any]]],
    primary_rows: list[dict[str, Any]],
    primary_pivot: list[dict[str, Any]],
    primary_pass_pivot: list[dict[str, Any]],
    primary_filtered_pivot: list[dict[str, Any]],
    primary_filtered_stats: list[dict[str, Any]],
    primary_filtered_pass_pivot: list[dict[str, Any]],
    primary_filtered_pass_stats: list[dict[str, Any]],
    primary_rankings_map: dict[str, list[dict[str, Any]]],
    secondary_rows: list[dict[str, Any]],
    secondary_pivot: list[dict[str, Any]],
    secondary_rankings_map: dict[str, list[dict[str, Any]]],
    key_takeaways: list[dict[str, Any]],
    scene_stats: dict[str, Any],
    phase_analysis: dict[str, Any],
) -> None:
    model_order = [row["model"] for row in model_summary]
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    best_session = max(model_summary, key=lambda row: row.get("session_metric_avg") or 0) if model_summary else {}
    best_text_task = max(
        model_summary,
        key=lambda row: row.get("text_task_weighted_avg") or 0,
    ) if model_summary else {}

    html = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Phase5 模型横向对比</title>
  <style>
    :root {{
      --bg: #f7f8fb;
      --panel: #ffffff;
      --text: #17202a;
      --muted: #64748b;
      --line: #dbe3ee;
      --accent: #2563eb;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family: "Segoe UI", "Microsoft YaHei", Arial, sans-serif;
      line-height: 1.45;
    }}
    header {{
      padding: 28px 32px 20px;
      background: #ffffff;
      border-bottom: 1px solid var(--line);
    }}
    h1 {{ margin: 0 0 8px; font-size: 28px; }}
    h2 {{ margin: 0 0 16px; font-size: 22px; }}
    h3 {{ margin: 18px 0 12px; font-size: 17px; }}
    main {{ padding: 24px 32px 48px; }}
    section {{
      margin-bottom: 24px;
      padding: 20px;
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
    }}
    .section-note {{
      margin: -6px 0 14px;
      color: var(--muted);
      font-size: 14px;
    }}
    .nav {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-top: 16px;
    }}
    .nav a {{
      color: var(--accent);
      text-decoration: none;
      padding: 6px 10px;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #f8fbff;
    }}
    .meta {{
      display: flex;
      flex-wrap: wrap;
      gap: 12px;
      color: var(--muted);
      font-size: 14px;
    }}
    .kpis {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
      gap: 12px;
      margin-bottom: 24px;
    }}
    .kpi {{
      padding: 16px;
      background: #ffffff;
      border: 1px solid var(--line);
      border-radius: 8px;
    }}
    .kpi .label {{ color: var(--muted); font-size: 13px; }}
    .kpi .value {{ margin-top: 6px; font-size: 26px; font-weight: 700; }}
    .table-scroll {{ overflow: auto; border: 1px solid var(--line); border-radius: 8px; }}
    table {{
      border-collapse: collapse;
      width: 100%;
      min-width: 880px;
      background: #ffffff;
      font-size: 13px;
    }}
    th, td {{
      border-bottom: 1px solid var(--line);
      border-right: 1px solid #edf2f7;
      padding: 8px 10px;
      text-align: left;
      vertical-align: middle;
      white-space: nowrap;
    }}
    th {{
      background: #f1f5f9;
      position: sticky;
      top: 0;
      z-index: 1;
      font-weight: 650;
    }}
    .heatmap th:first-child, .ability-table th:first-child {{
      position: sticky;
      left: 0;
      z-index: 2;
      background: #f1f5f9;
    }}
    #leaderboard th:first-child,
    #leaderboard td:first-child {{
      position: sticky;
      left: 0;
      z-index: 2;
      min-width: 170px;
      background: #ffffff;
      box-shadow: 1px 0 0 var(--line);
    }}
    #leaderboard th:first-child {{
      z-index: 4;
      background: #f1f5f9;
    }}
    .metric-table th:nth-child(1) {{
      min-width: 112px;
      width: 112px;
      left: 0;
      z-index: 3;
      box-shadow: 1px 0 0 var(--line);
    }}
    .metric-table th:nth-child(2) {{
      position: sticky;
      left: 112px;
      z-index: 3;
      min-width: 210px;
      width: 210px;
      background: #f1f5f9;
      box-shadow: 1px 0 0 var(--line);
    }}
    .metric-table thead th:nth-child(1),
    .metric-table thead th:nth-child(2) {{
      z-index: 5;
    }}
    .heat {{
      text-align: center;
      font-variant-numeric: tabular-nums;
      min-width: 72px;
    }}
    .blank {{ background: #f8fafc; }}
    .bar-cell {{
      position: relative;
      min-width: 130px;
      height: 22px;
      background: #eef2f7;
      border-radius: 4px;
      overflow: hidden;
    }}
    .bar {{
      position: absolute;
      inset: 0 auto 0 0;
      opacity: 0.86;
    }}
    .bar-cell span {{
      position: relative;
      display: block;
      padding: 2px 6px;
      font-variant-numeric: tabular-nums;
      font-weight: 650;
    }}
    .bar-cell small {{
      display: inline-block;
      margin-left: 6px;
      font-size: 11px;
      font-weight: 700;
      opacity: 0.72;
    }}
    .rank-heat span {{ display: block; font-weight: 700; }}
    .rank-heat small {{ display: block; margin-top: 1px; opacity: 0.86; }}
    .muted {{ color: var(--muted); }}
    .stat-note {{
      margin: 10px 0 12px;
      padding: 10px 12px;
      border: 1px solid #bfdbfe;
      border-radius: 8px;
      background: #eff6ff;
      color: #1e3a8a;
      font-size: 13px;
    }}
    .stats-table {{
      min-width: 1500px;
    }}
    .metric-stats-table {{
      min-width: 2300px;
    }}
    .metric-stats-table th:nth-child(1),
    .metric-stats-table td:nth-child(1) {{
      position: sticky;
      left: 0;
      z-index: 3;
      min-width: 112px;
      width: 112px;
      background: #ffffff;
      box-shadow: 1px 0 0 var(--line);
    }}
    .metric-stats-table th:nth-child(1) {{
      background: #f1f5f9;
      z-index: 5;
    }}
    .metric-stats-table th:nth-child(2),
    .metric-stats-table td:nth-child(2) {{
      position: sticky;
      left: 112px;
      z-index: 3;
      min-width: 210px;
      width: 210px;
      background: #ffffff;
      box-shadow: 1px 0 0 var(--line);
    }}
    .metric-stats-table th:nth-child(2) {{
      background: #f1f5f9;
      z-index: 5;
    }}
    .scene-matrix {{
      min-width: 1400px;
    }}
    .rank-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(310px, 1fr));
      gap: 14px;
    }}
    .metric-ranks {{
      grid-template-columns: repeat(auto-fit, minmax(430px, 1fr));
    }}
    .rank-table-scroll {{
      max-width: 100%;
      overflow-x: auto;
    }}
    .chart-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(420px, 1fr));
      gap: 14px;
      margin: 12px 0 18px;
    }}
    .chart-card {{
      margin: 14px 0;
      padding: 14px;
      background: #fbfdff;
      border: 1px solid var(--line);
      border-radius: 8px;
    }}
    .chart-card h3 {{ margin-top: 0; }}
    .small-chart {{ margin: 0; }}
    .chart {{
      display: block;
      width: 100%;
      height: auto;
      overflow: visible;
    }}
    .chart-viewport {{
      width: 100%;
      overflow: auto;
      border-radius: 6px;
    }}
    .chart-controls, .table-controls {{
      display: flex;
      flex-wrap: wrap;
      align-items: center;
      gap: 8px;
      margin: 6px 0 10px;
    }}
    .chart-controls button, .table-controls button, .sort-btn, .col-sort-btn {{
      border: 1px solid var(--line);
      background: #ffffff;
      color: #334155;
      border-radius: 5px;
      padding: 2px 7px;
      font: inherit;
      font-size: 12px;
      cursor: pointer;
    }}
    .sort-btn, .col-sort-btn {{
      margin-left: 6px;
      padding: 1px 5px;
      line-height: 1.2;
    }}
    th.sort-active {{
      background: #dbeafe;
    }}
    .chart-data {{
      cursor: pointer;
      transition: opacity 120ms ease, filter 120ms ease, stroke-width 120ms ease;
    }}
    .chart.has-hover .chart-data.dimmed {{
      opacity: 0.08 !important;
    }}
    .chart-data.is-highlight {{
      opacity: 1 !important;
      filter: drop-shadow(0 0 5px rgba(37, 99, 235, 0.55));
      stroke-width: 3.2px !important;
    }}
    .chart-tooltip {{
      position: fixed;
      z-index: 9999;
      max-width: 360px;
      padding: 7px 9px;
      background: rgba(15, 23, 42, 0.93);
      color: #ffffff;
      border-radius: 6px;
      font-size: 12px;
      line-height: 1.35;
      pointer-events: none;
      display: none;
      box-shadow: 0 10px 24px rgba(15, 23, 42, 0.22);
    }}
    .chart-detail svg {{ margin-top: 10px; }}
    details {{
      margin: 8px 0;
      padding: 10px 12px;
      background: #fbfdff;
      border: 1px solid var(--line);
      border-radius: 8px;
    }}
    summary {{ cursor: pointer; font-weight: 650; }}
    li {{ margin: 6px 0; }}
    .mini-table {{
      min-width: 0;
      font-size: 12px;
      margin-top: 10px;
    }}
    .mini-table th, .mini-table td {{ padding: 6px 8px; }}
    .rank-list {{
      max-height: 720px;
      overflow: auto;
      padding-right: 4px;
    }}
    .series-detail > summary {{
      font-size: 16px;
      color: var(--accent);
    }}
    .toolbar {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      margin-bottom: 12px;
    }}
    input[type="search"] {{
      width: min(360px, 100%);
      padding: 8px 10px;
      border: 1px solid var(--line);
      border-radius: 6px;
      font: inherit;
    }}
  </style>
</head>
<body>
  <header>
    <h1>Phase5 模型横向对比</h1>
    <div class="meta">
      <span>数据根目录：{escape(str(root))}</span>
      <span>生成时间：{escape(generated_at)}</span>
      <span>模型数：{len(model_summary)}</span>
    </div>
    <nav class="nav">
      <a href="#task-low-section">任务低分</a>
      <a href="#scene-score-section">场景得分</a>
      <a href="#phase4-section">失败归因</a>
      <a href="#leaderboard-section">总榜口径</a>
      <a href="#tasks-section">任务对比</a>
      <a href="#metrics-section">整体指标</a>
      <a href="#primary-section">Primary 能力</a>
      <a href="#secondary-section">Secondary 能力</a>
      <a href="#scene-section">场景分类</a>
      <a href="#series-section">系列内比较</a>
      <a href="#takeaways-section">关键看点</a>
    </nav>
  </header>
  <main>
    <div class="kpis">
      <div class="kpi"><div class="label">文本任务加权均分最高</div><div class="value">{escape(str(best_text_task.get('model', '')))}</div></div>
      <div class="kpi"><div class="label">最高文本任务加权均分</div><div class="value">{fmt(best_text_task.get('text_task_weighted_avg'))}</div></div>
      <div class="kpi"><div class="label">Session 均值最高</div><div class="value">{escape(str(best_session.get('model', '')))}</div></div>
      <div class="kpi"><div class="label">任务 / Primary / Secondary</div><div class="value">{len(tasks)} / {len(primary_pivot)} / {len(secondary_pivot)}</div></div>
    </div>

    <section id="leaderboard-section">
      <h2>覆盖与口径说明</h2>
      <p class="section-note">
        这里不再给“综合总榜”。之前的综合均分只是把“四、整体分数概览”中 8 个 Turn-Level 平均分和 4 个 Session-Level 平均分做简单宏平均，
        不是把所有原始轮次/会话分数放进同一个池子后的平均值；它也不能自然解决 Gemini 有四个任务而其他模型通常只有两个任务的问题。
        下表这些派生值全部来自每个报告“七、任务与模态分析 / 任务维度”表格。
        Turn 样本数和 Session 样本数分别为该表全部任务的样本数求和；
        Turn / Session 指标加权均值分别为对应均分按对应样本数加权。
        文本、TTS、全部任务加权均分则把匹配任务中的 Turn 均分×Turn 样本数 和 Session 均分×Session 样本数合并加权。
      </p>
      <div class="table-scroll">
        <table id="leaderboard" data-row-sortable="true">
          <thead>
            <tr><th>Model</th><th>Series</th><th>任务覆盖</th><th>Turn 样本数</th><th>Session 样本数</th><th>Turn 指标加权均值</th><th>Session 指标加权均值</th><th>文本任务加权均分</th><th>TTS 任务加权均分</th><th>全部任务加权均分</th><th>覆盖任务</th></tr>
          </thead>
          <tbody>{render_leaderboard_rows(model_summary)}</tbody>
        </table>
      </div>
    </section>

    <section id="tasks-section">
      <h2>任务均分与任务内排名</h2>
      <p class="section-note">
        来源为“三、关键看点速览”里的最强/最弱任务表；同一模型同一任务只保留一个均分。单元格显示“均分”和该任务下全模型排名。
      </p>
      {render_task_line_chart(task_matrix, tasks)}
      {render_task_bar_charts(task_rankings_map, tasks)}
      {render_task_matrix(task_matrix, tasks)}
      <h3>每个任务的模型排名</h3>
      {render_task_rankings(task_rankings_map, tasks)}
    </section>

    <section id="metrics-section">
      <h2>整体分数概览：平均分与低分率</h2>
      <p class="section-note">
        平均分热力图越绿越高；低分率热力图越绿越低。低分率使用每个模型原始评分中的对应指标分数计算：
        <code>低分率 = 指标分数 &lt; 4 的样本数 / 该指标有效样本数</code>。因为指标分数为 1-5 整数，
        它等价于 <code>&lt;=3</code>。通过率为 <code>1 - 低分率</code>，即 <code>&gt;=4</code> 的比例。
        下面的排行榜按每个指标的平均分从高到低排序，并同时列出低分率。
      </p>
      <div class="chart-grid">
        {render_turn_session_scatter(model_summary)}
      </div>
      {render_metric_profile_chart(metric_score_pivot, model_summary)}
      <h3>平均分热力图</h3>
      {render_metric_heatmap(metric_score_pivot, model_order)}
      <h3>低分率热力图</h3>
      {render_metric_heatmap(metric_low_rate_pivot, model_order, as_percent=True, reverse=True)}
      <h3>通过率热力图</h3>
      {render_metric_heatmap(metric_pass_rate_pivot, model_order, as_percent=True)}
      {render_pivot_pass_rate_line_chart(metric_pass_rate_pivot, model_summary, "12 个整体指标通过率折线图（>=4 分）", metric_pass_row_label)}
      <h3>Turn / Session 级整体通过率热力图</h3>
      <p class="section-note">
        这张图按样本整体均分计算，而不是按单个指标分别计算。Turn-Level 使用每条成功 turn 的 8 个 Turn-Level 指标分数求均值，
        <code>均值 &gt;= 4</code> 记为通过；Session-Level 使用每条成功 session 的 4 个 Session-Level 指标分数求均值，
        <code>均值 &gt;= 4</code> 记为通过。最终 <code>通过率 = 通过样本数 / 有效样本数</code>。
      </p>
      {render_matrix_heatmap(phase_analysis.get("level_pass_rate_pivot", []), model_order, "level", "Level", as_percent=True)}
      {render_metric_threshold_section(phase_analysis, model_order)}
      <h3>12 个整体指标的统计学诊断</h3>
      {render_metric_distribution_stats(metric_distribution_stats)}
      <p class="section-note">
        下面三张表从“模型自身”角度看稳定性：Turn 表只在 8 个 Turn-Level 指标之间统计，
        Session 表只在 4 个 Session-Level 指标之间统计，All 表在 12 个指标之间统计。
        这里的方差/标准差越大，表示该模型不同评测维度之间越不均衡。
      </p>
      {render_model_metric_self_stats(model_metric_self_stats, "Turn-Level", "各模型自身 8 个 Turn-Level 维度统计学诊断")}
      {render_model_metric_self_stats(model_metric_self_stats, "Session-Level", "各模型自身 4 个 Session-Level 维度统计学诊断")}
      {render_model_metric_self_stats(model_metric_self_stats, "All 12 Metrics", "各模型自身 12 个整体指标统计学诊断")}
      <h3>每个指标 Top 模型条形图</h3>
      {render_metric_rank_bar_charts(metric_rankings_map)}
      <h3>每个指标的模型排名</h3>
      {render_metric_rankings(metric_rankings_map)}
    </section>

    <section id="task-low-section">
      <h2>任务维度低分率与失败梯度</h2>
      <p class="section-note">
        这里把 phase3 的 Turn-Level 原始均分直接映射到任务层面，专门观察不同任务的失分风险。
        低分标准沿用同一条口径：<code>&lt;4</code> 表示进入低分区，<code>&lt;3</code> 表示更明显失分，
        <code>&lt;2</code> 表示极端低分。这样可以区分“轻微失分较多”和“尾部极差样本很多”这两类不同问题。
      </p>
      {render_task_low_score_section(phase_analysis, model_order)}
    </section>

    <section id="primary-section">
      <h2>Primary Category：整体均分热力图与能力排名</h2>
      <p class="section-note">
        这里不再显示 Mean/Spread，避免混淆；每个单元格就是对应模型在该 Primary 能力上的“整体均分”。
        根据 <code>tools/RUBRIC-MME/aggregation.py</code> 的实际逻辑，能力维度只使用 Turn-Level 成功样本：
        先按 <code>primary_category</code> 分组，再把该能力下所有成功 turn 的 8 个 Turn-Level 指标分数放入同一个池子求平均。
        如果每个 turn 都有完整 8 项指标，这等价于“每个 turn 先取 8 维均值，再对该能力下所有 turn 求平均”。
      </p>
      {render_ability_boxplot(primary_rows, model_summary, "Primary 能力分布箱线图")}
      <div class="chart-grid">
        {render_ability_stat_bars(primary_pivot, "mean_score", "Primary 共同短板：全模型均分最低的能力", ascending=True, lower=1.0, upper=5.0)}
        {render_ability_stat_bars(primary_pivot, "spread", "Primary 分化最大：最高分与最低分差距", ascending=False, lower=0.0, upper=4.0)}
      </div>
      <div class="toolbar">
        <span class="muted">热力图按全模型均分从低到高排序，便于先看共同短板。</span>
        <input id="primaryFilter" type="search" placeholder="筛选 Primary 能力名或模型名" />
      </div>
      <div id="primaryHeatmap">
        {render_ability_heatmap(primary_pivot, model_order)}
      </div>
      <h3>Primary 能力通过率热力图：Turn 均分 &gt; 4</h3>
      <p class="section-note">
        能力通过率直接回到逐 turn 结果计算：对某模型某 Primary 能力，统计该能力下成功 turn 的 <code>avg_score</code>。
        <code>通过率 = avg_score &gt; 4 的 turn 数 / 该能力成功 turn 总数</code>。这里按你的要求使用严格 <code>&gt;4</code>。
      </p>
      {render_ability_heatmap(primary_pass_pivot, model_order, as_percent=True)}
      <h3>Primary 能力过滤热力图：非 Gemini 模型样本数 >= 30</h3>
      <p class="section-note">
        该新增视图只保留在所有非 Gemini 模型报告的 Primary Category 表中样本数均不低于 30 的能力维度，
        以减少极小样本能力对横向比较的干扰；保留后仍展示全部模型在这些能力上的分数。
      </p>
      {render_ability_profile_line_chart(primary_filtered_pivot, model_summary, "过滤后 Primary 能力折线图")}
      <div id="primaryFilteredHeatmap">
        {render_ability_heatmap(primary_filtered_pivot, model_order)}
      </div>
      <h3>过滤后 Primary 能力通过率热力图</h3>
      {render_ability_heatmap(primary_filtered_pass_pivot, model_order, as_percent=True)}
      {render_pivot_pass_rate_line_chart(primary_filtered_pass_pivot, model_summary, "过滤后 Primary 能力通过率折线图（>=4 分）", ability_row_label)}
      {render_distribution_stats_table(primary_filtered_stats, [("Ability", "ability")], "过滤后 Primary 能力统计学诊断")}
      {render_distribution_stats_table(primary_filtered_pass_stats, [("Ability", "ability")], "过滤后 Primary 能力通过率统计学诊断")}
      <h3>每个 Primary 能力的模型排名</h3>
      <div id="primaryRankings">{render_ability_rankings(primary_rankings_map)}</div>
    </section>

    <section id="secondary-section">
      <h2>Secondary Category：整体均分热力图与能力排名</h2>
      <p class="section-note">
        每个单元格是对应模型在该 Secondary 能力上的“整体均分”；排名同样按该能力分数从高到低。
      </p>
      {render_ability_boxplot(secondary_rows, model_summary, "Secondary 能力分布箱线图")}
      <div class="chart-grid">
        {render_ability_stat_bars(secondary_pivot, "mean_score", "Secondary 共同短板：全模型均分最低的能力", ascending=True, lower=1.0, upper=5.0)}
        {render_ability_stat_bars(secondary_pivot, "spread", "Secondary 分化最大：最高分与最低分差距", ascending=False, lower=0.0, upper=4.0)}
      </div>
      <div class="toolbar">
        <span class="muted">Secondary 能力粒度更细，建议用右侧搜索框定位能力。</span>
        <input id="secondaryFilter" type="search" placeholder="筛选 Secondary 能力名或模型名" />
      </div>
      <div id="secondaryHeatmap">
        {render_ability_heatmap(secondary_pivot, model_order)}
      </div>
      <h3>每个 Secondary 能力的模型排名</h3>
      <div id="secondaryRankings">{render_ability_rankings(secondary_rankings_map)}</div>
    </section>

    {render_scene_environment_section(scene_stats)}
    {render_scene_score_section(phase_analysis, model_order)}
    {render_phase4_failure_section(phase_analysis, model_order)}

    <section id="series-section">
      <h2>系列内比较</h2>
      <p class="section-note">
        这一部分不单独发明新指标，而是把上面的覆盖、任务、整体指标、Primary/Secondary 热力图按模型系列过滤后重排。
      </p>
      <div class="chart-grid">
        {render_series_task_chart(task_matrix, tasks)}
        {render_series_metric_profile(metric_score_pivot, model_summary)}
      </div>
      {render_series_sections(model_summary, task_matrix, tasks, task_rankings_map, metric_score_pivot, primary_pivot, secondary_pivot)}
    </section>

    <section id="takeaways-section">
      <h2>关键看点摘录</h2>
      {render_takeaways(key_takeaways)}
    </section>
  </main>
  <script>
    function parseCellValue(cell) {{
      if (!cell) return {{ value: null, text: '' }};
      const raw = (cell.dataset.value || cell.textContent || '').trim();
      const compact = raw.replace(/\\s+/g, '');
      const ratio = compact.match(/^(-?\\d+(?:\\.\\d+)?)\\/(-?\\d+(?:\\.\\d+)?)/);
      if (ratio && Number(ratio[2]) !== 0) {{
        return {{ value: Number(ratio[1]) / Number(ratio[2]), text: raw }};
      }}
      const shouldNumeric = cell.matches('.heat') || !!cell.querySelector('.bar-cell') || /^-?\\d/.test(compact) || raw.includes('%');
      const number = raw.replace(/,/g, '').match(/-?\\d+(?:\\.\\d+)?/);
      if (shouldNumeric && number) {{
        let value = Number(number[0]);
        if (raw.includes('%')) value = value / 100;
        return {{ value, text: raw }};
      }}
      return {{ value: null, text: raw.toLowerCase() }};
    }}

    function compareCells(a, b, dir) {{
      const av = a.parsed.value;
      const bv = b.parsed.value;
      if (av !== null && bv !== null) {{
        return dir === 'asc' ? av - bv : bv - av;
      }}
      if (av !== null) return -1;
      if (bv !== null) return 1;
      return dir === 'asc'
        ? a.parsed.text.localeCompare(b.parsed.text)
        : b.parsed.text.localeCompare(a.parsed.text);
    }}

    function enhanceRowSorting(table) {{
      if (table.dataset.rowSortingReady === 'true') return;
      table.dataset.rowSortingReady = 'true';
      const headRow = table.tHead && table.tHead.rows[0];
      const body = table.tBodies[0];
      if (!headRow || !body) return;
      Array.from(body.rows).forEach((row, index) => row.dataset.originalRow = String(index));
      Array.from(headRow.cells).forEach((th, index) => {{
        const button = document.createElement('button');
        button.type = 'button';
        button.className = 'sort-btn';
        button.textContent = '↕';
        button.title = '按此列排序';
        button.addEventListener('click', event => {{
          event.stopPropagation();
          const previous = table.dataset.sortCol === String(index) ? table.dataset.sortDir : '';
          const dir = previous === 'desc' ? 'asc' : 'desc';
          table.dataset.sortCol = String(index);
          table.dataset.sortDir = dir;
          Array.from(headRow.cells).forEach(cell => {{
            cell.classList.remove('sort-active');
            const btn = cell.querySelector('.sort-btn');
            if (btn) btn.textContent = '↕';
          }});
          th.classList.add('sort-active');
          button.textContent = dir === 'desc' ? '↓' : '↑';
          const rows = Array.from(body.rows).map(row => ({{
            row,
            parsed: parseCellValue(row.cells[index]),
          }}));
          rows.sort((a, b) => compareCells(a, b, dir));
          rows.forEach(item => body.appendChild(item.row));
        }});
        th.appendChild(button);
      }});
      if (table.dataset.rowSortable === 'true') {{
        const host = table.closest('.table-scroll') || table;
        const controls = document.createElement('div');
        controls.className = 'table-controls';
        const reset = document.createElement('button');
        reset.type = 'button';
        reset.textContent = '重置行顺序';
        reset.addEventListener('click', () => {{
          Array.from(body.rows)
            .sort((a, b) => Number(a.dataset.originalRow) - Number(b.dataset.originalRow))
            .forEach(row => body.appendChild(row));
          Array.from(headRow.cells).forEach(cell => {{
            cell.classList.remove('sort-active');
            const btn = cell.querySelector('.sort-btn');
            if (btn) btn.textContent = '↕';
          }});
          delete table.dataset.sortCol;
          delete table.dataset.sortDir;
        }});
        controls.appendChild(reset);
        host.parentNode.insertBefore(controls, host);
      }}
    }}

    function applyColumnOrder(table, frozen, order) {{
      Array.from(table.rows).forEach(row => {{
        const movable = Array.from(row.children).slice(frozen);
        order.forEach(index => {{
          if (movable[index]) row.appendChild(movable[index]);
        }});
      }});
    }}

    function resetColumnOrder(table, frozen) {{
      Array.from(table.rows).forEach(row => {{
        const movable = Array.from(row.children).slice(frozen);
        movable
          .sort((a, b) => Number(a.dataset.originalCol) - Number(b.dataset.originalCol))
          .forEach(cell => row.appendChild(cell));
      }});
      table.querySelectorAll('.col-sort-btn').forEach(btn => btn.textContent = '列↕');
    }}

    function enhanceColumnSorting(table) {{
      if (table.dataset.colSortingReady === 'true') return;
      table.dataset.colSortingReady = 'true';
      const frozen = Number(table.dataset.frozenCols || 1);
      Array.from(table.rows).forEach(row => {{
        Array.from(row.children).forEach((cell, index) => {{
          if (index >= frozen) cell.dataset.originalCol = String(index - frozen);
        }});
      }});
      const host = table.closest('.table-scroll') || table;
      const controls = document.createElement('div');
      controls.className = 'table-controls';
      const reset = document.createElement('button');
      reset.type = 'button';
      reset.textContent = '重置模型列顺序';
      reset.addEventListener('click', () => resetColumnOrder(table, frozen));
      controls.appendChild(reset);
      host.parentNode.insertBefore(controls, host);

      Array.from(table.tBodies[0]?.rows || []).forEach(row => {{
        const labelCell = row.cells[0];
        if (!labelCell) return;
        const button = document.createElement('button');
        button.type = 'button';
        button.className = 'col-sort-btn';
        button.textContent = '列↕';
        button.title = '按本行分数重排模型列';
        button.addEventListener('click', event => {{
          event.stopPropagation();
          const dir = row.dataset.colSortDir === 'desc' ? 'asc' : 'desc';
          row.dataset.colSortDir = dir;
          table.querySelectorAll('.col-sort-btn').forEach(btn => btn.textContent = '列↕');
          button.textContent = dir === 'desc' ? '列↓' : '列↑';
          const movable = Array.from(row.children).slice(frozen);
          const order = movable
            .map((cell, index) => ({{ index, parsed: parseCellValue(cell) }}))
            .sort((a, b) => compareCells(a, b, dir))
            .map(item => item.index);
          applyColumnOrder(table, frozen, order);
        }});
        labelCell.appendChild(button);
      }});
    }}

    function enhanceTables() {{
      document.querySelectorAll('table[data-row-sortable="true"]').forEach(enhanceRowSorting);
      document.querySelectorAll('table:not([data-col-sortable="true"])').forEach(table => {{
        const headers = Array.from(table.querySelectorAll('thead th')).map(th => th.textContent.trim());
        if (headers.includes('Model')) enhanceRowSorting(table);
      }});
      document.querySelectorAll('table[data-col-sortable="true"]').forEach(enhanceColumnSorting);
    }}

    function enhanceCharts() {{
      const tooltip = document.createElement('div');
      tooltip.className = 'chart-tooltip';
      document.body.appendChild(tooltip);

      document.querySelectorAll('svg.chart').forEach(svg => {{
        if (svg.dataset.chartReady === 'true') return;
        svg.dataset.chartReady = 'true';
        let scale = 1;
        const viewport = document.createElement('div');
        viewport.className = 'chart-viewport';
        svg.parentNode.insertBefore(viewport, svg);
        viewport.appendChild(svg);
        const controls = document.createElement('div');
        controls.className = 'chart-controls';
        const zoomOut = document.createElement('button');
        const zoomIn = document.createElement('button');
        const reset = document.createElement('button');
        zoomOut.type = zoomIn.type = reset.type = 'button';
        zoomOut.textContent = '缩小';
        zoomIn.textContent = '放大';
        reset.textContent = '重置缩放';
        function applyZoom() {{
          svg.style.width = (scale * 100).toFixed(0) + '%';
        }}
        zoomOut.addEventListener('click', () => {{
          scale = Math.max(0.6, scale - 0.2);
          applyZoom();
        }});
        zoomIn.addEventListener('click', () => {{
          scale = Math.min(3, scale + 0.2);
          applyZoom();
        }});
        reset.addEventListener('click', () => {{
          scale = 1;
          applyZoom();
        }});
        controls.append(zoomOut, zoomIn, reset);
        viewport.parentNode.insertBefore(controls, viewport);
      }});

      document.querySelectorAll('.chart-data').forEach(element => {{
        element.addEventListener('mouseenter', event => {{
          const target = event.currentTarget;
          const svg = target.closest('svg.chart');
          const key = target.dataset.key || '';
          const peers = Array.from(svg.querySelectorAll('.chart-data'));
          svg.classList.add('has-hover');
          peers.forEach(peer => {{
            const same = key ? peer.dataset.key === key : peer === target;
            peer.classList.toggle('is-highlight', same);
            peer.classList.toggle('dimmed', !same);
          }});
          const text = target.dataset.tooltip || key;
          if (text) {{
            tooltip.textContent = text;
            tooltip.style.display = 'block';
          }}
        }});
        element.addEventListener('mousemove', event => {{
          tooltip.style.left = (event.clientX + 14) + 'px';
          tooltip.style.top = (event.clientY + 14) + 'px';
        }});
        element.addEventListener('mouseleave', event => {{
          const svg = event.currentTarget.closest('svg.chart');
          svg.classList.remove('has-hover');
          svg.querySelectorAll('.chart-data').forEach(peer => {{
            peer.classList.remove('is-highlight', 'dimmed');
          }});
          tooltip.style.display = 'none';
        }});
      }});
    }}

    function bindFilter(inputId, containerId) {{
      const filter = document.getElementById(inputId);
      const container = document.getElementById(containerId);
      if (!filter || !container) return;
      filter.addEventListener('input', () => {{
        const needle = filter.value.trim().toLowerCase();
        container.querySelectorAll('tbody tr, details.rank-detail').forEach(row => {{
          row.style.display = row.textContent.toLowerCase().includes(needle) ? '' : 'none';
        }});
      }});
    }}
    enhanceTables();
    enhanceCharts();
    bindFilter('primaryFilter', 'primary-section');
    bindFilter('secondaryFilter', 'secondary-section');
  </script>
</body>
</html>
"""
    out_path.write_text(html, encoding="utf-8")


def markdown_table(rows: list[dict[str, Any]], columns: list[tuple[str, str]], limit: int | None = None) -> str:
    selected = rows[:limit] if limit else rows
    header = "| " + " | ".join(title for title, _ in columns) + " |"
    sep = "| " + " | ".join("---" for _ in columns) + " |"
    body = []
    for row in selected:
        cells = []
        for _, key in columns:
            value = row.get(key)
            if isinstance(value, float):
                cells.append(f"{value:.3f}")
            elif value is None:
                cells.append("")
            else:
                cells.append(str(value))
        body.append("| " + " | ".join(cells) + " |")
    return "\n".join([header, sep, *body])


def render_markdown_report(
    out_path: Path,
    root: Path,
    model_summary: list[dict[str, Any]],
    series_summary: list[dict[str, Any]],
    ability_leaders: list[dict[str, Any]],
    primary_pivot: list[dict[str, Any]],
) -> None:
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    common_weak = sorted(
        primary_pivot,
        key=lambda row: (row.get("mean_score") is None, row.get("mean_score") or 99),
    )[:12]

    content = f"""# Phase5 模型横向比较汇总

- 数据根目录：`{root}`
- 纳入模型数：{len(model_summary)}
- 生成时间：{generated_at}
- 口径：默认只读取每个一级模型目录下的 `phase5/benchmark_report.md`，不读取嵌套的 video_frames 报告。
- 不再给综合总榜：报告没有原生 Overall 字段，且 Gemini 四任务与其他模型两任务不可直接用单一宏平均排序。
- 覆盖与加权口径均来自 `## 七、任务与模态分析 / ### 任务维度` 表格。
- Turn/Session 样本数为该表各任务样本数求和；Turn/Session 指标加权均值为对应均分按对应样本数加权。
- 文本/TTS/全部任务加权均分会把匹配任务中的 Turn 与 Session 分数按各自样本数合并加权。

## 覆盖与可比口径概览

{markdown_table(model_summary, [
    ("Model", "model"),
    ("Series", "series"),
    ("Task Coverage", "task_coverage"),
    ("Turn N", "turn_sample_count_est"),
    ("Session N", "session_sample_count_est"),
    ("Text Task Weighted", "text_task_weighted_avg"),
    ("TTS Task Weighted", "tts_task_weighted_avg"),
    ("All Task Weighted", "all_task_weighted_avg"),
    ("Turn Weighted", "turn_metric_avg"),
    ("Session Weighted", "session_metric_avg"),
])}

## Primary Category：模型差距最大的能力

{markdown_table(ability_leaders, [
    ("Ability", "ability"),
    ("最高-最低差值", "spread"),
    ("Best Model", "best_model"),
    ("Best", "best_score"),
    ("Worst Model", "worst_model"),
    ("Worst", "worst_score"),
], limit=15)}

## Primary Category：全模型共同短板

{markdown_table(common_weak, [
    ("Ability", "ability"),
    ("全模型均分", "mean_score"),
    ("标准差", "std_score"),
    ("Best Model", "best_model"),
    ("Best", "best_score"),
    ("Worst Model", "worst_model"),
    ("Worst", "worst_score"),
], limit=12)}

## 输出文件说明

- `model_comparison_dashboard.html`：可直接打开的横向对比看板。
- `model_summary.csv`：每个模型的总体派生指标与排名。
- `overall_metrics_long.csv` / `overall_metrics_pivot.csv`：第“四、整体分数概览”的长表与透视表。
- `primary_category_long.csv` / `primary_category_pivot.csv`：第“八、能力维度分析 / Primary Category”的长表与透视表。
- `secondary_category_long.csv` / `secondary_category_pivot.csv`：Secondary Category 的长表与透视表。
- `task_top_tables.csv`：第“三、关键看点速览”的最强/最弱任务表。
- `key_takeaways.csv`：第“三、关键看点速览”的文字要点摘录。
- `all_extracted_tables.json`：目标章节所有解析出的原始表格，便于后续自定义处理。
"""
    out_path.write_text(content, encoding="utf-8")


def write_outputs(
    root: Path,
    out_dir: Path,
    reports: list[ModelInfo],
    data: dict[str, Any],
    gemini_text_only: bool = False,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    model_summary = compute_model_summary(
        reports,
        data["metric_rows"],
        data["ability_primary_rows"],
        data["task_rows"],
        data["task_dimension_rows"],
    )
    series_summary = compute_series_summary(model_summary)
    model_order = [row["model"] for row in model_summary]
    task_matrix, tasks, task_rankings_map = task_pivot_rows(data["task_rows"], model_summary)
    metric_rankings_map = metric_rankings(data["metric_rows"])
    primary_pivot, primary_leaders = compute_ability_comparison(
        data["ability_primary_rows"], model_order
    )
    primary_filtered_rows, primary_filtered_abilities = filter_primary_by_non_gemini_sample_count(
        data["ability_primary_rows"],
        model_summary,
        min_sample_count=30,
    )
    primary_filtered_pivot, primary_filtered_leaders = compute_ability_comparison(
        primary_filtered_rows,
        model_order,
    )
    primary_ability_pass_rows_placeholder: list[dict[str, Any]] = []
    primary_filtered_stats = distribution_stats_for_pivot(
        primary_filtered_pivot,
        model_order,
        ["ability"],
        "整体均分",
        lower_is_better=False,
    )
    primary_rankings_map = ability_rankings(data["ability_primary_rows"])
    secondary_pivot, secondary_leaders = compute_ability_comparison(
        data["ability_secondary_rows"], model_order
    )
    secondary_rankings_map = ability_rankings(data["ability_secondary_rows"])
    metrics_pivot = metric_pivot_rows(data["metric_rows"], model_order)
    metrics_low_rate_pivot = metric_pivot_rows(data["metric_rows"], model_order, value_key="low_rate")
    metrics_pass_rate_pivot = complement_rate_pivot(metrics_low_rate_pivot, model_order, ["level", "metric"])
    metric_distribution_stats = compute_metric_distribution_stats(
        metrics_pivot,
        metrics_low_rate_pivot,
        model_order,
    )
    model_metric_self_stats = compute_model_metric_self_stats(data["metric_rows"])
    scene_stats = load_scene_environment_analysis(root.parent)
    phase_analysis = collect_multistage_phase_analysis(
        root,
        reports,
        model_summary,
        scene_stats,
        gemini_text_only=gemini_text_only,
    )
    primary_ability_pass_rows = phase_analysis.get("primary_ability_pass_rows", primary_ability_pass_rows_placeholder)
    primary_order = [str(row.get("ability")) for row in primary_pivot]
    primary_pass_pivot = pass_rate_pivot_from_rows(
        primary_ability_pass_rows,
        model_order,
        "ability",
        preferred_order=primary_order,
    )
    primary_filtered_pass_pivot = [
        row for row in primary_pass_pivot if row.get("ability") in set(primary_filtered_abilities)
    ]
    primary_filtered_pass_stats = distribution_stats_for_pivot(
        primary_filtered_pass_pivot,
        model_order,
        ["ability"],
        "通过率",
        lower_is_better=False,
    )

    rows_to_csv(out_dir / "model_summary.csv", model_summary)
    rows_to_csv(out_dir / "series_summary.csv", series_summary)
    rows_to_csv(out_dir / "key_takeaways.csv", data["key_takeaways"])
    rows_to_csv(out_dir / "task_top_tables.csv", data["task_rows"])
    rows_to_csv(out_dir / "overall_metrics_long.csv", data["metric_rows"])
    rows_to_csv(out_dir / "overall_metrics_pivot.csv", metrics_pivot)
    rows_to_csv(out_dir / "overall_metrics_pass_rate_pivot.csv", metrics_pass_rate_pivot)
    rows_to_csv(out_dir / "overall_metric_distribution_stats.csv", metric_distribution_stats)
    rows_to_csv(out_dir / "model_metric_self_stats.csv", model_metric_self_stats)
    rows_to_csv(out_dir / "primary_category_long.csv", data["ability_primary_rows"])
    rows_to_csv(out_dir / "primary_category_pivot.csv", primary_pivot)
    rows_to_csv(out_dir / "primary_category_pass_rate_long.csv", primary_ability_pass_rows)
    rows_to_csv(out_dir / "primary_category_pass_rate_pivot.csv", primary_pass_pivot)
    rows_to_csv(out_dir / "primary_category_leaders.csv", primary_leaders)
    rows_to_csv(out_dir / "primary_category_filtered_sample_ge30_pivot.csv", primary_filtered_pivot)
    rows_to_csv(out_dir / "primary_category_filtered_sample_ge30_stats.csv", primary_filtered_stats)
    rows_to_csv(out_dir / "primary_category_filtered_sample_ge30_pass_rate_pivot.csv", primary_filtered_pass_pivot)
    rows_to_csv(out_dir / "primary_category_filtered_sample_ge30_pass_rate_stats.csv", primary_filtered_pass_stats)
    rows_to_csv(out_dir / "primary_category_filtered_sample_ge30_leaders.csv", primary_filtered_leaders)
    rows_to_csv(out_dir / "secondary_category_long.csv", data["ability_secondary_rows"])
    rows_to_csv(out_dir / "secondary_category_pivot.csv", secondary_pivot)
    rows_to_csv(out_dir / "secondary_category_leaders.csv", secondary_leaders)
    rows_to_csv(out_dir / "scene_environment_dataset_summary.csv", scene_stats["dataset_rows"])
    rows_to_csv(out_dir / "scene_environment_major_distribution.csv", scene_stats["major_rows"])
    rows_to_csv(out_dir / "scene_environment_detail_distribution.csv", scene_stats["detail_rows"])
    rows_to_csv(out_dir / "scene_primary_category_distribution.csv", scene_stats["primary_rows"])
    rows_to_csv(out_dir / "scene_environment_primary_matrix_long.csv", scene_stats["matrix_rows"])
    rows_to_csv(out_dir / "phase12_generation_quality.csv", phase_analysis["generation_rows"])
    rows_to_csv(out_dir / "metric_threshold_low_rate_long.csv", phase_analysis["metric_threshold_rows"])
    rows_to_csv(out_dir / "metric_threshold_overall.csv", phase_analysis["metric_threshold_overall_rows"])
    rows_to_csv(out_dir / "level_pass_rate_long.csv", phase_analysis["level_pass_rate_rows"])
    rows_to_csv(out_dir / "level_pass_rate_pivot.csv", phase_analysis["level_pass_rate_pivot"])
    rows_to_csv(out_dir / "task_turn_low_rate_long.csv", phase_analysis["task_turn_low_rows"])
    rows_to_csv(out_dir / "task_turn_low_rate_overall.csv", phase_analysis["task_turn_threshold_overall_rows"])
    rows_to_csv(out_dir / "scene_major_model_scores.csv", phase_analysis["scene_major_model_rows"])
    rows_to_csv(out_dir / "scene_major_score_pivot.csv", phase_analysis["scene_major_score_pivot"])
    rows_to_csv(out_dir / "scene_major_low_rate_overall.csv", phase_analysis["scene_major_overall_rows"])
    rows_to_csv(out_dir / "scene_major_score_stats.csv", phase_analysis["scene_major_score_stats"])
    rows_to_csv(out_dir / "scene_major_lt4_stats.csv", phase_analysis["scene_major_lt4_stats"])
    rows_to_csv(out_dir / "scene_major_pass_rate_pivot.csv", phase_analysis["scene_major_pass_rate_pivot"])
    rows_to_csv(out_dir / "scene_major_pass_stats.csv", phase_analysis["scene_major_pass_stats"])
    rows_to_csv(out_dir / "scene_detail_model_scores.csv", phase_analysis["scene_detail_model_rows"])
    rows_to_csv(out_dir / "scene_detail_score_pivot.csv", phase_analysis["scene_detail_score_pivot"])
    rows_to_csv(out_dir / "scene_detail_low_rate_overall.csv", phase_analysis["scene_detail_overall_rows"])
    rows_to_csv(out_dir / "scene_detail_score_stats.csv", phase_analysis["scene_detail_score_stats"])
    rows_to_csv(out_dir / "scene_detail_lt4_stats.csv", phase_analysis["scene_detail_lt4_stats"])
    rows_to_csv(out_dir / "scene_detail_pass_rate_pivot.csv", phase_analysis["scene_detail_pass_rate_pivot"])
    rows_to_csv(out_dir / "scene_detail_pass_stats.csv", phase_analysis["scene_detail_pass_stats"])
    rows_to_csv(out_dir / "scene_major_primary_score_matrix.csv", phase_analysis["scene_primary_score_rows"])
    rows_to_csv(out_dir / "scene_major_primary_lt4_matrix.csv", phase_analysis["scene_primary_low_rate_rows"])
    rows_to_csv(out_dir / "scene_major_primary_pass_rate_matrix.csv", phase_analysis["scene_primary_pass_rate_rows"])
    rows_to_csv(out_dir / "phase4_model_low_pool_summary.csv", phase_analysis["phase4_model_rows"])
    rows_to_csv(out_dir / "phase4_turn_primary_error_pivot.csv", phase_analysis["phase4_turn_primary_pivot"])
    rows_to_csv(out_dir / "phase4_session_primary_error_pivot.csv", phase_analysis["phase4_session_primary_pivot"])
    rows_to_csv(out_dir / "phase4_turn_primary_error_count_pivot.csv", phase_analysis["phase4_turn_primary_count_pivot"])
    rows_to_csv(out_dir / "phase4_session_primary_error_count_pivot.csv", phase_analysis["phase4_session_primary_count_pivot"])
    rows_to_csv(out_dir / "phase4_turn_primary_error_overall.csv", phase_analysis["phase4_turn_primary_overall_rows"])
    rows_to_csv(out_dir / "phase4_session_primary_error_overall.csv", phase_analysis["phase4_session_primary_overall_rows"])
    rows_to_csv(out_dir / "phase4_turn_secondary_error_overall.csv", phase_analysis["phase4_turn_secondary_overall_rows"])
    rows_to_csv(out_dir / "phase4_session_secondary_error_overall.csv", phase_analysis["phase4_session_secondary_overall_rows"])
    rows_to_csv(out_dir / "phase4_turn_secondary_error_pivot.csv", phase_analysis["phase4_turn_secondary_pivot"])
    rows_to_csv(out_dir / "phase4_session_secondary_error_pivot.csv", phase_analysis["phase4_session_secondary_pivot"])
    rows_to_csv(out_dir / "phase4_turn_secondary_error_count_pivot.csv", phase_analysis["phase4_turn_secondary_count_pivot"])
    rows_to_csv(out_dir / "phase4_session_secondary_error_count_pivot.csv", phase_analysis["phase4_session_secondary_count_pivot"])
    rows_to_csv(out_dir / "phase4_turn_metric_primary_matrix.csv", phase_analysis["phase4_turn_metric_primary_rows"])
    rows_to_csv(out_dir / "phase4_session_metric_primary_matrix.csv", phase_analysis["phase4_session_metric_primary_rows"])
    rows_to_csv(out_dir / "phase4_turn_metric_secondary_matrix.csv", phase_analysis["phase4_turn_metric_secondary_rows"])
    rows_to_csv(out_dir / "phase4_session_metric_secondary_matrix.csv", phase_analysis["phase4_session_metric_secondary_rows"])
    rows_to_csv(out_dir / "phase4_turn_task_primary_matrix.csv", phase_analysis["phase4_turn_task_primary_rows"])
    rows_to_csv(out_dir / "phase4_session_task_primary_matrix.csv", phase_analysis["phase4_session_task_primary_rows"])

    json_payload = {
        "root": str(root),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "reports": [
            {
                "model": info.model_name,
                "model_dir": info.model_dir,
                "series": info.series,
                "report_path": relpath(info.report_path, root),
            }
            for info in reports
        ],
        "all_extracted_tables": data["all_tables"],
    }
    (out_dir / "all_extracted_tables.json").write_text(
        json.dumps(json_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    render_markdown_report(
        out_dir / "model_comparison_report.md",
        root,
        model_summary,
        series_summary,
        primary_leaders,
        primary_pivot,
    )
    render_html(
        out_dir / "model_comparison_dashboard.html",
        root,
        model_summary,
        task_matrix,
        tasks,
        task_rankings_map,
        metrics_pivot,
        metrics_low_rate_pivot,
        metrics_pass_rate_pivot,
        metric_distribution_stats,
        model_metric_self_stats,
        metric_rankings_map,
        data["ability_primary_rows"],
        primary_pivot,
        primary_pass_pivot,
        primary_filtered_pivot,
        primary_filtered_stats,
        primary_filtered_pass_pivot,
        primary_filtered_pass_stats,
        primary_rankings_map,
        data["ability_secondary_rows"],
        secondary_pivot,
        secondary_rankings_map,
        data["key_takeaways"],
        scene_stats,
        phase_analysis,
    )


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    default_out_name = (
        "_phase5_model_comparison_gemini_text_only"
        if args.gemini_text_only
        else "_phase5_model_comparison"
    )
    out_dir = (args.out or (root / default_out_name)).resolve()

    if not root.exists():
        raise SystemExit(f"Root directory does not exist: {root}")

    reports = discover_reports(root, include_nested=args.include_nested)
    if not reports:
        raise SystemExit(f"No phase5 benchmark_report.md files found under: {root}")

    data = collect_data(root, reports)
    if args.gemini_text_only:
        data = recompute_gemini_text_only_data(root, reports, data)
    write_outputs(root, out_dir, reports, data, gemini_text_only=args.gemini_text_only)

    print(f"Processed reports: {len(reports)}")
    print(f"Output directory: {out_dir}")
    print("Key artifacts:")
    print(f"  - {out_dir / 'model_comparison_dashboard.html'}")
    print(f"  - {out_dir / 'model_comparison_report.md'}")
    print(f"  - {out_dir / 'model_summary.csv'}")
    print(f"  - {out_dir / 'primary_category_pivot.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
