#!/usr/bin/env python3
"""Render text-only metric pass-rate radar charts.

Inputs are the Gemini-text-only comparison CSV artifacts. The script selects one
representative model per series for Turn-Level and Session-Level separately,
then writes a combined HTML page plus standalone HTML/SVG files for export.
"""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import Any


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

SERIES_ORDER = ["GPT", "Claude", "Gemini", "Doubao", "Qwen3-VL", "Qwen3.5", "Qwen2.5-VL"]

SERIES_COLORS = {
    "GPT": "#6f9fd8",
    "Claude": "#79b69b",
    "Gemini": "#e3b36b",
    "Doubao": "#e58b8b",
    "Qwen3-VL": "#a890d8",
    "Qwen3.5": "#75b8c6",
    "Qwen2.5-VL": "#9aa8b8",
    "Other": "#94a3b8",
}

METRIC_LABELS = {
    "accuracy": "Accuracy",
    "completeness": "Completeness",
    "relevance": "Relevance",
    "conciseness": "Conciseness",
    "naturalness": "Naturalness",
    "proactiveness_helpfulness": "Proactiveness Helpfulness",
    "intent_understanding_depth": "Intent Understanding Depth",
    "user_state_adaptation": "User State Adaptation",
    "session_consistency": "Session Consistency",
    "intent_fulfillment": "Intent Fulfillment",
    "persona_adaptation": "Persona Adaptation",
    "overall_helpfulness_trustworthiness": "Overall Helpfulness Trustworthiness",
}


@dataclass
class ModelLine:
    series: str
    model: str
    values: list[float]
    mean_value: float
    color: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render text-only pass-rate radar charts.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("result_judge(2.5pro)") / "_phase5_model_comparison_gemini_text_only",
        help="Gemini text-only comparison artifact directory.",
    )
    parser.add_argument("--html-name", default="text_only_metric_pass_rate_radars.html")
    parser.add_argument("--turn-html-name", default="text_only_turn_level_pass_rate_radar.html")
    parser.add_argument("--session-html-name", default="text_only_session_level_pass_rate_radar.html")
    parser.add_argument("--turn-svg-name", default="text_only_turn_level_pass_rate_radar.svg")
    parser.add_argument("--session-svg-name", default="text_only_session_level_pass_rate_radar.svg")
    return parser.parse_args()


def maybe_float(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def model_series_map(summary_rows: list[dict[str, str]]) -> dict[str, str]:
    return {row["model"]: row.get("series") or "Other" for row in summary_rows if row.get("model")}


def metric_lookup(metric_rows: list[dict[str, str]], level: str) -> dict[str, dict[str, float]]:
    lookup: dict[str, dict[str, float]] = {}
    for row in metric_rows:
        if row.get("level") != level:
            continue
        metric = row.get("metric")
        if not metric:
            continue
        lookup[metric] = {
            model: value
            for model, raw in row.items()
            if model not in {"level", "metric"} and (value := maybe_float(raw)) is not None
        }
    return lookup


def select_best_by_series(
    metric_rows: list[dict[str, str]],
    summary_rows: list[dict[str, str]],
    level: str,
    metrics: list[str],
) -> list[ModelLine]:
    by_metric = metric_lookup(metric_rows, level)
    series_by_model = model_series_map(summary_rows)
    candidates: dict[str, list[tuple[str, list[float], float]]] = {}
    for model, series in series_by_model.items():
        values = [by_metric.get(metric, {}).get(model) for metric in metrics]
        if any(value is None for value in values):
            continue
        clean_values = [float(value) for value in values if value is not None]
        mean_value = sum(clean_values) / len(clean_values)
        candidates.setdefault(series, []).append((model, clean_values, mean_value))

    selected: list[ModelLine] = []
    for series in SERIES_ORDER + sorted(series for series in candidates if series not in SERIES_ORDER):
        rows = candidates.get(series) or []
        if not rows:
            continue
        model, values, mean_value = max(rows, key=lambda item: (item[2], item[0]))
        selected.append(
            ModelLine(
                series=series,
                model=model,
                values=values,
                mean_value=mean_value,
                color=SERIES_COLORS.get(series, SERIES_COLORS["Other"]),
            )
        )
    return selected


def polar(cx: float, cy: float, radius: float, angle: float) -> tuple[float, float]:
    radians = math.radians(angle)
    return cx + radius * math.cos(radians), cy + radius * math.sin(radians)


def polygon_points(values: list[float], cx: float, cy: float, radius: float) -> str:
    count = len(values)
    points = []
    for index, value in enumerate(values):
        angle = -90 + index * 360 / count
        x, y = polar(cx, cy, radius * max(0.0, min(1.0, value)), angle)
        points.append(f"{x:.2f},{y:.2f}")
    return " ".join(points)


def radar_grid(metric_count: int, cx: float, cy: float, radius: float) -> str:
    chunks = []
    for step in range(1, 6):
        ratio = step / 5
        points = []
        for index in range(metric_count):
            angle = -90 + index * 360 / metric_count
            x, y = polar(cx, cy, radius * ratio, angle)
            points.append(f"{x:.2f},{y:.2f}")
        stroke = "#dbe5ef" if step < 5 else "#b8c7d8"
        chunks.append(
            f'<polygon points="{" ".join(points)}" fill="none" stroke="{stroke}" stroke-width="1.15" />'
        )
    for index in range(metric_count):
        angle = -90 + index * 360 / metric_count
        x, y = polar(cx, cy, radius, angle)
        chunks.append(f'<line x1="{cx}" y1="{cy}" x2="{x:.2f}" y2="{y:.2f}" stroke="#e2e8f0" stroke-width="1" />')
    return "".join(chunks)


def legend(lines: list[ModelLine], x: int, y: int) -> str:
    chunks = []
    for index, line in enumerate(lines):
        row_y = y + index * 28
        chunks.append(
            f'<rect x="{x}" y="{row_y - 11}" width="14" height="14" rx="4" fill="{line.color}" opacity="0.9" />'
            f'<text x="{x + 22}" y="{row_y}" class="legend-model">{escape(line.series)}: {escape(line.model)}</text>'
            f'<text x="{x + 22}" y="{row_y + 13}" class="legend-score">mean pass rate {line.mean_value:.1%}</text>'
        )
    return "".join(chunks)


def metric_label(metric: str) -> str:
    return METRIC_LABELS.get(metric, " ".join(part.capitalize() for part in metric.split("_")))


def wrap_label(label: str, max_words: int = 2) -> list[str]:
    words = label.split()
    if len(words) <= max_words:
        return [label]
    return [" ".join(words[index : index + max_words]) for index in range(0, len(words), max_words)]


def metric_axis_labels(metrics: list[str], cx: float, cy: float, radius: float) -> str:
    chunks = []
    count = len(metrics)
    label_radius = radius + 72
    for index, metric in enumerate(metrics):
        angle = -90 + index * 360 / count
        x, y = polar(cx, cy, label_radius, angle)
        if -100 <= angle <= -80 or 80 <= angle <= 100 or angle >= 260:
            anchor = "middle"
        elif -80 < angle < 80:
            anchor = "start"
        else:
            anchor = "end"
        lines = wrap_label(metric_label(metric))
        y_start = y - (len(lines) - 1) * 7
        tspans = []
        for line_index, line in enumerate(lines):
            dy = 0 if line_index == 0 else 15
            tspans.append(
                f'<tspan x="{x:.1f}" dy="{dy}">{escape(line)}</tspan>'
            )
        chunks.append(
            f'<text x="{x:.1f}" y="{y_start:.1f}" text-anchor="{anchor}" class="axis-label">'
            f'{"".join(tspans)}</text>'
        )
    return "".join(chunks)


def radar_svg(title: str, subtitle: str, metrics: list[str], lines: list[ModelLine]) -> str:
    width, height = 1320, 880
    cx, cy, radius = 455, 465, 230
    axis_count = len(metrics)
    grid = radar_grid(axis_count, cx, cy, radius)
    axis_labels = metric_axis_labels(metrics, cx, cy, radius)
    filled_polys = []
    strokes = []
    for line in lines:
        points = polygon_points(line.values, cx, cy, radius)
        filled_polys.append(
            f'<polygon points="{points}" fill="{line.color}" fill-opacity="0.105" stroke="none" />'
        )
        strokes.append(
            f'<polygon points="{points}" fill="none" stroke="{line.color}" stroke-width="3.2" '
            f'stroke-linejoin="round" opacity="0.92" />'
        )
        for index, value in enumerate(line.values):
            angle = -90 + index * 360 / axis_count
            x, y = polar(cx, cy, radius * value, angle)
            strokes.append(
                f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4.1" fill="{line.color}" '
                f'stroke="#ffffff" stroke-width="1.6"><title>{escape(line.model)}: {value:.1%}</title></circle>'
            )
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-label="{escape(title)}">
  <style>
    .title {{ font-family: Segoe UI, Arial, sans-serif; font-size: 28px; font-weight: 760; fill: #172033; }}
    .subtitle {{ font-family: Segoe UI, Arial, sans-serif; font-size: 14px; fill: #66758a; }}
    .legend-model {{ font-family: Segoe UI, Arial, sans-serif; font-size: 13px; font-weight: 680; fill: #1f2a3a; }}
    .legend-score {{ font-family: Segoe UI, Arial, sans-serif; font-size: 11px; fill: #64748b; }}
    .axis-label {{ font-family: Segoe UI, Arial, sans-serif; font-size: 13px; font-weight: 720; fill: #344256; }}
    .ring-value {{ font-family: Segoe UI, Arial, sans-serif; font-size: 11px; fill: #8a96a8; }}
    .center-label {{ font-family: Segoe UI, Arial, sans-serif; font-size: 13px; font-weight: 700; fill: #64748b; }}
  </style>
  <rect x="0" y="0" width="{width}" height="{height}" rx="24" fill="#fbfdff" />
  <rect x="26" y="26" width="{width - 52}" height="{height - 52}" rx="22" fill="#ffffff" stroke="#dbe5ef" />
  <text x="54" y="72" class="title">{escape(title)}</text>
  <text x="54" y="100" class="subtitle">{escape(subtitle)}</text>
  <g>
    {grid}
    {axis_labels}
    {''.join(filled_polys)}
    {''.join(strokes)}
    <circle cx="{cx}" cy="{cy}" r="4" fill="#94a3b8" />
    <text x="{cx}" y="{cy + radius + 42}" text-anchor="middle" class="center-label">Pass Rate Scale: 0% to 100%</text>
  </g>
  <g>{legend(lines, 900, 190)}</g>
</svg>
"""


def standalone_html(title: str, svg: str) -> str:
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{escape(title)}</title>
  <style>
    @page {{ size: 12.3in 8.55in; margin: 0; }}
    body {{
      margin: 0;
      background: #ffffff;
      display: grid;
      place-items: center;
      min-height: 100vh;
    }}
    svg {{ width: 100vw; height: auto; display: block; }}
    @media print {{ body {{ min-height: 0; }} svg {{ width: 100%; }} }}
  </style>
</head>
<body>{svg}</body>
</html>
"""


def combined_html(turn_svg: str, session_svg: str) -> str:
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Text-only Metric Pass Rate Radars</title>
  <style>
    body {{
      margin: 0;
      background: #eef3f8;
      color: #172033;
      font-family: "Segoe UI", Arial, sans-serif;
    }}
    main {{
      max-width: 1240px;
      margin: 0 auto;
      padding: 24px 20px 36px;
    }}
    h1 {{ margin: 0 0 8px; font-size: 28px; }}
    p {{ margin: 0 0 20px; color: #64748b; }}
    .chart-card {{
      margin-bottom: 22px;
      padding: 12px;
      border: 1px solid #dbe5ef;
      border-radius: 18px;
      background: #ffffff;
    }}
    svg {{ display: block; width: 100%; height: auto; }}
  </style>
</head>
<body>
<main>
  <h1>Text-only Metric Pass Rate Radars</h1>
  <p>One representative model is selected per series. Turn and Session representatives are selected separately by mean pass rate within each level.</p>
  <section class="chart-card">{turn_svg}</section>
  <section class="chart-card">{session_svg}</section>
</main>
</body>
</html>
"""


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    metric_rows = read_csv(input_dir / "overall_metrics_pass_rate_pivot.csv")
    summary_rows = read_csv(input_dir / "model_summary.csv")

    turn_lines = select_best_by_series(metric_rows, summary_rows, "Turn-Level", TURN_METRICS)
    session_lines = select_best_by_series(metric_rows, summary_rows, "Session-Level", SESSION_METRICS)

    turn_svg = radar_svg(
        "Turn-Level Pass Rate Radar",
        "Text-only report, one top representative per model series",
        TURN_METRICS,
        turn_lines,
    )
    session_svg = radar_svg(
        "Session-Level Pass Rate Radar",
        "Text-only report, one top representative per model series",
        SESSION_METRICS,
        session_lines,
    )

    (input_dir / args.turn_svg_name).write_text(turn_svg, encoding="utf-8")
    (input_dir / args.session_svg_name).write_text(session_svg, encoding="utf-8")
    (input_dir / args.turn_html_name).write_text(
        standalone_html("Turn-Level Pass Rate Radar", turn_svg),
        encoding="utf-8",
    )
    (input_dir / args.session_html_name).write_text(
        standalone_html("Session-Level Pass Rate Radar", session_svg),
        encoding="utf-8",
    )
    (input_dir / args.html_name).write_text(combined_html(turn_svg, session_svg), encoding="utf-8")

    print(f"HTML: {input_dir / args.html_name}")
    print(f"Turn export HTML: {input_dir / args.turn_html_name}")
    print(f"Session export HTML: {input_dir / args.session_html_name}")
    print(f"Turn SVG: {input_dir / args.turn_svg_name}")
    print(f"Session SVG: {input_dir / args.session_svg_name}")
    print("Turn representatives:")
    for line in turn_lines:
        print(f"  {line.series}: {line.model} ({line.mean_value:.4f})")
    print("Session representatives:")
    for line in session_lines:
        print(f"  {line.series}: {line.model} ({line.mean_value:.4f})")


if __name__ == "__main__":
    main()
