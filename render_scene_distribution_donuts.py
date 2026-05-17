#!/usr/bin/env python3
"""Render concentric donut charts for RUBRIC-MME scene distribution.

The script reads the two source dataset JSON files and writes a standalone HTML
page plus, when a local Chromium-based browser is available, a PDF export.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from collections import Counter, defaultdict
from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import Any


TASK_SPECS = [
    ("multi_image", "Multi-image Task", "omnibench_dataset/image_final_with_mimt_category.json"),
    ("streaming_video", "Streaming Video Task", "omnibench_dataset/video_final_with_vqa_category.json"),
]

MAJOR_COLORS = {
    "personal_living_space": "#2563eb",
    "health_and_physical_activity": "#059669",
    "outdoor_public_space": "#d97706",
    "work_and_office_environment": "#7c3aed",
    "retail_and_consumption_space": "#dc2626",
    "family_care_and_home_assistance": "#0891b2",
    "industrial_and_production_site": "#64748b",
}

TASK_COLORS = {
    "multi_image": "#2563eb",
    "streaming_video": "#f59e0b",
}


@dataclass
class Segment:
    key: str
    label: str
    count: int
    start: float
    end: float
    color: str
    parent: str | None = None

    @property
    def percent(self) -> float:
        return 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render scene distribution concentric donut charts.")
    parser.add_argument("--root", type=Path, default=Path("."), help="Project root.")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("result_judge(2.5pro)") / "_phase5_model_comparison",
        help="Directory for generated HTML/PDF.",
    )
    parser.add_argument("--html-name", default="scene_distribution_concentric_donuts.html")
    parser.add_argument("--pdf-name", default="scene_distribution_concentric_donuts.pdf")
    parser.add_argument("--no-pdf", action="store_true", help="Only write HTML.")
    return parser.parse_args()


def title_label(value: str) -> str:
    return " ".join(part.capitalize() for part in str(value).replace("_", " ").split())


def split_environment(label: str | None) -> tuple[str, str]:
    text = str(label or "unknown").strip() or "unknown"
    if "-" not in text:
        return text, "unknown"
    major, detail = text.split("-", 1)
    return major or "unknown", detail or "unknown"


def load_counts(root: Path) -> dict[str, Any]:
    task_counts: Counter[str] = Counter()
    major_counts: Counter[str] = Counter()
    detail_counts: Counter[tuple[str, str]] = Counter()
    detail_by_major: dict[str, Counter[str]] = defaultdict(Counter)

    for task_key, _, rel_path in TASK_SPECS:
        path = root / rel_path
        data = json.loads(path.read_text(encoding="utf-8"))
        task_counts[task_key] = len(data)
        for item in data:
            major, detail = split_environment(item.get("environment"))
            major_counts[major] += 1
            detail_counts[(major, detail)] += 1
            detail_by_major[major][detail] += 1

    total = sum(task_counts.values())
    return {
        "total": total,
        "task_counts": task_counts,
        "major_counts": major_counts,
        "detail_counts": detail_counts,
        "detail_by_major": detail_by_major,
    }


def polar(cx: float, cy: float, radius: float, angle: float) -> tuple[float, float]:
    radians = math.radians(angle)
    return cx + radius * math.cos(radians), cy + radius * math.sin(radians)


def donut_path(
    cx: float,
    cy: float,
    inner_r: float,
    outer_r: float,
    start: float,
    end: float,
    gap: float = 0.45,
) -> str:
    span = max(0.001, end - start)
    actual_gap = min(gap, span * 0.18)
    start += actual_gap
    end -= actual_gap
    if end <= start:
        end = start + 0.001

    large = 1 if end - start > 180 else 0
    ox1, oy1 = polar(cx, cy, outer_r, start)
    ox2, oy2 = polar(cx, cy, outer_r, end)
    ix2, iy2 = polar(cx, cy, inner_r, end)
    ix1, iy1 = polar(cx, cy, inner_r, start)
    return (
        f"M {ox1:.2f} {oy1:.2f} "
        f"A {outer_r:.2f} {outer_r:.2f} 0 {large} 1 {ox2:.2f} {oy2:.2f} "
        f"L {ix2:.2f} {iy2:.2f} "
        f"A {inner_r:.2f} {inner_r:.2f} 0 {large} 0 {ix1:.2f} {iy1:.2f} Z"
    )


def hex_to_rgb(value: str) -> tuple[int, int, int]:
    text = value.strip().lstrip("#")
    return int(text[0:2], 16), int(text[2:4], 16), int(text[4:6], 16)


def rgb_to_hex(rgb: tuple[int, int, int]) -> str:
    return "#" + "".join(f"{max(0, min(255, channel)):02x}" for channel in rgb)


def mix(color: str, target: str, amount: float) -> str:
    r1, g1, b1 = hex_to_rgb(color)
    r2, g2, b2 = hex_to_rgb(target)
    return rgb_to_hex(
        (
            round(r1 + (r2 - r1) * amount),
            round(g1 + (g2 - g1) * amount),
            round(b1 + (b2 - b1) * amount),
        )
    )


def text_color(background: str) -> str:
    r, g, b = hex_to_rgb(background)
    luminance = (0.2126 * r + 0.7152 * g + 0.0722 * b) / 255
    return "#0f172a" if luminance > 0.62 else "#ffffff"


def ordered_items(counter: Counter[Any], preferred: list[Any] | None = None) -> list[tuple[Any, int]]:
    if preferred:
        preferred_set = set(preferred)
        items = [(key, counter[key]) for key in preferred if key in counter]
        items.extend(sorted((key, value) for key, value in counter.items() if key not in preferred_set))
        return items
    return sorted(counter.items(), key=lambda item: (-item[1], str(item[0])))


def make_segments(
    items: list[tuple[str, int]],
    colors: dict[str, str],
    total: int,
    start_angle: float = -90.0,
    end_angle: float = 270.0,
    parent: str | None = None,
) -> list[Segment]:
    segments: list[Segment] = []
    cursor = start_angle
    span_total = end_angle - start_angle
    for index, (key, count) in enumerate(items):
        span = span_total * count / total if total else 0
        end = end_angle if index == len(items) - 1 else cursor + span
        color = colors.get(key, "#64748b")
        segments.append(
            Segment(
                key=str(key),
                label=title_label(str(key)),
                count=int(count),
                start=cursor,
                end=end,
                color=color,
                parent=parent,
            )
        )
        cursor = end
    return segments


def major_order(major_counts: Counter[str]) -> list[str]:
    return [key for key, _ in ordered_items(major_counts)]


def task_segments(counts: dict[str, Any]) -> list[Segment]:
    task_counts: Counter[str] = counts["task_counts"]
    colors = {key: TASK_COLORS[key] for key in TASK_COLORS}
    labels = {key: label for key, label, _ in TASK_SPECS}
    segments = make_segments(
        [(key, task_counts[key]) for key, _, _ in TASK_SPECS],
        colors,
        counts["total"],
    )
    for segment in segments:
        segment.label = labels.get(segment.key, segment.label)
    return segments


def major_segments(counts: dict[str, Any]) -> list[Segment]:
    major_counts: Counter[str] = counts["major_counts"]
    order = major_order(major_counts)
    return make_segments(
        [(key, major_counts[key]) for key in order],
        MAJOR_COLORS,
        counts["total"],
    )


def detail_segments(counts: dict[str, Any], major_ring: list[Segment]) -> list[Segment]:
    detail_by_major: dict[str, Counter[str]] = counts["detail_by_major"]
    segments: list[Segment] = []
    for major_segment in major_ring:
        details = ordered_items(detail_by_major[major_segment.key])
        base = MAJOR_COLORS.get(major_segment.key, "#64748b")
        shades = [mix(base, "#ffffff", amount) for amount in (0.06, 0.22, 0.38, 0.54)]
        color_map = {str(key): shades[index % len(shades)] for index, (key, _) in enumerate(details)}
        local_total = sum(value for _, value in details)
        local_segments = make_segments(
            [(str(key), int(value)) for key, value in details],
            color_map,
            local_total,
            start_angle=major_segment.start,
            end_angle=major_segment.end,
            parent=major_segment.key,
        )
        segments.extend(local_segments)
    return segments


def segment_path_svg(segment: Segment, cx: float, cy: float, inner_r: float, outer_r: float, total: int) -> str:
    pct = segment.count / total if total else 0
    title = f"{segment.label}: {segment.count} samples, {pct:.1%}"
    return (
        f'<path d="{donut_path(cx, cy, inner_r, outer_r, segment.start, segment.end)}" '
        f'fill="{segment.color}" class="arc">'
        f"<title>{escape(title)}</title></path>"
    )


def inside_label(segment: Segment, cx: float, cy: float, radius: float, total: int, min_pct: float = 0.08) -> str:
    pct = segment.count / total if total else 0
    if pct < min_pct:
        return ""
    mid = (segment.start + segment.end) / 2
    x, y = polar(cx, cy, radius, mid)
    color = text_color(segment.color)
    label = escape(segment.label)
    return (
        f'<text x="{x:.1f}" y="{y - 4:.1f}" text-anchor="middle" class="ring-label" fill="{color}">{label}</text>'
        f'<text x="{x:.1f}" y="{y + 13:.1f}" text-anchor="middle" class="ring-sub" fill="{color}">'
        f'{segment.count} / {pct:.1%}</text>'
    )


def callout_labels(
    segments: list[Segment],
    cx: float,
    cy: float,
    anchor_r: float,
    label_r: float,
    total: int,
    *,
    min_pct: float = 0.0,
    max_labels: int | None = None,
) -> str:
    selected = [segment for segment in segments if (segment.count / total if total else 0) >= min_pct]
    selected.sort(key=lambda segment: segment.count, reverse=True)
    if max_labels is not None:
        selected = selected[:max_labels]
    selected.sort(key=lambda segment: (segment.start + segment.end) / 2)

    points = []
    for segment in selected:
        mid = (segment.start + segment.end) / 2
        ax, ay = polar(cx, cy, anchor_r, mid)
        lx, ly = polar(cx, cy, label_r, mid)
        points.append({"segment": segment, "mid": mid, "ax": ax, "ay": ay, "lx": lx, "ly": ly})

    for side in (-1, 1):
        side_points = [point for point in points if (point["lx"] - cx) * side >= 0]
        side_points.sort(key=lambda point: point["ly"])
        min_gap = 32
        previous = -10_000.0
        for point in side_points:
            point["ly"] = max(point["ly"], previous + min_gap)
            previous = point["ly"]
        overflow = (side_points[-1]["ly"] - (cy + label_r)) if side_points else 0
        if overflow > 0:
            for point in side_points:
                point["ly"] -= overflow

    chunks = []
    for point in points:
        segment = point["segment"]
        pct = segment.count / total if total else 0
        side = 1 if point["lx"] >= cx else -1
        ex = point["lx"] + side * 16
        anchor = "start" if side > 0 else "end"
        text_x = ex + side * 8
        label = escape(segment.label)
        chunks.append(
            f'<polyline points="{point["ax"]:.1f},{point["ay"]:.1f} {point["lx"]:.1f},{point["ly"]:.1f} {ex:.1f},{point["ly"]:.1f}" '
            f'stroke="{segment.color}" stroke-width="1.4" fill="none" opacity="0.78" />'
            f'<circle cx="{point["ax"]:.1f}" cy="{point["ay"]:.1f}" r="3.2" fill="{segment.color}" />'
            f'<text x="{text_x:.1f}" y="{point["ly"] - 3:.1f}" text-anchor="{anchor}" class="callout-label">{label}</text>'
            f'<text x="{text_x:.1f}" y="{point["ly"] + 13:.1f}" text-anchor="{anchor}" class="callout-sub">'
            f'{segment.count} samples, {pct:.1%}</text>'
        )
    return "".join(chunks)


def overview_svg(counts: dict[str, Any]) -> str:
    total = counts["total"]
    cx, cy = 390, 345
    tasks = task_segments(counts)
    majors = major_segments(counts)
    task_paths = "".join(segment_path_svg(segment, cx, cy, 92, 152, total) for segment in tasks)
    major_paths = "".join(segment_path_svg(segment, cx, cy, 176, 266, total) for segment in majors)
    task_labels = "".join(inside_label(segment, cx, cy, 122, total, min_pct=0.02) for segment in tasks)
    major_callouts = callout_labels(majors, cx, cy, 270, 328, total)
    return f"""
<svg class="donut-svg overview-svg" viewBox="0 0 1040 700" role="img" aria-label="Two-ring scene distribution overview">
  <defs>
    <filter id="softShadow" x="-20%" y="-20%" width="140%" height="140%">
      <feDropShadow dx="0" dy="8" stdDeviation="10" flood-color="#0f172a" flood-opacity="0.12" />
    </filter>
  </defs>
  <rect x="0" y="0" width="1040" height="700" rx="22" fill="#ffffff" />
  <g filter="url(#softShadow)">
    {task_paths}
    {major_paths}
  </g>
  <circle cx="{cx}" cy="{cy}" r="68" fill="#f8fafc" stroke="#e2e8f0" stroke-width="1.2" />
  <text x="{cx}" y="{cy - 10}" text-anchor="middle" class="center-title">734</text>
  <text x="{cx}" y="{cy + 14}" text-anchor="middle" class="center-sub">Total Sessions</text>
  {task_labels}
  {major_callouts}
  <text x="48" y="54" class="svg-title">Task and Major Scene Distribution</text>
  <text x="48" y="82" class="svg-note">Inner ring: task type. Outer ring: seven major scene categories.</text>
</svg>
"""


def nested_svg(counts: dict[str, Any]) -> str:
    total = counts["total"]
    cx, cy = 390, 350
    tasks = task_segments(counts)
    majors = major_segments(counts)
    details = detail_segments(counts, majors)
    task_paths = "".join(segment_path_svg(segment, cx, cy, 72, 120, total) for segment in tasks)
    major_paths = "".join(segment_path_svg(segment, cx, cy, 138, 198, total) for segment in majors)
    detail_paths = "".join(segment_path_svg(segment, cx, cy, 214, 288, total) for segment in details)
    task_labels = "".join(inside_label(segment, cx, cy, 96, total, min_pct=0.02) for segment in tasks)
    major_labels = callout_labels(majors, cx, cy, 201, 322, total, min_pct=0.0)
    detail_labels = callout_labels(details, cx, cy, 292, 376, total, min_pct=0.035, max_labels=12)
    return f"""
<svg class="donut-svg nested-svg" viewBox="0 0 1120 760" role="img" aria-label="Three-ring nested scene distribution">
  <rect x="0" y="0" width="1120" height="760" rx="22" fill="#ffffff" />
  <g>
    {task_paths}
    {major_paths}
    {detail_paths}
  </g>
  <circle cx="{cx}" cy="{cy}" r="54" fill="#f8fafc" stroke="#e2e8f0" stroke-width="1.1" />
  <text x="{cx}" y="{cy - 8}" text-anchor="middle" class="center-title">3 Rings</text>
  <text x="{cx}" y="{cy + 15}" text-anchor="middle" class="center-sub">Task | Major | Detail</text>
  {task_labels}
  {major_labels}
  {detail_labels}
  <text x="48" y="54" class="svg-title">Nested Scene Distribution</text>
  <text x="48" y="82" class="svg-note">Outer detail arcs are constrained inside their parent major scene arc.</text>
</svg>
"""


def major_table(counts: dict[str, Any]) -> str:
    total = counts["total"]
    rows = []
    for key, count in ordered_items(counts["major_counts"]):
        color = MAJOR_COLORS.get(key, "#64748b")
        rows.append(
            "<tr>"
            f'<td><span class="swatch" style="background:{color}"></span>{escape(title_label(key))}</td>'
            f"<td>{count}</td>"
            f"<td>{count / total:.1%}</td>"
            "</tr>"
        )
    return (
        '<table class="summary-table">'
        "<thead><tr><th>Major Scene</th><th>Samples</th><th>Share</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )


def detail_table(counts: dict[str, Any]) -> str:
    total = counts["total"]
    detail_counts: Counter[tuple[str, str]] = counts["detail_counts"]
    rows = []
    for (major, detail), count in sorted(detail_counts.items(), key=lambda item: (-item[1], item[0][0], item[0][1])):
        color = MAJOR_COLORS.get(major, "#64748b")
        rows.append(
            "<tr>"
            f'<td><span class="swatch" style="background:{color}"></span>{escape(title_label(major))}</td>'
            f"<td>{escape(title_label(detail))}</td>"
            f"<td>{count}</td>"
            f"<td>{count / total:.1%}</td>"
            "</tr>"
        )
    return (
        '<table class="summary-table detail-table">'
        "<thead><tr><th>Parent Major Scene</th><th>Scene Detail</th><th>Samples</th><th>Share</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )


def task_cards(counts: dict[str, Any]) -> str:
    total = counts["total"]
    cards = []
    for key, label, _ in TASK_SPECS:
        count = counts["task_counts"][key]
        color = TASK_COLORS[key]
        cards.append(
            '<div class="metric-card">'
            f'<span class="metric-pill" style="background:{color}"></span>'
            f"<div><div class=\"metric-label\">{escape(label)}</div>"
            f"<div class=\"metric-value\">{count}</div>"
            f"<div class=\"metric-note\">{count / total:.1%} of all sessions</div></div>"
            "</div>"
        )
    return "".join(cards)


def render_html(counts: dict[str, Any]) -> str:
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>RUBRIC-MME Scene Distribution Donuts</title>
  <style>
    @page {{ size: A4 landscape; margin: 8mm; }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: #eef2f7;
      color: #0f172a;
      font-family: "Segoe UI", Arial, sans-serif;
      line-height: 1.4;
    }}
    main {{
      width: min(1480px, calc(100vw - 32px));
      margin: 0 auto;
      padding: 24px 0 36px;
    }}
    .page {{
      min-height: 780px;
      margin: 0 0 26px;
      padding: 24px;
      background: #f8fafc;
      border: 1px solid #dbe3ee;
      border-radius: 18px;
      page-break-after: always;
    }}
    .page:last-child {{ page-break-after: auto; }}
    .header {{
      display: flex;
      justify-content: space-between;
      align-items: flex-end;
      gap: 24px;
      margin-bottom: 18px;
    }}
    h1, h2, p {{ margin: 0; }}
    h1 {{ font-size: 30px; letter-spacing: 0; }}
    h2 {{ font-size: 21px; margin-bottom: 12px; }}
    .subtle {{ color: #64748b; font-size: 14px; max-width: 720px; }}
    .metric-grid {{
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 12px;
      margin: 14px 0 18px;
    }}
    .metric-card {{
      display: flex;
      gap: 12px;
      align-items: center;
      padding: 13px 14px;
      border: 1px solid #dbe3ee;
      border-radius: 12px;
      background: #ffffff;
    }}
    .metric-pill {{
      width: 12px;
      align-self: stretch;
      border-radius: 999px;
      box-shadow: inset 0 0 0 1px rgba(15, 23, 42, 0.08);
    }}
    .metric-label {{ color: #475569; font-size: 13px; }}
    .metric-value {{ font-size: 26px; font-weight: 750; }}
    .metric-note {{ color: #64748b; font-size: 12px; }}
    .layout {{
      display: grid;
      grid-template-columns: minmax(760px, 1fr) 360px;
      gap: 20px;
      align-items: start;
    }}
    .chart-panel {{
      padding: 10px;
      background: #ffffff;
      border: 1px solid #dbe3ee;
      border-radius: 18px;
      overflow: hidden;
    }}
    .donut-svg {{ display: block; width: 100%; height: auto; }}
    .svg-title {{ fill: #0f172a; font-size: 25px; font-weight: 760; }}
    .svg-note {{ fill: #64748b; font-size: 14px; }}
    .center-title {{ fill: #0f172a; font-size: 24px; font-weight: 780; }}
    .center-sub {{ fill: #64748b; font-size: 12px; font-weight: 650; }}
    .ring-label {{ font-size: 12px; font-weight: 760; pointer-events: none; }}
    .ring-sub {{ font-size: 11px; font-weight: 650; pointer-events: none; opacity: 0.94; }}
    .callout-label {{ fill: #0f172a; font-size: 12px; font-weight: 720; }}
    .callout-sub {{ fill: #64748b; font-size: 11px; font-weight: 620; }}
    .arc {{ stroke: #ffffff; stroke-width: 1.4; transition: opacity .15s ease, filter .15s ease; }}
    .arc:hover {{ opacity: 0.84; filter: brightness(1.03); }}
    .side-card {{
      padding: 14px;
      border: 1px solid #dbe3ee;
      border-radius: 14px;
      background: #ffffff;
    }}
    .summary-table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 12px;
    }}
    th, td {{
      padding: 7px 6px;
      border-bottom: 1px solid #e2e8f0;
      text-align: left;
      vertical-align: top;
    }}
    th {{ color: #475569; font-size: 11px; text-transform: uppercase; letter-spacing: .04em; }}
    td:nth-child(n+2), th:nth-child(n+2) {{ text-align: right; }}
    .detail-table td:nth-child(2), .detail-table th:nth-child(2) {{ text-align: left; }}
    .swatch {{
      display: inline-block;
      width: 9px;
      height: 9px;
      margin-right: 7px;
      border-radius: 2px;
      vertical-align: 0;
    }}
    .note-box {{
      margin-top: 12px;
      padding: 11px 12px;
      border: 1px solid #bfdbfe;
      background: #eff6ff;
      color: #1e3a8a;
      border-radius: 12px;
      font-size: 12px;
    }}
    @media print {{
      body {{ background: #ffffff; }}
      main {{ width: 100%; padding: 0; }}
      .page {{ border: 0; border-radius: 0; margin: 0; min-height: 0; }}
      .chart-panel, .side-card, .metric-card {{ break-inside: avoid; }}
    }}
  </style>
</head>
<body>
<main>
  <section class="page">
    <div class="header">
      <div>
        <h1>RUBRIC-MME Scene Distribution</h1>
        <p class="subtle">Concentric donut view of all dataset sessions. The visualization uses session-level environment labels from the multi-image and streaming-video source JSON files.</p>
      </div>
      <p class="subtle">Total sessions: <strong>{counts["total"]}</strong></p>
    </div>
    <div class="metric-grid">
      {task_cards(counts)}
      <div class="metric-card">
        <span class="metric-pill" style="background:#0f172a"></span>
        <div><div class="metric-label">Major Scene Categories</div><div class="metric-value">{len(counts["major_counts"])}</div><div class="metric-note">Each major category has four details</div></div>
      </div>
    </div>
    <div class="layout">
      <div class="chart-panel">{overview_svg(counts)}</div>
      <div class="side-card">
        <h2>Major Scene Shares</h2>
        {major_table(counts)}
        <div class="note-box">The first page keeps the chart clean: inner ring shows task type, outer ring shows the seven major scene categories.</div>
      </div>
    </div>
  </section>

  <section class="page">
    <div class="header">
      <div>
        <h1>Nested Scene Detail View</h1>
        <p class="subtle">The outer detail ring is nested within the corresponding major-scene arc. This makes the parent-child relationship visible without turning the chart into a flat pie chart.</p>
      </div>
    </div>
    <div class="layout">
      <div class="chart-panel">{nested_svg(counts)}</div>
      <div class="side-card">
        <h2>Scene Detail Distribution</h2>
        {detail_table(counts)}
      </div>
    </div>
  </section>
</main>
</body>
</html>
"""


def find_browser() -> Path | None:
    candidates = [
        Path("C:/Program Files/Microsoft/Edge/Application/msedge.exe"),
        Path("C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe"),
        Path("C:/Program Files/Google/Chrome/Application/chrome.exe"),
        Path("C:/Program Files (x86)/Google/Chrome/Application/chrome.exe"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def export_pdf(html_path: Path, pdf_path: Path) -> bool:
    browser = find_browser()
    if browser is None:
        return False
    command = [
        str(browser),
        "--headless=new",
        "--disable-gpu",
        "--no-first-run",
        f"--print-to-pdf={str(pdf_path)}",
        "--print-to-pdf-no-header",
        html_path.resolve().as_uri(),
    ]
    result = subprocess.run(
        command,
        cwd=html_path.parent,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        timeout=120,
    )
    if result.returncode == 0 and pdf_path.exists() and pdf_path.stat().st_size > 0:
        return True
    fallback = command[:]
    fallback[1] = "--headless"
    result = subprocess.run(
        fallback,
        cwd=html_path.parent,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        timeout=120,
    )
    return result.returncode == 0 and pdf_path.exists() and pdf_path.stat().st_size > 0


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    out_dir = (root / args.out_dir).resolve() if not args.out_dir.is_absolute() else args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    html_path = out_dir / args.html_name
    pdf_path = out_dir / args.pdf_name

    counts = load_counts(root)
    html_path.write_text(render_html(counts), encoding="utf-8")
    pdf_ok = False if args.no_pdf else export_pdf(html_path, pdf_path)

    print(f"HTML: {html_path}")
    if args.no_pdf:
        print("PDF: skipped")
    else:
        print(f"PDF: {pdf_path if pdf_ok else 'not generated'}")
    print(f"Total sessions: {counts['total']}")
    print(f"Major scenes: {len(counts['major_counts'])}")
    print(f"Scene details: {len(counts['detail_counts'])}")


if __name__ == "__main__":
    main()
