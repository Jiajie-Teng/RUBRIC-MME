from __future__ import annotations

import csv
import html
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path("result_judge(2.5pro)") / "_phase5_model_comparison"
OUT_MD = ROOT / "paper_insights_report.md"
OUT_HTML = ROOT / "paper_insights_report.html"
OUT_JSON = ROOT / "paper_insights_data.json"


METRIC_LABELS = {
    "Turn-Level / accuracy": "Turn 准确性",
    "Turn-Level / completeness": "Turn 完整性",
    "Turn-Level / relevance": "Turn 相关性",
    "Turn-Level / conciseness": "Turn 简洁性",
    "Turn-Level / naturalness": "Turn 自然度",
    "Turn-Level / proactiveness_helpfulness": "Turn 主动帮助性",
    "Turn-Level / intent_understanding_depth": "Turn 意图理解深度",
    "Turn-Level / user_state_adaptation": "Turn 用户状态适配",
    "Session-Level / intent_fulfillment": "Session 意图满足",
    "Session-Level / overall_helpfulness_trustworthiness": "Session 总体帮助可信",
    "Session-Level / persona_adaptation": "Session 人设适配",
    "Session-Level / session_consistency": "Session 会话一致性",
}


SERIES_COLORS = {
    "Claude": "#6f5bff",
    "GPT": "#0f9f7a",
    "Gemini": "#2f7de1",
    "Doubao": "#d66b2f",
    "Qwen3-VL": "#b53c6d",
    "Qwen3.5": "#7a8d15",
    "Qwen2.5-VL": "#7b5a3d",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def fnum(value: Any) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def inum(value: Any) -> int:
    if value is None or value == "":
        return 0
    return int(float(value))


def fmt(value: float | None, digits: int = 3) -> str:
    if value is None or not math.isfinite(value):
        return "-"
    return f"{value:.{digits}f}"


def pct(value: float | None, digits: int = 1) -> str:
    if value is None or not math.isfinite(value):
        return "-"
    return f"{value * 100:.{digits}f}%"


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def pstdev(values: list[float]) -> float:
    return statistics.pstdev(values) if len(values) > 1 else 0.0


def corr(xs: list[float], ys: list[float]) -> float:
    mx = mean(xs)
    my = mean(ys)
    denom = math.sqrt(sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys))
    if not denom:
        return float("nan")
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / denom


def linear_residuals(data: list[dict[str, Any]]) -> list[dict[str, Any]]:
    xs = [row["turn"] for row in data]
    ys = [row["session"] for row in data]
    mx = mean(xs)
    my = mean(ys)
    denom = sum((x - mx) ** 2 for x in xs)
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / denom if denom else 0.0
    intercept = my - slope * mx
    out = []
    for row in data:
        expected = intercept + slope * row["turn"]
        item = dict(row)
        item["expected_session"] = expected
        item["session_residual"] = row["session"] - expected
        out.append(item)
    return out


def table_md(headers: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def table_html(headers: list[str], rows: list[list[str]], klass: str = "") -> str:
    head = "".join(f"<th>{html.escape(h)}</th>" for h in headers)
    body = "\n".join(
        "<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>" for row in rows
    )
    return f'<div class="table-scroll"><table class="{klass}"><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def esc(value: Any) -> str:
    return html.escape(str(value))


def rank(values: list[tuple[str, float]], reverse: bool = True) -> dict[str, int]:
    sorted_values = sorted(values, key=lambda x: x[1], reverse=reverse)
    return {name: i + 1 for i, (name, _) in enumerate(sorted_values)}


def bar_rows(
    rows: list[dict[str, Any]],
    label_key: str,
    value_key: str,
    subtitle_key: str | None = None,
    max_value: float | None = None,
    low_is_bad: bool = False,
) -> str:
    vals = [float(row[value_key]) for row in rows if row.get(value_key) is not None]
    maxv = max_value if max_value is not None else (max(vals) if vals else 1.0)
    parts = []
    for row in rows:
        value = float(row[value_key])
        width = max(2.0, min(100.0, value / maxv * 100 if maxv else 0.0))
        label = esc(row[label_key])
        subtitle = esc(row.get(subtitle_key, "")) if subtitle_key else ""
        cls = "bad" if low_is_bad else "good"
        parts.append(
            f"""
            <div class="bar-row">
              <div class="bar-label"><strong>{label}</strong><span>{subtitle}</span></div>
              <div class="bar-track"><div class="bar {cls}" style="width:{width:.2f}%"></div></div>
              <div class="bar-value">{fmt(value)}</div>
            </div>
            """
        )
    return '<div class="bar-list">' + "\n".join(parts) + "</div>"


def scatter_svg(points: list[dict[str, Any]]) -> str:
    if not points:
        return ""
    width, height = 860, 520
    pad_l, pad_r, pad_t, pad_b = 70, 30, 38, 62
    xs = [p["turn"] for p in points]
    ys = [p["session"] for p in points]
    xmin = min(xs) - 0.08
    xmax = max(xs) + 0.08
    ymin = min(ys) - 0.08
    ymax = max(ys) + 0.08

    def sx(x: float) -> float:
        return pad_l + (x - xmin) / (xmax - xmin) * (width - pad_l - pad_r)

    def sy(y: float) -> float:
        return height - pad_b - (y - ymin) / (ymax - ymin) * (height - pad_t - pad_b)

    grid = []
    for i in range(6):
        xval = xmin + (xmax - xmin) * i / 5
        yval = ymin + (ymax - ymin) * i / 5
        x = sx(xval)
        y = sy(yval)
        grid.append(f'<line class="grid" x1="{x:.1f}" y1="{pad_t}" x2="{x:.1f}" y2="{height-pad_b}" />')
        grid.append(f'<text class="axis tick" x="{x:.1f}" y="{height-pad_b+24}" text-anchor="middle">{xval:.1f}</text>')
        grid.append(f'<line class="grid" x1="{pad_l}" y1="{y:.1f}" x2="{width-pad_r}" y2="{y:.1f}" />')
        grid.append(f'<text class="axis tick" x="{pad_l-12}" y="{y+4:.1f}" text-anchor="end">{yval:.1f}</text>')

    line_min = max(xmin, ymin)
    line_max = min(xmax, ymax)
    diag = (
        f'<line class="diag" x1="{sx(line_min):.1f}" y1="{sy(line_min):.1f}" '
        f'x2="{sx(line_max):.1f}" y2="{sy(line_max):.1f}" />'
    )

    point_html = []
    for p in points:
        color = SERIES_COLORS.get(p["series"], "#666")
        x = sx(p["turn"])
        y = sy(p["session"])
        title = (
            f"{p['model']} | {p['series']} | Turn {p['turn']:.3f} | "
            f"Session {p['session']:.3f} | Gap {p['gap']:.3f}"
        )
        point_html.append(
            f'<g class="pt"><circle cx="{x:.1f}" cy="{y:.1f}" r="6" fill="{color}">'
            f"<title>{esc(title)}</title></circle></g>"
        )
    legend = []
    for i, (series, color) in enumerate(SERIES_COLORS.items()):
        x = pad_l + i * 108
        legend.append(
            f'<circle cx="{x}" cy="{height-18}" r="5" fill="{color}" />'
            f'<text class="axis legend" x="{x+10}" y="{height-14}">{esc(series)}</text>'
        )
    return f"""
    <svg class="scatter" viewBox="0 0 {width} {height}" role="img" aria-label="Turn Session scatter">
      {''.join(grid)}
      {diag}
      <line class="axis-line" x1="{pad_l}" y1="{height-pad_b}" x2="{width-pad_r}" y2="{height-pad_b}" />
      <line class="axis-line" x1="{pad_l}" y1="{pad_t}" x2="{pad_l}" y2="{height-pad_b}" />
      <text class="axis title" x="{width/2}" y="{height-33}" text-anchor="middle">Turn 指标加权均值</text>
      <text class="axis title" transform="translate(22 {height/2}) rotate(-90)" text-anchor="middle">Session 指标加权均值</text>
      <text class="axis hint" x="{width-pad_r}" y="{pad_t+12}" text-anchor="end">虚线为 Session = Turn</text>
      {''.join(point_html)}
      {''.join(legend)}
    </svg>
    """


def mini_matrix_html(rows: list[dict[str, Any]], metrics: list[str]) -> str:
    values = [row[m] for row in rows for m in metrics if row.get(m) is not None]
    lo, hi = min(values), max(values)

    def cell(v: float | None) -> str:
        if v is None:
            return '<td class="na">-</td>'
        frac = (v - lo) / (hi - lo) if hi > lo else 0.5
        hue = 12 + frac * 140
        bg = f"hsl({hue:.0f} 66% 88%)"
        return f'<td style="background:{bg}">{fmt(v, 2)}</td>'

    headers = ["Model"] + metrics
    trs = []
    for row in rows:
        trs.append(
            "<tr><th>" + esc(row["model"]) + "</th>" + "".join(cell(row.get(m)) for m in metrics) + "</tr>"
        )
    return (
        '<div class="table-scroll heatmap-mini"><table><thead><tr>'
        + "".join(f"<th>{esc(h)}</th>" for h in headers)
        + "</tr></thead><tbody>"
        + "\n".join(trs)
        + "</tbody></table></div>"
    )


def main() -> None:
    summary = read_csv(ROOT / "model_summary.csv")
    metrics = read_csv(ROOT / "overall_metrics_long.csv")
    primary = read_csv(ROOT / "primary_category_long.csv")
    task_rows_raw = read_csv(ROOT / "task_top_tables.csv")

    by_model_metric: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_metric: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in metrics:
        key = f"{row['level']} / {row['metric']}"
        item = {
            "model": row["model"],
            "series": row["series"],
            "key": key,
            "label": METRIC_LABELS.get(key, key),
            "avg": fnum(row["avg_score"]),
            "low": fnum(row["low_rate"]),
        }
        by_model_metric[row["model"]].append(item)
        by_metric[key].append(item)

    summary_points = []
    for row in summary:
        turn = fnum(row.get("turn_metric_avg"))
        session = fnum(row.get("session_metric_avg"))
        all_task = fnum(row.get("all_task_weighted_avg"))
        if turn is None or session is None:
            continue
        summary_points.append(
            {
                "model": row["model"],
                "series": row["series"],
                "turn": turn,
                "session": session,
                "gap": turn - session,
                "ratio": session / turn if turn else None,
                "all_task": all_task,
                "text_task": fnum(row.get("text_task_weighted_avg")),
                "tts_task": fnum(row.get("tts_task_weighted_avg")),
                "turn_n": inum(row.get("turn_sample_count_est")),
                "session_n": inum(row.get("session_sample_count_est")),
                "primary": fnum(row.get("primary_weighted_score")),
            }
        )

    residual_points = linear_residuals(summary_points)
    turn_session_corr = corr([p["turn"] for p in summary_points], [p["session"] for p in summary_points])
    mean_gap = mean([p["gap"] for p in summary_points])
    session_above_turn = [p for p in summary_points if p["gap"] < 0]
    largest_gaps = sorted(summary_points, key=lambda p: p["gap"], reverse=True)[:8]
    smallest_gaps = sorted(summary_points, key=lambda p: p["gap"])[:8]
    positive_residuals = sorted(residual_points, key=lambda p: p["session_residual"], reverse=True)[:6]
    negative_residuals = sorted(residual_points, key=lambda p: p["session_residual"])[:6]

    metric_stats = []
    for key, rows in by_metric.items():
        avgs = [row["avg"] for row in rows if row["avg"] is not None]
        lows = [row["low"] for row in rows if row["low"] is not None]
        best = max(rows, key=lambda row: row["avg"] if row["avg"] is not None else -1)
        worst = min(rows, key=lambda row: row["avg"] if row["avg"] is not None else 99)
        low_best = min(rows, key=lambda row: row["low"] if row["low"] is not None else 99)
        low_worst = max(rows, key=lambda row: row["low"] if row["low"] is not None else -1)
        metric_stats.append(
            {
                "key": key,
                "label": METRIC_LABELS.get(key, key),
                "level": key.split(" / ")[0],
                "mean_avg": mean(avgs),
                "std_avg": pstdev(avgs),
                "mean_low": mean(lows),
                "std_low": pstdev(lows),
                "best_model": best["model"],
                "best_avg": best["avg"],
                "worst_model": worst["model"],
                "worst_avg": worst["avg"],
                "lowest_low_model": low_best["model"],
                "lowest_low": low_best["low"],
                "highest_low_model": low_worst["model"],
                "highest_low": low_worst["low"],
            }
        )
    metric_hardest = sorted(metric_stats, key=lambda row: row["mean_avg"])[:6]
    metric_easiest = sorted(metric_stats, key=lambda row: row["mean_avg"], reverse=True)[:4]
    lowrate_hardest = sorted(metric_stats, key=lambda row: row["mean_low"], reverse=True)[:6]

    profile_ranges = []
    for model, rows in by_model_metric.items():
        avgs = [row["avg"] for row in rows if row["avg"] is not None]
        lows = [row["low"] for row in rows if row["low"] is not None]
        best = max(rows, key=lambda row: row["avg"] if row["avg"] is not None else -1)
        worst = min(rows, key=lambda row: row["avg"] if row["avg"] is not None else 99)
        high_low = max(rows, key=lambda row: row["low"] if row["low"] is not None else -1)
        low_low = min(rows, key=lambda row: row["low"] if row["low"] is not None else 99)
        turn_rows = [row for row in rows if row["key"].startswith("Turn-Level")]
        session_rows = [row for row in rows if row["key"].startswith("Session-Level")]
        profile_ranges.append(
            {
                "model": model,
                "series": rows[0]["series"],
                "score_range": max(avgs) - min(avgs),
                "low_range": max(lows) - min(lows),
                "best_metric": best["label"],
                "best_metric_avg": best["avg"],
                "worst_metric": worst["label"],
                "worst_metric_avg": worst["avg"],
                "highest_low_metric": high_low["label"],
                "highest_low": high_low["low"],
                "lowest_low_metric": low_low["label"],
                "lowest_low": low_low["low"],
                "turn_mean_12": mean([row["avg"] for row in turn_rows if row["avg"] is not None]),
                "session_mean_12": mean([row["avg"] for row in session_rows if row["avg"] is not None]),
            }
        )
    largest_profile_ranges = sorted(profile_ranges, key=lambda row: row["score_range"], reverse=True)[:10]
    largest_low_ranges = sorted(profile_ranges, key=lambda row: row["low_range"], reverse=True)[:10]

    top_by_all_task = sorted(summary_points, key=lambda row: row["all_task"] or -1, reverse=True)[:12]
    top_profile_rows = []
    profile_by_model = {row["model"]: row for row in profile_ranges}
    for point in top_by_all_task:
        profile = profile_by_model[point["model"]]
        top_profile_rows.append(
            {
                "model": point["model"],
                "series": point["series"],
                "all_task": point["all_task"],
                "turn_mean_12": profile["turn_mean_12"],
                "session_mean_12": profile["session_mean_12"],
                "gap_12": profile["turn_mean_12"] - profile["session_mean_12"],
                "worst_metric": profile["worst_metric"],
                "worst_metric_avg": profile["worst_metric_avg"],
                "best_metric": profile["best_metric"],
                "best_metric_avg": profile["best_metric_avg"],
            }
        )

    primary_by_ability: dict[str, list[dict[str, Any]]] = defaultdict(list)
    primary_by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in primary:
        item = {
            "model": row["model"],
            "series": row["series"],
            "ability": row["ability"],
            "score": fnum(row["overall_score"]),
            "samples": inum(row.get("sample_count")),
        }
        primary_by_ability[row["ability"]].append(item)
        primary_by_model[row["model"]].append(item)

    ability_stats = []
    for ability, rows in primary_by_ability.items():
        scores = [row["score"] for row in rows if row["score"] is not None]
        samples = sum(row["samples"] for row in rows)
        best = max(rows, key=lambda row: row["score"] if row["score"] is not None else -1)
        worst = min(rows, key=lambda row: row["score"] if row["score"] is not None else 99)
        ability_stats.append(
            {
                "ability": ability,
                "mean": mean(scores),
                "std": pstdev(scores),
                "range": max(scores) - min(scores),
                "low_lt_3": sum(score < 3 for score in scores),
                "low_lt_35": sum(score < 3.5 for score in scores),
                "model_count": len(scores),
                "samples": samples,
                "best_model": best["model"],
                "best_score": best["score"],
                "worst_model": worst["model"],
                "worst_score": worst["score"],
            }
        )
    hardest_abilities = sorted(ability_stats, key=lambda row: row["mean"])[:15]
    robust_hardest_abilities = sorted(
        [row for row in ability_stats if row["samples"] >= 500], key=lambda row: row["mean"]
    )[:15]
    discriminative_abilities = sorted(ability_stats, key=lambda row: row["std"], reverse=True)[:15]
    robust_discriminative_abilities = sorted(
        [row for row in ability_stats if row["samples"] >= 500], key=lambda row: row["std"], reverse=True
    )[:15]

    primary_summary = {row["model"]: fnum(row.get("primary_weighted_score")) for row in summary}
    robust_cliffs = []
    fine_cliffs = []
    for model, rows in primary_by_model.items():
        primary_score = primary_summary.get(model)
        if primary_score is None or primary_score < 4.0:
            continue
        robust_rows = [row for row in rows if row["samples"] >= 10 and row["score"] is not None]
        fine_rows = [row for row in rows if row["score"] is not None]
        robust_worst = sorted(robust_rows, key=lambda row: row["score"])[:3]
        fine_worst = sorted(fine_rows, key=lambda row: row["score"])[:3]
        if robust_worst and robust_worst[0]["score"] <= 3.3:
            robust_cliffs.append(
                {
                    "model": model,
                    "series": rows[0]["series"],
                    "primary": primary_score,
                    "worst": robust_worst,
                }
            )
        if fine_worst and fine_worst[0]["score"] <= 3.0:
            fine_cliffs.append(
                {
                    "model": model,
                    "series": rows[0]["series"],
                    "primary": primary_score,
                    "worst": fine_worst,
                }
            )
    robust_cliffs = sorted(robust_cliffs, key=lambda row: (row["worst"][0]["score"], -row["primary"]))
    fine_cliffs = sorted(fine_cliffs, key=lambda row: (row["worst"][0]["score"], -row["primary"]))[:12]

    task_best: dict[tuple[str, str], dict[str, str]] = {}
    for row in task_rows_raw:
        key = (row["model"], row["task"])
        if key not in task_best:
            task_best[key] = row
    task_by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in task_best.values():
        task_by_task[row["task"]].append(
            {
                "model": row["model"],
                "series": row["series"],
                "task": row["task"],
                "score": fnum(row["avg_score"]),
                "samples": inum(row.get("sample_count")),
            }
        )
    task_rankings = {}
    for task, rows in task_by_task.items():
        task_rankings[task] = sorted(rows, key=lambda row: row["score"] or -1, reverse=True)
    image_video_gaps = []
    for model in sorted({row["model"] for row in task_best.values()}):
        image = task_best.get((model, "omnibench_image_multi_text"))
        video = task_best.get((model, "omnibench_video_stream_text"))
        if image and video:
            image_video_gaps.append(
                {
                    "model": model,
                    "series": image["series"],
                    "gap": fnum(image["avg_score"]) - fnum(video["avg_score"]),
                    "image": fnum(image["avg_score"]),
                    "video": fnum(video["avg_score"]),
                }
            )
    tts_deltas = []
    for model in sorted({row["model"] for row in task_best.values()}):
        for task_text, task_tts, label in [
            ("omnibench_image_multi_text", "omnibench_image_multi_tts", "image"),
            ("omnibench_video_stream_text", "omnibench_video_stream_tts", "video"),
        ]:
            text_row = task_best.get((model, task_text))
            tts_row = task_best.get((model, task_tts))
            if text_row and tts_row:
                tts_deltas.append(
                    {
                        "model": model,
                        "series": text_row["series"],
                        "modality": label,
                        "delta": fnum(tts_row["avg_score"]) - fnum(text_row["avg_score"]),
                        "text": fnum(text_row["avg_score"]),
                        "tts": fnum(tts_row["avg_score"]),
                    }
                )

    by_series: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for point in summary_points:
        by_series[point["series"]].append(point)
    series_findings = []
    for series, rows in sorted(by_series.items()):
        ordered = sorted(rows, key=lambda row: row["all_task"] or -1, reverse=True)
        best = ordered[0]
        worst = ordered[-1]
        series_findings.append(
            {
                "series": series,
                "count": len(rows),
                "best": best,
                "worst": worst,
                "range": (best["all_task"] or 0) - (worst["all_task"] or 0),
                "mean_gap": mean([row["gap"] for row in rows]),
                "ordered": ordered,
            }
        )

    artifacts = {
        "turn_session": {
            "corr": turn_session_corr,
            "mean_gap": mean_gap,
            "largest_gaps": largest_gaps,
            "smallest_gaps": smallest_gaps,
            "positive_session_residuals": positive_residuals,
            "negative_session_residuals": negative_residuals,
        },
        "metrics": {
            "metric_stats": metric_stats,
            "hardest": metric_hardest,
            "easiest": metric_easiest,
            "lowrate_hardest": lowrate_hardest,
            "largest_profile_ranges": largest_profile_ranges,
            "largest_low_ranges": largest_low_ranges,
            "top_profile_rows": top_profile_rows,
        },
        "primary": {
            "hardest": hardest_abilities,
            "robust_hardest": robust_hardest_abilities,
            "discriminative": discriminative_abilities,
            "robust_discriminative": robust_discriminative_abilities,
            "robust_cliffs": robust_cliffs,
            "fine_cliffs": fine_cliffs,
        },
        "tasks": {
            "task_rankings": task_rankings,
            "image_video_gaps": image_video_gaps,
            "tts_deltas": tts_deltas,
        },
        "series": series_findings,
    }
    OUT_JSON.write_text(json.dumps(artifacts, ensure_ascii=False, indent=2), encoding="utf-8")

    gap_rows = [
        [esc(p["model"]), esc(p["series"]), fmt(p["turn"]), fmt(p["session"]), fmt(p["gap"]), pct(1 - (p["ratio"] or 0))]
        for p in largest_gaps
    ]
    stable_rows = [
        [esc(p["model"]), esc(p["series"]), fmt(p["turn"]), fmt(p["session"]), fmt(p["gap"]), fmt(p["ratio"])]
        for p in smallest_gaps
    ]
    residual_rows = [
        [esc(p["model"]), esc(p["series"]), fmt(p["turn"]), fmt(p["session"]), fmt(p["session_residual"])]
        for p in positive_residuals
    ]
    metric_rows = [
        [
            esc(row["label"]),
            fmt(row["mean_avg"]),
            fmt(row["std_avg"]),
            pct(row["mean_low"]),
            esc(row["best_model"]),
            fmt(row["best_avg"]),
            esc(row["worst_model"]),
            fmt(row["worst_avg"]),
        ]
        for row in metric_stats
    ]
    metric_hard_rows = [
        [
            esc(row["label"]),
            fmt(row["mean_avg"]),
            pct(row["mean_low"]),
            esc(row["best_model"]),
            fmt(row["best_avg"]),
            esc(row["worst_model"]),
            fmt(row["worst_avg"]),
        ]
        for row in metric_hardest
    ]
    low_rows = [
        [
            esc(row["label"]),
            pct(row["mean_low"]),
            fmt(row["mean_avg"]),
            esc(row["lowest_low_model"]),
            pct(row["lowest_low"]),
            esc(row["highest_low_model"]),
            pct(row["highest_low"]),
        ]
        for row in lowrate_hardest
    ]
    profile_range_rows = [
        [
            esc(row["model"]),
            esc(row["series"]),
            fmt(row["score_range"]),
            esc(row["best_metric"]),
            fmt(row["best_metric_avg"]),
            esc(row["worst_metric"]),
            fmt(row["worst_metric_avg"]),
        ]
        for row in largest_profile_ranges
    ]
    top_profile_table_rows = [
        [
            esc(row["model"]),
            esc(row["series"]),
            fmt(row["all_task"]),
            fmt(row["turn_mean_12"]),
            fmt(row["session_mean_12"]),
            fmt(row["gap_12"]),
            esc(row["worst_metric"]),
            fmt(row["worst_metric_avg"]),
        ]
        for row in top_profile_rows
    ]
    ability_hard_rows = [
        [
            esc(row["ability"]),
            fmt(row["mean"]),
            fmt(row["std"]),
            str(row["low_lt_3"]),
            str(row["samples"]),
            esc(row["best_model"]),
            fmt(row["best_score"]),
            esc(row["worst_model"]),
            fmt(row["worst_score"]),
        ]
        for row in robust_hardest_abilities
    ]
    ability_disc_rows = [
        [
            esc(row["ability"]),
            fmt(row["std"]),
            fmt(row["range"]),
            fmt(row["mean"]),
            str(row["samples"]),
            esc(row["best_model"]),
            fmt(row["best_score"]),
            esc(row["worst_model"]),
            fmt(row["worst_score"]),
        ]
        for row in robust_discriminative_abilities
    ]

    def worst_list(items: list[dict[str, Any]]) -> str:
        return "<br>".join(
            f"{esc(x['ability'])}: {fmt(x['score'])} (n={x['samples']})" for x in items
        )

    robust_cliff_rows = [
        [esc(row["model"]), esc(row["series"]), fmt(row["primary"]), worst_list(row["worst"])]
        for row in robust_cliffs
    ]
    fine_cliff_rows = [
        [esc(row["model"]), esc(row["series"]), fmt(row["primary"]), worst_list(row["worst"])]
        for row in fine_cliffs
    ]

    task_top_rows = []
    for task, rows in sorted(task_rankings.items()):
        top3 = rows[:3]
        task_top_rows.append(
            [
                esc(task),
                str(len(rows)),
                "<br>".join(f"{esc(row['model'])}: {fmt(row['score'])}" for row in top3),
                "<br>".join(f"{esc(row['model'])}: {fmt(row['score'])}" for row in rows[-3:]),
            ]
        )
    tts_rows = [
        [esc(row["model"]), esc(row["modality"]), fmt(row["text"]), fmt(row["tts"]), fmt(row["delta"])]
        for row in sorted(tts_deltas, key=lambda row: row["delta"])
    ]

    series_rows = []
    for row in series_findings:
        ordered = row["ordered"]
        series_rows.append(
            [
                esc(row["series"]),
                str(row["count"]),
                esc(row["best"]["model"]),
                fmt(row["best"]["all_task"]),
                esc(row["worst"]["model"]),
                fmt(row["worst"]["all_task"]),
                fmt(row["range"]),
                fmt(row["mean_gap"]),
                "<br>".join(
                    f"{esc(item['model'])}: all {fmt(item['all_task'])}, gap {fmt(item['gap'])}"
                    for item in ordered
                ),
            ]
        )

    md_parts = [
        "# Phase5 Benchmark 论文向实验发现分析",
        "",
        "本报告基于 `_phase5_model_comparison` 下已抽取的数据生成，不改动任何原始模型报告。分析重点不是重新排名，而是寻找多任务、多指标、多能力分层评测带来的差异化证据。",
        "",
        "## 一、核心发现速览",
        "",
        f"1. Turn 与 Session 总体高度相关（Pearson r={fmt(turn_session_corr)}），但并不是同一个信号：29 个模型的 Turn-Session 平均差为 {fmt(mean_gap)}，说明会话级长期一致性和帮助可信度通常比单轮回答更难。",
        f"2. 只有 {', '.join(p['model'] for p in session_above_turn) if session_above_turn else '没有模型'} 出现 Session 高于 Turn。Claude Opus 4.6 是最清晰的反例，Session={fmt(session_above_turn[0]['session']) if session_above_turn else '-'}，Turn={fmt(session_above_turn[0]['turn']) if session_above_turn else '-'}，可作为论文中“会话能力不等价于单轮能力”的正例。",
        f"3. 12 个整体指标里，最难的是 {metric_hardest[0]['label']}（跨模型均值 {fmt(metric_hardest[0]['mean_avg'])}，平均低分率 {pct(metric_hardest[0]['mean_low'])}），最容易的是 {metric_easiest[0]['label']}（均值 {fmt(metric_easiest[0]['mean_avg'])}）。这说明表层表达质量和深层帮助可信之间存在明显分层。",
        f"4. Primary 能力维度里，稳健样本量下最困难的是 {robust_hardest_abilities[0]['ability']}（均值 {fmt(robust_hardest_abilities[0]['mean'])}），其次集中在自我中心导航、物体交互、预测和空间/时序推理。",
        f"5. Gemini 系列的 TTS 任务普遍低于 text 任务，最大落差来自 {sorted(tts_deltas, key=lambda row: row['delta'])[0]['model']} 的 {sorted(tts_deltas, key=lambda row: row['delta'])[0]['modality']}（delta={fmt(sorted(tts_deltas, key=lambda row: row['delta'])[0]['delta'])}），说明相同视觉任务换成 TTS 输入会引入额外压力。",
        "6. 系列内不是单调大模型/新模型必胜：GPT-5.4 总体和 Turn 更强，但 GPT-5 在 GPT 系列里 Session 均值最高；Qwen3-VL 235B Instruct 明显强于 Thinking；Claude Opus 4.6 是全局最强的会话稳定模型。",
        "",
        "## 二、覆盖口径：Turn 与 Session 的关系",
        "",
        f"Turn 与 Session 的相关性很高（r={fmt(turn_session_corr)}），但 Session 系统性低于 Turn。这个结果很适合写成 benchmark 的设计价值：单轮高分不能自动推出多轮会话可靠。",
        "",
        table_md(["Model", "Series", "Turn", "Session", "Gap", "Session 相对下降"], [[html.unescape(c) for c in row] for row in gap_rows]),
        "",
        "Session 保持较好的模型：",
        "",
        table_md(["Model", "Series", "Turn", "Session", "Gap", "Session/Turn"], [[html.unescape(c) for c in row] for row in stable_rows]),
        "",
        "在同等 Turn 水平下，Session 表现高于回归预期的模型：",
        "",
        table_md(["Model", "Series", "Turn", "Session", "Session residual"], [[html.unescape(c) for c in row] for row in residual_rows]),
        "",
        "## 三、12 个整体指标：平均分画像",
        "",
        table_md(["指标", "跨模型均值", "模型间std", "平均低分率", "最高模型", "最高分", "最低模型", "最低分"], [[html.unescape(c) for c in row] for row in metric_hard_rows]),
        "",
        "高分模型的最弱项也多集中在 Session 总体帮助可信，说明平均分热力图中最值得解释的并不是谁高谁低，而是同一模型在表达类指标与会话可靠性指标之间的落差。",
        "",
        table_md(["Model", "Series", "全部任务加权", "Turn均值(12指标)", "Session均值(12指标)", "Gap", "最弱指标", "最弱分"], [[html.unescape(c) for c in row] for row in top_profile_table_rows]),
        "",
        "## 四、低分率：可靠性视角",
        "",
        table_md(["指标", "平均低分率", "平均分", "最低低分率模型", "最低低分率", "最高低分率模型", "最高低分率"], [[html.unescape(c) for c in row] for row in low_rows]),
        "",
        "低分率强调的是失败风险。自然度、简洁性这类指标均值高且低分率低；总体帮助可信、会话一致性、意图满足的低分率显著更高，说明多轮层面的错误不是少数极端样本，而是更广泛的稳定性问题。",
        "",
        "## 五、Primary Category 能力发现",
        "",
        "稳健样本量（总样本数 >=500）下较困难的能力：",
        "",
        table_md(["Ability", "Mean", "Std", "低于3模型数", "总样本", "Best", "Best分", "Worst", "Worst分"], [[html.unescape(c) for c in row] for row in ability_hard_rows[:10]]),
        "",
        "稳健样本量下最能区分模型的能力：",
        "",
        table_md(["Ability", "Std", "Range", "Mean", "总样本", "Best", "Best分", "Worst", "Worst分"], [[html.unescape(c) for c in row] for row in ability_disc_rows[:10]]),
        "",
        "高整体 Primary 分数但存在稳健能力短板的模型：",
        "",
        table_md(["Model", "Series", "Primary加权", "低谷能力(sample>=10)"], [[html.unescape(c.replace('<br>', '; ')) for c in row] for row in robust_cliff_rows]),
        "",
        "## 六、任务与系列内观察",
        "",
        table_md(["Task", "模型数", "Top3", "Bottom3"], [[html.unescape(c.replace('<br>', '; ')) for c in row] for row in task_top_rows]),
        "",
        "Gemini TTS 相对 text 的变化：",
        "",
        table_md(["Model", "Modality", "Text", "TTS", "Delta"], [[html.unescape(c) for c in row] for row in tts_rows]),
        "",
        "系列内比较：",
        "",
        table_md(["Series", "数量", "最佳", "最佳All", "最低", "最低All", "系列Range", "平均Turn-Session gap", "内部顺序"], [[html.unescape(c.replace('<br>', '; ')) for c in row] for row in series_rows]),
        "",
        "## 七、可以写进论文的表述角度",
        "",
        "1. 本 benchmark 的分层指标能揭示“流畅但不可靠”的模型行为：自然度/简洁性普遍高，但会话帮助可信和一致性显著更难。",
        "2. Turn-Level 与 Session-Level 是相关但不等价的评测层：Claude Opus 4.6 说明会话层可强于单轮层，而 Qwen 系列多个模型说明单轮能力提升后，会话层仍可能滞后。",
        "3. 能力维度提供比总分更细的诊断：自我中心、空间/时序、物体交互和幻觉检测等能力暴露了总分无法解释的结构性短板。",
        "4. 同一系列内部也存在非单调与能力迁移问题：例如 GPT-5.x 的 Turn/Session 权衡、Gemini 的 TTS 退化、Qwen3-VL Instruct/Thinking 的差异。",
    ]
    OUT_MD.write_text("\n".join(md_parts), encoding="utf-8")

    html_doc = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Phase5 Benchmark 论文实验发现分析</title>
  <style>
    :root {{
      --bg: #f6f7f9;
      --panel: #ffffff;
      --text: #1d2430;
      --muted: #667085;
      --line: #d9dee7;
      --accent: #245bdb;
      --good: #0f9f7a;
      --bad: #d45a38;
      --warn: #b98000;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Microsoft YaHei", sans-serif;
      line-height: 1.55;
    }}
    header {{
      padding: 34px 42px 24px;
      background: #111827;
      color: white;
    }}
    header h1 {{ margin: 0 0 8px; font-size: 30px; letter-spacing: 0; }}
    header p {{ margin: 0; max-width: 1100px; color: #d1d5db; }}
    main {{ padding: 28px 42px 60px; max-width: 1500px; margin: 0 auto; }}
    section {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 22px;
      margin: 0 0 22px;
    }}
    h2 {{ margin: 0 0 12px; font-size: 22px; }}
    h3 {{ margin: 22px 0 10px; font-size: 17px; }}
    .note {{ color: var(--muted); margin: 4px 0 16px; }}
    .grid-cards {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
      gap: 12px;
      margin: 16px 0 6px;
    }}
    .card {{
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 14px;
      background: #fbfcfe;
    }}
    .card .k {{ font-size: 13px; color: var(--muted); }}
    .card .v {{ font-size: 24px; font-weight: 750; margin-top: 4px; }}
    .card .s {{ font-size: 13px; color: var(--muted); margin-top: 4px; }}
    .finding-list {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
      gap: 10px;
      padding: 0;
      margin: 16px 0 0;
      list-style: none;
    }}
    .finding-list li {{
      border-left: 4px solid var(--accent);
      background: #f8faff;
      padding: 12px 14px;
      border-radius: 6px;
    }}
    .table-scroll {{ overflow: auto; border: 1px solid var(--line); border-radius: 8px; margin: 12px 0; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 13px; min-width: 760px; }}
    th, td {{ border-bottom: 1px solid #e7ebf2; padding: 8px 10px; text-align: left; vertical-align: top; }}
    th {{ background: #f1f4f8; position: sticky; top: 0; z-index: 1; }}
    tr:last-child td {{ border-bottom: 0; }}
    .two-col {{ display: grid; grid-template-columns: minmax(0, 1fr) minmax(0, 1fr); gap: 16px; }}
    @media (max-width: 1000px) {{ .two-col {{ grid-template-columns: 1fr; }} main {{ padding: 18px; }} header {{ padding: 28px 18px 20px; }} }}
    .bar-list {{ display: grid; gap: 8px; }}
    .bar-row {{ display: grid; grid-template-columns: minmax(160px, 300px) 1fr 64px; align-items: center; gap: 10px; }}
    .bar-label strong {{ display: block; font-size: 13px; }}
    .bar-label span {{ display: block; color: var(--muted); font-size: 12px; }}
    .bar-track {{ height: 12px; background: #eef2f7; border-radius: 999px; overflow: hidden; }}
    .bar {{ height: 100%; border-radius: 999px; }}
    .bar.good {{ background: linear-gradient(90deg, #69d4b2, #0f9f7a); }}
    .bar.bad {{ background: linear-gradient(90deg, #f2b69d, #d45a38); }}
    .bar-value {{ font-variant-numeric: tabular-nums; font-size: 12px; color: var(--muted); }}
    .scatter {{ width: 100%; height: auto; background: #fbfcfe; border: 1px solid var(--line); border-radius: 8px; }}
    .grid {{ stroke: #dfe4ec; stroke-width: 1; }}
    .diag {{ stroke: #9aa4b2; stroke-width: 1.5; stroke-dasharray: 5 5; }}
    .axis-line {{ stroke: #98a2b3; stroke-width: 1.2; }}
    .axis {{ fill: #667085; font-size: 12px; }}
    .title {{ font-weight: 700; fill: #475467; }}
    .hint {{ fill: #98a2b3; }}
    .pt circle {{ stroke: white; stroke-width: 1.5; opacity: .9; }}
    .pt:hover circle {{ stroke: #111827; stroke-width: 3; opacity: 1; }}
    .heatmap-mini table {{ min-width: 980px; }}
    .heatmap-mini th:first-child {{ min-width: 170px; }}
    .na {{ color: #98a2b3; background: #f3f4f6; }}
    .callout {{
      background: #fff8e6;
      border: 1px solid #f0d389;
      border-radius: 8px;
      padding: 12px 14px;
      color: #473a16;
      margin: 12px 0;
    }}
    code {{ background: #eef2f7; padding: 1px 4px; border-radius: 4px; }}
  </style>
</head>
<body>
  <header>
    <h1>Phase5 Benchmark 论文实验发现分析</h1>
    <p>基于 29 个模型报告的横向汇总结果生成。重点是提炼 benchmark 分层评测带来的差异化发现，而不是继续修改原 dashboard。</p>
  </header>
  <main>
    <section>
      <h2>核心发现速览</h2>
      <div class="grid-cards">
        <div class="card"><div class="k">Turn / Session 相关性</div><div class="v">{fmt(turn_session_corr)}</div><div class="s">相关但不等价</div></div>
        <div class="card"><div class="k">平均 Turn-Session 差</div><div class="v">{fmt(mean_gap)}</div><div class="s">Session 通常更难</div></div>
        <div class="card"><div class="k">最难整体指标</div><div class="v">{esc(metric_hardest[0]['label'])}</div><div class="s">均值 {fmt(metric_hardest[0]['mean_avg'])}，低分率 {pct(metric_hardest[0]['mean_low'])}</div></div>
        <div class="card"><div class="k">稳健样本下最难 Primary 能力</div><div class="v">{esc(robust_hardest_abilities[0]['ability'])}</div><div class="s">均值 {fmt(robust_hardest_abilities[0]['mean'])}</div></div>
      </div>
      <ul class="finding-list">
        <li>单轮表现和会话表现高度相关，但 29 个模型平均仍有 {fmt(mean_gap)} 分差，说明多轮可靠性不是单轮质量的简单外推。</li>
        <li>Claude Opus 4.6 是唯一 Session 高于 Turn 的模型，适合用作“会话层能力独立存在”的正例。</li>
        <li>自然度、简洁性普遍容易；总体帮助可信、会话一致性、意图满足更难，体现了表层表达与深层任务完成之间的分层。</li>
        <li>Primary 能力的难点集中在自我中心行为/导航、物体交互、空间时序和幻觉检测，能解释为什么总分相近的模型在细粒度能力上不同。</li>
        <li>Gemini 的 TTS 任务普遍低于 text 任务，说明同一视觉任务的输入形式变化会显著改变模型表现。</li>
      </ul>
    </section>

    <section>
      <h2>一、覆盖与口径：Turn 和 Session 的关系</h2>
      <p class="note">这里使用总表里由“七、任务与模态分析 / 任务维度”计算出的 Turn 与 Session 加权均值。</p>
      {scatter_svg(summary_points)}
      <div class="two-col">
        <div>
          <h3>Turn 到 Session 下降最明显的模型</h3>
          {table_html(["Model", "Series", "Turn", "Session", "Gap", "Session相对下降"], gap_rows)}
        </div>
        <div>
          <h3>Session 保持较好的模型</h3>
          {table_html(["Model", "Series", "Turn", "Session", "Gap", "Session/Turn"], stable_rows)}
        </div>
      </div>
      <div class="callout">论文可写法：Turn 与 Session 不是冗余指标。小模型和部分 Qwen 系列在 Turn 提升后，Session 仍明显滞后；Claude Opus 4.6 则显示会话层稳定性可以成为独立优势。</div>
      <h3>同等 Turn 水平下，Session 高于回归预期的模型</h3>
      {table_html(["Model", "Series", "Turn", "Session", "Session residual"], residual_rows)}
    </section>

    <section>
      <h2>二、平均分热力图与 12 个整体指标画像</h2>
      <p class="note">这部分对应原 dashboard 的平均分热力图和 12 指标模型画像。重点看同一模型内部指标差异，以及不同指标的整体难度。</p>
      <div class="two-col">
        <div>
          <h3>最难指标</h3>
          {bar_rows(metric_hardest, "label", "mean_avg", "best_model", max_value=5.0)}
        </div>
        <div>
          <h3>低分率最高的指标</h3>
          {bar_rows(lowrate_hardest, "label", "mean_low", "highest_low_model", max_value=1.0, low_is_bad=True)}
        </div>
      </div>
      <h3>难指标证据表</h3>
      {table_html(["指标", "跨模型均值", "模型间std", "平均低分率", "最高模型", "最高分", "最低模型", "最低分"], metric_rows)}
      <h3>高分模型也暴露出的最弱项</h3>
      {table_html(["Model", "Series", "全部任务加权", "Turn均值(12指标)", "Session均值(12指标)", "Gap", "最弱指标", "最弱分"], top_profile_table_rows)}
      <h3>指标内部落差最大的模型</h3>
      {table_html(["Model", "Series", "12指标Range", "最佳指标", "最佳分", "最弱指标", "最弱分"], profile_range_rows)}
    </section>

    <section>
      <h2>三、低分率：失败风险视角</h2>
      <p class="note">平均分回答“通常表现如何”，低分率回答“失败风险有多大”。两者结合能解释模型是否稳定。</p>
      {table_html(["指标", "平均低分率", "平均分", "最低低分率模型", "最低低分率", "最高低分率模型", "最高低分率"], low_rows)}
      <div class="callout">低分率最突出的不是自然度或简洁性，而是 Session 总体帮助可信、会话一致性、意图满足。这支持 benchmark 将 Turn-Level 与 Session-Level 分开评估。</div>
      <h3>低分率内部落差最大的模型</h3>
      {table_html(
        ["Model", "Series", "低分率Range", "最高低分率指标", "最高低分率", "最低低分率指标", "最低低分率"],
        [
          [esc(row["model"]), esc(row["series"]), fmt(row["low_range"]), esc(row["highest_low_metric"]), pct(row["highest_low"]), esc(row["lowest_low_metric"]), pct(row["lowest_low"])]
          for row in largest_low_ranges
        ],
      )}
    </section>

    <section>
      <h2>四、Primary Category：能力维度发现</h2>
      <p class="note">Secondary Category 暂不展开。Primary 部分重点看普遍难点、模型区分度，以及高总体模型的能力低谷。</p>
      <div class="two-col">
        <div>
          <h3>稳健样本量下最困难的能力</h3>
          {bar_rows(robust_hardest_abilities[:10], "ability", "mean", "worst_model", max_value=5.0)}
        </div>
        <div>
          <h3>稳健样本量下最能区分模型的能力</h3>
          {bar_rows(robust_discriminative_abilities[:10], "ability", "std", "best_model", max_value=max(row["std"] for row in robust_discriminative_abilities[:10]))}
        </div>
      </div>
      <h3>稳健样本量下较困难能力</h3>
      {table_html(["Ability", "Mean", "Std", "低于3模型数", "总样本", "Best", "Best分", "Worst", "Worst分"], ability_hard_rows)}
      <h3>稳健样本量下最有区分度能力</h3>
      {table_html(["Ability", "Std", "Range", "Mean", "总样本", "Best", "Best分", "Worst", "Worst分"], ability_disc_rows)}
      <h3>高 Primary 总体分但存在稳健能力低谷</h3>
      {table_html(["Model", "Series", "Primary加权", "低谷能力(sample>=10)"], robust_cliff_rows)}
      <h3>细粒度低谷提醒：样本较少，但适合做 qualitative case study</h3>
      {table_html(["Model", "Series", "Primary加权", "低谷能力"], fine_cliff_rows)}
      <div class="callout">论文可写法：Primary 维度说明总分不是充分解释。即使整体 Primary 加权分超过 4.0，模型仍可能在自我中心行为推理、物体交互、幻觉检测、实体位置排序等能力上出现明显低谷。</div>
    </section>

    <section>
      <h2>五、任务差异与 TTS 影响</h2>
      <p class="note">任务均分来自“关键看点速览”的最强/最弱任务表。这里用于辅助说明 benchmark 的任务覆盖带来的差异化。</p>
      {table_html(["Task", "模型数", "Top3", "Bottom3"], task_top_rows)}
      <h3>Gemini TTS 相对 text 的变化</h3>
      {table_html(["Model", "Modality", "Text", "TTS", "Delta"], tts_rows)}
      <div class="callout">TTS delta 全部为负，且 video TTS 的下降通常更明显。这说明同一视觉理解问题在不同输入通道下不是等价任务。</div>
    </section>

    <section>
      <h2>六、系列内比较</h2>
      <p class="note">系列内观察把上面的 Turn/Session、整体指标和任务均分压缩到同一视角，方便写成模型家族分析。</p>
      {table_html(["Series", "数量", "最佳", "最佳All", "最低", "最低All", "系列Range", "平均Turn-Session gap", "内部顺序"], series_rows)}
      <div class="callout">
        关键系列现象：GPT-5.4 总体与 Turn 更强，但 GPT-5 在 GPT 系列里 Session 均值最高；Claude Opus 4.6 是全局最稳的会话模型；Gemini 3 Flash 在 text 任务上非常强，但 TTS 仍下降；Qwen3-VL 随规模提升明显，但 Session gap 长期偏大，且 235B Instruct 高于 Thinking。
      </div>
    </section>

    <section>
      <h2>七、可直接转化为论文论点的角度</h2>
      <ol>
        <li>分层评测揭示了“表达质量”和“可靠完成任务”的分离：自然度、简洁性容易高分，但总体帮助可信和会话一致性仍是主要瓶颈。</li>
        <li>Turn-Level 与 Session-Level 是相关但不等价的评测层。单轮高分模型并不必然有好的会话稳定性。</li>
        <li>Primary 能力维度能解释总分相近模型之间的结构性差异，尤其是自我中心、空间/时序、物体交互和幻觉检测。</li>
        <li>多任务输入形式带来真实差异：Gemini 的 TTS 退化说明文本、图像、视频、语音式输入不应被同质化处理。</li>
        <li>系列内非单调结果说明 benchmark 对模型演进具有诊断价值，而不是只复述参数规模或版本新旧。</li>
      </ol>
    </section>
  </main>
</body>
</html>
"""
    OUT_HTML.write_text(html_doc, encoding="utf-8")

    print(f"Wrote {OUT_MD}")
    print(f"Wrote {OUT_HTML}")
    print(f"Wrote {OUT_JSON}")


if __name__ == "__main__":
    main()
