from __future__ import annotations

import html
from typing import Any, Dict, List, Sequence


def _text(value: Any) -> str:
    return "" if value is None else str(value)


def _fmt_score(value: Any, digits: int = 3) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except Exception:
        return _text(value)


def _join(values: Sequence[Any], sep: str = "、") -> str:
    items = [_text(value) for value in values if _text(value)]
    return sep.join(items) if items else "暂无"


def _md_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    if not rows:
        return "暂无数据。"
    header_line = "| " + " | ".join(headers) + " |"
    sep_line = "| " + " | ".join(["---"] * len(headers)) + " |"
    body_lines = ["| " + " | ".join(_text(cell) for cell in row) + " |" for row in rows]
    return "\n".join([header_line, sep_line] + body_lines)


def _html_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    if not rows:
        return "<p>暂无数据。</p>"
    thead = "".join(f"<th>{html.escape(_text(header))}</th>" for header in headers)
    body_rows = []
    for row in rows:
        cells = "".join(f"<td>{html.escape(_text(cell))}</td>" for cell in row)
        body_rows.append(f"<tr>{cells}</tr>")
    return f"<table><thead><tr>{thead}</tr></thead><tbody>{''.join(body_rows)}</tbody></table>"


def _metric_rows(metric_summary: Dict[str, Any]) -> List[List[Any]]:
    rows: List[List[Any]] = []
    for metric_name, payload in metric_summary.items():
        rows.append([metric_name, _fmt_score(payload.get("avg_score", 0.0)), _fmt_score(payload.get("min_score", 0.0)), _fmt_score(payload.get("max_score", 0.0)), int(payload.get("low_score_count", 0) or 0), _fmt_score(payload.get("low_score_rate", 0.0))])
    return rows


def _group_score_rows(group_summary: Dict[str, Any]) -> List[List[Any]]:
    rows: List[List[Any]] = []
    for name, payload in group_summary.items():
        turn = payload.get("turn", {})
        session = payload.get("session", {})
        rows.append([name, _fmt_score(turn.get("overall_avg_score", {}).get("avg_score", 0.0)), int(turn.get("record_count", 0) or 0), _fmt_score(session.get("overall_avg_score", {}).get("avg_score", 0.0)), int(session.get("record_count", 0) or 0)])
    rows.sort(key=lambda row: (row[1], row[3], row[0]), reverse=True)
    return rows


def _ability_rows(ability_summary: Dict[str, Any], key: str) -> List[List[Any]]:
    rows: List[List[Any]] = []
    for ability_name, payload in ability_summary.get(key, {}).items():
        metrics = payload.get("metrics", {})
        rows.append([ability_name, _fmt_score(payload.get("overall_avg_score", {}).get("avg_score", 0.0)), int(payload.get("record_count", 0) or 0), _fmt_score(metrics.get("accuracy", {}).get("avg_score", 0.0)), _fmt_score(metrics.get("proactiveness_helpfulness", {}).get("avg_score", 0.0)), _fmt_score(metrics.get("intent_understanding_depth", {}).get("avg_score", 0.0))])
    rows.sort(key=lambda row: (row[1], row[2], row[0]))
    return rows


def _quick_rank_rows(items: Sequence[Dict[str, Any]], key: str) -> List[List[Any]]:
    rows: List[List[Any]] = []
    for item in items:
        rows.append([item.get(key, ""), _fmt_score(item.get("avg_score", 0.0)), int(item.get("count", 0) or 0)])
    return rows


def _error_reason_task_rows(summary: Dict[str, Any], *, limit: int = 12) -> List[List[Any]]:
    rows: List[List[Any]] = []
    for task_name, payload in summary.get("by_task", {}).items():
        turn = payload.get("turn", {})
        session = payload.get("session", {})
        turn_primary = next(iter((turn.get("primary_error_category_counts") or {}).items()), ("暂无", 0))
        session_primary = next(iter((session.get("primary_error_category_counts") or {}).items()), ("暂无", 0))
        rows.append([task_name, int(turn.get("candidate_count", 0) or 0), turn_primary[0], turn_primary[1], int(session.get("candidate_count", 0) or 0), session_primary[0], session_primary[1]])
    rows.sort(key=lambda row: (-(int(row[1]) + int(row[4])), row[0]))
    return rows[:limit]


def _error_reason_mode_rows(summary: Dict[str, Any], key: str, *, limit: int = 12) -> List[List[Any]]:
    rows: List[List[Any]] = []
    for scope_name, payload in summary.get(key, {}).items():
        turn = payload.get("turn", {})
        primary = next(iter((turn.get("primary_error_category_counts") or {}).items()), ("暂无", 0))
        rows.append([scope_name, int(turn.get("candidate_count", 0) or 0), primary[0], primary[1], _join(list((turn.get("low_metric_counts") or {}).keys())[:3])])
    rows.sort(key=lambda row: (-int(row[1]), row[0]))
    return rows[:limit]


def _error_reason_ability_rows(summary: Dict[str, Any], key: str, *, limit: int = 12) -> List[List[Any]]:
    rows: List[List[Any]] = []
    for scope_name, payload in summary.get(key, {}).items():
        primary = next(iter((payload.get("primary_error_category_counts") or {}).items()), ("暂无", 0))
        rows.append([scope_name, int(payload.get("candidate_count", 0) or 0), primary[0], primary[1], _join(list((payload.get("low_metric_counts") or {}).keys())[:3])])
    rows.sort(key=lambda row: (-int(row[1]), row[0]))
    return rows[:limit]


def _metric_error_rows(summary: Dict[str, Any], level: str) -> List[List[Any]]:
    rows: List[List[Any]] = []
    for metric_name, payload in summary.get(level, {}).items():
        primary = next(iter((payload.get("primary_error_category_counts") or {}).items()), ("暂无", 0))
        rows.append([metric_name, int(payload.get("record_count", 0) or 0), primary[0], primary[1], _join(list((payload.get("task_counts") or {}).keys())[:3])])
    rows.sort(key=lambda row: (-int(row[1]), row[0]))
    return rows


def _high_score_rows(high_score_summary: Dict[str, Any]) -> List[List[Any]]:
    overall = high_score_summary.get("overall", {})
    return [["turn", int(overall.get("turn", {}).get("high_score_count", 0) or 0), _fmt_score(overall.get("turn", {}).get("high_score_rate", 0.0)), _join(list((overall.get("turn", {}).get("task_counts") or {}).keys())[:4]), _join(list((overall.get("turn", {}).get("media_mode_counts") or {}).keys())[:4])], ["session", int(overall.get("session", {}).get("high_score_count", 0) or 0), _fmt_score(overall.get("session", {}).get("high_score_rate", 0.0)), _join(list((overall.get("session", {}).get("task_counts") or {}).keys())[:4]), _join(list((overall.get("session", {}).get("media_mode_counts") or {}).keys())[:4])]]


def _simple_list_markdown(items: Sequence[str]) -> str:
    if not items:
        return "- 暂无"
    return "\n".join(f"- {item}" for item in items)


def _simple_list_html(items: Sequence[str]) -> str:
    if not items:
        return "<ul><li>暂无</li></ul>"
    return "<ul>" + "".join(f"<li>{html.escape(_text(item))}</li>" for item in items) + "</ul>"


def _metric_findings_markdown(items: Sequence[Dict[str, Any]]) -> str:
    if not items:
        return "暂无。"
    lines: List[str] = []
    for item in items:
        lines.extend([f"### {item.get('metric', '')}", item.get("assessment", ""), f"证据：{item.get('evidence', '')}", f"常见错误：{_join(item.get('common_errors', []))}", ""])
    return "\n".join(lines).strip()


def _metric_findings_html(items: Sequence[Dict[str, Any]]) -> str:
    if not items:
        return "<p>暂无。</p>"
    cards = []
    for item in items:
        cards.append("<div class='card'><h4>" + html.escape(_text(item.get('metric', ''))) + "</h4>" + f"<p>{html.escape(_text(item.get('assessment', '')))}</p>" + f"<p><strong>证据：</strong>{html.escape(_text(item.get('evidence', '')))}</p>" + f"<p><strong>常见错误：</strong>{html.escape(_join(item.get('common_errors', [])))}</p></div>")
    return "<div class='grid'>" + "".join(cards) + "</div>"


def _scoped_findings_markdown(items: Sequence[Dict[str, Any]], *, ability: bool = False) -> str:
    if not items:
        return "暂无。"
    lines: List[str] = []
    for item in items:
        lines.append(f"### {item.get('scope', '')}")
        if ability:
            lines.append(item.get("assessment", ""))
            lines.append(f"常见错误：{_join(item.get('common_errors', []))}")
        else:
            lines.append(f"优势：{item.get('strengths', '')}")
            lines.append(f"短板：{item.get('weaknesses', '')}")
        lines.append(f"证据：{item.get('evidence', '')}")
        lines.append("")
    return "\n".join(lines).strip()


def _scoped_findings_html(items: Sequence[Dict[str, Any]], *, ability: bool = False) -> str:
    if not items:
        return "<p>暂无。</p>"
    cards = []
    for item in items:
        if ability:
            body = [f"<p>{html.escape(_text(item.get('assessment', '')))}</p>", f"<p><strong>常见错误：</strong>{html.escape(_join(item.get('common_errors', [])))}</p>"]
        else:
            body = [f"<p><strong>优势：</strong>{html.escape(_text(item.get('strengths', '')))}</p>", f"<p><strong>短板：</strong>{html.escape(_text(item.get('weaknesses', '')))}</p>"]
        body.append(f"<p><strong>证据：</strong>{html.escape(_text(item.get('evidence', '')))}</p>")
        cards.append("<div class='card'><h4>" + html.escape(_text(item.get('scope', ''))) + "</h4>" + "".join(body) + "</div>")
    return "<div class='grid'>" + "".join(cards) + "</div>"

def _root_causes_markdown(items: Sequence[Dict[str, Any]]) -> str:
    if not items:
        return "暂无。"
    lines: List[str] = []
    for item in items:
        lines.extend([f"### {item.get('category', '')}", item.get("explanation", ""), f"影响指标：{_join(item.get('affected_metrics', []))}", f"影响范围：{_join(item.get('affected_scopes', []))}", f"证据：{item.get('evidence', '')}", ""])
    return "\n".join(lines).strip()


def _root_causes_html(items: Sequence[Dict[str, Any]]) -> str:
    if not items:
        return "<p>暂无。</p>"
    cards = []
    for item in items:
        cards.append("<div class='card'><h4>" + html.escape(_text(item.get('category', ''))) + "</h4>" + f"<p>{html.escape(_text(item.get('explanation', '')))}</p>" + f"<p><strong>影响指标：</strong>{html.escape(_join(item.get('affected_metrics', [])))}</p>" + f"<p><strong>影响范围：</strong>{html.escape(_join(item.get('affected_scopes', [])))}</p>" + f"<p><strong>证据：</strong>{html.escape(_text(item.get('evidence', '')))}</p></div>")
    return "<div class='grid'>" + "".join(cards) + "</div>"


def _recommendations_markdown(items: Sequence[Dict[str, Any]]) -> str:
    if not items:
        return "暂无建议。"
    lines: List[str] = []
    for item in items:
        lines.extend([f"### {item.get('priority', '')} {item.get('title', '')}", f"原因：{item.get('rationale', '')}", f"具体动作：{_join(item.get('actions', []), '；')}", f"预期收益：{item.get('expected_gain', '')}", f"目标指标：{_join(item.get('target_metrics', []))}", f"目标范围：{_join(item.get('target_scopes', []))}", ""])
    return "\n".join(lines).strip()


def _recommendations_html(items: Sequence[Dict[str, Any]]) -> str:
    if not items:
        return "<p>暂无建议。</p>"
    cards = []
    for item in items:
        cards.append("<div class='card'><h4>" + html.escape(_text(item.get('priority', ''))) + " " + html.escape(_text(item.get('title', ''))) + "</h4>" + f"<p><strong>原因：</strong>{html.escape(_text(item.get('rationale', '')))}</p>" + f"<div class='sublist'><strong>具体动作</strong>{_simple_list_html(item.get('actions', []))}</div>" + f"<p><strong>预期收益：</strong>{html.escape(_text(item.get('expected_gain', '')))}</p>" + f"<p><strong>目标指标：</strong>{html.escape(_join(item.get('target_metrics', [])))}</p>" + f"<p><strong>目标范围：</strong>{html.escape(_join(item.get('target_scopes', [])))}</p></div>")
    return "<div class='grid'>" + "".join(cards) + "</div>"


def _case_lookup(representative_cases: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    lookup: Dict[str, Dict[str, Any]] = {}
    for group_name in ["low_turn_cases", "low_session_cases", "high_turn_cases", "high_session_cases"]:
        for item in representative_cases.get(group_name, []):
            for case_id in [
                _text(item.get("case_id")),
                _text(item.get("candidate_id")),
                _text(item.get("judgement_id")),
                _text(item.get("dialogue_id")),
                _text(item.get("round_id")),
            ]:
                if case_id:
                    lookup[case_id] = item
    return lookup

def _case_sections_markdown(rep: Dict[str, Any], representative_cases: Dict[str, Any]) -> str:
    lookup = _case_lookup(representative_cases)

    def _block(title: str, items: Sequence[Dict[str, Any]]) -> List[str]:
        lines: List[str] = [f"### {title}", ""]
        if not items:
            lines.extend(["暂无。", ""])
            return lines
        for item in items:
            detail = lookup.get(_text(item.get("case_id")), {})
            is_session_case = "session" in title
            lines.extend([
                f"#### {item.get('case_id', '')}",
                f"- 代表原因：{item.get('why_representative', '')}",
                f"- 案例启示：{item.get('lesson', '')}",
                f"- 任务/模态：{detail.get('task_name', '')} / {detail.get('task_mode', '')} / {detail.get('media_mode', '')} / {detail.get('question_mode', '')}",
                f"- 评分概览：avg_score={_fmt_score(detail.get('avg_score', 0.0))}，主错误={detail.get('primary_error_category', '暂无')}，低分指标={_join(detail.get('low_score_metrics', []))}",
            ])
            if is_session_case:
                preview = _join(detail.get('dialogue_preview', []), '；')
                signals = _join(detail.get('key_dialogue_signals', []), '；')
                if preview != '暂无':
                    lines.append(f"- 对话预览：{preview}")
                if signals != '暂无':
                    lines.append(f"- 关键信号：{signals}")
            else:
                lines.extend([
                    f"- 用户问题：{detail.get('question_text', '暂无')}",
                    f"- 参考答案：{detail.get('reference_answer', '暂无')}",
                    f"- 模型回答：{detail.get('prediction', '暂无')}",
                ])
            lines.extend([
                f"- 归因摘要：{detail.get('attribution_summary') or detail.get('overall_summary') or '暂无'}",
                "",
            ])
        return lines

    lines: List[str] = []
    lines.extend(_block("代表性低分单轮案例", rep.get("low_turn_cases", [])))
    lines.extend(_block("代表性低分整段案例", rep.get("low_session_cases", [])))
    lines.extend(_block("代表性高分单轮案例", rep.get("high_turn_cases", [])))
    lines.extend(_block("代表性高分整段案例", rep.get("high_session_cases", [])))
    return "\n".join(lines).strip()

def _case_sections_html(rep: Dict[str, Any], representative_cases: Dict[str, Any]) -> str:
    lookup = _case_lookup(representative_cases)

    def _cards(title: str, items: Sequence[Dict[str, Any]]) -> str:
        if not items:
            return f"<h3>{html.escape(title)}</h3><p>暂无。</p>"
        cards = []
        for item in items:
            detail = lookup.get(_text(item.get("case_id")), {})
            is_session_case = "session" in title
            body = [
                f"<p><strong>代表原因：</strong>{html.escape(_text(item.get('why_representative', '')))}</p>",
                f"<p><strong>案例启示：</strong>{html.escape(_text(item.get('lesson', '')))}</p>",
                f"<p><strong>任务/模态：</strong>{html.escape(_text(detail.get('task_name', '')))} / {html.escape(_text(detail.get('task_mode', '')))} / {html.escape(_text(detail.get('media_mode', '')))} / {html.escape(_text(detail.get('question_mode', '')))}</p>",
                f"<p><strong>评分概览：</strong>avg_score={html.escape(_fmt_score(detail.get('avg_score', 0.0)))}，主错误={html.escape(_text(detail.get('primary_error_category', '暂无')))}，低分指标={html.escape(_join(detail.get('low_score_metrics', [])))}</p>",
            ]
            if is_session_case:
                preview = _join(detail.get('dialogue_preview', []), '；')
                signals = _join(detail.get('key_dialogue_signals', []), '；')
                if preview != '暂无':
                    body.append(f"<p><strong>对话预览：</strong>{html.escape(preview)}</p>")
                if signals != '暂无':
                    body.append(f"<p><strong>关键信号：</strong>{html.escape(signals)}</p>")
            else:
                body.extend([
                    f"<p><strong>用户问题：</strong>{html.escape(_text(detail.get('question_text', '暂无')))}</p>",
                    f"<p><strong>参考答案：</strong>{html.escape(_text(detail.get('reference_answer', '暂无')))}</p>",
                    f"<p><strong>模型回答：</strong>{html.escape(_text(detail.get('prediction', '暂无')))}</p>",
                ])
            body.append(f"<p><strong>归因摘要：</strong>{html.escape(_text(detail.get('attribution_summary') or detail.get('overall_summary') or '暂无'))}</p>")
            cards.append("<div class='card'><h4>" + html.escape(_text(item.get('case_id', ''))) + "</h4>" + "".join(body) + "</div>")
        return f"<h3>{html.escape(title)}</h3><div class='grid'>{''.join(cards)}</div>"

    return ''.join([_cards("代表性低分单轮案例", rep.get("low_turn_cases", [])), _cards("代表性低分整段案例", rep.get("low_session_cases", [])), _cards("代表性高分单轮案例", rep.get("high_turn_cases", [])), _cards("代表性高分整段案例", rep.get("high_session_cases", []))])

def _file_guide_markdown(manifest: Dict[str, Any]) -> str:
    rows = [["report_payload.json", "Phase 4 全量证据快照", manifest.get("report_payload_path", "")], ["report_digest.json", "给分析模型看的高信息密度摘要", manifest.get("report_digest_path", "")], ["report_step_results.json", "分步分析的逐步状态、原始输出与结构化结果", manifest.get("report_step_results_path", "")], ["report_analysis_raw.txt", "按步骤拼接的模型原始输出", manifest.get("report_raw_text_path", "")], ["report_analysis.json", "最终结构化分析结果", manifest.get("report_analysis_path", "")], ["benchmark_report.md/html", "最终可读报告", manifest.get("report_markdown_path", "")]]
    return _md_table(["文件", "作用", "路径"], rows)


def _file_guide_html(manifest: Dict[str, Any]) -> str:
    rows = [["report_payload.json", "Phase 4 全量证据快照", manifest.get("report_payload_path", "")], ["report_digest.json", "给分析模型看的高信息密度摘要", manifest.get("report_digest_path", "")], ["report_step_results.json", "分步分析的逐步状态、原始输出与结构化结果", manifest.get("report_step_results_path", "")], ["report_analysis_raw.txt", "按步骤拼接的模型原始输出", manifest.get("report_raw_text_path", "")], ["report_analysis.json", "最终结构化分析结果", manifest.get("report_analysis_path", "")], ["benchmark_report.md/html", "最终可读报告", manifest.get("report_markdown_path", "")]]
    return _html_table(["文件", "作用", "路径"], rows)


def build_markdown_report(manifest: Dict[str, Any], payload: Dict[str, Any], analysis: Dict[str, Any]) -> str:
    score_summary = payload.get("score_summary", {})
    ability_summary = payload.get("ability_score_summary", {})
    error_reason_by_task = payload.get("error_reason_by_task_summary", {})
    error_reason_by_mode = payload.get("error_reason_by_mode_summary", {})
    error_reason_by_ability = payload.get("error_reason_by_ability_summary", {})
    metric_error_cross = payload.get("metric_error_cross_summary", {})
    high_score_summary = payload.get("high_score_summary", {})
    representative_cases = payload.get("representative_cases", {})
    validation_summary = payload.get("validation_summary", {})
    benchmark_overview = analysis.get("benchmark_overview", {})
    overall_assessment = analysis.get("overall_assessment", {})
    quick = payload.get("quick_insights", {})
    analysis_relation = "相同" if manifest.get("analysis_vs_tested_same_model") else "不同"

    sections: List[str] = ["# RUBRIC-MME 自动分析报告", ""]
    if str(manifest.get("analysis_status", "") or "") != "success":
        sections.extend([f"> 注意：本次报告分析状态为 `{manifest.get('analysis_status', '')}`，正文中包含回退分析内容。", ""])
    sections.extend([
        "## 阅读导航",
        "- [一、执行摘要](#一执行摘要)",
        "- [二、评测与分析设置](#二评测与分析设置)",
        "- [三、关键看点速览](#三关键看点速览)",
        "- [四、整体分数概览](#四整体分数概览)",
        "- [五、Turn-Level 详细分析](#五turn-level-详细分析)",
        "- [六、Session-Level 详细分析](#六session-level-详细分析)",
        "- [七、任务与模态分析](#七任务与模态分析)",
        "- [八、能力维度分析](#八能力维度分析)",
        "- [九、低分原因与交叉统计](#九低分原因与交叉统计)",
        "- [十、高分对照统计](#十高分对照统计)",
        "- [十一、代表性案例分析](#十一代表性案例分析)",
        "- [十二、改进建议与路线图](#十二改进建议与路线图)",
        "- [十三、产物说明](#十三产物说明)",
        "",
        "## 一、执行摘要",
        analysis.get("executive_summary", "暂无。"),
        "",
        "## 二、评测与分析设置",
        _md_table(["项目", "内容"], [["被测模型", manifest.get("tested_model_name", "")], ["被测模型提供方", manifest.get("tested_provider", "")], ["分析模型", manifest.get("analysis_model_name", "")], ["分析后端", manifest.get("analysis_backend", "")], ["模型关系", f"被测模型与分析模型{analysis_relation}"], ["分析状态", manifest.get("analysis_status", "")], ["分析阶段数", manifest.get("analysis_stage_count", 0)], ["生成时间", manifest.get("generated_at", "")]]),
        "",
        f"- 评测范围：{benchmark_overview.get('scope_summary', '暂无。')}",
        f"- 覆盖情况：{benchmark_overview.get('coverage_summary', '暂无。')}",
        f"- 结果解读：{benchmark_overview.get('evaluation_note', '暂无。')}",
        f"- Phase 4 覆盖：turn={validation_summary.get('expected_turn_candidate_count', 0)}，session={validation_summary.get('expected_session_candidate_count', 0)}，错误记录={validation_summary.get('error_record_count', 0)}",
        "",
        "## 三、关键看点速览",
        f"- 综合结论：{overall_assessment.get('verdict', '暂无。')}",
        f"- Turn-Level：{overall_assessment.get('turn_level_summary', '暂无。')}",
        f"- Session-Level：{overall_assessment.get('session_level_summary', '暂无。')}",
        f"- 主要强信号：{_join(overall_assessment.get('strongest_signals', []))}",
        f"- 主要弱信号：{_join(overall_assessment.get('weakest_signals', []))}",
        "",
        "### 最强任务 Top 5",
        _md_table(["任务", "均分", "样本数"], _quick_rank_rows(quick.get("best_tasks", []), "task_name")),
        "",
        "### 最弱任务 Top 5",
        _md_table(["任务", "均分", "样本数"], _quick_rank_rows(quick.get("weakest_tasks", []), "task_name")),
        "",
        "## 四、整体分数概览",
        "### Turn-Level 指标",
        _md_table(["指标", "平均分", "最低分", "最高分", "低分数", "低分率"], _metric_rows(score_summary.get("overall", {}).get("turn", {}).get("metrics", {}))),
        "",
        "### Session-Level 指标",
        _md_table(["指标", "平均分", "最低分", "最高分", "低分数", "低分率"], _metric_rows(score_summary.get("overall", {}).get("session", {}).get("metrics", {}))),
        "",
        "## 五、Turn-Level 详细分析",
        _metric_findings_markdown(analysis.get("turn_level_findings", [])),
        "",
        "## 六、Session-Level 详细分析",
        _metric_findings_markdown(analysis.get("session_level_findings", [])),
        "",
        "## 七、任务与模态分析",
        "### 任务维度",
        _md_table(["任务", "Turn 均分", "Turn 样本数", "Session 均分", "Session 样本数"], _group_score_rows(score_summary.get("by_task", {}))),
        "",
        _scoped_findings_markdown(analysis.get("task_findings", []), ability=False),
        "",
        "### Task Mode",
        _md_table(["模式", "Turn 均分", "Turn 样本数", "Session 均分", "Session 样本数"], _group_score_rows(score_summary.get("by_task_mode", {}))),
        "",
        "### Question Mode",
        _md_table(["模式", "Turn 均分", "Turn 样本数", "Session 均分", "Session 样本数"], _group_score_rows(score_summary.get("by_question_mode", {}))),
        "",
        "### Media Mode",
        _md_table(["模式", "Turn 均分", "Turn 样本数", "Session 均分", "Session 样本数"], _group_score_rows(score_summary.get("by_media_mode", {}))),
        "",
        _scoped_findings_markdown(analysis.get("mode_findings", []), ability=False),
        "",
        "## 八、能力维度分析",
        "### Primary Category",
        _md_table(["能力", "整体均分", "样本数", "Accuracy", "Proactiveness", "Intent Depth"], _ability_rows(ability_summary, "by_primary_category")),
        "",
        "### Secondary Category",
        _md_table(["能力", "整体均分", "样本数", "Accuracy", "Proactiveness", "Intent Depth"], _ability_rows(ability_summary, "by_secondary_category")),
        "",
        _scoped_findings_markdown(analysis.get("ability_findings", []), ability=True),
        "",
        "## 九、低分原因与交叉统计",
        "### 各任务的主要低分原因",
        _md_table(["任务", "Turn 候选数", "Turn 主错误", "次数", "Session 候选数", "Session 主错误", "次数"], _error_reason_task_rows(error_reason_by_task)),
        "",
        "### 各 Task Mode 的主要低分原因",
        _md_table(["模式", "Turn 候选数", "主错误", "次数", "常见低分指标"], _error_reason_mode_rows(error_reason_by_mode, "by_task_mode")),
        "",
        "### 能力层的主要低分原因",
        _md_table(["能力", "候选数", "主错误", "次数", "常见低分指标"], _error_reason_ability_rows(error_reason_by_ability, "by_primary_category")),
        "",
        "### 指标与错误类型交叉",
        _md_table(["指标", "关联记录数", "主错误", "次数", "常见任务"], _metric_error_rows(metric_error_cross, "turn")),
        "",
        _root_causes_markdown(analysis.get("root_causes", [])),
        "",
        "## 十、高分对照统计",
        _md_table(["层级", "高分样本数", "高分率", "主要任务", "主要模态"], _high_score_rows(high_score_summary)),
        "",
        "## 十一、代表性案例分析",
        _case_sections_markdown(analysis.get("representative_case_analysis", {}), representative_cases),
        "",
        "## 十二、改进建议与路线图",
        _recommendations_markdown(analysis.get("recommendations", [])),
        "",
        "### P0",
        _simple_list_markdown(analysis.get("priority_roadmap", {}).get("p0", [])),
        "",
        "### P1",
        _simple_list_markdown(analysis.get("priority_roadmap", {}).get("p1", [])),
        "",
        "### P2",
        _simple_list_markdown(analysis.get("priority_roadmap", {}).get("p2", [])),
        "",
        "## 十三、产物说明",
        _file_guide_markdown(manifest),
        "",
        "## 十四、结论",
        analysis.get("report_closing", "暂无。"),
        "",
    ])
    return "\n".join(sections)


def build_html_report(manifest: Dict[str, Any], payload: Dict[str, Any], analysis: Dict[str, Any]) -> str:
    score_summary = payload.get("score_summary", {})
    ability_summary = payload.get("ability_score_summary", {})
    error_reason_by_task = payload.get("error_reason_by_task_summary", {})
    error_reason_by_mode = payload.get("error_reason_by_mode_summary", {})
    error_reason_by_ability = payload.get("error_reason_by_ability_summary", {})
    metric_error_cross = payload.get("metric_error_cross_summary", {})
    high_score_summary = payload.get("high_score_summary", {})
    representative_cases = payload.get("representative_cases", {})
    validation_summary = payload.get("validation_summary", {})
    benchmark_overview = analysis.get("benchmark_overview", {})
    overall_assessment = analysis.get("overall_assessment", {})
    analysis_status = str(manifest.get("analysis_status", "") or "")
    analysis_relation = "相同" if manifest.get("analysis_vs_tested_same_model") else "不同"

    styles = """
    body{font-family:'Segoe UI','PingFang SC','Microsoft YaHei',sans-serif;background:#f6f4ee;color:#172033;margin:0;padding:0;}
    .wrap{max-width:1360px;margin:0 auto;padding:28px 20px 48px;}
    .hero{background:linear-gradient(135deg,#0f766e,#1d4ed8);color:#fff;border-radius:22px;padding:30px 32px;margin-bottom:20px;box-shadow:0 20px 40px rgba(15,118,110,.18);} .hero h1{margin:0 0 16px;font-size:34px;line-height:1.2;}
    .meta,.nav,.cards,.grid{display:grid;gap:14px;} .meta{grid-template-columns:repeat(auto-fit,minmax(240px,1fr));font-size:14px;} .nav{grid-template-columns:repeat(auto-fit,minmax(180px,1fr));margin-bottom:18px;} .cards{grid-template-columns:repeat(auto-fit,minmax(220px,1fr));margin-bottom:18px;} .grid{grid-template-columns:repeat(auto-fit,minmax(300px,1fr));}
    .nav a,.mini-card,.section,.card{background:#fff;border-radius:18px;box-shadow:0 8px 24px rgba(15,23,42,.06);} .nav a{padding:12px 14px;color:#0f172a;text-decoration:none;font-weight:600;} .mini-card{padding:16px 18px;} .mini-card .label{font-size:13px;color:#64748b;margin-bottom:6px;} .mini-card .value{font-size:26px;font-weight:700;color:#0f172a;}
    .section{padding:22px 24px;margin-bottom:18px;} .card{padding:14px 16px;border:1px solid #e5e7eb;background:#fcfcfb;}
    .warn{background:#fff7ed;border:1px solid #fdba74;color:#9a3412;border-radius:14px;padding:14px 16px;margin-bottom:20px;}
    h2{margin:0 0 14px;font-size:22px;} h3{margin:18px 0 10px;font-size:18px;} h4{margin:0 0 8px;font-size:16px;} p{line-height:1.7;margin:8px 0;}
    table{width:100%;border-collapse:collapse;margin-top:8px;font-size:13px;} th,td{border:1px solid #e5e7eb;padding:8px 10px;text-align:left;vertical-align:top;} th{background:#f3f4f6;} .mono{font-family:Consolas,Monaco,monospace;}
    """

    summary_cards = [["Turn 低分候选", payload.get("phase4_manifest", {}).get("turn_candidate_count", 0)], ["Session 低分候选", payload.get("phase4_manifest", {}).get("session_candidate_count", 0)], ["高分 Turn 样本", high_score_summary.get("overall", {}).get("turn", {}).get("high_score_count", 0)], ["高分 Session 样本", high_score_summary.get("overall", {}).get("session", {}).get("high_score_count", 0)]]
    cards_html = "".join(f"<div class='mini-card'><div class='label'>{html.escape(_text(label))}</div><div class='value'>{html.escape(_text(value))}</div></div>" for label, value in summary_cards)
    warning_html = ""
    if analysis_status != "success":
        warning_html = f"<div class='warn'><strong>提示：</strong>本次报告分析状态为 <span class='mono'>{html.escape(analysis_status)}</span>，当前报告包含回退分析内容。</div>"
    nav_links = [("summary", "执行摘要"), ("setup", "评测设置"), ("overview", "关键看点"), ("scores", "整体分数"), ("turn", "Turn 分析"), ("session", "Session 分析"), ("task", "任务与模态"), ("ability", "能力分析"), ("errors", "低分原因"), ("high", "高分对照"), ("cases", "代表性案例"), ("roadmap", "建议路线图"), ("files", "产物说明")]
    nav_html = "".join(f"<a href='#{anchor}'>{html.escape(title)}</a>" for anchor, title in nav_links)

    return "".join([
        "<!DOCTYPE html><html lang='zh-CN'><head><meta charset='utf-8'>", "<title>RUBRIC-MME 自动分析报告</title>", f"<style>{styles}</style></head><body><div class='wrap'>",
        "<section class='hero'><h1>RUBRIC-MME 自动分析报告</h1><div class='meta'>",
        f"<div>被测模型：<span class='mono'>{html.escape(_text(manifest.get('tested_model_name', '')))}</span></div>",
        f"<div>被测模型提供方：<span class='mono'>{html.escape(_text(manifest.get('tested_provider', '')))}</span></div>",
        f"<div>分析模型：<span class='mono'>{html.escape(_text(manifest.get('analysis_model_name', '')))}</span></div>",
        f"<div>分析后端：<span class='mono'>{html.escape(_text(manifest.get('analysis_backend', '')))}</span></div>",
        f"<div>模型关系：<span class='mono'>被测模型与分析模型{html.escape(analysis_relation)}</span></div>",
        f"<div>分析状态：<span class='mono'>{html.escape(analysis_status)}</span></div>",
        "</div></section>", warning_html, f"<div class='nav'>{nav_html}</div>", f"<div class='cards'>{cards_html}</div>",
        f"<section class='section' id='summary'><h2>执行摘要</h2><p>{html.escape(_text(analysis.get('executive_summary', '暂无。')))}</p></section>",
        "<section class='section' id='setup'><h2>评测与分析设置</h2>",
        _html_table(["项目", "内容"], [["被测模型", manifest.get("tested_model_name", "")], ["被测模型提供方", manifest.get("tested_provider", "")], ["分析模型", manifest.get("analysis_model_name", "")], ["分析后端", manifest.get("analysis_backend", "")], ["模型关系", f"被测模型与分析模型{analysis_relation}"], ["分析状态", analysis_status], ["分析阶段数", manifest.get("analysis_stage_count", 0)], ["生成时间", manifest.get("generated_at", "")]]),
        f"<p><strong>评测范围：</strong>{html.escape(_text(benchmark_overview.get('scope_summary', '暂无。')))}</p>", f"<p><strong>覆盖情况：</strong>{html.escape(_text(benchmark_overview.get('coverage_summary', '暂无。')))}</p>", f"<p><strong>结果解读：</strong>{html.escape(_text(benchmark_overview.get('evaluation_note', '暂无。')))}</p>", f"<p><strong>Phase 4 覆盖：</strong>turn={html.escape(_text(validation_summary.get('expected_turn_candidate_count', 0)))}，session={html.escape(_text(validation_summary.get('expected_session_candidate_count', 0)))}，错误记录={html.escape(_text(validation_summary.get('error_record_count', 0)))}</p></section>",
        "<section class='section' id='overview'><h2>关键看点速览</h2>", f"<p><strong>综合结论：</strong>{html.escape(_text(overall_assessment.get('verdict', '暂无。')))}</p>", f"<p><strong>Turn-Level：</strong>{html.escape(_text(overall_assessment.get('turn_level_summary', '暂无。')))}</p>", f"<p><strong>Session-Level：</strong>{html.escape(_text(overall_assessment.get('session_level_summary', '暂无。')))}</p>", f"<div class='grid'><div class='card'><h4>主要强信号</h4>{_simple_list_html(overall_assessment.get('strongest_signals', []))}</div><div class='card'><h4>主要弱信号</h4>{_simple_list_html(overall_assessment.get('weakest_signals', []))}</div></div>", "<h3>任务快速排名</h3>", _html_table(["任务", "均分", "样本数"], _quick_rank_rows(payload.get("quick_insights", {}).get("weakest_tasks", []), "task_name")), "</section>",
        f"<section class='section' id='scores'><h2>整体分数概览</h2><h3>Turn-Level 指标</h3>{_html_table(['指标', '平均分', '最低分', '最高分', '低分数', '低分率'], _metric_rows(score_summary.get('overall', {}).get('turn', {}).get('metrics', {})))}<h3>Session-Level 指标</h3>{_html_table(['指标', '平均分', '最低分', '最高分', '低分数', '低分率'], _metric_rows(score_summary.get('overall', {}).get('session', {}).get('metrics', {})))}</section>",
        f"<section class='section' id='turn'><h2>Turn-Level 详细分析</h2>{_metric_findings_html(analysis.get('turn_level_findings', []))}</section>",
        f"<section class='section' id='session'><h2>Session-Level 详细分析</h2>{_metric_findings_html(analysis.get('session_level_findings', []))}</section>",
        f"<section class='section' id='task'><h2>任务与模态分析</h2><h3>任务维度</h3>{_html_table(['任务', 'Turn 均分', 'Turn 样本数', 'Session 均分', 'Session 样本数'], _group_score_rows(score_summary.get('by_task', {})))}{_scoped_findings_html(analysis.get('task_findings', []), ability=False)}<h3>Task Mode</h3>{_html_table(['模式', 'Turn 均分', 'Turn 样本数', 'Session 均分', 'Session 样本数'], _group_score_rows(score_summary.get('by_task_mode', {})))}<h3>Question Mode</h3>{_html_table(['模式', 'Turn 均分', 'Turn 样本数', 'Session 均分', 'Session 样本数'], _group_score_rows(score_summary.get('by_question_mode', {})))}<h3>Media Mode</h3>{_html_table(['模式', 'Turn 均分', 'Turn 样本数', 'Session 均分', 'Session 样本数'], _group_score_rows(score_summary.get('by_media_mode', {})))}{_scoped_findings_html(analysis.get('mode_findings', []), ability=False)}</section>",
        f"<section class='section' id='ability'><h2>能力维度分析</h2><h3>Primary Category</h3>{_html_table(['能力', '整体均分', '样本数', 'Accuracy', 'Proactiveness', 'Intent Depth'], _ability_rows(ability_summary, 'by_primary_category'))}<h3>Secondary Category</h3>{_html_table(['能力', '整体均分', '样本数', 'Accuracy', 'Proactiveness', 'Intent Depth'], _ability_rows(ability_summary, 'by_secondary_category'))}{_scoped_findings_html(analysis.get('ability_findings', []), ability=True)}</section>",
        f"<section class='section' id='errors'><h2>低分原因与交叉统计</h2><h3>各任务的主要低分原因</h3>{_html_table(['任务', 'Turn 候选数', 'Turn 主错误', '次数', 'Session 候选数', 'Session 主错误', '次数'], _error_reason_task_rows(error_reason_by_task))}<h3>各 Task Mode 的主要低分原因</h3>{_html_table(['模式', 'Turn 候选数', '主错误', '次数', '常见低分指标'], _error_reason_mode_rows(error_reason_by_mode, 'by_task_mode'))}<h3>能力层的主要低分原因</h3>{_html_table(['能力', '候选数', '主错误', '次数', '常见低分指标'], _error_reason_ability_rows(error_reason_by_ability, 'by_primary_category'))}<h3>指标与错误类型交叉</h3>{_html_table(['指标', '关联记录数', '主错误', '次数', '常见任务'], _metric_error_rows(metric_error_cross, 'turn'))}{_root_causes_html(analysis.get('root_causes', []))}</section>",
        f"<section class='section' id='high'><h2>高分对照统计</h2>{_html_table(['层级', '高分样本数', '高分率', '主要任务', '主要模态'], _high_score_rows(high_score_summary))}</section>",
        f"<section class='section' id='cases'><h2>代表性案例分析</h2>{_case_sections_html(analysis.get('representative_case_analysis', {}), representative_cases)}</section>",
        f"<section class='section' id='roadmap'><h2>改进建议与路线图</h2>{_recommendations_html(analysis.get('recommendations', []))}<div class='grid'><div class='card'><h4>P0</h4>{_simple_list_html(analysis.get('priority_roadmap', {}).get('p0', []))}</div><div class='card'><h4>P1</h4>{_simple_list_html(analysis.get('priority_roadmap', {}).get('p1', []))}</div><div class='card'><h4>P2</h4>{_simple_list_html(analysis.get('priority_roadmap', {}).get('p2', []))}</div></div><p>{html.escape(_text(analysis.get('report_closing', '暂无。')))}</p></section>",
        f"<section class='section' id='files'><h2>产物说明</h2>{_file_guide_html(manifest)}</section>",
        "</div></body></html>",
    ])



