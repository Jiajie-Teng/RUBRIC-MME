from __future__ import annotations

import json
from typing import Any, Dict, Sequence

from error_taxonomy import (
    SESSION_ERROR_CATEGORIES_CN,
    TURN_ERROR_CATEGORIES_CN,
    build_taxonomy_prompt_block,
)

TURN_PROMPT_VERSION = "rubric_mme_phase4_turn_cn_v1"
SESSION_PROMPT_VERSION = "rubric_mme_phase4_session_cn_v1"


def _json_block(payload: Dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2)


def build_turn_attribution_prompt(candidate: Dict[str, Any]) -> str:
    taxonomy_block = build_taxonomy_prompt_block(TURN_ERROR_CATEGORIES_CN)
    observed_payload = {
        "task_name": candidate.get("task_name", ""),
        "task_mode": candidate.get("task_mode", ""),
        "question_mode": candidate.get("question_mode", ""),
        "media_mode": candidate.get("media_mode", ""),
        "dialogue_id": candidate.get("dialogue_id", ""),
        "round_id": candidate.get("round_id", ""),
        "round_index": candidate.get("round_index", 0),
        "primary_category": candidate.get("primary_category", ""),
        "secondary_categories": candidate.get("secondary_categories", []),
        "severity": candidate.get("severity", ""),
        "status": candidate.get("status", ""),
        "trigger_reasons": candidate.get("trigger_reasons", []),
        "low_score_metrics": candidate.get("low_score_metrics", []),
        "critical_metrics": candidate.get("critical_metrics", []),
        "avg_score": candidate.get("avg_score"),
        "score_vector": candidate.get("score_vector", {}),
        "reason_vector": candidate.get("reason_vector", {}),
        "overall_summary": candidate.get("overall_summary", ""),
        "question_text": candidate.get("question_text", ""),
        "reference_answer": candidate.get("reference_answer", ""),
        "prediction": candidate.get("prediction", ""),
        "error": candidate.get("error", ""),
        "error_type": candidate.get("error_type", ""),
    }
    return f"""
你是 RUBRIC-MME 第四阶段的低分归因专家。你需要对一个单轮低分样本做错误原因打标。

任务目标：
1. 结合问题、参考答案、被测模型回答、第三阶段评分结果及其理由，判断低分最主要的错误原因；
2. 必须从给定的 turn-level 错误原因集合中选择，不允许发明集合外标签；
3. 输出一个主错误大类，并从该大类下选择 1 到 3 个最合适的二级错误原因；
4. 同时指出最受影响的评分维度，并给出简洁、可落地的归因总结。

判定原则：
- 第三阶段的 score_vector、reason_vector、overall_summary 是重要证据；
- 参考答案仍然是主要参照，但如果被测模型回答包含参考答案未显式写出、却与题意和评分理由一致的有效信息，不要机械误判；
- 不要把多个含义高度重复的错误原因同时贴上；
- 如果样本本身是 api_error / parse_error / skipped 类失败，也要从现有标签中选择最贴近的原因，并在 summary 中明确指出是流程失败还是回答质量问题。

可选错误原因集合（turn-level）：
{taxonomy_block}

待归因样本：
{_json_block(observed_payload)}

请只返回 JSON，不要输出 markdown。JSON 字段必须包含：
- primary_error_category: 字符串，必须是上面集合中的一级大类
- secondary_error_categories: 字符串数组，长度 1 到 3，且必须都属于该一级大类
- affected_metrics: 字符串数组，从当前样本的评分维度中选择最相关的 1 到 4 个
- attribution_summary: 字符串，2 到 4 句中文，概括低分主要原因及其与评分的关系
""".strip()



def build_session_attribution_prompt(candidate: Dict[str, Any], dialogue_context: Dict[str, Any] | None = None) -> str:
    taxonomy_block = build_taxonomy_prompt_block(SESSION_ERROR_CATEGORIES_CN)
    dialogue_context = dialogue_context or {}
    observed_payload = {
        "task_name": candidate.get("task_name", ""),
        "task_mode": candidate.get("task_mode", ""),
        "question_mode": candidate.get("question_mode", ""),
        "media_mode": candidate.get("media_mode", ""),
        "dialogue_id": candidate.get("dialogue_id", ""),
        "round_count": candidate.get("round_count", 0),
        "severity": candidate.get("severity", ""),
        "status": candidate.get("status", ""),
        "trigger_reasons": candidate.get("trigger_reasons", []),
        "low_score_metrics": candidate.get("low_score_metrics", []),
        "critical_metrics": candidate.get("critical_metrics", []),
        "avg_score": candidate.get("avg_score"),
        "score_vector": candidate.get("score_vector", {}),
        "reason_vector": candidate.get("reason_vector", {}),
        "overall_summary": candidate.get("overall_summary", ""),
        "key_dialogue_signals": candidate.get("key_dialogue_signals", []),
        "error": candidate.get("error", ""),
        "error_type": candidate.get("error_type", ""),
        "interaction_goal": dialogue_context.get("interaction_goal", {}),
        "user_persona": dialogue_context.get("user_persona", {}),
    }
    return f"""
你是 RUBRIC-MME 第四阶段的低分归因专家。你需要对一个整段对话的低分样本做错误原因打标。

任务目标：
1. 结合整段对话的 session-level 评分结果及其理由，找出导致整段分数偏低的主要原因；
2. 必须从给定的 session-level 错误原因集合中选择，不允许发明集合外标签；
3. 输出一个主错误大类，并从该大类下选择 1 到 3 个最合适的二级错误原因；
4. 结合 interaction_goal、user_persona、overall_summary、key_dialogue_signals 判断是“目标推进问题”“连贯性问题”还是“用户适配问题”等。

判定原则：
- session-level 的 score_vector、reason_vector、overall_summary、key_dialogue_signals 是核心证据；
- interaction_goal 可以帮助判断 intent_fulfillment 相关问题；
- user_persona 只能作为辅助背景，不应被机械当成硬扣分标准；
- 如果整段对话整体不错，但存在某个明确短板，应该聚焦真正拉低总分的主因，而不是把所有可能问题都贴上。

可选错误原因集合（session-level）：
{taxonomy_block}

待归因样本：
{_json_block(observed_payload)}

请只返回 JSON，不要输出 markdown。JSON 字段必须包含：
- primary_error_category: 字符串，必须是上面集合中的一级大类
- secondary_error_categories: 字符串数组，长度 1 到 3，且必须都属于该一级大类
- affected_metrics: 字符串数组，从当前样本的 session 评分维度中选择最相关的 1 到 4 个
- attribution_summary: 字符串，2 到 4 句中文，概括整段对话低分主因及其与评分的关系
- improvement_focus: 字符串数组，长度 1 到 3，总结最值得优先改进的方向
""".strip()
