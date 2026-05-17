from __future__ import annotations

"""RUBRIC-MME 第三阶段裁判提示词构造。
真正可直接修改的中文提示词模板位于 `prompt_templates/` 目录下。
如果后续需要调 prompt，优先修改模板文件，而不是改本文件的拼装逻辑。
"""

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Sequence

TURN_PROMPT_VERSION = "rubric_mme_phase3_turn_cn_v4"
SESSION_PROMPT_VERSION = "rubric_mme_phase3_session_cn_v4"
MAX_FIELD_CHARS = 1200
PROMPT_TEMPLATE_DIR = Path(__file__).resolve().parent / "prompt_templates"
TURN_TEMPLATE_PATH = PROMPT_TEMPLATE_DIR / "turn_core_cn.txt"
SESSION_TEMPLATE_PATH = PROMPT_TEMPLATE_DIR / "session_core_cn.txt"


def clip_text(text: Any, *, max_chars: int = MAX_FIELD_CHARS) -> str:
    normalized = str(text or "").strip()
    if len(normalized) <= max_chars:
        return normalized
    return normalized[: max_chars - 8].rstrip() + "...[已截断]"


def json_block(payload: Any) -> str:
    if payload in (None, "", {}, []):
        return "{}"
    return json.dumps(payload, ensure_ascii=False, indent=2)


@lru_cache(maxsize=4)
def load_template(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def format_history_rounds(history_rounds: Sequence[Dict[str, Any]]) -> str:
    if not history_rounds:
        return "无历史轮次"
    parts: List[str] = []
    for item in history_rounds:
        round_number = int(item.get("round_index", 0)) + 1
        parts.append(
            "\n".join(
                [
                    f"第{round_number}轮用户问题：{clip_text(item.get('question_text', ''))}",
                    f"第{round_number}轮模型回答：{clip_text(item.get('prediction', ''))}",
                ]
            )
        )
    return "\n\n".join(parts)


def format_dialogue_transcript(rounds: Sequence[Dict[str, Any]]) -> str:
    parts: List[str] = []
    for item in rounds:
        round_number = int(item.get("round_index", 0)) + 1
        parts.append(
            "\n".join(
                [
                    f"第{round_number}轮用户问题：{clip_text(item.get('question_text', ''))}",
                    f"第{round_number}轮模型回答：{clip_text(item.get('prediction', ''))}",
                    f"第{round_number}轮参考答案：{clip_text(item.get('reference_answer', ''))}",
                ]
            )
        )
    return "\n\n".join(parts) if parts else "无有效对话"


def build_turn_prompt(round_record: Dict[str, Any], history_rounds: Sequence[Dict[str, Any]]) -> str:
    template = load_template(TURN_TEMPLATE_PATH)
    return template.format(
        task_name=round_record.get("task_name", ""),
        task_mode=round_record.get("task_mode", ""),
        round_id=round_record.get("round_id", ""),
        primary_category=round_record.get("primary_category", ""),
        history_rounds=format_history_rounds(history_rounds),
        question_text=clip_text(round_record.get("question_text", "")),
        reference_answer=clip_text(round_record.get("reference_answer", "")),
        prediction=clip_text(round_record.get("prediction", "")),
    )


def build_session_prompt(dialogue_record: Dict[str, Any]) -> str:
    interaction_setup = dialogue_record.get("interaction_setup") or {}
    template = load_template(SESSION_TEMPLATE_PATH)
    return template.format(
        task_name=dialogue_record.get("task_name", ""),
        task_mode=dialogue_record.get("task_mode", ""),
        dialogue_id=dialogue_record.get("dialogue_id", ""),
        round_count=dialogue_record.get("round_count", 0),
        interaction_goal=json_block(interaction_setup.get("interaction_goal") or {}),
        user_persona=json_block(interaction_setup.get("user_persona") or {}),
        dialogue_transcript=format_dialogue_transcript(dialogue_record.get("rounds") or []),
    )

