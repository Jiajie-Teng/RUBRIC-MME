from __future__ import annotations

from typing import Dict, List, Tuple

TURN_ERROR_CATEGORIES_CN: Dict[str, Dict[str, str]] = {
    "事实性错误": {
        "幻觉与捏造": "凭空生成图片/视频中不存在的内容、事件或并不存在的细节",
        "对象与属性错误": "识别错误的物体、人物、颜色、形状、位置等静态特征",
        "时空与数值错误": "时间顺序混乱、空间方位错误、计数统计错误或数值识别偏差",
        "内容与媒体不符": "回答内容描述与图片/视频实际画面明显矛盾或冲突",
        "专业知识/常识错误": "违反常识或特定领域的专业知识错误",
    },
    "完整性缺失": {
        "核心信息遗漏": "未回答问题的关键点，遗漏主要事实或关键实体",
        "步骤与流程缺失": "在说明书或操作演练类问题中，遗漏关键步骤或环节",
        "细节与背景缺失": "遗漏必要的描述性细节、背景信息或情境说明",
        "未回答所有子问题": "对于复合型问题，遗漏了部分子问题的回答",
    },
    "指令与相关性": {
        "指令理解错误": "未遵循用户的特定格式、角色设定、字数限制或否定约束",
        "偏离主题": "回答偏离问题核心，答非所问，或错误地抓住了次要问题",
        "上下文忽略": "忽略多轮对话的历史上下文，或未识别出用户的真实意图",
        "包含无关信息": "引入了与问题或图片/视频内容无关的冗余信息或噪音",
    },
    "回复质量与深度": {
        "过于笼统肤浅": "回答正确但缺乏深度，内容空泛，止步于表面，或通过模板回复",
        "过于被动挤牙膏": "仅回答字面意思，极其简短，缺乏必要的主动性、扩展性或连贯性",
        "缺乏合理引申": "未能根据媒体内容提供有价值的建议、推论或下一步预测",
        "引申误导或错误": "主动提供的建议、猜测或引申内容存在事实错误或误导性",
    },
    "逻辑与表达": {
        "逻辑推理错误": "回答内部逻辑自相矛盾，因果倒置，或无法自圆其说",
        "表达啰嗦重复": "存在大量车轱辘话、信息重复、语义冗余或表达累赘",
        "格式与风格生硬": "回答机器感强，语气生硬，或不符合预期的语言风格",
    },
}

SESSION_ERROR_CATEGORIES_CN: Dict[str, Dict[str, str]] = {
    "对话连贯性问题": {
        "历史信息断裂": "未能延续前文已建立的事实、人物、场景或任务状态",
        "前后自相矛盾": "不同轮次回答之间出现明显冲突或互相否定",
        "长期上下文利用不足": "只关注局部轮次，无法整合整段对话中的关键信息",
    },
    "目标达成问题": {
        "核心目标推进不足": "多轮结束后仍未有效推动用户的核心任务或问题解决",
        "关键需求覆盖不全": "虽然单轮回答尚可，但整段对话遗漏了用户反复关注的核心点",
        "总结收束能力不足": "在对话后期未能对前文信息进行归纳、确认或形成有用结论",
    },
    "用户适配问题": {
        "显式用户状态适配不足": "未根据用户在对话中显式表现出的焦虑、求确认、谨慎或目标导向调整回应",
        "语气与角色不匹配": "整体语气、风格或互动方式与用户状态不协调",
        "互动策略单一": "整段对话中缺乏根据用户变化而做出的策略调整",
    },
    "可信度与帮助性问题": {
        "持续顺从用户错误前提": "多轮中反复顺着用户错误猜测作答，未及时纠偏",
        "整体事实可靠性不足": "虽然局部可用，但整段对话存在较多不稳定或可疑信息",
        "帮助性不成体系": "单轮可能提供了帮助，但整体上缺乏稳定、系统的支持效果",
    },
}


def flatten_taxonomy(taxonomy: Dict[str, Dict[str, str]]) -> List[Dict[str, str]]:
    entries: List[Dict[str, str]] = []
    for primary_category, secondary_map in taxonomy.items():
        for secondary_category, description in secondary_map.items():
            entries.append(
                {
                    "primary_category": primary_category,
                    "secondary_category": secondary_category,
                    "description": description,
                }
            )
    return entries


def build_taxonomy_prompt_block(taxonomy: Dict[str, Dict[str, str]]) -> str:
    lines: List[str] = []
    for primary_category, secondary_map in taxonomy.items():
        lines.append(f"{primary_category}:")
        for secondary_category, description in secondary_map.items():
            lines.append(f"- {secondary_category}: {description}")
    return "\n".join(lines)


def valid_secondary_categories(taxonomy: Dict[str, Dict[str, str]]) -> List[str]:
    return [entry["secondary_category"] for entry in flatten_taxonomy(taxonomy)]


def valid_primary_secondary_pairs(taxonomy: Dict[str, Dict[str, str]]) -> List[Tuple[str, str]]:
    return [
        (entry["primary_category"], entry["secondary_category"])
        for entry in flatten_taxonomy(taxonomy)
    ]
