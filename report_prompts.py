from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

REPORT_PROMPT_VERSION = "rubric_mme_phase5_cn_v5_multistep_refined"
_TEMPLATE_DIR = Path(__file__).resolve().parent / "prompt_templates"
_STAGE_TO_TEMPLATE = {
    "step1_overview": "report_phase5_step1_cn.txt",
    "step2_scope_findings": "report_phase5_step2_scope_cn.txt",
    "step3_root_causes_cases": "report_phase5_step3_causes_cn.txt",
    "step4_recommendations": "report_phase5_step4_recommendations_cn.txt",
}
_PLACEHOLDER = "{evidence_json}"


def _load_template(stage_name: str) -> str:
    template_name = _STAGE_TO_TEMPLATE[stage_name]
    return (_TEMPLATE_DIR / template_name).read_text(encoding="utf-8")


def build_report_step_prompt(stage_name: str, payload: Dict[str, Any]) -> str:
    if stage_name not in _STAGE_TO_TEMPLATE:
        raise KeyError(f"Unsupported Phase 5 stage: {stage_name}")
    evidence_json = json.dumps(payload, ensure_ascii=False, indent=2)
    return _load_template(stage_name).replace(_PLACEHOLDER, evidence_json)
