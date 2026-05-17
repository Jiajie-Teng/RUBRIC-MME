from __future__ import annotations

import argparse
from pathlib import Path

from attribution_runner import DEFAULT_INTERNAL_API_URL, build_internal_backend
from phase5_pipeline import STEP_SEQUENCE, run_phase5_pipeline

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PHASE4_DIR = REPO_ROOT / "logs" / "rubric_mme_phase4_internal"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "logs" / "rubric_mme_phase5_internal"
DEFAULT_MODEL = "gemini-3.1-pro-preview"


def _split_steps(value: str) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行 RUBRIC-MME 第五阶段自动分析报告（内部接口版）")
    parser.add_argument("--phase4-dir", default=str(DEFAULT_PHASE4_DIR), help="Phase 4 结果目录")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Phase 5 输出目录")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="用于分析报告生成的 Gemini 模型")
    parser.add_argument("--api-url", default=DEFAULT_INTERNAL_API_URL, help="内部接口地址")
    parser.add_argument("--api-key-env", default="MATRIXLLM_API_KEY", help="内部接口 token 的环境变量名")
    parser.add_argument("--timeout", type=int, default=180, help="HTTP 请求超时时间（秒）")
    parser.add_argument("--max-retries", type=int, default=5, help="单次分析请求的最大重试次数")
    parser.add_argument("--retry-sleep", type=float, default=3.0, help="普通错误重试前的等待秒数")
    parser.add_argument("--rate-limit-retry-sleep", type=float, default=12.0, help="遇到 429 后的初始等待秒数")
    parser.add_argument("--rate-limit-max-sleep", type=float, default=60.0, help="遇到 429 后的最大等待秒数")
    parser.add_argument("--temperature", type=float, default=0.0, help="分析模型 temperature")
    parser.add_argument("--top-p", type=float, default=0.95, help="分析模型 top_p")
    parser.add_argument("--max-output-tokens", type=int, default=8192, help="单个分析 step 的输出 token 上限")
    parser.add_argument("--keep-existing", action="store_true", help="保留已有 Phase 5 输出目录内容")
    parser.add_argument("--resume", action="store_true", help="从已有 step 结果继续运行，跳过成功 step")
    parser.add_argument("--repair-failed", action="store_true", help="只修复失败的 step")
    parser.add_argument("--repair-steps", default="", help=f"仅修复指定 step，多个 step 用逗号分隔：{', '.join(STEP_SEQUENCE)}")
    parser.add_argument("--save-prompt-text", action="store_true", help="额外保存各分析 step 的 prompt 文本")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    backend = build_internal_backend(
        model_name=args.model,
        api_url=args.api_url,
        api_key_env=args.api_key_env,
        timeout=args.timeout,
        max_retries=args.max_retries,
        retry_sleep=args.retry_sleep,
        rate_limit_retry_sleep=args.rate_limit_retry_sleep,
        rate_limit_max_sleep=args.rate_limit_max_sleep,
        temperature=args.temperature,
        top_p=args.top_p,
        max_output_tokens=args.max_output_tokens,
    )
    try:
        run_phase5_pipeline(
            Path(args.phase4_dir),
            Path(args.output_dir),
            backend,
            clear_output=not (args.keep_existing or args.resume or args.repair_failed or args.repair_steps),
            save_prompt_text=args.save_prompt_text,
            resume=args.resume,
            repair_failed=args.repair_failed,
            repair_steps=_split_steps(args.repair_steps),
        )
    finally:
        if hasattr(backend, "close"):
            backend.close()


if __name__ == "__main__":
    main()
