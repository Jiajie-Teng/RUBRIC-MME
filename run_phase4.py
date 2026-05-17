from __future__ import annotations

import argparse
from pathlib import Path

from attribution_runner import build_official_backend
from low_score_selector import DEFAULT_AVG_THRESHOLD, DEFAULT_CRITICAL_THRESHOLD, DEFAULT_METRIC_THRESHOLD
from phase4_pipeline import run_phase4_pipeline

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PHASE3_DIR = REPO_ROOT / "logs" / "rubric_mme_phase3"
DEFAULT_PHASE2_DIR = REPO_ROOT / "logs" / "rubric_mme_phase2_normalized"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "logs" / "rubric_mme_phase4"
DEFAULT_MODEL = "gemini-3.1-pro-preview"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行 RUBRIC-MME 第四阶段官方 Gemini 低分归因。")
    parser.add_argument("--phase3-dir", default=str(DEFAULT_PHASE3_DIR), help="Phase 3 输出目录，需要包含 turn_judgements.jsonl 和 session_judgements.jsonl。")
    parser.add_argument("--phase2-dir", default=str(DEFAULT_PHASE2_DIR), help="可选的 Phase 2 输出目录，用于给 session-level 归因补充 interaction_goal / user_persona。")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Phase 4 输出目录。")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="官方 Gemini 归因模型名称。")
    parser.add_argument("--api-key-env", default="GOOGLE_API_KEY", help="官方 Gemini key 所在环境变量名。")
    parser.add_argument("--use-response-schema", action="store_true", help="优先使用 Gemini response_schema；失败时仍会回退到 prompt-only JSON。")
    parser.add_argument("--timeout-seconds", type=int, default=300, help="官方 Gemini 文件处理/请求超时时间（秒）。")
    parser.add_argument("--poll-interval", type=float, default=3.0, help="官方 Gemini 轮询间隔（秒）。")
    parser.add_argument("--max-retries", type=int, default=5, help="归因调用失败后的最大重试次数。")
    parser.add_argument("--retry-sleep", type=float, default=3.0, help="普通错误重试基础等待时间（秒）。")
    parser.add_argument("--rate-limit-retry-sleep", type=float, default=12.0, help="429 限流时的基础退避时间（秒）。")
    parser.add_argument("--rate-limit-max-sleep", type=float, default=60.0, help="429 限流时的最大退避时间（秒）。")
    parser.add_argument("--inter-request-sleep", type=float, default=1.0, help="两次归因请求之间的冷却时间（秒）。")
    parser.add_argument("--max-workers", type=int, default=1, help="candidate 级并行 worker 数；默认 1 表示串行。")
    parser.add_argument("--repair-passes", type=int, default=0, help="首轮结束后，对失败项再做多少轮补跑。默认关闭。")
    parser.add_argument("--repair-pass-cooldown", type=float, default=0.0, help="补跑轮次之间的冷却时间（秒）。")
    parser.add_argument("--temperature", type=float, default=0.0, help="归因模型采样温度。")
    parser.add_argument("--max-output-tokens", type=int, default=1024, help="归因模型最大输出 token 数。")
    parser.add_argument("--dialogue-id", default="", help="只处理指定的 dialogue_id。")
    parser.add_argument("--limit-dialogues", type=int, default=None, help="只处理前 N 个 dialogue。")
    parser.add_argument("--resume", action="store_true", help="继续已有的 Phase 4 输出；仅跳过已成功项，失败项允许补跑。")
    parser.add_argument("--keep-existing", action="store_true", help="保留已有 Phase 4 输出文件，不先清空。")
    parser.add_argument("--save-prompt-text", action="store_true", help="将归因 prompt 文本一起写入结果文件。")
    parser.add_argument("--avg-threshold", type=float, default=DEFAULT_AVG_THRESHOLD, help="低分样本的 avg_score 阈值。")
    parser.add_argument("--metric-threshold", type=int, default=DEFAULT_METRIC_THRESHOLD, help="单个指标触发低分归因的阈值（<=）。")
    parser.add_argument("--critical-threshold", type=int, default=DEFAULT_CRITICAL_THRESHOLD, help="critical 严重等级阈值（<=）。")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    def build_backend():
        return build_official_backend(
            model_name=args.model,
            api_key_env=args.api_key_env,
            use_response_schema=args.use_response_schema,
            timeout_seconds=args.timeout_seconds,
            poll_interval=args.poll_interval,
            max_retries=args.max_retries,
            retry_sleep=args.retry_sleep,
            rate_limit_retry_sleep=args.rate_limit_retry_sleep,
            rate_limit_max_sleep=args.rate_limit_max_sleep,
            temperature=args.temperature,
            max_output_tokens=args.max_output_tokens,
        )

    backend = build_backend()
    try:
        run_phase4_pipeline(
            Path(args.phase3_dir),
            Path(args.output_dir),
            backend,
            phase2_dir=Path(args.phase2_dir) if str(args.phase2_dir).strip() else None,
            dialogue_id=args.dialogue_id,
            limit_dialogues=args.limit_dialogues,
            clear_output=not args.keep_existing,
            resume=args.resume,
            save_prompt_text=args.save_prompt_text,
            avg_threshold=args.avg_threshold,
            metric_threshold=args.metric_threshold,
            critical_threshold=args.critical_threshold,
            inter_request_sleep=args.inter_request_sleep,
            repair_passes=args.repair_passes,
            repair_pass_cooldown=args.repair_pass_cooldown,
            max_workers=args.max_workers,
            backend_factory=build_backend,
        )
    finally:
        close = getattr(backend, "close", None)
        if callable(close):
            close()


if __name__ == "__main__":
    main()
