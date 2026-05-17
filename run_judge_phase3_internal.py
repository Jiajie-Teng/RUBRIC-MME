from __future__ import annotations

import argparse
from pathlib import Path

from judge_pipeline import run_phase3_pipeline
from judge_runner import DEFAULT_INTERNAL_API_URL, MatrixLLMJudgeBackend

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PHASE2_DIR = REPO_ROOT / "logs" / "rubric_mme_phase2_normalized"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "logs" / "rubric_mme_phase3_internal"
DEFAULT_MODEL = "gemini-3.1-pro-preview"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行 RUBRIC-MME 第三阶段内部版裁判评分")
    parser.add_argument("--phase2-dir", default=str(DEFAULT_PHASE2_DIR), help="Phase 2 输出目录，内部需要包含 rounds.jsonl 和 dialogues.jsonl。")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Phase 3 输出目录。")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="内部 judge 使用的模型名称。")
    parser.add_argument("--api-url", default=DEFAULT_INTERNAL_API_URL, help="内部 judge 接口地址。")
    parser.add_argument("--api-key-env", default="MATRIXLLM_API_KEY", help="内部 judge token 所在的环境变量名。")
    parser.add_argument("--timeout", type=int, default=180, help="HTTP 请求超时时间（秒）。")
    parser.add_argument("--max-retries", type=int, default=5, help="judge 调用失败后的最大重试次数。")
    parser.add_argument("--retry-sleep", type=float, default=3.0, help="普通错误重试基础等待时间（秒）。")
    parser.add_argument("--rate-limit-retry-sleep", type=float, default=12.0, help="429 限流时的基础退避时间（秒）。")
    parser.add_argument("--rate-limit-max-sleep", type=float, default=60.0, help="429 限流时的最大退避时间（秒）。")
    parser.add_argument("--inter-request-sleep", type=float, default=1.0, help="两次 judge 请求之间的冷却时间（秒）。")
    parser.add_argument("--max-workers", type=int, default=1, help="session 级并行 worker 数；默认 1 表示串行。")
    parser.add_argument("--repair-passes", type=int, default=0, help="首轮结束后，对失败项再做多少轮补跑。默认关闭，后续也可用 --repair-failed 单独修复。")
    parser.add_argument("--repair-pass-cooldown", type=float, default=0.0, help="补跑轮次之间的冷却时间（秒）。")
    parser.add_argument("--turn-repair-mode", choices=["full_context", "current_turn_only"], default="full_context", help="turn-level repair 模式。full_context 使用完整历史；current_turn_only 仅使用当前轮视觉和文本。")
    parser.add_argument("--session-repair-mode", choices=["full_context", "light_context"], default="full_context", help="session-level repair 模式。full_context 使用整组视觉；light_context 保留完整文本但只取代表性视觉轮次。")
    parser.add_argument("--temperature", type=float, default=0.0, help="judge 采样温度。")
    parser.add_argument("--top-p", type=float, default=0.95, help="judge top_p。")
    parser.add_argument("--max-output-tokens", type=int, default=2048, help="judge 最大输出 token 数。")
    parser.add_argument("--dialogue-id", default="", help="只评测指定的 dialogue_id。")
    parser.add_argument("--limit-dialogues", type=int, default=None, help="只评测前 N 个 dialogue。")
    parser.add_argument("--resume", action="store_true", help="继续已有 Phase 3 输出；仅跳过已成功项，失败项允许补跑。")
    parser.add_argument("--repair-failed", action="store_true", help="只修复当前输出目录里已经存在且失败的 turn/session judgement。")
    parser.add_argument("--keep-existing", action="store_true", help="保留已有 Phase 3 输出文件，不先清空。")
    parser.add_argument("--save-prompt-text", action="store_true", help="把 judge prompt 文本一并写入结果文件。")
    parser.add_argument("--allow-incomplete-dialogues", action="store_true", help="即使某些 round 缺失预测，也继续做 session-level judge。")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    def build_backend() -> MatrixLLMJudgeBackend:
        return MatrixLLMJudgeBackend(
            api_url=args.api_url,
            api_key_env=args.api_key_env,
            timeout=args.timeout,
            top_p=args.top_p,
            model_name=args.model,
            max_retries=args.max_retries,
            retry_sleep=args.retry_sleep,
            rate_limit_retry_sleep=args.rate_limit_retry_sleep,
            rate_limit_max_sleep=args.rate_limit_max_sleep,
            temperature=args.temperature,
            max_output_tokens=args.max_output_tokens,
        )

    backend = build_backend()
    try:
        run_phase3_pipeline(
            Path(args.phase2_dir),
            Path(args.output_dir),
            backend,
            dialogue_id=args.dialogue_id,
            limit_dialogues=args.limit_dialogues,
            clear_output=not args.keep_existing and not args.repair_failed,
            resume=args.resume,
            repair_failed=args.repair_failed,
            max_workers=args.max_workers,
            save_prompt_text=args.save_prompt_text,
            allow_incomplete_dialogues=args.allow_incomplete_dialogues,
            inter_request_sleep=args.inter_request_sleep,
            repair_passes=args.repair_passes,
            repair_pass_cooldown=args.repair_pass_cooldown,
            turn_repair_mode=args.turn_repair_mode,
            session_repair_mode=args.session_repair_mode,
            backend_factory=build_backend,
        )
    finally:
        close = getattr(backend, "close", None)
        if callable(close):
            close()


if __name__ == "__main__":
    main()
