
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
TOOLS_DIR = Path(__file__).resolve().parent

STAGE_ORDER = ["phase1", "phase2", "phase3", "phase4", "phase5"]
PHASE5_PARTIAL_FILENAMES = [
    "report_payload.json",
    "report_digest.json",
    "report_step_results.json",
    "report_analysis.json",
    "report_analysis_raw.txt",
    "report_prompt.txt",
    "benchmark_report.md",
    "benchmark_report.html",
]
ALL_TASKS = [
    "omnibench_image_multi_text",
    "omnibench_image_multi_tts",
    "omnibench_video_stream_text",
    "omnibench_video_stream_tts",
]
GPT_SUPPORTED_TASKS = [
    "omnibench_image_multi_text",
    "omnibench_video_stream_text",
]
CLAUDE_SUPPORTED_TASKS = [
    "omnibench_image_multi_text",
    "omnibench_video_stream_text",
]
QWEN_SUPPORTED_TASKS = [
    "omnibench_image_multi_text",
    "omnibench_video_stream_text",
]
GLM_SUPPORTED_TASKS = [
    "omnibench_image_multi_text",
    "omnibench_video_stream_text",
]


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def parse_csv(raw: str) -> list[str]:
    if not raw:
        return []
    return [item.strip() for item in raw.split(",") if item.strip()]


def normalize_tasks(raw: str) -> list[str]:
    tasks = parse_csv(raw)
    if not tasks:
        return list(ALL_TASKS)
    lowered = {task.lower() for task in tasks}
    if lowered & {"rubric-mme", "rubric_mme", "omnibench", "all"}:
        return list(ALL_TASKS)
    return tasks


def normalize_tasks_for_phase1_backend(raw: str, phase1_backend: str) -> list[str]:
    tasks = normalize_tasks(raw)
    if phase1_backend == "gpt_openai_compatible":
        return [task for task in tasks if task in GPT_SUPPORTED_TASKS]
    if phase1_backend in {"claude_vision_openai_compatible", "claude_openai_sdk"}:
        return [task for task in tasks if task in CLAUDE_SUPPORTED_TASKS]
    if phase1_backend in {"qwen_openai_compatible", "qwen_antchat_openai_sdk"}:
        return [task for task in tasks if task in QWEN_SUPPORTED_TASKS]
    if phase1_backend == "glm_antchat_openai_sdk":
        return [task for task in tasks if task in GLM_SUPPORTED_TASKS]
    return tasks


def stage_range(start_stage: str, end_stage: str) -> list[str]:
    start_idx = STAGE_ORDER.index(start_stage)
    end_idx = STAGE_ORDER.index(end_stage)
    if start_idx > end_idx:
        raise ValueError(f"start-stage {start_stage} must be <= end-stage {end_stage}")
    return STAGE_ORDER[start_idx : end_idx + 1]


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def command_str(parts: list[str]) -> str:
    return subprocess.list2cmdline(parts)


def render_progress_bar(current: int, total: int, width: int = 24) -> str:
    total = max(total, 1)
    current = max(0, min(current, total))
    filled = int(width * current / total)
    return f"[{'#' * filled}{'-' * (width - filled)}] {current}/{total}"


@dataclass
class StageAudit:
    stage: str
    success: bool
    status: str
    summary: str
    details: dict[str, Any]
    remaining_count: int


class PipelineLogger:
    def __init__(self, output_root: Path) -> None:
        self.output_root = output_root
        self.events_path = output_root / "pipeline_events.jsonl"

    def log(self, event_type: str, **payload: Any) -> None:
        append_jsonl(self.events_path, {"timestamp": now_iso(), "event_type": event_type, **payload})


class PipelineRunner:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.output_root = Path(args.output_root).resolve()
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.logger = PipelineLogger(self.output_root)
        self.stage_statuses: dict[str, dict[str, Any]] = {}
        self.tasks = normalize_tasks_for_phase1_backend(args.tasks, args.phase1_backend)
        self.stage_dirs = {
            "phase1": Path(args.phase1_dir).resolve() if args.phase1_dir else self.output_root / "phase1",
            "phase2": Path(args.phase2_dir).resolve() if args.phase2_dir else self.output_root / "phase2",
            "phase3": Path(args.phase3_dir).resolve() if args.phase3_dir else self.output_root / "phase3",
            "phase4": Path(args.phase4_dir).resolve() if args.phase4_dir else self.output_root / "phase4",
            "phase5": Path(args.phase5_dir).resolve() if args.phase5_dir else self.output_root / "phase5",
        }

    def run(self) -> int:
        selected_stages = stage_range(self.args.start_stage, self.args.end_stage)
        total_stages = len(selected_stages)
        self.logger.log(
            "pipeline_started",
            benchmark_name="RUBRIC-MME",
            selected_stages=selected_stages,
            tested_model=self.args.tested_model,
            judge_model=self.args.judge_model,
            attribution_model=self.args.attribution_model,
            analysis_model=self.args.analysis_model,
            output_root=str(self.output_root),
        )

        print(f"\n开始执行 RUBRIC-MME 总控流程，共 {total_stages} 个阶段。")
        overall_success = True
        for stage_index, stage in enumerate(selected_stages, start=1):
            handler = getattr(self, f"run_{stage}")
            stage_started_at = time.time()
            progress = render_progress_bar(stage_index - 1, total_stages)
            print(f"\n{progress} 准备进入 {stage}，输出目录：{self.stage_dirs[stage]}")
            self.logger.log("stage_started", stage=stage, output_dir=str(self.stage_dirs[stage]))
            try:
                audit = handler()
            except Exception as exc:
                audit = StageAudit(
                    stage=stage,
                    success=False,
                    status="exception",
                    summary=str(exc),
                    details={"exception": repr(exc)},
                    remaining_count=1,
                )
            stage_elapsed = round(time.time() - stage_started_at, 2)
            self.stage_statuses[stage] = {
                "status": audit.status,
                "success": audit.success,
                "summary": audit.summary,
                "remaining_count": audit.remaining_count,
                "elapsed_seconds": stage_elapsed,
                "output_dir": str(self.stage_dirs[stage]),
                "details": audit.details,
            }
            self.logger.log(
                "stage_completed",
                stage=stage,
                success=audit.success,
                status=audit.status,
                elapsed_seconds=stage_elapsed,
                remaining_count=audit.remaining_count,
            )
            progress = render_progress_bar(stage_index, total_stages)
            print(f"{progress} {stage} 结束，状态：{audit.status}，剩余失败数：{audit.remaining_count}。")
            print(f"阶段摘要：{audit.summary}")
            if not audit.success:
                overall_success = False
                if self.args.fail_policy == "stop":
                    print(f"因 fail-policy=stop，{stage} 未完全成功，流水线停止。")
                    break

        pipeline_status = "success" if overall_success else ("best_effort_completed" if self.args.fail_policy == "best_effort" else "failed")
        manifest = {
            "pipeline_name": self.args.pipeline_name,
            "benchmark_name": "RUBRIC-MME",
            "generated_at": now_iso(),
            "status": pipeline_status,
            "selected_stages": selected_stages,
            "tasks": self.tasks,
            "tested_model": self.args.tested_model,
            "judge_model": self.args.judge_model,
            "attribution_model": self.args.attribution_model,
            "analysis_model": self.args.analysis_model,
            "phase1_backend": self.args.phase1_backend,
            "phase3_backend": self.args.phase3_backend,
            "phase4_backend": self.args.phase4_backend,
            "phase5_backend": self.args.phase5_backend,
            "output_root": str(self.output_root),
            "stage_dirs": {stage: str(path) for stage, path in self.stage_dirs.items()},
            "stage_statuses": self.stage_statuses,
        }
        write_json(self.output_root / "pipeline_manifest.json", manifest)
        write_json(self.output_root / "pipeline_stage_status.json", self.stage_statuses)
        self.write_summary_markdown(pipeline_status)
        self.logger.log("pipeline_completed", status=pipeline_status)
        print(f"\n总控流程结束，最终状态：{pipeline_status}。")
        return 0 if overall_success or self.args.fail_policy == "best_effort" else 1

    def run_command(self, stage: str, command: list[str]) -> int:
        pretty = command_str(command)
        print(f"  -> [{stage}] 开始执行命令")
        print(f"     {pretty}")
        self.logger.log("command_started", stage=stage, command=pretty)
        completed = subprocess.run(command, cwd=str(REPO_ROOT))
        self.logger.log("command_completed", stage=stage, command=pretty, returncode=completed.returncode)
        print(f"  -> [{stage}] 命令结束，返回码：{completed.returncode}")
        return completed.returncode

    def phase1_script(self) -> Path:
        if self.args.phase1_backend == "internal":
            return TOOLS_DIR / "run_gemini_phase1_internal.py"
        if self.args.phase1_backend == "official":
            return TOOLS_DIR / "run_gemini_phase1.py"
        if self.args.phase1_backend == "gpt_openai_compatible":
            return TOOLS_DIR / "run_gpt_openai_compatible_phase1.py"
        if self.args.phase1_backend == "claude_vision_openai_compatible":
            return TOOLS_DIR / "run_claude_vision_openai_compatible_phase1.py"
        if self.args.phase1_backend == "claude_openai_sdk":
            return TOOLS_DIR / "run_claude_openai_sdk_phase1.py"
        if self.args.phase1_backend == "qwen_openai_compatible":
            return TOOLS_DIR / "run_qwen25_vl_openai_compatible_phase1.py"
        if self.args.phase1_backend == "qwen_antchat_openai_sdk":
            return TOOLS_DIR / "run_qwen_antchat_openai_sdk_phase1.py"
        if self.args.phase1_backend == "glm_antchat_openai_sdk":
            return TOOLS_DIR / "run_glm_antchat_openai_sdk_phase1.py"
        return TOOLS_DIR / "run_openai_compatible_phase1.py"

    def phase3_script(self) -> Path:
        return TOOLS_DIR / ("run_judge_phase3_internal.py" if self.args.phase3_backend == "internal" else "run_judge_phase3.py")

    def phase4_script(self) -> Path:
        return TOOLS_DIR / ("run_phase4_internal.py" if self.args.phase4_backend == "internal" else "run_phase4.py")

    def phase5_script(self) -> Path:
        return TOOLS_DIR / ("run_phase5_internal.py" if self.args.phase5_backend == "internal" else "run_phase5.py")

    def base_python_command(self, script: Path) -> list[str]:
        return [sys.executable, str(script)]

    def phase1_worker_count(self, *, repair_failed: bool) -> int:
        workers = max(1, int(self.args.phase1_workers))
        if repair_failed and self.args.phase1_backend == "openai_compatible":
            return 1
        return workers

    def build_phase1_command(self, *, repair_failed: bool = False, repair_mode: str = "", include_resume: bool = False) -> list[str]:
        phase1_dir = self.stage_dirs["phase1"]
        command = self.base_python_command(self.phase1_script())
        self.maybe_add_common_selector_args(command, tasks_override=",".join(self.tasks))
        command.extend(["--model", self.args.tested_model, "--output-dir", str(phase1_dir), "--max-workers", str(self.phase1_worker_count(repair_failed=repair_failed))])
        if repair_failed:
            command.append("--repair-failed")
            if repair_mode:
                command.extend(["--repair-mode", repair_mode])
        elif include_resume:
            command.append("--resume")
        if self.args.data_root:
            command.extend(["--data-root", self.args.data_root])
        if self.args.phase1_backend in {"internal", "openai_compatible", "gpt_openai_compatible", "claude_vision_openai_compatible", "claude_openai_sdk", "qwen_openai_compatible", "qwen_antchat_openai_sdk", "glm_antchat_openai_sdk"} and self.args.media_root:
            command.extend(["--media-root", self.args.media_root])
        if self.args.phase1_api_key_env:
            command.extend(["--api-key-env", self.args.phase1_api_key_env])
        if self.args.phase1_api_url and self.args.phase1_backend in {"internal", "openai_compatible", "gpt_openai_compatible", "claude_vision_openai_compatible", "claude_openai_sdk", "qwen_openai_compatible", "qwen_antchat_openai_sdk", "glm_antchat_openai_sdk"}:
            command.extend(["--api-url", self.args.phase1_api_url])
        if self.args.phase1_backend == "openai_compatible":
            if self.args.phase1_image_input_mode:
                command.extend(["--image-input-mode", self.args.phase1_image_input_mode])
            if self.args.phase1_video_input_mode:
                command.extend(["--video-input-mode", self.args.phase1_video_input_mode])
            if self.args.phase1_video_history_mode:
                command.extend(["--video-history-mode", self.args.phase1_video_history_mode])
            if self.args.phase1_video_precompressed_mode:
                command.extend(["--video-precompressed-mode", self.args.phase1_video_precompressed_mode])
            if self.args.phase1_video_precompressed_field:
                command.extend(["--video-precompressed-field", self.args.phase1_video_precompressed_field])
            if self.args.phase1_video_compress_mode:
                command.extend(["--video-compress-mode", self.args.phase1_video_compress_mode])
            if self.args.phase1_video_max_inline_bytes:
                command.extend(["--video-max-inline-bytes", str(self.args.phase1_video_max_inline_bytes)])
        if self.args.phase1_backend in {"gpt_openai_compatible", "claude_vision_openai_compatible", "claude_openai_sdk"}:
            if self.args.phase1_image_input_mode:
                command.extend(["--image-input-mode", self.args.phase1_image_input_mode])
            if self.args.phase1_video_history_mode:
                command.extend(["--video-history-mode", self.args.phase1_video_history_mode])
            if self.args.phase1_video_prepared_frame_mode:
                command.extend(["--video-prepared-frame-mode", self.args.phase1_video_prepared_frame_mode])
            if self.args.phase1_video_prepared_dir_field:
                command.extend(["--video-prepared-dir-field", self.args.phase1_video_prepared_dir_field])
            if self.args.phase1_video_prepared_paths_field:
                command.extend(["--video-prepared-paths-field", self.args.phase1_video_prepared_paths_field])
            if self.args.phase1_video_prepared_profile_field:
                command.extend(["--video-prepared-profile-field", self.args.phase1_video_prepared_profile_field])
            if self.args.phase1_video_prepared_total_bytes_field:
                command.extend(["--video-prepared-total-bytes-field", self.args.phase1_video_prepared_total_bytes_field])
            if self.args.phase1_video_prepared_strategy_field:
                command.extend(["--video-prepared-strategy-field", self.args.phase1_video_prepared_strategy_field])
            if self.args.phase1_video_prepared_source_size_field:
                command.extend(["--video-prepared-source-size-field", self.args.phase1_video_prepared_source_size_field])
            if self.args.phase1_video_prepared_duration_field:
                command.extend(["--video-prepared-duration-field", self.args.phase1_video_prepared_duration_field])
            if self.args.phase1_video_frame_root_name:
                command.extend(["--video-frame-root-name", self.args.phase1_video_frame_root_name])
            if self.args.phase1_video_frame_count:
                command.extend(["--video-frame-count", str(self.args.phase1_video_frame_count)])
            if self.args.phase1_video_frame_max_side:
                command.extend(["--video-frame-max-side", str(self.args.phase1_video_frame_max_side)])
            if self.args.phase1_video_frame_jpeg_quality:
                command.extend(["--video-frame-jpeg-quality", str(self.args.phase1_video_frame_jpeg_quality)])
            if self.args.phase1_video_frame_max_inline_bytes:
                command.extend(["--video-frame-max-inline-bytes", str(self.args.phase1_video_frame_max_inline_bytes)])
            if self.args.phase1_video_frame_sampling_strategy:
                command.extend(["--video-frame-sampling-strategy", self.args.phase1_video_frame_sampling_strategy])
            if self.args.phase1_video_max_images_per_request:
                command.extend(["--video-max-images-per-request", str(self.args.phase1_video_max_images_per_request)])
            if self.args.phase1_video_history_max_frames_per_round:
                command.extend(["--video-history-max-frames-per-round", str(self.args.phase1_video_history_max_frames_per_round)])
        if self.args.phase1_backend in {"qwen_openai_compatible", "qwen_antchat_openai_sdk", "glm_antchat_openai_sdk"}:
            if self.args.phase1_image_input_mode:
                command.extend(["--image-input-mode", self.args.phase1_image_input_mode])
            if self.args.phase1_video_route:
                command.extend(["--video-route", self.args.phase1_video_route])
            if self.args.phase1_video_input_mode:
                command.extend(["--video-input-mode", self.args.phase1_video_input_mode])
            if self.args.phase1_video_history_mode:
                command.extend(["--video-history-mode", self.args.phase1_video_history_mode])
            if self.args.phase1_video_history_max_visual_rounds:
                command.extend(["--video-history-max-visual-rounds", str(self.args.phase1_video_history_max_visual_rounds)])
            if self.args.phase1_video_history_max_inline_bytes_total:
                command.extend(["--video-history-max-inline-bytes-total", str(self.args.phase1_video_history_max_inline_bytes_total)])
            if self.args.phase1_video_precompressed_mode:
                command.extend(["--video-precompressed-mode", self.args.phase1_video_precompressed_mode])
            if self.args.phase1_video_precompressed_field:
                command.extend(["--video-precompressed-field", self.args.phase1_video_precompressed_field])
            if self.args.phase1_video_compress_mode:
                command.extend(["--video-compress-mode", self.args.phase1_video_compress_mode])
            if self.args.phase1_video_max_inline_bytes:
                command.extend(["--video-max-inline-bytes", str(self.args.phase1_video_max_inline_bytes)])
            if self.args.phase1_video_prepared_frame_mode:
                command.extend(["--video-prepared-frame-mode", self.args.phase1_video_prepared_frame_mode])
            if self.args.phase1_video_prepared_dir_field:
                command.extend(["--video-prepared-dir-field", self.args.phase1_video_prepared_dir_field])
            if self.args.phase1_video_prepared_paths_field:
                command.extend(["--video-prepared-paths-field", self.args.phase1_video_prepared_paths_field])
            if self.args.phase1_video_prepared_profile_field:
                command.extend(["--video-prepared-profile-field", self.args.phase1_video_prepared_profile_field])
            if self.args.phase1_video_prepared_total_bytes_field:
                command.extend(["--video-prepared-total-bytes-field", self.args.phase1_video_prepared_total_bytes_field])
            if self.args.phase1_video_prepared_strategy_field:
                command.extend(["--video-prepared-strategy-field", self.args.phase1_video_prepared_strategy_field])
            if self.args.phase1_video_prepared_source_size_field:
                command.extend(["--video-prepared-source-size-field", self.args.phase1_video_prepared_source_size_field])
            if self.args.phase1_video_prepared_duration_field:
                command.extend(["--video-prepared-duration-field", self.args.phase1_video_prepared_duration_field])
            if self.args.phase1_video_frame_root_name:
                command.extend(["--video-frame-root-name", self.args.phase1_video_frame_root_name])
            if self.args.phase1_video_frame_count:
                command.extend(["--video-frame-count", str(self.args.phase1_video_frame_count)])
            if self.args.phase1_video_frame_max_side:
                command.extend(["--video-frame-max-side", str(self.args.phase1_video_frame_max_side)])
            if self.args.phase1_video_frame_jpeg_quality:
                command.extend(["--video-frame-jpeg-quality", str(self.args.phase1_video_frame_jpeg_quality)])
            if self.args.phase1_video_frame_max_inline_bytes:
                command.extend(["--video-frame-max-inline-bytes", str(self.args.phase1_video_frame_max_inline_bytes)])
            if self.args.phase1_video_frame_sampling_strategy:
                command.extend(["--video-frame-sampling-strategy", self.args.phase1_video_frame_sampling_strategy])
            if self.args.phase1_video_max_frame_images_per_request:
                command.extend(["--video-max-frame-images-per-request", str(self.args.phase1_video_max_frame_images_per_request)])
            if self.args.phase1_video_history_max_frames_per_round:
                command.extend(["--video-history-max-frames-per-round", str(self.args.phase1_video_history_max_frames_per_round)])
        return command

    def maybe_add_common_selector_args(self, command: list[str], tasks_override: str = "") -> list[str]:
        command.extend(["--tasks", tasks_override or self.args.tasks])
        if self.args.limit is not None:
            command.extend(["--limit", str(self.args.limit)])
        if self.args.dialogue_id:
            command.extend(["--dialogue-id", self.args.dialogue_id])
        return command
    def audit_phase1(self) -> StageAudit:
        phase1_dir = self.stage_dirs["phase1"]
        task_details: dict[str, Any] = {}
        missing_tasks: list[str] = []
        failed_rounds_total = 0
        missing_samples = 0
        for task in self.tasks:
            task_dir = phase1_dir / task
            summaries = sorted(task_dir.glob("*_summary.json"), key=lambda path: path.stat().st_mtime, reverse=True)
            if not summaries:
                missing_tasks.append(task)
                continue
            summary = read_json(summaries[0])
            samples_path = Path(summary.get("samples_path", "")) if summary.get("samples_path") else None
            samples_exists = bool(samples_path and samples_path.exists())
            failed_rounds = int(summary.get("failed_rounds", 0) or 0)
            failed_rounds_total += failed_rounds
            if not samples_exists:
                missing_samples += 1
            task_details[task] = {
                "summary_path": str(summaries[0]),
                "samples_path": str(samples_path) if samples_path else "",
                "samples_exists": samples_exists,
                "failed_rounds": failed_rounds,
                "repair_mode": summary.get("repair_mode", ""),
            }
        success = not missing_tasks and missing_samples == 0 and failed_rounds_total == 0
        remaining = failed_rounds_total + len(missing_tasks) + missing_samples
        status = "success" if success else "incomplete"
        summary = f"Phase 1 failed_rounds={failed_rounds_total}, missing_tasks={len(missing_tasks)}, missing_samples={missing_samples}"
        return StageAudit("phase1", success, status, summary, {"tasks": task_details, "missing_tasks": missing_tasks}, remaining)

    def audit_phase2(self) -> StageAudit:
        manifest_path = self.stage_dirs["phase2"] / "manifest.json"
        if not manifest_path.exists():
            return StageAudit("phase2", False, "missing_manifest", "Phase 2 manifest.json not found", {}, 1)
        manifest = read_json(manifest_path)
        dialogues_path = Path(manifest.get("dialogues_path", ""))
        rounds_path = Path(manifest.get("rounds_path", ""))
        errors_path = Path(manifest.get("errors_path", "")) if manifest.get("errors_path") else None
        success = dialogues_path.exists() and rounds_path.exists()
        remaining = 0 if success else 1
        summary = f"Phase 2 dialogues={manifest.get('dialogue_count', 0)}, rounds={manifest.get('round_count', 0)}, error_rounds={manifest.get('error_round_count', 0)}"
        return StageAudit(
            "phase2",
            success,
            "success" if success else "incomplete",
            summary,
            {
                "manifest_path": str(manifest_path),
                "dialogues_path": str(dialogues_path),
                "rounds_path": str(rounds_path),
                "errors_path": str(errors_path) if errors_path else "",
                "dialogue_count": manifest.get("dialogue_count", 0),
                "round_count": manifest.get("round_count", 0),
                "error_round_count": manifest.get("error_round_count", 0),
            },
            remaining,
        )

    def audit_phase3(self) -> StageAudit:
        validation_path = self.stage_dirs["phase3"] / "validation_summary.json"
        manifest_path = self.stage_dirs["phase3"] / "manifest.json"
        if not validation_path.exists() or not manifest_path.exists():
            return StageAudit("phase3", False, "missing_outputs", "Phase 3 validation_summary.json or manifest.json not found", {}, 1)
        validation = read_json(validation_path)
        manifest = read_json(manifest_path)
        success = bool(validation.get("turn_coverage_complete")) and bool(validation.get("session_coverage_complete")) and int(validation.get("error_record_count", 0) or 0) == 0
        remaining = max(int(manifest.get("expected_round_count", 0) or 0) - int(manifest.get("turn_success_count", 0) or 0), 0)
        remaining += max(int(manifest.get("expected_dialogue_count", 0) or 0) - int(manifest.get("session_success_count", 0) or 0), 0)
        remaining += int(validation.get("error_record_count", 0) or 0)
        summary = (
            f"Phase 3 turn={manifest.get('turn_success_count', 0)}/{manifest.get('expected_round_count', 0)}, "
            f"session={manifest.get('session_success_count', 0)}/{manifest.get('expected_dialogue_count', 0)}, "
            f"errors={validation.get('error_record_count', 0)}"
        )
        return StageAudit("phase3", success, "success" if success else "incomplete", summary, {"manifest": manifest, "validation": validation}, remaining)

    def audit_phase4(self) -> StageAudit:
        validation_path = self.stage_dirs["phase4"] / "validation_summary.json"
        manifest_path = self.stage_dirs["phase4"] / "manifest.json"
        if not validation_path.exists() or not manifest_path.exists():
            return StageAudit("phase4", False, "missing_outputs", "Phase 4 validation_summary.json or manifest.json not found", {}, 1)
        validation = read_json(validation_path)
        manifest = read_json(manifest_path)
        success = bool(validation.get("turn_coverage_complete")) and bool(validation.get("session_coverage_complete")) and int(validation.get("error_record_count", 0) or 0) == 0
        remaining = max(int(manifest.get("turn_candidate_count", 0) or 0) - int(manifest.get("turn_success_count", 0) or 0), 0)
        remaining += max(int(manifest.get("session_candidate_count", 0) or 0) - int(manifest.get("session_success_count", 0) or 0), 0)
        remaining += int(validation.get("error_record_count", 0) or 0)
        summary = (
            f"Phase 4 turn_attrib={manifest.get('turn_success_count', 0)}/{manifest.get('turn_candidate_count', 0)}, "
            f"session_attrib={manifest.get('session_success_count', 0)}/{manifest.get('session_candidate_count', 0)}, "
            f"errors={validation.get('error_record_count', 0)}"
        )
        return StageAudit("phase4", success, "success" if success else "incomplete", summary, {"manifest": manifest, "validation": validation}, remaining)

    def audit_phase5(self) -> StageAudit:
        manifest_path = self.stage_dirs["phase5"] / "manifest.json"
        step_results_path = self.stage_dirs["phase5"] / "report_step_results.json"
        if not manifest_path.exists():
            return StageAudit("phase5", False, "missing_manifest", "Phase 5 manifest.json not found", {}, 1)
        manifest = read_json(manifest_path)
        step_results = read_json(step_results_path) if step_results_path.exists() else {}
        analysis_status = manifest.get("analysis_status", "unknown")
        step_statuses = manifest.get("step_statuses", {}) or {}
        failed_steps = [name for name, status in step_statuses.items() if status != "success"]
        success = analysis_status == "success"
        status = analysis_status
        summary = f"Phase 5 analysis_status={analysis_status}, failed_steps={','.join(failed_steps) if failed_steps else 'none'}"
        remaining = len(failed_steps) if failed_steps else (0 if success else 1)
        return StageAudit("phase5", success, status, summary, {"manifest": manifest, "step_results": step_results, "failed_steps": failed_steps}, remaining)
    def run_phase1(self) -> StageAudit:
        phase1_dir = self.stage_dirs["phase1"]
        phase1_dir.mkdir(parents=True, exist_ok=True)
        command = self.build_phase1_command(include_resume=self.args.resume)

        print("[phase1] 开始主跑。")
        self.run_command("phase1", command)
        audit = self.audit_phase1()
        if audit.success:
            return audit

        strict_mode = parse_csv(self.args.phase1_repair_chain)[0] if parse_csv(self.args.phase1_repair_chain) else "resume_from_failure"
        light_modes = parse_csv(self.args.phase1_repair_chain)[1:] or ["current_turn_only"]

        print(f"[phase1] 主跑后仍有失败，先进行严格修复：{strict_mode}")
        strict_command = self.build_phase1_command(repair_failed=True, repair_mode=strict_mode)
        self.run_command("phase1", strict_command)
        audit = self.audit_phase1()
        if audit.success:
            return audit

        previous_remaining = audit.remaining_count
        for cycle in range(self.args.phase1_repair_cycles):
            light_mode = light_modes[min(cycle, len(light_modes) - 1)]
            print(f"[phase1] 进入轻量修复，第 {cycle + 1} 轮，模式：{light_mode}")
            repair_command = self.build_phase1_command(repair_failed=True, repair_mode=light_mode)
            self.run_command("phase1", repair_command)
            audit = self.audit_phase1()
            if audit.success:
                return audit
            if audit.remaining_count >= previous_remaining:
                print("[phase1] 轻量修复后剩余失败数未继续下降，停止后续修复。")
                break
            previous_remaining = audit.remaining_count
        return audit

    def run_phase2(self) -> StageAudit:
        phase2_dir = self.stage_dirs["phase2"]
        phase2_dir.mkdir(parents=True, exist_ok=True)
        command = self.base_python_command(TOOLS_DIR / "normalize_phase1_outputs.py")
        command.extend(["--input", str(self.stage_dirs["phase1"]), "--output-dir", str(phase2_dir)])
        if self.args.resume:
            command.append("--keep-existing")
        self.run_command("phase2", command)
        return self.audit_phase2()

    def phase3_repair_modes(self) -> tuple[tuple[str, str], tuple[str, str]]:
        turn_modes = parse_csv(self.args.phase3_turn_repair_chain) or ["full_context", "current_turn_only"]
        session_modes = parse_csv(self.args.phase3_session_repair_chain) or ["full_context", "light_context"]
        strict_pair = (turn_modes[0], session_modes[0])
        light_pair = (turn_modes[1] if len(turn_modes) > 1 else turn_modes[-1], session_modes[1] if len(session_modes) > 1 else session_modes[-1])
        return strict_pair, light_pair

    def run_phase3(self) -> StageAudit:
        phase3_dir = self.stage_dirs["phase3"]
        phase3_dir.mkdir(parents=True, exist_ok=True)
        command = self.base_python_command(self.phase3_script())
        command.extend(["--phase2-dir", str(self.stage_dirs["phase2"]), "--output-dir", str(phase3_dir), "--model", self.args.judge_model, "--max-workers", str(self.args.phase3_workers)])
        if self.args.resume:
            command.append("--resume")
        if self.args.dialogue_id:
            command.extend(["--dialogue-id", self.args.dialogue_id])
        elif self.args.limit is not None:
            command.extend(["--limit-dialogues", str(self.args.limit)])
        if self.args.phase3_allow_incomplete_dialogues:
            command.append("--allow-incomplete-dialogues")
        if self.args.save_prompt_text:
            command.append("--save-prompt-text")
        if self.args.phase3_api_key_env:
            command.extend(["--api-key-env", self.args.phase3_api_key_env])
        if self.args.phase3_api_url and self.args.phase3_backend == "internal":
            command.extend(["--api-url", self.args.phase3_api_url])

        print("[phase3] 开始主跑。")
        self.run_command("phase3", command)
        audit = self.audit_phase3()
        if audit.success:
            return audit

        strict_pair, light_pair = self.phase3_repair_modes()
        print(f"[phase3] 主跑后仍有失败，先进行严格修复：turn={strict_pair[0]}，session={strict_pair[1]}")
        strict_command = self.base_python_command(self.phase3_script())
        strict_command.extend(["--phase2-dir", str(self.stage_dirs["phase2"]), "--output-dir", str(phase3_dir), "--model", self.args.judge_model, "--repair-failed", "--turn-repair-mode", strict_pair[0], "--session-repair-mode", strict_pair[1], "--max-workers", str(self.args.phase3_workers)])
        if self.args.dialogue_id:
            strict_command.extend(["--dialogue-id", self.args.dialogue_id])
        elif self.args.limit is not None:
            strict_command.extend(["--limit-dialogues", str(self.args.limit)])
        if self.args.phase3_allow_incomplete_dialogues:
            strict_command.append("--allow-incomplete-dialogues")
        if self.args.save_prompt_text:
            strict_command.append("--save-prompt-text")
        if self.args.phase3_api_key_env:
            strict_command.extend(["--api-key-env", self.args.phase3_api_key_env])
        if self.args.phase3_api_url and self.args.phase3_backend == "internal":
            strict_command.extend(["--api-url", self.args.phase3_api_url])
        self.run_command("phase3", strict_command)
        audit = self.audit_phase3()
        if audit.success:
            return audit

        previous_remaining = audit.remaining_count
        for cycle in range(self.args.phase3_repair_cycles):
            print(f"[phase3] 进入轻量修复，第 {cycle + 1} 轮，turn={light_pair[0]}，session={light_pair[1]}")
            repair_command = self.base_python_command(self.phase3_script())
            repair_command.extend(["--phase2-dir", str(self.stage_dirs["phase2"]), "--output-dir", str(phase3_dir), "--model", self.args.judge_model, "--repair-failed", "--turn-repair-mode", light_pair[0], "--session-repair-mode", light_pair[1], "--max-workers", str(self.args.phase3_workers)])
            if self.args.dialogue_id:
                repair_command.extend(["--dialogue-id", self.args.dialogue_id])
            elif self.args.limit is not None:
                repair_command.extend(["--limit-dialogues", str(self.args.limit)])
            if self.args.phase3_allow_incomplete_dialogues:
                repair_command.append("--allow-incomplete-dialogues")
            if self.args.save_prompt_text:
                repair_command.append("--save-prompt-text")
            if self.args.phase3_api_key_env:
                repair_command.extend(["--api-key-env", self.args.phase3_api_key_env])
            if self.args.phase3_api_url and self.args.phase3_backend == "internal":
                repair_command.extend(["--api-url", self.args.phase3_api_url])
            self.run_command("phase3", repair_command)
            audit = self.audit_phase3()
            if audit.success:
                return audit
            if audit.remaining_count >= previous_remaining:
                print("[phase3] 轻量修复后剩余失败数未继续下降，停止后续修复。")
                break
            previous_remaining = audit.remaining_count
        return audit

    def run_phase4(self) -> StageAudit:
        phase4_dir = self.stage_dirs["phase4"]
        phase4_dir.mkdir(parents=True, exist_ok=True)
        command = self.base_python_command(self.phase4_script())
        command.extend(["--phase3-dir", str(self.stage_dirs["phase3"]), "--phase2-dir", str(self.stage_dirs["phase2"]), "--output-dir", str(phase4_dir), "--model", self.args.attribution_model, "--max-workers", str(self.args.phase4_workers)])
        if self.args.resume:
            command.append("--resume")
        if self.args.dialogue_id:
            command.extend(["--dialogue-id", self.args.dialogue_id])
        elif self.args.limit is not None:
            command.extend(["--limit-dialogues", str(self.args.limit)])
        if self.args.save_prompt_text:
            command.append("--save-prompt-text")
        if self.args.phase4_api_key_env:
            command.extend(["--api-key-env", self.args.phase4_api_key_env])
        if self.args.phase4_api_url and self.args.phase4_backend == "internal":
            command.extend(["--api-url", self.args.phase4_api_url])
        self.run_command("phase4", command)
        audit = self.audit_phase4()
        if audit.success:
            return audit
        previous_remaining = audit.remaining_count
        for _ in range(self.args.phase4_repair_cycles):
            repair_command = self.base_python_command(self.phase4_script())
            repair_command.extend(["--phase3-dir", str(self.stage_dirs["phase3"]), "--phase2-dir", str(self.stage_dirs["phase2"]), "--output-dir", str(phase4_dir), "--model", self.args.attribution_model, "--resume", "--repair-passes", "1", "--max-workers", str(self.args.phase4_workers)])
            if self.args.dialogue_id:
                repair_command.extend(["--dialogue-id", self.args.dialogue_id])
            elif self.args.limit is not None:
                repair_command.extend(["--limit-dialogues", str(self.args.limit)])
            if self.args.save_prompt_text:
                repair_command.append("--save-prompt-text")
            if self.args.phase4_api_key_env:
                repair_command.extend(["--api-key-env", self.args.phase4_api_key_env])
            if self.args.phase4_api_url and self.args.phase4_backend == "internal":
                repair_command.extend(["--api-url", self.args.phase4_api_url])
            self.run_command("phase4", repair_command)
            audit = self.audit_phase4()
            if audit.success:
                return audit
            if audit.remaining_count >= previous_remaining:
                break
            previous_remaining = audit.remaining_count
        return audit

    def run_phase5(self) -> StageAudit:
        phase5_dir = self.stage_dirs["phase5"]
        phase5_dir.mkdir(parents=True, exist_ok=True)
        if self.args.resume:
            self._reset_partial_phase5_resume_state(phase5_dir)
        command = self.base_python_command(self.phase5_script())
        command.extend(["--phase4-dir", str(self.stage_dirs["phase4"]), "--output-dir", str(phase5_dir), "--model", self.args.analysis_model])
        if self.args.resume:
            command.append("--resume")
        if self.args.save_prompt_text:
            command.append("--save-prompt-text")
        if self.args.phase5_api_key_env:
            command.extend(["--api-key-env", self.args.phase5_api_key_env])
        if self.args.phase5_api_url and self.args.phase5_backend == "internal":
            command.extend(["--api-url", self.args.phase5_api_url])
        self.run_command("phase5", command)
        audit = self.audit_phase5()
        if audit.success:
            return audit
        previous_remaining = audit.remaining_count
        for _ in range(self.args.phase5_repair_cycles):
            failed_steps = audit.details.get("failed_steps", [])
            if not failed_steps:
                break
            repair_command = self.base_python_command(self.phase5_script())
            repair_command.extend(["--phase4-dir", str(self.stage_dirs["phase4"]), "--output-dir", str(phase5_dir), "--model", self.args.analysis_model, "--resume", "--repair-failed", "--repair-steps", ",".join(failed_steps)])
            if self.args.save_prompt_text:
                repair_command.append("--save-prompt-text")
            if self.args.phase5_api_key_env:
                repair_command.extend(["--api-key-env", self.args.phase5_api_key_env])
            if self.args.phase5_api_url and self.args.phase5_backend == "internal":
                repair_command.extend(["--api-url", self.args.phase5_api_url])
            self.run_command("phase5", repair_command)
            audit = self.audit_phase5()
            if audit.success:
                return audit
            if audit.remaining_count >= previous_remaining:
                break
            previous_remaining = audit.remaining_count
        return audit

    def _reset_partial_phase5_resume_state(self, phase5_dir: Path) -> None:
        manifest_path = phase5_dir / "manifest.json"
        if manifest_path.exists():
            return
        partial_paths = [phase5_dir / name for name in PHASE5_PARTIAL_FILENAMES]
        existing_partial_paths = [path for path in partial_paths if path.exists()]
        if not existing_partial_paths:
            return
        print("[phase5] 检测到缺少 manifest 的半成品目录，已清理旧产物后重新执行。")
        for path in existing_partial_paths:
            path.unlink()

    def write_summary_markdown(self, pipeline_status: str) -> None:
        lines = [
            "# RUBRIC-MME Pipeline Summary",
            "",
            f"- Generated at: {now_iso()}",
            f"- Pipeline status: {pipeline_status}",
            f"- Output root: `{self.output_root}`",
            f"- Tasks: `{', '.join(self.tasks)}`",
            f"- Tested model: `{self.args.tested_model}`",
            f"- Judge model: `{self.args.judge_model}`",
            f"- Attribution model: `{self.args.attribution_model}`",
            f"- Analysis model: `{self.args.analysis_model}`",
            "",
            "## Stage Results",
            "",
        ]
        for stage in STAGE_ORDER:
            if stage not in self.stage_statuses:
                continue
            info = self.stage_statuses[stage]
            lines.extend([
                f"### {stage}",
                "",
                f"- Status: `{info['status']}`",
                f"- Success: `{info['success']}`",
                f"- Elapsed seconds: `{info['elapsed_seconds']}`",
                f"- Remaining count: `{info['remaining_count']}`",
                f"- Output dir: `{info['output_dir']}`",
                f"- Summary: {info['summary']}",
                "",
            ])
        (self.output_root / "pipeline_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the RUBRIC-MME Phase 1-5 pipeline with audit and repair orchestration.")
    parser.add_argument("--pipeline-name", default="rubric_mme_pipeline", help="Name written into pipeline manifest files.")
    parser.add_argument("--output-root", default=str(REPO_ROOT / "logs" / "rubric_mme_pipeline_run"), help="Root output directory for the pipeline controller.")
    parser.add_argument("--start-stage", choices=STAGE_ORDER, default="phase1", help="First stage to run.")
    parser.add_argument("--end-stage", choices=STAGE_ORDER, default="phase5", help="Last stage to run.")
    parser.add_argument("--fail-policy", choices=["stop", "best_effort"], default="stop", help="Whether to stop immediately when a stage still fails after repair.")
    parser.add_argument("--resume", action="store_true", help="Resume existing stage outputs when possible.")

    parser.add_argument("--tasks", default="rubric-mme", help="Task list. Use rubric-mme/omnibench/all for all four tasks.")
    parser.add_argument("--limit", type=int, default=None, help="Optional per-task/per-stage limit.")
    parser.add_argument("--dialogue-id", default="", help="Optional single dialogue id selector.")
    parser.add_argument("--data-root", default="", help="Optional Phase 1 data root override.")
    parser.add_argument("--media-root", default="", help="Optional Phase 1 media root override for internal backend.")

    parser.add_argument("--tested-model", default="gemini-2.5-pro", help="Tested model used in Phase 1.")
    parser.add_argument("--judge-model", default="gemini-3.1-pro-preview", help="Judge model used in Phase 3.")
    parser.add_argument("--attribution-model", default="gemini-3.1-pro-preview", help="Attribution model used in Phase 4.")
    parser.add_argument("--analysis-model", default="gemini-3.1-pro-preview", help="Analysis model used in Phase 5.")

    parser.add_argument("--phase1-backend", choices=["internal", "official", "openai_compatible", "gpt_openai_compatible", "claude_vision_openai_compatible", "claude_openai_sdk", "qwen_openai_compatible", "qwen_antchat_openai_sdk", "glm_antchat_openai_sdk"], default="internal")
    parser.add_argument("--phase3-backend", choices=["internal", "official"], default="internal")
    parser.add_argument("--phase4-backend", choices=["internal", "official"], default="internal")
    parser.add_argument("--phase5-backend", choices=["internal", "official"], default="internal")

    parser.add_argument("--phase1-dir", default="", help="Optional existing Phase 1 directory. Defaults to <output-root>/phase1")
    parser.add_argument("--phase2-dir", default="", help="Optional existing Phase 2 directory. Defaults to <output-root>/phase2")
    parser.add_argument("--phase3-dir", default="", help="Optional existing Phase 3 directory. Defaults to <output-root>/phase3")
    parser.add_argument("--phase4-dir", default="", help="Optional existing Phase 4 directory. Defaults to <output-root>/phase4")
    parser.add_argument("--phase5-dir", default="", help="Optional existing Phase 5 directory. Defaults to <output-root>/phase5")

    parser.add_argument("--phase1-workers", type=int, default=1)
    parser.add_argument("--phase3-workers", type=int, default=1)
    parser.add_argument("--phase4-workers", type=int, default=1)

    parser.add_argument("--phase1-repair-chain", default="resume_from_failure,current_turn_only", help="Comma separated Phase 1 repair mode chain.")
    parser.add_argument("--phase1-repair-cycles", type=int, default=3, help="严格修复一次之后，轻量修复最多循环多少轮。")
    parser.add_argument("--phase3-turn-repair-chain", default="full_context,current_turn_only", help="Comma separated Phase 3 turn repair modes.")
    parser.add_argument("--phase3-session-repair-chain", default="full_context,light_context", help="Comma separated Phase 3 session repair modes.")
    parser.add_argument("--phase3-repair-cycles", type=int, default=3, help="严格修复一次之后，轻量修复最多循环多少轮。")
    parser.add_argument("--phase4-repair-cycles", type=int, default=1, help="How many Phase 4 repair attempts to run after the main pass.")
    parser.add_argument("--phase5-repair-cycles", type=int, default=1, help="How many Phase 5 step repair attempts to run after the main pass.")
    parser.add_argument("--phase3-allow-incomplete-dialogues", action="store_true", help="Pass through to Phase 3 when continuing despite incomplete Phase 1 results.")
    parser.add_argument("--save-prompt-text", action="store_true", help="Pass prompt-saving flags through to Phases 3-5.")

    parser.add_argument("--phase1-api-url", default="")
    parser.add_argument("--phase3-api-url", default="")
    parser.add_argument("--phase4-api-url", default="")
    parser.add_argument("--phase5-api-url", default="")
    parser.add_argument("--phase1-api-key-env", default="")
    parser.add_argument("--phase1-image-input-mode", default="auto", help="Phase 1 openai_compatible 图片输入策略。")
    parser.add_argument("--phase1-video-route", default="auto", help="Phase 1 qwen_openai_compatible 视频通道策略：auto/video/frames。")
    parser.add_argument("--phase1-video-input-mode", default="auto", help="Phase 1 openai_compatible 视频输入策略。")
    parser.add_argument("--phase1-video-history-mode", default="text_only", help="Phase 1 openai_compatible 视频历史输入策略。")
    parser.add_argument("--phase1-video-history-max-visual-rounds", type=int, default=1, help="Phase 1 qwen_openai_compatible video 通道下最多保留多少个最近历史视频轮次。")
    parser.add_argument("--phase1-video-history-max-inline-bytes-total", type=int, default=5500000, help="Phase 1 qwen_openai_compatible video 通道下当前轮与历史轮视频原始体积总预算。")
    parser.add_argument("--phase1-video-precompressed-mode", default="prefer", help="Phase 1 openai_compatible 是否优先使用预压缩视频。")
    parser.add_argument("--phase1-video-precompressed-field", default="compressed_clip_path", help="Phase 1 openai_compatible 预压缩视频字段名。")
    parser.add_argument("--phase1-video-compress-mode", default="auto", help="Phase 1 openai_compatible 运行时视频压缩兜底策略。")
    parser.add_argument("--phase1-video-max-inline-bytes", type=int, default=5500000, help="Phase 1 openai_compatible 单个视频内联大小阈值。")
    parser.add_argument("--phase1-video-prepared-frame-mode", default="prefer", help="Phase 1 gpt_openai_compatible 是否优先使用预抽帧结果。")
    parser.add_argument("--phase1-video-prepared-dir-field", default="gpt_frame_dir", help="Phase 1 gpt_openai_compatible 预抽帧目录字段名。")
    parser.add_argument("--phase1-video-prepared-paths-field", default="gpt_frame_paths", help="Phase 1 gpt_openai_compatible 预抽帧路径列表字段名。")
    parser.add_argument("--phase1-video-prepared-profile-field", default="gpt_frame_profile", help="Phase 1 gpt_openai_compatible 预抽帧 profile 字段名。")
    parser.add_argument("--phase1-video-prepared-total-bytes-field", default="gpt_frame_total_bytes", help="Phase 1 gpt_openai_compatible 预抽帧总字节数字段名。")
    parser.add_argument("--phase1-video-prepared-strategy-field", default="gpt_frame_sampling_strategy", help="Phase 1 gpt_openai_compatible 预抽帧采样策略字段名。")
    parser.add_argument("--phase1-video-prepared-source-size-field", default="gpt_frame_source_size_bytes", help="Phase 1 gpt_openai_compatible 原视频大小字段名。")
    parser.add_argument("--phase1-video-prepared-duration-field", default="gpt_frame_duration_seconds", help="Phase 1 gpt_openai_compatible 视频时长字段名。")
    parser.add_argument("--phase1-video-frame-root-name", default="video_final_gpt_frames", help="Phase 1 gpt_openai_compatible 运行时 fallback 帧缓存目录名。")
    parser.add_argument("--phase1-video-frame-count", type=int, default=10, help="Phase 1 gpt_openai_compatible 基础目标帧数。")
    parser.add_argument("--phase1-video-frame-max-side", type=int, default=768, help="Phase 1 gpt_openai_compatible 帧图长边最大尺寸。")
    parser.add_argument("--phase1-video-frame-jpeg-quality", type=int, default=8, help="Phase 1 gpt_openai_compatible JPEG 质量参数。")
    parser.add_argument("--phase1-video-frame-max-inline-bytes", type=int, default=3000000, help="Phase 1 gpt_openai_compatible 单轮所有帧图总大小预算。")
    parser.add_argument("--phase1-video-frame-sampling-strategy", default="hybrid_tail", help="Phase 1 gpt_openai_compatible 抽帧采样策略。")
    parser.add_argument("--phase1-video-max-images-per-request", type=int, default=50, help="Phase 1 gpt_openai_compatible 单次请求允许的最大图片数量。")
    parser.add_argument("--phase1-video-max-frame-images-per-request", type=int, default=40, help="Phase 1 qwen_openai_compatible frames 通道单次请求允许的最大帧图数量。")
    parser.add_argument("--phase1-video-history-max-frames-per-round", type=int, default=4, help="Phase 1 gpt_openai_compatible 启用历史视频帧时，每个历史轮次最多保留多少帧。")
    parser.add_argument("--phase3-api-key-env", default="")
    parser.add_argument("--phase4-api-key-env", default="")
    parser.add_argument("--phase5-api-key-env", default="")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    runner = PipelineRunner(args)
    return runner.run()


if __name__ == "__main__":
    raise SystemExit(main())
