from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from gpt_video_frame_utils import (
    DEFAULT_VIDEO_FRAME_COUNT,
    DEFAULT_VIDEO_FRAME_JPEG_QUALITY,
    DEFAULT_VIDEO_FRAME_MAX_INLINE_BYTES,
    DEFAULT_VIDEO_FRAME_MAX_SIDE,
    DEFAULT_VIDEO_FRAME_ROOT_NAME,
    DEFAULT_VIDEO_FRAME_SAMPLING_STRATEGY,
    DEFAULT_VIDEO_PREPARED_DIR_FIELD,
    DEFAULT_VIDEO_PREPARED_DURATION_FIELD,
    DEFAULT_VIDEO_PREPARED_PATHS_FIELD,
    DEFAULT_VIDEO_PREPARED_PROFILE_FIELD,
    DEFAULT_VIDEO_PREPARED_SOURCE_SIZE_FIELD,
    DEFAULT_VIDEO_PREPARED_STRATEGY_FIELD,
    DEFAULT_VIDEO_PREPARED_TOTAL_BYTES_FIELD,
    prepare_video_frames,
)
from phase1_common import DEFAULT_DATA_ROOT, DEFAULT_MEDIA_ROOT, dump_json, utc_now


@dataclass(frozen=True)
class ClipTask:
    item_index: int
    conversation_index: int
    dialogue_id: str
    round_index: int
    clip_rel_path: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="预抽帧 GPT Phase 1 视频片段，并将帧路径回写到视频数据 JSON。")
    parser.add_argument(
        "--dataset-json",
        default=str((DEFAULT_DATA_ROOT / "video_final_with_vqa_category.json").resolve()),
        help="输入视频数据 JSON。",
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="输出 JSON 路径。默认原地覆盖 dataset-json。",
    )
    parser.add_argument(
        "--media-root",
        default=str(DEFAULT_MEDIA_ROOT.resolve()),
        help="媒体根目录，clip_path 相对路径基于该目录解析。",
    )
    parser.add_argument(
        "--frame-root-name",
        default=DEFAULT_VIDEO_FRAME_ROOT_NAME,
        help="预抽帧输出到 media-root 下的哪个子目录。",
    )
    parser.add_argument(
        "--prepared-dir-field",
        default=DEFAULT_VIDEO_PREPARED_DIR_FIELD,
        help="回写到 AI 轮次中的帧目录字段名。",
    )
    parser.add_argument(
        "--prepared-paths-field",
        default=DEFAULT_VIDEO_PREPARED_PATHS_FIELD,
        help="回写到 AI 轮次中的帧路径列表字段名。",
    )
    parser.add_argument(
        "--prepared-profile-field",
        default=DEFAULT_VIDEO_PREPARED_PROFILE_FIELD,
        help="回写到 AI 轮次中的抽帧 profile 字段名。",
    )
    parser.add_argument(
        "--prepared-total-bytes-field",
        default=DEFAULT_VIDEO_PREPARED_TOTAL_BYTES_FIELD,
        help="回写到 AI 轮次中的总字节数字段名。",
    )
    parser.add_argument(
        "--prepared-strategy-field",
        default=DEFAULT_VIDEO_PREPARED_STRATEGY_FIELD,
        help="回写到 AI 轮次中的采样策略字段名。",
    )
    parser.add_argument(
        "--prepared-source-size-field",
        default=DEFAULT_VIDEO_PREPARED_SOURCE_SIZE_FIELD,
        help="回写到 AI 轮次中的原视频大小字段名。",
    )
    parser.add_argument(
        "--prepared-duration-field",
        default=DEFAULT_VIDEO_PREPARED_DURATION_FIELD,
        help="回写到 AI 轮次中的视频时长字段名。",
    )
    parser.add_argument("--dialogue-id", default=None, help="只处理指定 dialogue_id。")
    parser.add_argument("--limit", type=int, default=None, help="最多处理多少个 dialogue。")
    parser.add_argument("--max-clips", type=int, default=None, help="最多处理多少个 AI 视频片段，用于小样本测试。")
    parser.add_argument("--max-workers", type=int, default=1, help="预抽帧并行 worker 数。")
    parser.add_argument("--video-frame-count", type=int, default=DEFAULT_VIDEO_FRAME_COUNT, help="基础目标帧数。")
    parser.add_argument("--video-frame-max-side", type=int, default=DEFAULT_VIDEO_FRAME_MAX_SIDE, help="帧图长边最大尺寸。")
    parser.add_argument("--video-frame-jpeg-quality", type=int, default=DEFAULT_VIDEO_FRAME_JPEG_QUALITY, help="JPEG 质量参数，数值越大体积通常越小。")
    parser.add_argument("--video-frame-max-inline-bytes", type=int, default=DEFAULT_VIDEO_FRAME_MAX_INLINE_BYTES, help="单轮所有帧图总大小预算。")
    parser.add_argument(
        "--video-frame-sampling-strategy",
        choices=["uniform", "hybrid_tail"],
        default=DEFAULT_VIDEO_FRAME_SAMPLING_STRATEGY,
        help="抽帧采样策略。",
    )
    parser.add_argument("--summary-path", default=None, help="摘要 JSON 路径。默认与 output-json 同目录。")
    return parser.parse_args()


def resolve_local_path(media_root: Path, relative_path: str) -> Optional[Path]:
    if not relative_path:
        return None
    candidate = Path(relative_path)
    if candidate.is_absolute():
        return candidate if candidate.exists() else None
    combined = (media_root / candidate).resolve()
    return combined if combined.exists() else None


def build_clip_tasks(
    raw_items: List[Dict[str, Any]],
    dialogue_id: Optional[str],
    limit: Optional[int],
    max_clips: Optional[int],
) -> List[ClipTask]:
    tasks: List[ClipTask] = []
    matched_dialogues = 0
    for item_index, item in enumerate(raw_items):
        current_dialogue_id = str(item.get("_ai_unique_id_", "") or "").strip()
        if dialogue_id and current_dialogue_id != dialogue_id:
            continue
        matched_dialogues += 1
        if limit is not None and matched_dialogues > limit:
            break
        round_index = 0
        for conversation_index, entry in enumerate((item.get("stream_conversation") or {}).get("conversations", [])):
            if entry.get("speaker") != "ai":
                continue
            clip_rel_path = str(entry.get("clip_path", "") or "").strip()
            if clip_rel_path:
                tasks.append(
                    ClipTask(
                        item_index=item_index,
                        conversation_index=conversation_index,
                        dialogue_id=current_dialogue_id,
                        round_index=round_index,
                        clip_rel_path=clip_rel_path,
                    )
                )
                if max_clips is not None and len(tasks) >= max_clips:
                    return tasks
            round_index += 1
    return tasks


def process_clip_task(task: ClipTask, args: argparse.Namespace, media_root: Path, prepared_root: Path) -> Dict[str, Any]:
    source_path = resolve_local_path(media_root, task.clip_rel_path)
    if source_path is None:
        raise FileNotFoundError(f"找不到本地视频：{task.clip_rel_path}")
    frame_paths, frame_meta = prepare_video_frames(
        source_path=source_path,
        media_root=media_root,
        prepared_root=prepared_root,
        base_frame_count=int(args.video_frame_count),
        base_max_side=int(args.video_frame_max_side),
        base_jpeg_quality=int(args.video_frame_jpeg_quality),
        max_inline_bytes=int(args.video_frame_max_inline_bytes),
        sampling_strategy=args.video_frame_sampling_strategy,
    )
    return {
        "task": task,
        "source_path": str(source_path),
        "frame_paths": [str(path) for path in frame_paths],
        "frame_meta": frame_meta,
    }


def main() -> None:
    args = parse_args()
    dataset_json = Path(args.dataset_json).resolve()
    output_json = Path(args.output_json).resolve() if args.output_json else dataset_json
    media_root = Path(args.media_root).resolve()
    prepared_root = (media_root / args.frame_root_name).resolve()

    with dataset_json.open("r", encoding="utf-8") as file:
        raw_items = json.load(file)
    if not isinstance(raw_items, list):
        raise ValueError(f"文件 {dataset_json} 不是 list JSON")

    clip_tasks = build_clip_tasks(raw_items, args.dialogue_id, args.limit, args.max_clips)
    results: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []

    if clip_tasks:
        max_workers = max(1, int(args.max_workers))
        if max_workers == 1 or len(clip_tasks) <= 1:
            for task in clip_tasks:
                try:
                    results.append(process_clip_task(task, args, media_root, prepared_root))
                except Exception as exc:
                    failures.append(
                        {
                            "dialogue_id": task.dialogue_id,
                            "round_index": task.round_index,
                            "clip_rel_path": task.clip_rel_path,
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )
        else:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_task = {
                    executor.submit(process_clip_task, task, args, media_root, prepared_root): task
                    for task in clip_tasks
                }
                for future in as_completed(future_to_task):
                    task = future_to_task[future]
                    try:
                        results.append(future.result())
                    except Exception as exc:
                        failures.append(
                            {
                                "dialogue_id": task.dialogue_id,
                                "round_index": task.round_index,
                                "clip_rel_path": task.clip_rel_path,
                                "error": f"{type(exc).__name__}: {exc}",
                            }
                        )

    result_index = {(result["task"].item_index, result["task"].conversation_index): result for result in results}

    prepared_count = 0
    reused_existing_count = 0
    for item_index, item in enumerate(raw_items):
        conversations = (item.get("stream_conversation") or {}).get("conversations", [])
        for conversation_index, entry in enumerate(conversations):
            if entry.get("speaker") != "ai":
                continue
            result = result_index.get((item_index, conversation_index))
            if result is None:
                continue
            frame_meta = result["frame_meta"]
            entry[args.prepared_dir_field] = str(frame_meta.get("prepared_local_path", ""))
            entry[args.prepared_paths_field] = list(frame_meta.get("prepared_rel_paths", []))
            entry[args.prepared_profile_field] = str(frame_meta.get("profile", ""))
            entry[args.prepared_total_bytes_field] = frame_meta.get("prepared_size_bytes")
            entry[args.prepared_strategy_field] = str(frame_meta.get("sampling_strategy", ""))
            entry[args.prepared_source_size_field] = frame_meta.get("source_size_bytes")
            entry[args.prepared_duration_field] = frame_meta.get("duration_seconds")
            prepared_count += 1
            if frame_meta.get("used_existing"):
                reused_existing_count += 1

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as file:
        json.dump(raw_items, file, ensure_ascii=False, indent=2)

    summary = {
        "benchmark_name": "RUBRIC-MME",
        "phase": "gpt_video_preextract",
        "generated_at": utc_now(),
        "dataset_json": str(dataset_json),
        "output_json": str(output_json),
        "media_root": str(media_root),
        "prepared_root": str(prepared_root),
        "dialogue_id": args.dialogue_id or "",
        "limit": args.limit,
        "max_clips": args.max_clips,
        "max_workers": max(1, int(args.max_workers)),
        "task_count": len(clip_tasks),
        "prepared_count": prepared_count,
        "reused_existing_count": reused_existing_count,
        "failure_count": len(failures),
        "video_frame_count": int(args.video_frame_count),
        "video_frame_max_side": int(args.video_frame_max_side),
        "video_frame_jpeg_quality": int(args.video_frame_jpeg_quality),
        "video_frame_max_inline_bytes": int(args.video_frame_max_inline_bytes),
        "video_frame_sampling_strategy": args.video_frame_sampling_strategy,
        "failures": failures,
    }
    summary_path = (
        Path(args.summary_path).resolve()
        if args.summary_path
        else output_json.with_name(f"{output_json.stem}_gpt_video_prepare_summary.json")
    )
    dump_json(summary_path, summary)


if __name__ == "__main__":
    main()
