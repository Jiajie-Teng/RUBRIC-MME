from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import imageio_ffmpeg  # type: ignore
except Exception:  # pragma: no cover
    imageio_ffmpeg = None

from phase1_common import DEFAULT_DATA_ROOT, DEFAULT_MEDIA_ROOT, dump_json, utc_now


DEFAULT_COMPRESSED_ROOT_NAME = 'video_final_doubao_compressed'
DEFAULT_COMPRESSED_FIELD = 'compressed_clip_path'
DEFAULT_VIDEO_MAX_INLINE_BYTES = 5_500_000


@dataclass(frozen=True)
class ClipTask:
    item_index: int
    conversation_index: int
    dialogue_id: str
    round_index: int
    clip_rel_path: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='预压缩 Doubao Phase 1 视频片段，并将相对路径回写到视频数据 JSON。')
    parser.add_argument('--dataset-json', default=str((DEFAULT_DATA_ROOT / 'video_final_with_vqa_category.json').resolve()), help='输入视频数据 JSON。')
    parser.add_argument('--output-json', default=None, help='输出 JSON 路径。默认原地覆盖 dataset-json。')
    parser.add_argument('--media-root', default=str(DEFAULT_MEDIA_ROOT.resolve()), help='媒体根目录，clip_path 相对路径基于该目录解析。')
    parser.add_argument('--compressed-root-name', default=DEFAULT_COMPRESSED_ROOT_NAME, help='压缩视频输出到 media-root 下的哪个子目录。')
    parser.add_argument('--compressed-field', default=DEFAULT_COMPRESSED_FIELD, help='回写到 AI 轮次中的压缩视频相对路径字段名。')
    parser.add_argument('--video-max-inline-bytes', type=int, default=DEFAULT_VIDEO_MAX_INLINE_BYTES, help='目标内联原始体积上限（字节）。需要为 base64 膨胀和请求 JSON 预留空间。超过该阈值的视频会被压缩。')
    parser.add_argument('--dialogue-id', default=None, help='只处理指定 dialogue_id。')
    parser.add_argument('--limit', type=int, default=None, help='最多处理多少个 dialogue。')
    parser.add_argument('--max-clips', type=int, default=None, help='最多处理多少个 AI 视频片段，用于小样本测试。')
    parser.add_argument('--max-workers', type=int, default=1, help='压缩并行 worker 数。')
    parser.add_argument('--summary-path', default=None, help='压缩摘要 JSON 路径。默认写到 output-json 同目录。')
    return parser.parse_args()


def resolve_local_path(media_root: Path, relative_path: str) -> Optional[Path]:
    if not relative_path:
        return None
    candidate = Path(relative_path)
    if candidate.is_absolute():
        return candidate if candidate.exists() else None
    combined = (media_root / candidate).resolve()
    return combined if combined.exists() else None


def resolve_ffmpeg_executable() -> Optional[str]:
    if imageio_ffmpeg is None:
        return None
    try:
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def relative_storage_dir_for_source(source_path: Path, media_root: Path, compressed_root: Path) -> Path:
    try:
        source_rel = source_path.resolve().relative_to(media_root.resolve())
    except ValueError:
        source_rel = Path(source_path.name)
    trailing_parent = Path(*source_rel.parts[1:-1]) if len(source_rel.parts) > 2 else Path()
    return compressed_root / trailing_parent


def transcode_video_candidates(source_path: Path, media_root: Path, compressed_root: Path) -> List[Tuple[Path, List[str], Dict[str, Any]]]:
    target_dir = relative_storage_dir_for_source(source_path, media_root, compressed_root)
    target_dir.mkdir(parents=True, exist_ok=True)
    try:
        source_rel = source_path.resolve().relative_to(media_root.resolve()).as_posix()
    except ValueError:
        source_rel = source_path.name
    stem_hash = hashlib.sha1(source_rel.encode('utf-8')).hexdigest()[:12]
    profiles = [
        ('p1', ['-vf', 'scale=w=min(960\\,iw):h=-2', '-crf', '32', '-b:a', '64k'], {'scale': 960, 'crf': 32, 'audio_bitrate': '64k'}),
        ('p2', ['-vf', 'scale=w=min(720\\,iw):h=-2', '-crf', '34', '-b:a', '48k'], {'scale': 720, 'crf': 34, 'audio_bitrate': '48k'}),
        ('p3', ['-vf', 'scale=w=min(480\\,iw):h=-2', '-crf', '36', '-b:a', '32k'], {'scale': 480, 'crf': 36, 'audio_bitrate': '32k'}),
        ('p4', ['-vf', 'scale=w=min(360\\,iw):h=-2', '-crf', '38', '-b:a', '24k'], {'scale': 360, 'crf': 38, 'audio_bitrate': '24k'}),
        ('p5', ['-vf', 'scale=w=min(320\\,iw):h=-2', '-crf', '40', '-b:a', '16k'], {'scale': 320, 'crf': 40, 'audio_bitrate': '16k'}),
    ]
    candidates: List[Tuple[Path, List[str], Dict[str, Any]]] = []
    for profile_name, extra_args, meta in profiles:
        target_path = target_dir / f'{source_path.stem}_{stem_hash}_{profile_name}.mp4'
        ffmpeg_args = [
            '-y',
            '-i',
            str(source_path),
            '-map',
            '0:v:0',
            '-map',
            '0:a:0?',
            '-c:v',
            'libx264',
            '-preset',
            'veryfast',
            *extra_args,
            '-pix_fmt',
            'yuv420p',
            '-c:a',
            'aac',
            '-ac',
            '1',
            '-movflags',
            '+faststart',
            str(target_path),
        ]
        candidates.append((target_path, ffmpeg_args, {'profile': profile_name, **meta}))
    return candidates


def compress_video_for_dataset(source_path: Path, media_root: Path, compressed_root: Path, max_inline_bytes: int) -> Tuple[Optional[Path], Dict[str, Any]]:
    original_size = source_path.stat().st_size
    if original_size <= max_inline_bytes:
        return None, {
            'needs_compression': False,
            'original_size_bytes': original_size,
            'prepared_size_bytes': original_size,
        }

    ffmpeg_exe = resolve_ffmpeg_executable()
    if not ffmpeg_exe:
        raise RuntimeError('当前环境没有可用 ffmpeg，无法预压缩超大视频。')

    profile_error_messages: List[str] = []
    for candidate_path, ffmpeg_args, profile_meta in transcode_video_candidates(source_path, media_root, compressed_root):
        if candidate_path.exists() and candidate_path.stat().st_size <= max_inline_bytes:
            return candidate_path, {
                'needs_compression': True,
                'used_existing': True,
                'original_size_bytes': original_size,
                'prepared_size_bytes': candidate_path.stat().st_size,
                **profile_meta,
            }
        command = [ffmpeg_exe, *ffmpeg_args]
        completed = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        if completed.returncode != 0 or not candidate_path.exists():
            stderr_text = completed.stderr.decode('utf-8', errors='ignore').strip()
            profile_error_messages.append(f"{profile_meta.get('profile')}: {stderr_text[-300:]}")
            continue
        prepared_size = candidate_path.stat().st_size
        if prepared_size <= max_inline_bytes:
            return candidate_path, {
                'needs_compression': True,
                'used_existing': False,
                'original_size_bytes': original_size,
                'prepared_size_bytes': prepared_size,
                **profile_meta,
            }

    joined_errors = ' | '.join(profile_error_messages[-3:]).strip()
    raise RuntimeError(
        f'视频压缩后仍超过内联上限 {max_inline_bytes} 字节，原始大小 {original_size} 字节。'
        + (f' 最近失败信息：{joined_errors}' if joined_errors else '')
    )


def build_clip_tasks(raw_items: List[Dict[str, Any]], dialogue_id: Optional[str], limit: Optional[int], max_clips: Optional[int]) -> List[ClipTask]:
    tasks: List[ClipTask] = []
    matched_dialogues = 0
    for item_index, item in enumerate(raw_items):
        current_dialogue_id = str(item.get('_ai_unique_id_', '') or '').strip()
        if dialogue_id and current_dialogue_id != dialogue_id:
            continue
        matched_dialogues += 1
        if limit is not None and matched_dialogues > limit:
            break
        round_index = 0
        for conversation_index, entry in enumerate((item.get('stream_conversation') or {}).get('conversations', [])):
            if entry.get('speaker') != 'ai':
                continue
            clip_rel_path = str(entry.get('clip_path', '') or '').strip()
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


def process_clip_task(task: ClipTask, media_root: Path, compressed_root: Path, max_inline_bytes: int) -> Dict[str, Any]:
    source_path = resolve_local_path(media_root, task.clip_rel_path)
    if source_path is None:
        raise FileNotFoundError(f'找不到本地视频：{task.clip_rel_path}')
    prepared_path, meta = compress_video_for_dataset(source_path, media_root, compressed_root, max_inline_bytes)
    return {
        'task': task,
        'source_path': str(source_path),
        'prepared_path': str(prepared_path) if prepared_path else '',
        'prepared_rel_path': str(prepared_path.resolve().relative_to(media_root.resolve())).replace('\\', '/') if prepared_path else '',
        'meta': meta,
    }


def main() -> None:
    args = parse_args()
    dataset_json = Path(args.dataset_json).resolve()
    output_json = Path(args.output_json).resolve() if args.output_json else dataset_json
    media_root = Path(args.media_root).resolve()
    compressed_root = (media_root / args.compressed_root_name).resolve()

    with dataset_json.open('r', encoding='utf-8') as file:
        raw_items = json.load(file)
    if not isinstance(raw_items, list):
        raise ValueError(f'文件 {dataset_json} 不是 list JSON。')

    clip_tasks = build_clip_tasks(raw_items, args.dialogue_id, args.limit, args.max_clips)
    results: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []
    if clip_tasks:
        max_workers = max(1, int(args.max_workers))
        if max_workers == 1 or len(clip_tasks) <= 1:
            for task in clip_tasks:
                try:
                    results.append(process_clip_task(task, media_root, compressed_root, int(args.video_max_inline_bytes)))
                except Exception as exc:
                    failures.append({'dialogue_id': task.dialogue_id, 'round_index': task.round_index, 'clip_rel_path': task.clip_rel_path, 'error': f'{type(exc).__name__}: {exc}'})
        else:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_task = {
                    executor.submit(process_clip_task, task, media_root, compressed_root, int(args.video_max_inline_bytes)): task
                    for task in clip_tasks
                }
                for future in as_completed(future_to_task):
                    task = future_to_task[future]
                    try:
                        results.append(future.result())
                    except Exception as exc:
                        failures.append({'dialogue_id': task.dialogue_id, 'round_index': task.round_index, 'clip_rel_path': task.clip_rel_path, 'error': f'{type(exc).__name__}: {exc}'})

    result_index = {(result['task'].item_index, result['task'].conversation_index): result for result in results}
    compressed_field = args.compressed_field
    compressed_count = 0
    reused_count = 0
    skipped_small_count = 0
    for item_index, item in enumerate(raw_items):
        conversations = (item.get('stream_conversation') or {}).get('conversations', [])
        for conversation_index, entry in enumerate(conversations):
            if entry.get('speaker') != 'ai':
                continue
            result = result_index.get((item_index, conversation_index))
            if result is None:
                continue
            meta = result['meta']
            if result['prepared_rel_path']:
                entry[compressed_field] = result['prepared_rel_path']
                entry[f'{compressed_field}_profile'] = meta.get('profile', '')
                entry[f'{compressed_field}_size_bytes'] = meta.get('prepared_size_bytes')
                entry[f'{compressed_field}_original_size_bytes'] = meta.get('original_size_bytes')
                compressed_count += 1
                if meta.get('used_existing'):
                    reused_count += 1
            else:
                entry.pop(compressed_field, None)
                entry.pop(f'{compressed_field}_profile', None)
                entry.pop(f'{compressed_field}_size_bytes', None)
                entry.pop(f'{compressed_field}_original_size_bytes', None)
                skipped_small_count += 1

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open('w', encoding='utf-8') as file:
        json.dump(raw_items, file, ensure_ascii=False, indent=2)

    summary = {
        'benchmark_name': 'RUBRIC-MME',
        'phase': 'doubao_video_precompress',
        'generated_at': utc_now(),
        'dataset_json': str(dataset_json),
        'output_json': str(output_json),
        'media_root': str(media_root),
        'compressed_root': str(compressed_root),
        'compressed_field': compressed_field,
        'video_max_inline_bytes': int(args.video_max_inline_bytes),
        'dialogue_id': args.dialogue_id or '',
        'limit': args.limit,
        'max_clips': args.max_clips,
        'max_workers': max(1, int(args.max_workers)),
        'task_count': len(clip_tasks),
        'compressed_count': compressed_count,
        'reused_existing_count': reused_count,
        'skipped_small_count': skipped_small_count,
        'failure_count': len(failures),
        'failures': failures,
    }
    summary_path = Path(args.summary_path).resolve() if args.summary_path else output_json.with_name(f'{output_json.stem}_doubao_video_prepare_summary.json')
    dump_json(summary_path, summary)


if __name__ == '__main__':
    main()
