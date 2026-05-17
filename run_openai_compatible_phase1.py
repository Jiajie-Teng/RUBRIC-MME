from __future__ import annotations

import argparse
import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import mimetypes
import os
import subprocess
from threading import Lock
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import requests
from requests.adapters import HTTPAdapter
try:
    import imageio_ffmpeg  # type: ignore
except Exception:  # pragma: no cover
    imageio_ffmpeg = None

from phase1_common import (
    DEFAULT_DATA_ROOT,
    DEFAULT_MEDIA_ROOT,
    IMAGE_EXTENSIONS,
    AUDIO_EXTENSIONS,
    VIDEO_EXTENSIONS,
    Phase1DialogueWorkItem,
    REPO_ROOT,
    TASK_SPECS,
    TaskSpec,
    build_phase1_work_items,
    dump_json,
    is_phase1_round_success,
    load_docs_for_task,
    merge_phase1_payloads,
    resolve_task_names,
    safe_name,
    utc_now,
    write_jsonl,
)

DEFAULT_API_URL = 'https://matrixllm.alipay.com/v1/chat/completions'
DEFAULT_MODEL = 'doubao-seed-2-0-pro-260215'
DEFAULT_OUTPUT_DIR = REPO_ROOT / 'logs' / 'rubric_mme_openai_compatible_phase1'
DEFAULT_API_KEY_ENV = 'MATRIXLLM_API_KEY'
DEFAULT_VIDEO_PRECOMPRESSED_FIELD = 'compressed_clip_path'
DEFAULT_VIDEO_MAX_INLINE_BYTES = 5_500_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='运行 OpenAI 兼容多模态接口版 RUBRIC-MME Phase 1。')
    parser.add_argument('--tasks', default='rubric-mme', help='逗号分隔的任务名；rubric-mme/omnibench/all 表示四个任务全部运行。')
    parser.add_argument('--model', default=DEFAULT_MODEL, help='被测试模型名称。')
    parser.add_argument('--data-root', default=str(DEFAULT_DATA_ROOT if DEFAULT_DATA_ROOT.exists() else REPO_ROOT), help='RUBRIC-MME JSON 数据根目录。')
    parser.add_argument('--media-root', default=str(DEFAULT_MEDIA_ROOT if DEFAULT_MEDIA_ROOT.exists() else REPO_ROOT), help='图片、视频、音频等相对路径所对应的媒体根目录。')
    parser.add_argument('--output-dir', default=str(DEFAULT_OUTPUT_DIR), help='输出目录。')
    parser.add_argument('--limit', type=int, default=None, help='每个任务最多处理多少个 dialogue。')
    parser.add_argument('--dialogue-id', default=None, help='只运行指定的 dialogue_id。')
    parser.add_argument('--resume', action='store_true', help='基于已有 samples.jsonl 跳过已完成 dialogue。')
    parser.add_argument('--repair-failed', action='store_true', help='只修复已有 samples.jsonl 中失败或未完整的 dialogue。')
    parser.add_argument('--repair-mode', choices=['resume_from_failure', 'current_turn_only'], default='resume_from_failure', help='repair 模式：resume_from_failure 从首个失败轮继续；current_turn_only 仅修当前失败轮。')
    parser.add_argument('--max-workers', type=int, default=1, help='session 级并行 worker 数，1 表示串行执行。')
    parser.add_argument('--api-url', default=DEFAULT_API_URL, help='OpenAI 兼容 chat/completions 接口地址。')
    parser.add_argument('--api-key-env', default=DEFAULT_API_KEY_ENV, help='保存 API key 的环境变量名。')
    parser.add_argument('--timeout', type=int, default=180, help='单次请求超时时间（秒）。')
    parser.add_argument('--max-retries', type=int, default=5, help='每轮请求最大重试次数。')
    parser.add_argument('--retry-sleep', type=float, default=3.0, help='普通重试基础等待时间（秒）。')
    parser.add_argument('--rate-limit-retry-sleep', type=float, default=20.0, help='429 限流时的基础退避时间（秒）。')
    parser.add_argument('--rate-limit-max-sleep', type=float, default=120.0, help='429 限流时的最大退避时间（秒）。')
    parser.add_argument('--inter-round-sleep', type=float, default=1.0, help='轮与轮之间的默认等待时间（秒）。')
    parser.add_argument('--temperature', type=float, default=0.0, help='采样温度。')
    parser.add_argument('--top-p', type=float, default=0.95, help='top_p。')
    parser.add_argument('--max-output-tokens', type=int, default=512, help='单轮最大输出 token。')
    parser.add_argument('--tts-input-mode', choices=['auto', 'text_fallback'], default='auto', help='TTS 任务的处理方式。当前 Doubao 脚本默认使用转写文本回退，不直接发送音频。')
    parser.add_argument('--image-input-mode', choices=['local_data_url', 'remote_url', 'auto'], default='auto', help='图片输入策略。local_data_url 直接发送本地图片 base64；remote_url 使用数据集中的远程 URL；auto 优先本地、其次远程。')
    parser.add_argument('--video-input-mode', choices=['local_data_url', 'remote_url', 'auto'], default='auto', help='视频输入策略。local_data_url 直接发送本地视频 base64；remote_url 使用原始视频 URL；auto 优先本地、其次远程。不会做抽帧。')
    parser.add_argument('--video-history-mode', choices=['text_only', 'full'], default='text_only', help='视频多轮历史的处理方式。text_only 仅保留历史轮文字与历史回答，避免重复发送旧视频导致请求过大；full 表示历史轮也携带完整视频。')
    parser.add_argument('--video-precompressed-mode', choices=['prefer', 'off'], default='prefer', help='视频输入时是否优先使用数据集中预先压缩好的视频路径。prefer 表示若存在压缩视频则优先使用；off 表示忽略预压缩字段。')
    parser.add_argument('--video-precompressed-field', default=DEFAULT_VIDEO_PRECOMPRESSED_FIELD, help='视频数据集中记录预压缩视频相对路径的字段名。')
    parser.add_argument('--video-compress-mode', choices=['auto', 'off'], default='auto', help='超大本地视频的处理方式。auto 会在不抽帧的前提下尝试转码压缩到可内联发送的体积；off 直接使用原视频。')
    parser.add_argument('--video-max-inline-bytes', type=int, default=DEFAULT_VIDEO_MAX_INLINE_BYTES, help='单个视频内联发送的目标原始体积上限（字节）。该阈值需要为 base64 膨胀和请求 JSON 预留空间。')
    parser.add_argument('--save-request-blueprint', action='store_true', help='将请求蓝图写入每轮记录，方便调试。')
    return parser.parse_args()


def resolve_local_path(media_root: Path, relative_path: str) -> Optional[Path]:
    if not relative_path:
        return None
    candidate = Path(relative_path)
    if candidate.is_absolute():
        return candidate if candidate.exists() else None
    combined = (media_root / candidate).resolve()
    return combined if combined.exists() else None


def encode_file_to_data_url(file_path: Path) -> Tuple[str, str, int]:
    mime_type, _ = mimetypes.guess_type(str(file_path))
    if not mime_type:
        suffix = file_path.suffix.lower()
        if suffix in IMAGE_EXTENSIONS:
            mime_type = 'image/jpeg' if suffix in {'.jpg', '.jpeg'} else f"image/{suffix.lstrip('.')}"
        elif suffix in AUDIO_EXTENSIONS:
            mime_type = 'audio/mpeg' if suffix == '.mp3' else f"audio/{suffix.lstrip('.')}"
        elif suffix in VIDEO_EXTENSIONS:
            mime_type = 'video/mp4' if suffix == '.mp4' else f"video/{suffix.lstrip('.')}"
        else:
            mime_type = 'application/octet-stream'
    raw = file_path.read_bytes()
    encoded = base64.b64encode(raw).decode('ascii')
    return f'data:{mime_type};base64,{encoded}', mime_type, len(raw)


def resolve_ffmpeg_executable() -> Optional[str]:
    if imageio_ffmpeg is None:
        return None
    try:
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def transcode_video_candidates(source_path: Path, output_dir: Path) -> List[Tuple[Path, List[str], Dict[str, Any]]]:
    stem_hash = hashlib.sha1(str(source_path).encode('utf-8')).hexdigest()[:12]
    output_dir.mkdir(parents=True, exist_ok=True)
    profiles = [
        ('p1', ['-vf', 'scale=w=min(960\\,iw):h=-2', '-crf', '32', '-b:a', '64k'], {'scale': 960, 'crf': 32, 'audio_bitrate': '64k'}),
        ('p2', ['-vf', 'scale=w=min(720\\,iw):h=-2', '-crf', '34', '-b:a', '48k'], {'scale': 720, 'crf': 34, 'audio_bitrate': '48k'}),
        ('p3', ['-vf', 'scale=w=min(480\\,iw):h=-2', '-crf', '36', '-b:a', '32k'], {'scale': 480, 'crf': 36, 'audio_bitrate': '32k'}),
        ('p4', ['-vf', 'scale=w=min(360\\,iw):h=-2', '-crf', '38', '-b:a', '24k'], {'scale': 360, 'crf': 38, 'audio_bitrate': '24k'}),
        ('p5', ['-vf', 'scale=w=min(320\\,iw):h=-2', '-crf', '40', '-b:a', '16k'], {'scale': 320, 'crf': 40, 'audio_bitrate': '16k'}),
    ]
    candidates: List[Tuple[Path, List[str], Dict[str, Any]]] = []
    for profile_name, extra_args, meta in profiles:
        target_path = output_dir / f'{source_path.stem}_{stem_hash}_{profile_name}.mp4'
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


def prepare_video_file_for_request(source_path: Path, args: argparse.Namespace) -> Tuple[Path, Dict[str, Any]]:
    original_size = source_path.stat().st_size
    max_inline_bytes = int(args.video_max_inline_bytes)
    if original_size <= max_inline_bytes or args.video_compress_mode == 'off':
        return source_path, {
            'used_compression': False,
            'original_size_bytes': original_size,
            'prepared_size_bytes': original_size,
            'video_compress_mode': args.video_compress_mode,
        }

    ffmpeg_exe = resolve_ffmpeg_executable()
    if not ffmpeg_exe:
        raise RuntimeError('检测到视频体积超出内联上限，但当前环境没有可用 ffmpeg，无法在不抽帧的前提下压缩视频。')

    cache_root = Path(args.output_dir).resolve() / '_video_cache'
    profile_error_messages: List[str] = []
    for candidate_path, ffmpeg_args, profile_meta in transcode_video_candidates(source_path, cache_root):
        if candidate_path.exists() and candidate_path.stat().st_size <= max_inline_bytes:
            prepared_size = candidate_path.stat().st_size
            return candidate_path, {
                'used_compression': True,
                'original_size_bytes': original_size,
                'prepared_size_bytes': prepared_size,
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
                'used_compression': True,
                'original_size_bytes': original_size,
                'prepared_size_bytes': prepared_size,
                **profile_meta,
            }

    joined_errors = ' | '.join(profile_error_messages[-3:]).strip()
    raise RuntimeError(
        f'视频压缩后仍超过内联上限 {max_inline_bytes} 字节，原始大小 {original_size} 字节。'
        + (f' 最近失败信息：{joined_errors}' if joined_errors else '')
    )


def build_system_prompt(spec: TaskSpec, question_delivery_mode: str) -> str:
    media_desc = '完整视频片段' if spec.media_mode == 'video' else '单张图像'
    if spec.question_mode == 'tts':
        question_desc = '用户问题原本以语音给出；本次请求中同时提供语音转写文本'
    else:
        question_desc = '用户问题以文本给出'
    if question_delivery_mode == 'transcript_text_fallback':
        question_desc += '，请基于转写文本回答，不要假设音频中存在额外信息。'
    return (
        '你正在执行 RUBRIC-MME 第一阶段多轮作答。'
        f'当前任务的视觉输入是{media_desc}，{question_desc}'
        '请严格遵守以下要求：'
        '1. 只能回答当前轮问题；'
        '2. 可以利用历史轮对话和历史轮视觉信息来理解上下文；'
        '3. 绝对不要假设未来轮的信息；'
        '4. 绝对不要输出你看不到的参考答案；'
        '5. 请始终使用中文直接作答，简洁、自然、信息充分。'
    )


def build_user_text(round_index: int, round_data: Dict[str, Any], question_delivery_mode: str, *, is_history: bool) -> str:
    prefix = f'这是第{round_index + 1}轮历史对话。' if is_history else f'这是当前第{round_index + 1}轮。'
    question_text = str(round_data.get('question_text', '') or '').strip()
    if question_delivery_mode == 'text':
        return f'{prefix}请结合提供的视觉信息回答这一轮问题。\n用户问题：{question_text}\n请只围绕这一轮作答。'
    return (
        f'{prefix}这一轮用户问题原本通过语音给出；当前接口不直接接收音频，以下是对应转写文本。'
        f'\n用户问题转写：{question_text}\n请只围绕这一轮作答。'
    )


def detect_question_delivery_mode(spec: TaskSpec, args: argparse.Namespace) -> str:
    if spec.question_mode != 'tts':
        return 'text'
    if args.tts_input_mode in {'auto', 'text_fallback'}:
        return 'transcript_text_fallback'
    return 'text'


def load_video_precompressed_index(dataset_path: Path, target_dialogue_ids: Sequence[str], field_name: str) -> Dict[Tuple[str, int], Dict[str, Any]]:
    if not target_dialogue_ids:
        return {}
    target_id_set = {str(dialogue_id).strip() for dialogue_id in target_dialogue_ids if str(dialogue_id).strip()}
    if not target_id_set:
        return {}
    with dataset_path.open('r', encoding='utf-8') as file:
        raw_items = json.load(file)
    if not isinstance(raw_items, list):
        return {}

    index: Dict[Tuple[str, int], Dict[str, Any]] = {}
    for item in raw_items:
        dialogue_id = str(item.get('_ai_unique_id_', '') or '').strip()
        if not dialogue_id or dialogue_id not in target_id_set:
            continue
        round_index = 0
        for entry in (item.get('stream_conversation') or {}).get('conversations', []):
            if entry.get('speaker') != 'ai':
                continue
            compressed_path = str(entry.get(field_name, '') or '').strip()
            if compressed_path:
                index[(dialogue_id, round_index)] = {
                    'video_precompressed_local_path': compressed_path,
                    'video_precompressed_profile': str(entry.get(f'{field_name}_profile', '') or '').strip(),
                    'video_precompressed_size_bytes': entry.get(f'{field_name}_size_bytes'),
                    'video_precompressed_original_size_bytes': entry.get(f'{field_name}_original_size_bytes'),
                }
            round_index += 1
    return index


def attach_video_precompressed_metadata(docs: Sequence[Dict[str, Any]], dataset_path: Path, field_name: str) -> List[Dict[str, Any]]:
    dialogue_ids = [str(doc.get('dialogue_id', '') or '').strip() for doc in docs]
    metadata_index = load_video_precompressed_index(dataset_path, dialogue_ids, field_name)
    if not metadata_index:
        return list(docs)

    updated_docs: List[Dict[str, Any]] = []
    for doc in docs:
        dialogue_id = str(doc.get('dialogue_id', '') or '').strip()
        updated_rounds: List[Dict[str, Any]] = []
        for round_data in doc.get('rounds', []):
            updated_round = dict(round_data)
            round_index = int(updated_round.get('round_index', len(updated_rounds)) or 0)
            metadata = metadata_index.get((dialogue_id, round_index))
            if metadata:
                updated_round.update(metadata)
            updated_rounds.append(updated_round)
        updated_doc = dict(doc)
        updated_doc['rounds'] = updated_rounds
        updated_docs.append(updated_doc)
    return updated_docs


def build_media_content(spec: TaskSpec, round_data: Dict[str, Any], media_root: Path, args: argparse.Namespace) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    content: List[Dict[str, Any]] = []
    refs: List[Dict[str, Any]] = []
    if spec.media_mode == 'image':
        remote_url = str(round_data.get('image_remote_url', '') or '').strip()
        local_path = resolve_local_path(media_root, str(round_data.get('image_local_path', '') or ''))
        if args.image_input_mode in {'local_data_url', 'auto'} and local_path is not None:
            data_url, mime_type, size = encode_file_to_data_url(local_path)
            content.append({'type': 'image_url', 'image_url': {'url': data_url}})
            refs.append({'kind': 'image', 'transport': 'data_url', 'local_path': str(local_path), 'mime_type': mime_type, 'size_bytes': size})
            return content, refs
        if remote_url:
            content.append({'type': 'image_url', 'image_url': {'url': remote_url}})
            refs.append({'kind': 'image', 'transport': 'remote_url', 'local_path': str(local_path) if local_path else '', 'remote_url': remote_url})
            return content, refs
        raise FileNotFoundError(f'图片轮次缺少可用图像资源: {round_data}')

    remote_url = str(round_data.get('video_remote_url', '') or '').strip()
    original_local_rel_path = str(round_data.get('video_local_path', '') or '').strip()
    precompressed_rel_path = str(round_data.get('video_precompressed_local_path', '') or '').strip()
    original_local_path = resolve_local_path(media_root, original_local_rel_path)
    precompressed_local_path = resolve_local_path(media_root, precompressed_rel_path) if args.video_precompressed_mode == 'prefer' else None
    selected_local_path = precompressed_local_path or original_local_path
    selected_source = 'precompressed_local_path' if precompressed_local_path is not None else 'original_local_path'
    if args.video_input_mode in {'local_data_url', 'auto'} and selected_local_path is not None:
        prepared_video_path, prepare_meta = prepare_video_file_for_request(selected_local_path, args)
        data_url, mime_type, size = encode_file_to_data_url(prepared_video_path)
        content.append({'type': 'video_url', 'video_url': {'url': data_url}})
        refs.append(
            {
                'kind': 'video',
                'transport': 'data_url',
                'local_path': str(selected_local_path),
                'prepared_local_path': str(prepared_video_path),
                'mime_type': mime_type,
                'size_bytes': size,
                'selected_source': selected_source,
                'original_local_path': str(original_local_path) if original_local_path else '',
                'precompressed_local_path': str(precompressed_local_path) if precompressed_local_path else '',
                **prepare_meta,
            }
        )
        return content, refs
    if remote_url:
        content.append({'type': 'video_url', 'video_url': {'url': remote_url}})
        refs.append({'kind': 'video', 'transport': 'remote_url', 'local_path': str(local_path) if local_path else '', 'remote_url': remote_url})
        return content, refs
    raise FileNotFoundError(f'视频轮次缺少可用视频资源；当前脚本不做抽帧，但要求本地视频或可访问的视频 URL: {round_data}')


def extract_text_response(payload: Dict[str, Any]) -> str:
    choices = payload.get('choices')
    if not isinstance(choices, list) or not choices:
        return ''
    message = choices[0].get('message', {})
    content = message.get('content', '')
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        text_parts: List[str] = []
        for item in content:
            if isinstance(item, dict) and item.get('type') == 'text' and isinstance(item.get('text'), str):
                text_parts.append(item['text'])
        return '\n'.join(part.strip() for part in text_parts if part.strip()).strip()
    return str(content or '').strip()


def extract_error_message(response: requests.Response) -> str:
    try:
        payload = response.json()
    except Exception:
        return response.text.strip()
    error = payload.get('error')
    if isinstance(error, dict):
        return json.dumps(error, ensure_ascii=False)
    return json.dumps(payload, ensure_ascii=False)


class OpenAICompatiblePhase1Runner:
    def __init__(self, *, model_name: str, api_url: str, api_key_env: str, timeout: int, max_retries: int, retry_sleep: float, rate_limit_retry_sleep: float, rate_limit_max_sleep: float, temperature: float, top_p: float, max_output_tokens: int) -> None:
        api_key = os.getenv(api_key_env, '').strip()
        if not api_key:
            raise RuntimeError(f'未找到 API key，请先设置环境变量 {api_key_env}')
        self.api_key = api_key
        self.model_name = model_name
        self.api_url = api_url
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_sleep = retry_sleep
        self.rate_limit_retry_sleep = rate_limit_retry_sleep
        self.rate_limit_max_sleep = rate_limit_max_sleep
        self.temperature = temperature
        self.top_p = top_p
        self.max_output_tokens = max_output_tokens
        self.session = self._build_session()

    def _build_session(self) -> requests.Session:
        session = requests.Session()
        adapter = HTTPAdapter(pool_connections=4, pool_maxsize=4, max_retries=0)
        session.mount('https://', adapter)
        session.mount('http://', adapter)
        session.headers.update(
            {
                'Authorization': f'Bearer {self.api_key}',
                'Content-Type': 'application/json',
                'Connection': 'keep-alive',
            }
        )
        return session

    def reset_session(self) -> None:
        try:
            self.session.close()
        except Exception:
            pass
        self.session = self._build_session()

    def close(self) -> None:
        self.session.close()

    def _build_messages(self, spec: TaskSpec, doc: Dict[str, Any], round_index: int, media_root: Path, args: argparse.Namespace, previous_predictions: Sequence[str]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        rounds = doc.get('rounds', [])
        question_delivery_mode = detect_question_delivery_mode(spec, args)
        system_prompt = build_system_prompt(spec, question_delivery_mode)
        messages: List[Dict[str, Any]] = [{'role': 'system', 'content': system_prompt}]
        blueprint: List[Dict[str, Any]] = [{'role': 'system', 'text': system_prompt}]

        for history_index in range(min(len(previous_predictions), round_index)):
            history_round = rounds[history_index]
            user_text = build_user_text(history_index, history_round, question_delivery_mode, is_history=True)
            if spec.media_mode == 'video' and args.video_history_mode == 'text_only':
                media_content = []
                media_refs = [{'kind': 'video', 'transport': 'omitted_history_video', 'note': '历史视频为控制请求体大小未重复发送，仅保留历史文本与历史回答。'}]
            else:
                media_content, media_refs = build_media_content(spec, history_round, media_root, args)
            user_content = [{'type': 'text', 'text': user_text}] + media_content
            messages.append({'role': 'user', 'content': user_content})
            messages.append({'role': 'assistant', 'content': str(previous_predictions[history_index])})
            blueprint.append({'role': 'user', 'round_index': history_index, 'is_history': True, 'text': user_text, 'media_refs': media_refs})
            blueprint.append({'role': 'assistant', 'round_index': history_index, 'is_history': True, 'text': str(previous_predictions[history_index])})

        current_round = rounds[round_index]
        current_text = build_user_text(round_index, current_round, question_delivery_mode, is_history=False)
        current_media_content, current_media_refs = build_media_content(spec, current_round, media_root, args)
        current_content = [{'type': 'text', 'text': current_text}] + current_media_content
        messages.append({'role': 'user', 'content': current_content})
        blueprint.append({'role': 'user', 'round_index': round_index, 'is_history': False, 'text': current_text, 'media_refs': current_media_refs})

        request_context = {
            'history_round_count': min(len(previous_predictions), round_index),
            'current_round_index': round_index,
            'question_mode': spec.question_mode,
            'media_mode': spec.media_mode,
            'question_delivery_mode': question_delivery_mode,
            'tts_input_mode_requested': args.tts_input_mode if spec.question_mode == 'tts' else '',
            'image_input_mode': args.image_input_mode if spec.media_mode == 'image' else '',
            'video_input_mode': args.video_input_mode if spec.media_mode == 'video' else '',
            'video_history_mode': args.video_history_mode if spec.media_mode == 'video' else '',
        }
        if current_media_refs:
            request_context['current_media_ref'] = current_media_refs[0]
        if args.save_request_blueprint:
            request_context['message_blueprint'] = blueprint
        return messages, request_context

    def generate_round(self, messages: List[Dict[str, Any]]) -> Tuple[str, Dict[str, Any], str, Dict[str, Any]]:
        payload = {
            'model': self.model_name,
            'messages': messages,
            'temperature': self.temperature,
            'top_p': self.top_p,
            'max_tokens': self.max_output_tokens,
            'stream': False,
        }
        last_error = ''
        last_error_info: Dict[str, Any] = {'status_code': None, 'error_type': '', 'retriable': False, 'retry_trace': []}
        for attempt in range(self.max_retries):
            try:
                response = self.session.post(self.api_url, json=payload, timeout=self.timeout)
            except requests.RequestException as exc:
                last_error = f'{type(exc).__name__}: {exc}'
                error_text = last_error.lower()
                is_connection_error = 'failed to establish a new connection' in error_text or 'winerror 10013' in error_text
                error_type = 'connection_exception' if is_connection_error else 'request_exception'
                sleep_seconds = self.retry_sleep
                if is_connection_error:
                    sleep_seconds = min(max(self.retry_sleep * (2 ** attempt), 10.0), self.rate_limit_max_sleep)
                retry_trace = list(last_error_info.get('retry_trace', []))
                retry_trace.append(
                    {
                        'attempt': attempt + 1,
                        'error_type': error_type,
                        'sleep_seconds': round(sleep_seconds, 2),
                        'message': last_error[-240:],
                    }
                )
                last_error_info = {
                    'status_code': None,
                    'error_type': error_type,
                    'retriable': attempt + 1 < self.max_retries,
                    'retry_trace': retry_trace,
                }
                if attempt + 1 < self.max_retries:
                    self.reset_session()
                    time.sleep(sleep_seconds)
                    continue
                return '', {}, last_error, last_error_info

            if response.status_code == 429:
                sleep_seconds = min(self.rate_limit_retry_sleep * (2 ** attempt), self.rate_limit_max_sleep)
                retry_trace = list(last_error_info.get('retry_trace', []))
                retry_trace.append({'attempt': attempt + 1, 'status_code': 429, 'error_type': 'rate_limit', 'sleep_seconds': round(sleep_seconds, 2)})
                last_error_info = {'status_code': 429, 'error_type': 'rate_limit', 'retriable': attempt + 1 < self.max_retries, 'retry_trace': retry_trace}
                last_error = f'HTTP 429: {extract_error_message(response)}'
                if attempt + 1 < self.max_retries:
                    time.sleep(sleep_seconds)
                    continue
                return '', {}, last_error, last_error_info

            if response.status_code >= 400:
                error_text = extract_error_message(response)
                retriable = response.status_code >= 500
                last_error = f'HTTP {response.status_code}: {error_text}'
                last_error_info = {'status_code': response.status_code, 'error_type': 'http_error', 'retriable': retriable and attempt + 1 < self.max_retries, 'retry_trace': list(last_error_info.get('retry_trace', []))}
                if retriable and attempt + 1 < self.max_retries:
                    time.sleep(self.retry_sleep)
                    continue
                return '', {}, last_error, last_error_info

            response_payload = response.json()
            prediction = extract_text_response(response_payload)
            usage = response_payload.get('usage', {}) if isinstance(response_payload.get('usage'), dict) else {}
            return prediction, usage, '', {'status_code': None, 'error_type': '', 'retriable': False, 'retry_trace': []}
        return '', {}, last_error, last_error_info


def build_runner(args: argparse.Namespace) -> OpenAICompatiblePhase1Runner:
    return OpenAICompatiblePhase1Runner(
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


def generate_round_record(args: argparse.Namespace, spec: TaskSpec, runner: OpenAICompatiblePhase1Runner, media_root: Path, doc: Dict[str, Any], round_index: int, previous_predictions: Sequence[str]) -> Tuple[str, Dict[str, Any]]:
    round_data = doc['rounds'][round_index]
    request_started = time.time()
    messages, request_context = runner._build_messages(spec, doc, round_index, media_root, args, previous_predictions)
    prediction, usage, error, error_info = runner.generate_round(messages)
    latency_seconds = round(time.time() - request_started, 3)
    media_local_path = str(round_data.get('image_local_path', '') or round_data.get('video_local_path', '') or '')
    effective_media_local_path = ''
    effective_media_source = ''
    current_media_ref = request_context.get('current_media_ref')
    if isinstance(current_media_ref, dict):
        effective_media_local_path = str(
            current_media_ref.get('prepared_local_path')
            or current_media_ref.get('local_path')
            or ''
        )
        effective_media_source = str(current_media_ref.get('selected_source') or current_media_ref.get('transport') or '')
    media_remote_url = str(round_data.get('image_remote_url', '') or round_data.get('video_remote_url', '') or '')
    media_cloud_path = str(round_data.get('video_cloud_path', '') or '')
    round_record: Dict[str, Any] = {
        'round_index': round_index,
        'timestamp': round_data.get('timestamp', ''),
        'category': round_data.get('category', ''),
        'question_text': round_data.get('question_text', ''),
        'question_audio_local_path': round_data.get('question_audio_local_path', ''),
        'question_audio_remote_url': round_data.get('question_audio_remote_url', ''),
        'question_audio_cloud_path': round_data.get('question_audio_cloud_path', ''),
        'reference_answer': round_data.get('reference_answer', ''),
        'prediction': prediction,
        'usage': usage,
        'attempt_count': 1,
        'latency_seconds': latency_seconds,
        'request_context': request_context,
        'media_local_path': media_local_path,
        'effective_media_local_path': effective_media_local_path,
        'effective_media_source': effective_media_source,
        'media_remote_url': media_remote_url,
        'media_cloud_path': media_cloud_path,
        'error_info': error_info,
    }
    if error:
        round_record['error'] = error
    return prediction, round_record


def process_dialogue_work_item(args: argparse.Namespace, spec: TaskSpec, media_root: Path, work_item: Phase1DialogueWorkItem, shared_runner: Optional[OpenAICompatiblePhase1Runner] = None) -> Dict[str, Any]:
    owns_runner = shared_runner is None
    runner = shared_runner or build_runner(args)
    try:
        doc = work_item.doc
        dialogue_id = work_item.dialogue_id
        all_rounds = doc.get('rounds', [])
        resume_round_index = min(max(work_item.resume_round_index, 0), len(all_rounds))
        existing_payload = work_item.existing_payload if isinstance(work_item.existing_payload, dict) else None
        use_current_turn_only_repair = args.repair_failed and args.repair_mode == 'current_turn_only' and existing_payload is not None and bool(work_item.failed_round_indices)

        if use_current_turn_only_repair:
            existing_rounds = list(existing_payload.get('rounds') or [])
            round_records: List[Dict[str, Any]] = list(existing_rounds)
            for round_index in work_item.failed_round_indices:
                _prediction, round_record = generate_round_record(args, spec, runner, media_root, doc, round_index, [])
                while len(round_records) <= round_index:
                    round_records.append({})
                round_records[round_index] = round_record
        else:
            predictions: List[str] = []
            round_records = []
            if existing_payload is not None and resume_round_index > 0:
                existing_rounds = existing_payload.get('rounds') or []
                reusable_rounds = list(existing_rounds[:resume_round_index])
                round_records.extend(reusable_rounds)
                predictions.extend(str(record.get('prediction', '')) for record in reusable_rounds)
            for round_index in range(resume_round_index, len(all_rounds)):
                prediction, round_record = generate_round_record(args, spec, runner, media_root, doc, round_index, predictions)
                predictions.append(prediction)
                round_records.append(round_record)
                if args.inter_round_sleep > 0 and round_index + 1 < len(all_rounds):
                    time.sleep(args.inter_round_sleep)

        final_round_records: List[Dict[str, Any]] = []
        for round_index in range(len(all_rounds)):
            if round_index < len(round_records) and isinstance(round_records[round_index], dict):
                final_round_records.append(round_records[round_index])
            else:
                final_round_records.append({'round_index': round_index, 'error': 'MissingRoundAfterRepair', 'prediction': ''})
        failed_rounds = sum(1 for record in final_round_records if not is_phase1_round_success(record))
        payload = {
            'benchmark_name': 'RUBRIC-MME',
            'phase': 'phase1_generation',
            'provider': 'openai_compatible_api',
            'task_name': spec.name,
            'task_alias': spec.task_alias,
            'model_name': args.model,
            'dialogue_id': dialogue_id,
            'source_type': doc.get('source_type', ''),
            'environment': doc.get('environment', ''),
            'interaction_setup': doc.get('interaction_setup'),
            'conversation_meta': doc.get('conversation_meta'),
            'round_count': len(final_round_records),
            'question_mode': spec.question_mode,
            'media_mode': spec.media_mode,
            'source_dataset_file': spec.dataset_file,
            'source_data_root': str(Path(args.data_root).resolve()),
            'generated_at': utc_now(),
            'tts_input_mode_requested': args.tts_input_mode if spec.question_mode == 'tts' else '',
            'tts_input_mode_effective': detect_question_delivery_mode(spec, args) if spec.question_mode == 'tts' else '',
            'rounds': final_round_records,
        }
        return {'dialogue_id': dialogue_id, 'doc_index': work_item.doc_index, 'payload': payload, 'failed_rounds': failed_rounds}
    finally:
        if owns_runner:
            runner.close()


def run_task(args: argparse.Namespace, spec: TaskSpec) -> Dict[str, Any]:
    data_root = Path(args.data_root).resolve()
    media_root = Path(args.media_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    task_output_dir = output_dir / spec.name
    samples_path = task_output_dir / f"{safe_name(args.model)}_samples.jsonl"
    summary_path = task_output_dir / f"{safe_name(args.model)}_summary.json"
    docs = load_docs_for_task(spec, data_root, args.dialogue_id, args.limit)
    if spec.media_mode == 'video' and args.video_precompressed_mode == 'prefer':
        docs = attach_video_precompressed_metadata(docs, (data_root / spec.dataset_file).resolve(), args.video_precompressed_field)
    work_items, samples_index, skipped = build_phase1_work_items(docs, samples_path, resume=args.resume, repair_failed=args.repair_failed, repair_mode=args.repair_mode)
    started_at = utc_now()
    attempted = len(work_items)
    completed = 0
    failed_rounds = 0
    max_workers = max(1, int(args.max_workers))
    results_by_index: Dict[int, Dict[str, Any]] = {}
    written_payloads_by_id: Dict[str, Dict[str, Any]] = {}
    write_lock = Lock()

    def persist_payload(payload: Dict[str, Any]) -> None:
        dialogue_id = str(payload.get('dialogue_id', '') or '')
        with write_lock:
            written_payloads_by_id[dialogue_id] = payload
            if args.repair_failed or args.resume:
                merged_payloads = merge_phase1_payloads(samples_index, written_payloads_by_id, docs)
                write_jsonl(samples_path, merged_payloads)
            else:
                current_payloads = [written_payloads_by_id[str(doc.get('dialogue_id', '') or '')] for doc in docs if str(doc.get('dialogue_id', '') or '') in written_payloads_by_id]
                write_jsonl(samples_path, current_payloads)

    if max_workers == 1 or len(work_items) <= 1:
        active_runner: Optional[OpenAICompatiblePhase1Runner] = None
        try:
            if work_items:
                active_runner = build_runner(args)
            for work_item in work_items:
                result = process_dialogue_work_item(args, spec, media_root, work_item, active_runner)
                results_by_index[work_item.doc_index] = result
                persist_payload(result['payload'])
        finally:
            if active_runner is not None:
                active_runner.close()
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_index = {executor.submit(process_dialogue_work_item, args, spec, media_root, work_item, None): work_item.doc_index for work_item in work_items}
            for future in as_completed(future_to_index):
                doc_index = future_to_index[future]
                result = future.result()
                results_by_index[doc_index] = result
                persist_payload(result['payload'])

    ordered_payloads: List[Dict[str, Any]] = []
    updated_payloads_by_id: Dict[str, Dict[str, Any]] = {}
    for work_item in work_items:
        result = results_by_index[work_item.doc_index]
        payload = result['payload']
        ordered_payloads.append(payload)
        updated_payloads_by_id[str(payload.get('dialogue_id', ''))] = payload
        completed += 1
        failed_rounds += int(result.get('failed_rounds', 0) or 0)

    if args.repair_failed or args.resume:
        final_payloads = merge_phase1_payloads(samples_index, updated_payloads_by_id, docs)
    else:
        final_payloads = ordered_payloads
    write_jsonl(samples_path, final_payloads)

    total_failed_rounds = 0
    for payload in final_payloads:
        total_failed_rounds += sum(1 for round_record in payload.get('rounds', []) if not is_phase1_round_success(round_record))

    summary = {
        'benchmark_name': 'RUBRIC-MME',
        'phase': 'phase1_generation',
        'provider': 'openai_compatible_api',
        'task_name': spec.name,
        'task_alias': spec.task_alias,
        'model_name': args.model,
        'started_at': started_at,
        'completed_at': utc_now(),
        'data_root': str(data_root),
        'media_root': str(media_root),
        'output_dir': str(task_output_dir),
        'attempted_dialogues': attempted,
        'completed_dialogues': completed,
        'skipped_dialogues': skipped,
        'dialogue_count_total': len(final_payloads),
        'failed_rounds': total_failed_rounds,
        'samples_path': str(samples_path),
        'api_url': args.api_url,
        'tts_input_mode_requested': args.tts_input_mode if spec.question_mode == 'tts' else '',
        'tts_input_mode_effective': detect_question_delivery_mode(spec, args) if spec.question_mode == 'tts' else '',
        'video_precompressed_mode': args.video_precompressed_mode if spec.media_mode == 'video' else '',
        'video_precompressed_field': args.video_precompressed_field if spec.media_mode == 'video' else '',
        'repair_mode': args.repair_mode if args.repair_failed else '',
    }
    dump_json(summary_path, summary)
    return summary


def main() -> None:
    args = parse_args()
    task_names = resolve_task_names(args.tasks)
    summaries: List[Dict[str, Any]] = []
    for task_name in task_names:
        summaries.append(run_task(args, TASK_SPECS[task_name]))
    output_dir = Path(args.output_dir).resolve()
    run_summary = {
        'benchmark_name': 'RUBRIC-MME',
        'phase': 'phase1_generation',
        'provider': 'openai_compatible_api',
        'model_name': args.model,
        'generated_at': utc_now(),
        'tasks': summaries,
    }
    dump_json(output_dir / f"{safe_name(args.model)}_run_summary.json", run_summary)


if __name__ == '__main__':
    main()


