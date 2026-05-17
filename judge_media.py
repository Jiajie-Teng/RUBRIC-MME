from __future__ import annotations

import base64
import mimetypes
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".wmv"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}


@dataclass
class JudgeRequest:
    prompt_text: str
    internal_messages: List[Dict[str, Any]]
    content_blueprint: List[Dict[str, Any]]
    media_refs: List[Dict[str, Any]]
    input_mode: str


def resolve_local_path(source_data_root: str, relative_path: str) -> Optional[Path]:
    if not relative_path:
        return None
    candidate = Path(relative_path)
    if candidate.is_absolute():
        return candidate if candidate.exists() else None
    base_root = Path(source_data_root).expanduser().resolve() if source_data_root else Path.cwd()
    resolved = (base_root / candidate).resolve()
    return resolved if resolved.exists() else None


def encode_file_to_data_url(file_path: Path) -> Tuple[str, str, int]:
    mime_type, _ = mimetypes.guess_type(str(file_path))
    if not mime_type:
        suffix = file_path.suffix.lower()
        if suffix in IMAGE_EXTENSIONS:
            mime_type = "image/jpeg" if suffix in {".jpg", ".jpeg"} else f"image/{suffix.lstrip('.')}"
        elif suffix in VIDEO_EXTENSIONS:
            mime_type = "video/mp4" if suffix == ".mp4" else f"video/{suffix.lstrip('.')}"
        else:
            mime_type = "application/octet-stream"
    raw = file_path.read_bytes()
    encoded = base64.b64encode(raw).decode("ascii")
    return f"data:{mime_type};base64,{encoded}", mime_type, len(raw)


def build_gs_uri(cloud_path: str) -> str:
    cleaned = str(cloud_path or "").strip()
    if not cleaned:
        return ""
    if cleaned.startswith("gs://"):
        return cleaned
    return f"gs://antgroup_matrix_storage/{cleaned.lstrip('/')}"


def _text_segment(text: str) -> Dict[str, Any]:
    return {"segment_type": "text", "text": text}


def _media_summary(
    *,
    kind: str,
    round_index: int,
    evidence_role: str,
    transport: str,
    local_path: str = "",
    remote_url: str = "",
    cloud_path: str = "",
    gs_uri: str = "",
    mime_type: str = "",
    size_bytes: Optional[int] = None,
) -> Dict[str, Any]:
    return {
        "kind": kind,
        "round_index": round_index,
        "evidence_role": evidence_role,
        "transport": transport,
        "local_path": local_path,
        "remote_url": remote_url,
        "cloud_path": cloud_path,
        "gs_uri": gs_uri,
        "mime_type": mime_type,
        "size_bytes": size_bytes,
    }


def _build_image_media_segment(round_record: Dict[str, Any], *, evidence_role: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    source_root = str(round_record.get("source_data_root", "") or "")
    local_path = resolve_local_path(source_root, str(round_record.get("media_local_path", "") or ""))
    remote_url = str(round_record.get("media_remote_url", "") or "")
    round_index = int(round_record.get("round_index", 0) or 0)

    if local_path is not None:
        data_url, mime_type, size_bytes = encode_file_to_data_url(local_path)
        return (
            [
                {
                    "segment_type": "media",
                    "kind": "image",
                    "round_index": round_index,
                    "evidence_role": evidence_role,
                    "transport": "data_url",
                    "local_path": str(local_path),
                    "remote_url": remote_url,
                    "mime_type": mime_type,
                    "size_bytes": size_bytes,
                    "data_url": data_url,
                }
            ],
            [
                _media_summary(
                    kind="image",
                    round_index=round_index,
                    evidence_role=evidence_role,
                    transport="data_url",
                    local_path=str(local_path),
                    remote_url=remote_url,
                    mime_type=mime_type,
                    size_bytes=size_bytes,
                )
            ],
        )

    if remote_url:
        return (
            [
                {
                    "segment_type": "media",
                    "kind": "image",
                    "round_index": round_index,
                    "evidence_role": evidence_role,
                    "transport": "remote_url",
                    "local_path": "",
                    "remote_url": remote_url,
                    "mime_type": "image/unknown",
                    "size_bytes": None,
                }
            ],
            [
                _media_summary(
                    kind="image",
                    round_index=round_index,
                    evidence_role=evidence_role,
                    transport="remote_url",
                    remote_url=remote_url,
                    mime_type="image/unknown",
                )
            ],
        )

    return ([_text_segment(f"第{round_index + 1}轮视觉证据缺失，无法直接查看该轮图像。")], [])


def _build_video_media_segment(round_record: Dict[str, Any], *, evidence_role: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    source_root = str(round_record.get("source_data_root", "") or "")
    local_path = resolve_local_path(source_root, str(round_record.get("media_local_path", "") or ""))
    remote_url = str(round_record.get("media_remote_url", "") or "")
    cloud_path = str(round_record.get("media_cloud_path", "") or "")
    gs_uri = build_gs_uri(cloud_path)
    round_index = int(round_record.get("round_index", 0) or 0)
    local_size = local_path.stat().st_size if local_path is not None else None

    if gs_uri:
        return (
            [
                {
                    "segment_type": "media",
                    "kind": "video",
                    "round_index": round_index,
                    "evidence_role": evidence_role,
                    "transport": "gs_uri_input_audio",
                    "local_path": str(local_path) if local_path is not None else "",
                    "remote_url": remote_url,
                    "cloud_path": cloud_path,
                    "gs_uri": gs_uri,
                    "mime_type": "video/mp4",
                    "size_bytes": local_size,
                }
            ],
            [
                _media_summary(
                    kind="video",
                    round_index=round_index,
                    evidence_role=evidence_role,
                    transport="gs_uri_input_audio",
                    local_path=str(local_path) if local_path is not None else "",
                    remote_url=remote_url,
                    cloud_path=cloud_path,
                    gs_uri=gs_uri,
                    mime_type="video/mp4",
                    size_bytes=local_size,
                )
            ],
        )

    if local_path is not None:
        data_url, mime_type, size_bytes = encode_file_to_data_url(local_path)
        return (
            [
                {
                    "segment_type": "media",
                    "kind": "video",
                    "round_index": round_index,
                    "evidence_role": evidence_role,
                    "transport": "data_url",
                    "local_path": str(local_path),
                    "remote_url": remote_url,
                    "cloud_path": cloud_path,
                    "gs_uri": "",
                    "mime_type": mime_type,
                    "size_bytes": size_bytes,
                    "data_url": data_url,
                }
            ],
            [
                _media_summary(
                    kind="video",
                    round_index=round_index,
                    evidence_role=evidence_role,
                    transport="data_url",
                    local_path=str(local_path),
                    remote_url=remote_url,
                    cloud_path=cloud_path,
                    mime_type=mime_type,
                    size_bytes=size_bytes,
                )
            ],
        )

    if remote_url:
        return (
            [
                {
                    "segment_type": "media",
                    "kind": "video",
                    "round_index": round_index,
                    "evidence_role": evidence_role,
                    "transport": "remote_url",
                    "local_path": "",
                    "remote_url": remote_url,
                    "cloud_path": cloud_path,
                    "gs_uri": "",
                    "mime_type": "video/mp4",
                    "size_bytes": None,
                }
            ],
            [
                _media_summary(
                    kind="video",
                    round_index=round_index,
                    evidence_role=evidence_role,
                    transport="remote_url",
                    remote_url=remote_url,
                    cloud_path=cloud_path,
                    mime_type="video/mp4",
                )
            ],
        )

    return ([_text_segment(f"第{round_index + 1}轮视觉证据缺失，无法直接查看该轮视频。")], [])


def _build_round_visual_segments(round_record: Dict[str, Any], *, evidence_role: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    round_index = int(round_record.get("round_index", 0) or 0)
    if evidence_role == "turn_history":
        intro = f"以下是第{round_index + 1}轮历史视觉证据，仅用于理解上下文："
    elif evidence_role == "turn_current":
        intro = f"以下是当前第{round_index + 1}轮视觉证据，请结合它判断当前回答质量："
    else:
        intro = f"以下是第{round_index + 1}轮视觉证据，可用于判断整段对话表现："

    segments: List[Dict[str, Any]] = [_text_segment(intro)]
    media_mode = str(round_record.get("media_mode", "") or "")
    if media_mode == "image":
        media_segments, media_refs = _build_image_media_segment(round_record, evidence_role=evidence_role)
    else:
        media_segments, media_refs = _build_video_media_segment(round_record, evidence_role=evidence_role)
    segments.extend(media_segments)
    return segments, media_refs


def _build_internal_messages(content_blueprint: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    content: List[Dict[str, Any]] = []
    for item in content_blueprint:
        segment_type = item.get("segment_type", "")
        if segment_type == "text":
            content.append({"type": "text", "text": str(item.get("text", "") or "")})
            continue

        kind = item.get("kind", "")
        transport = item.get("transport", "")
        if kind == "image":
            if transport == "data_url":
                content.append({"type": "image_url", "image_url": {"url": item["data_url"], "detail": "auto"}})
            elif transport == "remote_url":
                content.append({"type": "image_url", "image_url": {"url": item["remote_url"], "detail": "auto"}})
            continue

        if kind == "video":
            if transport == "gs_uri_input_audio":
                content.append({"type": "input_audio", "input_audio": {"format": "video/mp4", "data": item["gs_uri"]}})
            elif transport == "data_url":
                content.append({"type": "image_url", "image_url": {"url": item["data_url"]}})
            elif transport == "remote_url":
                content.append({"type": "image_url", "image_url": {"url": item["remote_url"]}})
            continue

    return [{"role": "user", "content": content}]



def _select_lightweight_session_visual_rounds(rounds: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    round_list = list(rounds)
    if len(round_list) <= 3:
        return round_list

    candidate_indices = [0, len(round_list) // 2, len(round_list) - 1]
    selected_indices: List[int] = []
    seen: set[int] = set()
    for index in candidate_indices:
        if index in seen:
            continue
        seen.add(index)
        selected_indices.append(index)
    return [round_list[index] for index in selected_indices]
def build_turn_judge_request(prompt_text: str, round_record: Dict[str, Any], history_rounds: Sequence[Dict[str, Any]]) -> JudgeRequest:
    content_blueprint: List[Dict[str, Any]] = []
    media_refs: List[Dict[str, Any]] = []

    for history_round in history_rounds:
        segments, refs = _build_round_visual_segments(history_round, evidence_role="turn_history")
        content_blueprint.extend(segments)
        media_refs.extend(refs)

    current_segments, current_refs = _build_round_visual_segments(round_record, evidence_role="turn_current")
    content_blueprint.extend(current_segments)
    media_refs.extend(current_refs)
    content_blueprint.append(_text_segment(prompt_text))

    input_mode = "text_plus_visual" if media_refs else "text_only"
    return JudgeRequest(
        prompt_text=prompt_text,
        internal_messages=_build_internal_messages(content_blueprint),
        content_blueprint=content_blueprint,
        media_refs=media_refs,
        input_mode=input_mode,
    )


def build_session_judge_request(
    prompt_text: str,
    dialogue_record: Dict[str, Any],
    *,
    visual_mode: str = "full_context",
) -> JudgeRequest:
    content_blueprint: List[Dict[str, Any]] = []
    media_refs: List[Dict[str, Any]] = []

    all_rounds = list(dialogue_record.get("rounds") or [])
    if visual_mode == "light_context":
        selected_rounds = _select_lightweight_session_visual_rounds(all_rounds)
        if len(selected_rounds) < len(all_rounds):
            content_blueprint.append(
                _text_segment(
                    f"以下仅提供 {len(selected_rounds)} 轮代表性视觉证据以降低请求负载；整段文本转录与评分标准仍保持完整。"
                )
            )
    else:
        selected_rounds = all_rounds

    for round_record in selected_rounds:
        segments, refs = _build_round_visual_segments(round_record, evidence_role="session_round")
        content_blueprint.extend(segments)
        media_refs.extend(refs)

    content_blueprint.append(_text_segment(prompt_text))
    input_mode = "text_plus_visual" if media_refs else "text_only"
    return JudgeRequest(
        prompt_text=prompt_text,
        internal_messages=_build_internal_messages(content_blueprint),
        content_blueprint=content_blueprint,
        media_refs=media_refs,
        input_mode=input_mode,
    )


