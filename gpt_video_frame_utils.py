from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:
    import imageio_ffmpeg  # type: ignore
except Exception:  # pragma: no cover
    imageio_ffmpeg = None


DEFAULT_VIDEO_FRAME_ROOT_NAME = "video_final_gpt_frames"
DEFAULT_VIDEO_PREPARED_DIR_FIELD = "gpt_frame_dir"
DEFAULT_VIDEO_PREPARED_PATHS_FIELD = "gpt_frame_paths"
DEFAULT_VIDEO_PREPARED_PROFILE_FIELD = "gpt_frame_profile"
DEFAULT_VIDEO_PREPARED_TOTAL_BYTES_FIELD = "gpt_frame_total_bytes"
DEFAULT_VIDEO_PREPARED_STRATEGY_FIELD = "gpt_frame_sampling_strategy"
DEFAULT_VIDEO_PREPARED_SOURCE_SIZE_FIELD = "gpt_frame_source_size_bytes"
DEFAULT_VIDEO_PREPARED_DURATION_FIELD = "gpt_frame_duration_seconds"
DEFAULT_VIDEO_FRAME_COUNT = 10
DEFAULT_VIDEO_FRAME_MAX_SIDE = 768
DEFAULT_VIDEO_FRAME_JPEG_QUALITY = 8
DEFAULT_VIDEO_FRAME_MAX_INLINE_BYTES = 3_000_000
DEFAULT_VIDEO_FRAME_SAMPLING_STRATEGY = "hybrid_tail"
SUPPORTED_VIDEO_FRAME_SAMPLING_STRATEGIES = ("uniform", "hybrid_tail")


def resolve_ffmpeg_executable() -> Optional[str]:
    if imageio_ffmpeg is not None:
        try:
            return imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            pass
    return shutil.which("ffmpeg")


def resolve_ffprobe_executable() -> Optional[str]:
    ffprobe_exe = shutil.which("ffprobe")
    if ffprobe_exe:
        return ffprobe_exe
    ffmpeg_exe = resolve_ffmpeg_executable()
    if ffmpeg_exe:
        ffmpeg_path = Path(ffmpeg_exe)
        sibling_candidates = [
            ffmpeg_path.with_name("ffprobe.exe"),
            ffmpeg_path.with_name("ffprobe"),
        ]
        for candidate in sibling_candidates:
            if candidate.exists():
                return str(candidate)
    return None


def probe_video_duration_seconds(source_path: Path) -> Optional[float]:
    ffprobe_exe = resolve_ffprobe_executable()
    if ffprobe_exe:
        command = [
            ffprobe_exe,
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=nokey=1:noprint_wrappers=1",
            str(source_path),
        ]
        completed = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        if completed.returncode == 0:
            try:
                duration = float(completed.stdout.decode("utf-8", errors="ignore").strip())
            except Exception:
                duration = None
            if duration and duration > 0:
                return duration

    ffmpeg_exe = resolve_ffmpeg_executable()
    if not ffmpeg_exe:
        return None
    command = [
        ffmpeg_exe,
        "-i",
        str(source_path),
        "-f",
        "null",
        "-",
    ]
    completed = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    stderr_text = completed.stderr.decode("utf-8", errors="ignore")
    marker = "Duration:"
    if marker not in stderr_text:
        return None
    try:
        duration_text = stderr_text.split(marker, 1)[1].split(",", 1)[0].strip()
        hours, minutes, seconds = duration_text.split(":")
        duration = int(hours) * 3600 + int(minutes) * 60 + float(seconds)
        return duration if duration > 0 else None
    except Exception:
        return None


def dynamic_target_frame_count(base_count: int, duration_seconds: Optional[float]) -> int:
    base = max(4, int(base_count))
    if duration_seconds is None:
        return base
    if duration_seconds <= 10:
        return max(base, 10)
    if duration_seconds <= 30:
        return max(base + 2, 12)
    if duration_seconds <= 90:
        return max(base + 6, 16)
    return max(base + 10, 20)


def _linspace_positions(start: float, end: float, count: int) -> List[float]:
    if count <= 0:
        return []
    if count == 1:
        return [round((start + end) / 2.0, 6)]
    step = (end - start) / float(count - 1)
    return [round(start + step * index, 6) for index in range(count)]


def build_sampling_positions(frame_count: int, strategy: str) -> List[float]:
    count = max(1, int(frame_count))
    if strategy == "uniform":
        return _linspace_positions(0.03, 0.97, count)
    global_count = max(2, min(count, int(round(count * 0.6))))
    tail_count = max(0, count - global_count)
    positions = _linspace_positions(0.03, 0.97, global_count) + _linspace_positions(0.55, 0.99, tail_count)
    deduped: List[float] = []
    seen: set[float] = set()
    for position in sorted(positions):
        rounded = round(position, 4)
        if rounded in seen:
            continue
        seen.add(rounded)
        deduped.append(position)
    if len(deduped) >= count:
        return deduped[:count]
    filler = _linspace_positions(0.08, 0.98, count * 2)
    for position in filler:
        rounded = round(position, 4)
        if rounded in seen:
            continue
        seen.add(rounded)
        deduped.append(position)
        if len(deduped) >= count:
            break
    return sorted(deduped[:count])


def build_sampling_timestamps(duration_seconds: float, frame_count: int, strategy: str) -> List[float]:
    if duration_seconds <= 0:
        return [0.0]
    max_timestamp = max(duration_seconds - 0.05, 0.0)
    timestamps: List[float] = []
    seen: set[float] = set()
    for position in build_sampling_positions(frame_count, strategy):
        timestamp = round(min(max(duration_seconds * position, 0.0), max_timestamp), 3)
        if timestamp in seen:
            continue
        seen.add(timestamp)
        timestamps.append(timestamp)
    if not timestamps:
        return [0.0]
    return timestamps


def select_frame_indices(total_count: int, target_count: int, strategy: str) -> List[int]:
    if total_count <= 0 or target_count <= 0:
        return []
    if target_count >= total_count:
        return list(range(total_count))
    positions = build_sampling_positions(target_count, strategy)
    indices: List[int] = []
    seen: set[int] = set()
    max_index = total_count - 1
    for position in positions:
        index = min(max_index, max(0, int(round(position * max_index))))
        if index in seen:
            continue
        seen.add(index)
        indices.append(index)
    if len(indices) >= target_count:
        return sorted(indices[:target_count])
    for index in range(total_count):
        if index in seen:
            continue
        seen.add(index)
        indices.append(index)
        if len(indices) >= target_count:
            break
    return sorted(indices[:target_count])


def subset_prepared_frame_paths(prepared_paths: Sequence[str], target_count: int, strategy: str) -> List[str]:
    if target_count <= 0:
        return []
    if len(prepared_paths) <= target_count:
        return list(prepared_paths)
    indices = select_frame_indices(len(prepared_paths), target_count, strategy)
    return [str(prepared_paths[index]) for index in indices]


def subset_frame_paths(frame_paths: Sequence[Path], target_count: int, strategy: str) -> List[Path]:
    if target_count <= 0:
        return []
    if len(frame_paths) <= target_count:
        return list(frame_paths)
    indices = select_frame_indices(len(frame_paths), target_count, strategy)
    return [frame_paths[index] for index in indices]


def frame_profile_candidates(
    *,
    base_frame_count: int,
    base_max_side: int,
    base_jpeg_quality: int,
    duration_seconds: Optional[float],
) -> List[Dict[str, int]]:
    target_frame_count = dynamic_target_frame_count(base_frame_count, duration_seconds)
    base_side = max(256, int(base_max_side))
    base_quality = max(2, int(base_jpeg_quality))
    candidates = [
        {"frame_count": target_frame_count, "max_side": base_side, "jpeg_quality": base_quality},
        {"frame_count": max(8, target_frame_count - 2), "max_side": min(base_side, 704), "jpeg_quality": max(base_quality + 2, 10)},
        {"frame_count": max(6, target_frame_count - 4), "max_side": min(base_side, 640), "jpeg_quality": max(base_quality + 4, 12)},
        {"frame_count": max(4, target_frame_count - 6), "max_side": min(base_side, 576), "jpeg_quality": max(base_quality + 6, 14)},
        {"frame_count": 4, "max_side": min(base_side, 448), "jpeg_quality": max(base_quality + 8, 16)},
    ]
    deduped: List[Dict[str, int]] = []
    seen: set[Tuple[int, int, int]] = set()
    for candidate in candidates:
        key = (candidate["frame_count"], candidate["max_side"], candidate["jpeg_quality"])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(candidate)
    return deduped


def list_existing_frames(target_dir: Path) -> List[Path]:
    if not target_dir.exists():
        return []
    return sorted(path for path in target_dir.glob("frame_*.jpg") if path.is_file())


def relative_storage_dir_for_source(source_path: Path, media_root: Path, prepared_root: Path) -> Path:
    try:
        source_rel = source_path.resolve().relative_to(media_root.resolve())
    except ValueError:
        source_rel = Path(source_path.name)
    trailing_parent = Path(*source_rel.parts[1:-1]) if len(source_rel.parts) > 2 else Path()
    return prepared_root / trailing_parent


def build_frame_bundle_dir(
    *,
    source_path: Path,
    media_root: Path,
    prepared_root: Path,
    profile: Dict[str, int],
    sampling_strategy: str,
) -> Path:
    target_dir = relative_storage_dir_for_source(source_path, media_root, prepared_root)
    try:
        source_rel = source_path.resolve().relative_to(media_root.resolve()).as_posix()
    except ValueError:
        source_rel = source_path.name
    source_hash = hashlib.sha1(source_rel.encode("utf-8")).hexdigest()[:12]
    profile_key = f"f{profile['frame_count']}_s{profile['max_side']}_q{profile['jpeg_quality']}"
    return target_dir / f"{source_path.stem}_{source_hash}_{sampling_strategy}_{profile_key}"


def extract_single_frame(
    *,
    ffmpeg_exe: str,
    source_path: Path,
    output_path: Path,
    timestamp_seconds: float,
    max_side: int,
    jpeg_quality: int,
) -> Optional[str]:
    command = [
        ffmpeg_exe,
        "-y",
        "-ss",
        f"{max(timestamp_seconds, 0.0):.3f}",
        "-i",
        str(source_path),
        "-frames:v",
        "1",
        "-vf",
        f"scale='min({max_side},iw)':-2",
        "-q:v",
        str(jpeg_quality),
        str(output_path),
    ]
    completed = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if completed.returncode != 0 or not output_path.exists():
        return completed.stderr.decode("utf-8", errors="ignore").strip()[-500:]
    return None


def extract_frames_at_timestamps(
    *,
    ffmpeg_exe: str,
    source_path: Path,
    target_dir: Path,
    timestamps: Sequence[float],
    max_side: int,
    jpeg_quality: int,
) -> Optional[str]:
    target_dir.mkdir(parents=True, exist_ok=True)
    for index, timestamp in enumerate(timestamps, start=1):
        output_path = target_dir / f"frame_{index:02d}.jpg"
        error = extract_single_frame(
            ffmpeg_exe=ffmpeg_exe,
            source_path=source_path,
            output_path=output_path,
            timestamp_seconds=float(timestamp),
            max_side=max_side,
            jpeg_quality=jpeg_quality,
        )
        if error:
            return error
    return None


def extract_uniform_frames_fallback(
    *,
    ffmpeg_exe: str,
    source_path: Path,
    target_dir: Path,
    frame_count: int,
    duration_seconds: Optional[float],
    max_side: int,
    jpeg_quality: int,
) -> Optional[str]:
    target_dir.mkdir(parents=True, exist_ok=True)
    if duration_seconds and duration_seconds > 0:
        fps = max(frame_count / duration_seconds, 0.2)
        vf = f"fps={fps:.6f},scale='min({max_side},iw)':-2"
    else:
        vf = f"fps=1,scale='min({max_side},iw)':-2"
    command = [
        ffmpeg_exe,
        "-y",
        "-i",
        str(source_path),
        "-vf",
        vf,
        "-q:v",
        str(jpeg_quality),
        "-frames:v",
        str(frame_count),
        str(target_dir / "frame_%02d.jpg"),
    ]
    completed = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if completed.returncode != 0:
        return completed.stderr.decode("utf-8", errors="ignore").strip()[-500:]
    return None


def dump_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)


def load_existing_frame_bundle(target_dir: Path, max_inline_bytes: int) -> Optional[Tuple[List[Path], Dict[str, Any]]]:
    manifest_path = target_dir / "_frame_manifest.json"
    if not manifest_path.exists():
        return None
    try:
        with manifest_path.open("r", encoding="utf-8") as file:
            manifest = json.load(file)
    except Exception:
        return None
    frame_paths = list_existing_frames(target_dir)
    if not frame_paths:
        return None
    total_size = sum(path.stat().st_size for path in frame_paths)
    if total_size > max_inline_bytes:
        return None
    manifest["prepared_size_bytes"] = total_size
    manifest["frame_count"] = len(frame_paths)
    manifest["prepared_local_path"] = str(target_dir)
    return frame_paths, manifest


def build_frame_manifest(
    *,
    source_path: Path,
    media_root: Path,
    frame_paths: Sequence[Path],
    prepared_size_bytes: int,
    profile: Dict[str, int],
    sampling_strategy: str,
    duration_seconds: Optional[float],
    timestamps: Sequence[float],
    used_existing: bool,
) -> Dict[str, Any]:
    try:
        source_rel = source_path.resolve().relative_to(media_root.resolve()).as_posix()
    except ValueError:
        source_rel = source_path.name
    prepared_rel_paths: List[str] = []
    for frame_path in frame_paths:
        try:
            rel_path = frame_path.resolve().relative_to(media_root.resolve()).as_posix()
        except ValueError:
            rel_path = frame_path.name
        prepared_rel_paths.append(rel_path)
    return {
        "source_local_path": str(source_path),
        "source_rel_path": source_rel,
        "source_size_bytes": source_path.stat().st_size,
        "prepared_size_bytes": prepared_size_bytes,
        "prepared_rel_paths": prepared_rel_paths,
        "prepared_local_path": str(frame_paths[0].parent if frame_paths else ""),
        "frame_count": len(frame_paths),
        "profile": f"f{profile['frame_count']}_s{profile['max_side']}_q{profile['jpeg_quality']}",
        "requested_frame_count": profile["frame_count"],
        "max_side": profile["max_side"],
        "jpeg_quality": profile["jpeg_quality"],
        "sampling_strategy": sampling_strategy,
        "duration_seconds": duration_seconds,
        "sampling_timestamps": list(timestamps),
        "used_existing": used_existing,
        "selected_source": "precomputed_video_frames",
    }


def prepare_video_frames(
    *,
    source_path: Path,
    media_root: Path,
    prepared_root: Path,
    base_frame_count: int,
    base_max_side: int,
    base_jpeg_quality: int,
    max_inline_bytes: int,
    sampling_strategy: str,
) -> Tuple[List[Path], Dict[str, Any]]:
    ffmpeg_exe = resolve_ffmpeg_executable()
    if not ffmpeg_exe:
        raise RuntimeError("当前环境没有可用 ffmpeg，无法对视频做抽帧。")
    if sampling_strategy not in SUPPORTED_VIDEO_FRAME_SAMPLING_STRATEGIES:
        raise ValueError(f"不支持的抽帧采样策略: {sampling_strategy}")

    duration_seconds = probe_video_duration_seconds(source_path)
    last_error = ""
    for profile in frame_profile_candidates(
        base_frame_count=base_frame_count,
        base_max_side=base_max_side,
        base_jpeg_quality=base_jpeg_quality,
        duration_seconds=duration_seconds,
    ):
        target_dir = build_frame_bundle_dir(
            source_path=source_path,
            media_root=media_root,
            prepared_root=prepared_root,
            profile=profile,
            sampling_strategy=sampling_strategy,
        )
        existing = load_existing_frame_bundle(target_dir, max_inline_bytes)
        if existing is not None:
            return existing

        if target_dir.exists():
            shutil.rmtree(target_dir, ignore_errors=True)
        target_dir.mkdir(parents=True, exist_ok=True)

        sampling_timestamps: List[float] = []
        if duration_seconds and duration_seconds > 0:
            sampling_timestamps = build_sampling_timestamps(duration_seconds, profile["frame_count"], sampling_strategy)
            error = extract_frames_at_timestamps(
                ffmpeg_exe=ffmpeg_exe,
                source_path=source_path,
                target_dir=target_dir,
                timestamps=sampling_timestamps,
                max_side=profile["max_side"],
                jpeg_quality=profile["jpeg_quality"],
            )
            if error:
                shutil.rmtree(target_dir, ignore_errors=True)
                last_error = error
                continue
        else:
            error = extract_uniform_frames_fallback(
                ffmpeg_exe=ffmpeg_exe,
                source_path=source_path,
                target_dir=target_dir,
                frame_count=profile["frame_count"],
                duration_seconds=duration_seconds,
                max_side=profile["max_side"],
                jpeg_quality=profile["jpeg_quality"],
            )
            if error:
                shutil.rmtree(target_dir, ignore_errors=True)
                last_error = error
                continue

        frame_paths = list_existing_frames(target_dir)
        if not frame_paths:
            shutil.rmtree(target_dir, ignore_errors=True)
            last_error = "ffmpeg 执行完成但没有产出任何帧图像。"
            continue
        prepared_size_bytes = sum(path.stat().st_size for path in frame_paths)
        if prepared_size_bytes > max_inline_bytes:
            shutil.rmtree(target_dir, ignore_errors=True)
            last_error = f"抽帧总大小 {prepared_size_bytes} 超过阈值 {max_inline_bytes}"
            continue
        manifest = build_frame_manifest(
            source_path=source_path,
            media_root=media_root,
            frame_paths=frame_paths,
            prepared_size_bytes=prepared_size_bytes,
            profile=profile,
            sampling_strategy=sampling_strategy,
            duration_seconds=duration_seconds,
            timestamps=sampling_timestamps,
            used_existing=False,
        )
        dump_json(target_dir / "_frame_manifest.json", manifest)
        return frame_paths, manifest

    raise RuntimeError(f"视频抽帧后仍超过可接受大小，文件：{source_path}。{last_error}")


def resolve_prepared_frame_paths(
    *,
    media_root: Path,
    prepared_paths: Sequence[str],
    max_inline_bytes: int,
) -> Optional[Tuple[List[Path], Dict[str, Any]]]:
    resolved_paths: List[Path] = []
    for rel_path in prepared_paths:
        if not rel_path:
            continue
        candidate = Path(rel_path)
        if not candidate.is_absolute():
            candidate = (media_root / candidate).resolve()
        if not candidate.exists():
            return None
        resolved_paths.append(candidate)
    if not resolved_paths:
        return None
    prepared_size_bytes = sum(path.stat().st_size for path in resolved_paths)
    if prepared_size_bytes > max_inline_bytes:
        return None
    manifest_path = resolved_paths[0].parent / "_frame_manifest.json"
    manifest: Dict[str, Any] = {}
    if manifest_path.exists():
        try:
            with manifest_path.open("r", encoding="utf-8") as file:
                manifest = json.load(file)
        except Exception:
            manifest = {}
    manifest["prepared_size_bytes"] = prepared_size_bytes
    manifest["frame_count"] = len(resolved_paths)
    manifest["prepared_local_path"] = str(resolved_paths[0].parent)
    manifest["selected_source"] = "precomputed_video_frames"
    return resolved_paths, manifest
