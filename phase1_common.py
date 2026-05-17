from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

from judge_pipeline import run_phase3_pipeline
from phase2_results import normalize_phase1_outputs


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_ROOT = REPO_ROOT / "omnibench_dataset"
DEFAULT_MEDIA_ROOT = DEFAULT_DATA_ROOT
VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".wmv")
AUDIO_EXTENSIONS = (".wav", ".mp3", ".m4a", ".flac", ".ogg", ".aac", ".opus", ".webm")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}


@dataclass(frozen=True)
class TaskSpec:
    name: str
    task_alias: str
    dataset_file: str
    source_type: str
    media_mode: str
    question_mode: str
    task_instruction: str = ""
    answer_instruction: str = ""
    spoken_question_placeholder: str = ""


@dataclass(frozen=True)
class Phase1DialogueWorkItem:
    doc_index: int
    dialogue_id: str
    doc: Dict[str, Any]
    existing_payload: Optional[Dict[str, Any]]
    resume_round_index: int = 0
    failed_round_indices: tuple[int, ...] = ()


@dataclass(frozen=True)
class Phase1SamplesIndex:
    ordered_ids: List[str]
    payloads_by_id: Dict[str, Dict[str, Any]]


TASK_SPECS: Dict[str, TaskSpec] = {
    "omnibench_video_stream_text": TaskSpec(
        name="omnibench_video_stream_text",
        task_alias="RUBRIC-MME Video Stream Text",
        dataset_file="video_final_with_vqa_category.json",
        source_type="video_stream",
        media_mode="video",
        question_mode="text",
        task_instruction="请结合提供的视频片段和已有多轮上下文，回答当前这一轮问题。",
        answer_instruction="请只回答当前这一轮，不要扩展到未来轮次。",
    ),
    "omnibench_video_stream_tts": TaskSpec(
        name="omnibench_video_stream_tts",
        task_alias="RUBRIC-MME Video Stream TTS",
        dataset_file="video_final_with_vqa_category.json",
        source_type="video_stream",
        media_mode="video",
        question_mode="tts",
        task_instruction="请结合提供的视频片段和当前的语音问题，回答这一轮问题。",
        answer_instruction="请只回答当前这一轮，不要扩展到未来轮次。",
        spoken_question_placeholder="当前这一轮的问题以语音形式给出，请结合附带的视频和问题音频直接作答。",
    ),
    "omnibench_image_multi_text": TaskSpec(
        name="omnibench_image_multi_text",
        task_alias="RUBRIC-MME Image Multi Text",
        dataset_file="image_final_with_mimt_category.json",
        source_type="image_multi",
        media_mode="image",
        question_mode="text",
        task_instruction="请结合提供的图像和已有多轮上下文，回答当前这一轮问题。",
        answer_instruction="请只回答当前这一轮，不要扩展到未来轮次。",
    ),
    "omnibench_image_multi_tts": TaskSpec(
        name="omnibench_image_multi_tts",
        task_alias="RUBRIC-MME Image Multi TTS",
        dataset_file="image_final_with_mimt_category.json",
        source_type="image_multi",
        media_mode="image",
        question_mode="tts",
        task_instruction="请结合提供的图像和当前的语音问题，回答这一轮问题。",
        answer_instruction="请只回答当前这一轮，不要扩展到未来轮次。",
        spoken_question_placeholder="当前这一轮的问题以语音形式给出，请结合附带的图像和问题音频直接作答。",
    ),
}


def resolve_task_names(tasks_arg: str) -> List[str]:
    raw = [part.strip() for part in tasks_arg.split(",") if part.strip()]
    if not raw or (len(raw) == 1 and raw[0].lower() in {"rubric-mme", "rubric_mme", "omnibench", "all"}):
        return list(TASK_SPECS.keys())
    unknown = [name for name in raw if name not in TASK_SPECS]
    if unknown:
        raise ValueError(f"未知任务: {unknown}; 支持: {sorted(TASK_SPECS)}")
    return raw


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def safe_name(name: str) -> str:
    return name.replace("/", "_").replace(":", "_")


def first_nonempty(*values: Optional[str]) -> str:
    for value in values:
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def load_json_list(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, list):
        raise ValueError(f"文件 {path} 不是 list JSON")
    return data


def read_completed_dialogue_ids(samples_path: Path) -> set[str]:
    completed: set[str] = set()
    if not samples_path.exists():
        return completed
    with samples_path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            dialogue_id = payload.get("dialogue_id")
            if dialogue_id:
                completed.add(str(dialogue_id))
    return completed


def load_phase1_samples_index(samples_path: Path) -> Phase1SamplesIndex:
    ordered_ids: List[str] = []
    payloads_by_id: Dict[str, Dict[str, Any]] = {}
    seen: set[str] = set()
    if not samples_path.exists():
        return Phase1SamplesIndex(ordered_ids=ordered_ids, payloads_by_id=payloads_by_id)

    with samples_path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            dialogue_id = str(payload.get("dialogue_id") or "").strip()
            if not dialogue_id:
                continue
            if dialogue_id not in seen:
                ordered_ids.append(dialogue_id)
                seen.add(dialogue_id)
            payloads_by_id[dialogue_id] = payload
    return Phase1SamplesIndex(ordered_ids=ordered_ids, payloads_by_id=payloads_by_id)


def append_jsonl(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(payload, ensure_ascii=False) + "\n")


def write_jsonl(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for payload in rows:
            file.write(json.dumps(payload, ensure_ascii=False) + "\n")


def dump_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)


def is_phase1_round_success(round_record: Dict[str, Any]) -> bool:
    if not isinstance(round_record, dict):
        return False
    if round_record.get("error"):
        return False
    prediction = round_record.get("prediction", "")
    return isinstance(prediction, str) and bool(prediction.strip())


def find_phase1_resume_round_index(expected_round_count: int, existing_payload: Optional[Dict[str, Any]]) -> int:
    if expected_round_count <= 0:
        return 0
    if not isinstance(existing_payload, dict):
        return 0

    existing_rounds = existing_payload.get("rounds")
    if not isinstance(existing_rounds, list):
        return 0

    upper_bound = min(expected_round_count, len(existing_rounds))
    for round_index in range(upper_bound):
        if not is_phase1_round_success(existing_rounds[round_index]):
            return round_index
    if len(existing_rounds) < expected_round_count:
        return len(existing_rounds)
    return expected_round_count


def is_phase1_dialogue_complete(doc: Dict[str, Any], existing_payload: Optional[Dict[str, Any]]) -> bool:
    expected_round_count = len(doc.get("rounds", []) or [])
    return find_phase1_resume_round_index(expected_round_count, existing_payload) >= expected_round_count


def collect_phase1_failed_round_indices(expected_round_count: int, existing_payload: Optional[Dict[str, Any]]) -> List[int]:
    if expected_round_count <= 0:
        return []
    if not isinstance(existing_payload, dict):
        return list(range(expected_round_count))

    existing_rounds = existing_payload.get("rounds")
    if not isinstance(existing_rounds, list):
        return list(range(expected_round_count))

    failed_indices: List[int] = []
    for round_index in range(expected_round_count):
        if round_index >= len(existing_rounds) or not is_phase1_round_success(existing_rounds[round_index]):
            failed_indices.append(round_index)
    return failed_indices


def build_phase1_work_items(
    docs: Sequence[Dict[str, Any]],
    samples_path: Path,
    *,
    resume: bool,
    repair_failed: bool,
    repair_mode: str = "resume_from_failure",
) -> tuple[List[Phase1DialogueWorkItem], Phase1SamplesIndex, int]:
    samples_index = load_phase1_samples_index(samples_path)
    work_items: List[Phase1DialogueWorkItem] = []
    skipped_dialogues = 0

    for doc_index, doc in enumerate(docs):
        dialogue_id = str(doc.get("dialogue_id", "")).strip()
        existing_payload = samples_index.payloads_by_id.get(dialogue_id)

        if repair_failed:
            if existing_payload is None:
                skipped_dialogues += 1
                continue
            if is_phase1_dialogue_complete(doc, existing_payload):
                skipped_dialogues += 1
                continue
            expected_round_count = len(doc.get("rounds", []) or [])
            resume_round_index = find_phase1_resume_round_index(expected_round_count, existing_payload)
            failed_round_indices = tuple(collect_phase1_failed_round_indices(expected_round_count, existing_payload))
            if repair_mode == "current_turn_only" and not failed_round_indices:
                skipped_dialogues += 1
                continue
            work_items.append(
                Phase1DialogueWorkItem(
                    doc_index=doc_index,
                    dialogue_id=dialogue_id,
                    doc=doc,
                    existing_payload=existing_payload,
                    resume_round_index=resume_round_index,
                    failed_round_indices=failed_round_indices,
                )
            )
            continue

        if resume and existing_payload is not None:
            skipped_dialogues += 1
            continue

        work_items.append(
            Phase1DialogueWorkItem(
                doc_index=doc_index,
                dialogue_id=dialogue_id,
                doc=doc,
                existing_payload=None,
                resume_round_index=0,
            )
        )

    return work_items, samples_index, skipped_dialogues


def merge_phase1_payloads(
    samples_index: Phase1SamplesIndex,
    updated_payloads_by_id: Dict[str, Dict[str, Any]],
    docs: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    merged_payloads: List[Dict[str, Any]] = []
    written_ids: set[str] = set()

    for dialogue_id in samples_index.ordered_ids:
        payload = updated_payloads_by_id.get(dialogue_id, samples_index.payloads_by_id.get(dialogue_id))
        if payload is None:
            continue
        merged_payloads.append(payload)
        written_ids.add(dialogue_id)

    for doc in docs:
        dialogue_id = str(doc.get("dialogue_id", "")).strip()
        if not dialogue_id or dialogue_id in written_ids:
            continue
        payload = updated_payloads_by_id.get(dialogue_id)
        if payload is None:
            continue
        merged_payloads.append(payload)
        written_ids.add(dialogue_id)

    return merged_payloads


def build_image_docs(raw_items: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    docs: List[Dict[str, Any]] = []
    for item in raw_items:
        rounds: List[Dict[str, Any]] = []
        for idx, conv in enumerate((item.get("image_conversation") or {}).get("conversations", [])):
            rounds.append(
                {
                    "round_index": idx,
                    "timestamp": conv.get("timestamp", ""),
                    "question_text": conv.get("human", ""),
                    "reference_answer": conv.get("ai", ""),
                    "category": conv.get("mimt_category", ""),
                    "image_local_path": conv.get("image_path", ""),
                    "image_remote_url": first_nonempty(conv.get("sign_url"), conv.get("gcs_url"), conv.get("oss_url")),
                    "question_audio_local_path": conv.get("human_tts_path", ""),
                    "question_audio_remote_url": first_nonempty(conv.get("tts_gcs_url")),
                    "question_audio_cloud_path": conv.get("tts_destPath", ""),
                }
            )
        docs.append(
            {
                "dialogue_id": item.get("_ai_unique_id_", ""),
                "benchmark_name": "RUBRIC-MME",
                "source_type": "image_multi",
                "environment": item.get("environment", ""),
                "interaction_setup": item.get("interaction_setup"),
                "conversation_meta": item.get("conversation_meta"),
                "rounds": rounds,
            }
        )
    return docs


def build_video_docs(raw_items: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    docs: List[Dict[str, Any]] = []
    for item in raw_items:
        rounds: List[Dict[str, Any]] = []
        pending_user: Optional[Dict[str, Any]] = None
        for entry in (item.get("stream_conversation") or {}).get("conversations", []):
            speaker = entry.get("speaker")
            if speaker == "user":
                pending_user = entry
                continue
            if speaker != "ai":
                continue
            question_turn = pending_user or {}
            rounds.append(
                {
                    "round_index": len(rounds),
                    "timestamp": question_turn.get("timestamp") or entry.get("timestamp", ""),
                    "question_text": question_turn.get("text", ""),
                    "reference_answer": entry.get("text", ""),
                    "category": entry.get("vqa_category", ""),
                    "video_local_path": entry.get("clip_path", ""),
                    "video_remote_url": first_nonempty(entry.get("gcs_url")),
                    "video_cloud_path": entry.get("destPath", ""),
                    "question_audio_local_path": entry.get("user_tts_path", ""),
                    "question_audio_remote_url": first_nonempty(entry.get("tts_gcs_url")),
                    "question_audio_cloud_path": entry.get("tts_destPath", ""),
                }
            )
            pending_user = None
        docs.append(
            {
                "dialogue_id": item.get("_ai_unique_id_", ""),
                "benchmark_name": "RUBRIC-MME",
                "source_type": "video_stream",
                "environment": item.get("environment", ""),
                "interaction_setup": item.get("interaction_setup"),
                "conversation_meta": item.get("conversation_meta"),
                "rounds": rounds,
            }
        )
    return docs


def load_docs_for_task(spec: TaskSpec, data_root: Path, dialogue_id: Optional[str], limit: Optional[int]) -> List[Dict[str, Any]]:
    if limit == 0:
        return []
    raw_items = load_json_list((data_root / spec.dataset_file).resolve())
    docs = build_video_docs(raw_items) if spec.source_type == "video_stream" else build_image_docs(raw_items)
    filtered: List[Dict[str, Any]] = []
    for doc in docs:
        if dialogue_id and doc.get("dialogue_id") != dialogue_id:
            continue
        filtered.append(doc)
        if limit is not None and len(filtered) >= limit:
            break
    return filtered


def convert_doc_for_official_runner(doc: Dict[str, Any]) -> Dict[str, Any]:
    rounds: List[Dict[str, Any]] = []
    for round_data in doc.get("rounds", []):
        rounds.append(
            {
                "round_index": round_data.get("round_index", len(rounds)),
                "timestamp": round_data.get("timestamp", ""),
                "question_text": round_data.get("question_text", ""),
                "answer_text": round_data.get("reference_answer", ""),
                "media_rel_path": first_nonempty(round_data.get("image_local_path", ""), round_data.get("video_local_path", "")),
                "question_audio_rel_path": round_data.get("question_audio_local_path", ""),
                "category": round_data.get("category", ""),
            }
        )
    return {
        "dialogue_id": doc.get("dialogue_id", ""),
        "benchmark_name": doc.get("benchmark_name", "RUBRIC-MME"),
        "source_type": doc.get("source_type", ""),
        "environment": doc.get("environment", ""),
        "round_count": len(rounds),
        "rounds": rounds,
        "interaction_setup": doc.get("interaction_setup"),
        "conversation_meta": doc.get("conversation_meta"),
    }


def run_phase1_followups(
    *,
    summaries: Sequence[Dict[str, Any]],
    run_summary: Dict[str, Any],
    phase2_output_dir: str,
    phase2_keep_existing: bool,
    phase3_output_dir: str,
    phase3_keep_existing: bool,
    dialogue_id: str,
    limit: Optional[int],
    phase3_save_prompt_text: bool,
    phase3_allow_incomplete_dialogues: bool,
    build_phase3_backend: Callable[[], Any],
) -> Dict[str, Any]:
    if phase3_output_dir and not phase2_output_dir:
        raise ValueError("使用 Phase 3 输出目录时，必须同时提供 Phase 2 输出目录。")

    if phase2_output_dir:
        phase2_output_path = Path(phase2_output_dir).resolve()
        phase1_sample_files = [Path(summary["samples_path"]).resolve() for summary in summaries if summary.get("samples_path")]
        phase2_manifest = normalize_phase1_outputs(
            phase1_sample_files,
            phase2_output_path,
            strict=False,
            clear_output=not phase2_keep_existing,
        )
        run_summary["phase2_output_dir"] = str(phase2_output_path)
        run_summary["phase2_manifest_path"] = str(phase2_output_path / "manifest.json")
        run_summary["phase2_schema_version"] = phase2_manifest.get("schema_version", "")

    if phase3_output_dir:
        phase3_output_path = Path(phase3_output_dir).resolve()
        phase2_output_path = Path(phase2_output_dir).resolve()
        phase3_manifest = run_phase3_pipeline(
            phase2_output_path,
            phase3_output_path,
            build_phase3_backend(),
            dialogue_id=dialogue_id or "",
            limit_dialogues=limit,
            clear_output=not phase3_keep_existing,
            resume=phase3_keep_existing,
            save_prompt_text=phase3_save_prompt_text,
            allow_incomplete_dialogues=phase3_allow_incomplete_dialogues,
        )
        run_summary["phase3_output_dir"] = str(phase3_output_path)
        run_summary["phase3_manifest_path"] = str(phase3_output_path / "manifest.json")
        run_summary["phase3_schema_version"] = phase3_manifest.get("schema_version", "")

    return run_summary
