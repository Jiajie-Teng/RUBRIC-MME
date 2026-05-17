from __future__ import annotations

import sys
from typing import List

import run_qwen_openai_compatible_phase1 as qwen_phase1


DEFAULT_MODEL = "qwen2.5-vl-72b-instruct"
SUPPORTED_TASKS = [
    "omnibench_image_multi_text",
    "omnibench_video_stream_text",
]


def _has_option(argv: List[str], option: str) -> bool:
    return option in argv


def _value_after(argv: List[str], option: str) -> str | None:
    try:
        index = argv.index(option)
    except ValueError:
        return None
    if index + 1 >= len(argv):
        return None
    return argv[index + 1]


def _normalize_tasks(tasks_value: str | None) -> str:
    if not tasks_value:
        return ",".join(SUPPORTED_TASKS)

    raw_tasks = [task.strip() for task in tasks_value.split(",") if task.strip()]
    lowered = {task.lower() for task in raw_tasks}
    if not raw_tasks or lowered & {"rubric-mme", "rubric_mme", "omnibench", "all"}:
        return ",".join(SUPPORTED_TASKS)

    unsupported = [task for task in raw_tasks if task not in SUPPORTED_TASKS]
    if unsupported:
        raise SystemExit(
            "Qwen2.5-VL Phase 1 当前只支持两个 text 任务："
            f"{SUPPORTED_TASKS}；收到不支持任务：{unsupported}"
        )
    return ",".join(raw_tasks)


def build_forward_argv(argv: List[str]) -> List[str]:
    forward = list(argv)

    if not _has_option(forward, "--model"):
        forward.extend(["--model", DEFAULT_MODEL])

    normalized_tasks = _normalize_tasks(_value_after(forward, "--tasks"))
    if _has_option(forward, "--tasks"):
        task_index = forward.index("--tasks")
        if task_index + 1 >= len(forward):
            raise SystemExit("--tasks 后缺少任务值")
        forward[task_index + 1] = normalized_tasks
    else:
        forward.extend(["--tasks", normalized_tasks])

    return forward


def main() -> None:
    sys.argv = [sys.argv[0], *build_forward_argv(sys.argv[1:])]
    qwen_phase1.main()


if __name__ == "__main__":
    main()
