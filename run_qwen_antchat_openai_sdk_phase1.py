from __future__ import annotations

import os
import sys
import time
from typing import Any, Dict, List, Tuple

import run_qwen_openai_compatible_phase1 as qwen_phase1


DEFAULT_MODEL = "Qwen2.5-VL-7B-Instruct"
DEFAULT_BASE_URL = "https://antchat.alipay.com/v1"
DEFAULT_API_KEY_ENV = "MATRIXLLM_API_KEY"
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
            "Qwen antchat Phase 1 当前只支持两个 text 任务："
            f"{SUPPORTED_TASKS}；收到不支持任务：{unsupported}"
        )
    return ",".join(raw_tasks)


def build_forward_argv(argv: List[str]) -> List[str]:
    forward = list(argv)

    if not _has_option(forward, "--model"):
        forward.extend(["--model", DEFAULT_MODEL])
    if not _has_option(forward, "--api-url"):
        forward.extend(["--api-url", DEFAULT_BASE_URL])
    if not _has_option(forward, "--api-key-env"):
        forward.extend(["--api-key-env", DEFAULT_API_KEY_ENV])

    normalized_tasks = _normalize_tasks(_value_after(forward, "--tasks"))
    if _has_option(forward, "--tasks"):
        task_index = forward.index("--tasks")
        if task_index + 1 >= len(forward):
            raise SystemExit("--tasks 后缺少任务值")
        forward[task_index + 1] = normalized_tasks
    else:
        forward.extend(["--tasks", normalized_tasks])

    return forward


def _lazy_openai_client(api_key: str, base_url: str, timeout: int) -> Any:
    try:
        from openai import OpenAI
    except Exception as exc:  # pragma: no cover - import guard
        raise RuntimeError(
            "当前环境未安装 openai 包，无法使用 antchat OpenAI SDK 路线。"
            "请先在实际运行环境安装 openai，例如：pip install openai"
        ) from exc
    return OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)


def _usage_to_dict(usage: Any) -> Dict[str, Any]:
    if usage is None:
        return {}
    if isinstance(usage, dict):
        return usage
    if hasattr(usage, "model_dump"):
        dumped = usage.model_dump()
        if isinstance(dumped, dict):
            return dumped
    if hasattr(usage, "dict"):
        dumped = usage.dict()
        if isinstance(dumped, dict):
            return dumped
    result: Dict[str, Any] = {}
    for key in ["prompt_tokens", "completion_tokens", "total_tokens", "completion_tokens_details"]:
        value = getattr(usage, key, None)
        if value is None:
            continue
        if hasattr(value, "model_dump"):
            value = value.model_dump()
        elif hasattr(value, "dict"):
            value = value.dict()
        result[key] = value
    return result


def _response_to_dict(response: Any) -> Dict[str, Any]:
    if response is None:
        return {}
    if isinstance(response, dict):
        return response
    if hasattr(response, "model_dump"):
        dumped = response.model_dump()
        if isinstance(dumped, dict):
            return dumped
    if hasattr(response, "dict"):
        dumped = response.dict()
        if isinstance(dumped, dict):
            return dumped
    return {}


def _extract_prediction_from_sdk_response(response: Any) -> str:
    choices = getattr(response, "choices", None)
    if not choices and isinstance(response, dict):
        choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    choice = choices[0]
    message = getattr(choice, "message", None)
    if message is None and isinstance(choice, dict):
        message = choice.get("message")
    content = getattr(message, "content", None)
    if content is None and isinstance(message, dict):
        content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
                continue
            if isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
                    continue
                if item.get("type") == "text" and isinstance(item.get("content"), str):
                    parts.append(item["content"])
                    continue
            text = getattr(item, "text", None)
            if isinstance(text, str):
                parts.append(text)
        return "".join(parts)
    return ""


def _normalize_content_item_for_antchat(item: Any) -> List[Any]:
    if not isinstance(item, dict):
        return [item]

    item_type = item.get("type")
    if item_type == "video" and isinstance(item.get("video"), list):
        normalized: List[Dict[str, Any]] = []
        for frame_url in item.get("video") or []:
            if not isinstance(frame_url, str) or not frame_url:
                continue
            normalized.append({"type": "image_url", "image_url": {"url": frame_url}})
        return normalized

    if item_type == "image_url":
        image_value = item.get("image_url")
        if isinstance(image_value, str):
            return [{"type": "image_url", "image_url": {"url": image_value}}]

    if item_type == "video_url":
        video_value = item.get("video_url")
        if isinstance(video_value, str):
            return [{"type": "video_url", "video_url": {"url": video_value}}]

    return [item]


def _normalize_messages_for_antchat(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    normalized_messages: List[Dict[str, Any]] = []
    for message in messages:
        if not isinstance(message, dict):
            normalized_messages.append(message)
            continue
        content = message.get("content")
        if not isinstance(content, list):
            normalized_messages.append(message)
            continue

        normalized_content: List[Any] = []
        for item in content:
            normalized_content.extend(_normalize_content_item_for_antchat(item))

        normalized_message = dict(message)
        normalized_message["content"] = normalized_content
        normalized_messages.append(normalized_message)
    return normalized_messages


class QwenAntchatSDKPhase1Runner(qwen_phase1.QwenCompatiblePhase1Runner):
    def __init__(self, *, model_name: str, api_url: str, api_key_env: str, timeout: int, max_retries: int, retry_sleep: float, rate_limit_retry_sleep: float, rate_limit_max_sleep: float, temperature: float, top_p: float, max_output_tokens: int, stream_mode: str) -> None:
        api_key = os.getenv(api_key_env, "").strip()
        if not api_key:
            raise RuntimeError(f"未找到 API key，请先设置环境变量 {api_key_env}")
        self.api_key = api_key
        self.model_name = model_name
        self.api_url = api_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_sleep = retry_sleep
        self.rate_limit_retry_sleep = rate_limit_retry_sleep
        self.rate_limit_max_sleep = rate_limit_max_sleep
        self.temperature = temperature
        self.top_p = top_p
        self.max_output_tokens = max_output_tokens
        self.stream_mode = stream_mode
        self._tts_auto_lock = qwen_phase1.Lock()
        self._tts_resolved_mode = None
        self._tts_unsupported_modes = set()
        self.client = _lazy_openai_client(self.api_key, self.api_url, self.timeout)

    def reset_session(self) -> None:
        self.client = _lazy_openai_client(self.api_key, self.api_url, self.timeout)

    def close(self) -> None:
        return None

    def get_stream_candidates(self) -> List[bool]:
        # antchat 先走稳定的非 stream 路线，和同事提供的 test_connection.py 对齐。
        return [False]

    def generate_round(self, messages: List[Dict[str, Any]]) -> Tuple[str, Dict[str, Any], str, Dict[str, Any]]:
        last_error = ""
        last_error_info: Dict[str, Any] = {"status_code": None, "error_type": "", "retriable": False, "retry_trace": []}
        completion_budgets = [self.max_output_tokens]
        for extra_budget in [2048, 4096]:
            if extra_budget not in completion_budgets:
                completion_budgets.append(extra_budget)

        for budget_index, completion_budget in enumerate(completion_budgets):
            parameter_profiles: List[Dict[str, Any]] = [
                {"temperature": self.temperature, "top_p": self.top_p, "max_tokens": completion_budget},
                {"temperature": self.temperature, "max_tokens": completion_budget},
                {"max_tokens": completion_budget},
                {"max_completion_tokens": completion_budget},
            ]
            deduped_profiles: List[Dict[str, Any]] = []
            seen_profiles: set[Tuple[Tuple[str, Any], ...]] = set()
            for profile in parameter_profiles:
                key = tuple(sorted(profile.items()))
                if key in seen_profiles:
                    continue
                seen_profiles.add(key)
                deduped_profiles.append(profile)

            for profile in deduped_profiles:
                payload: Dict[str, Any] = {
                    "model": self.model_name,
                    "messages": _normalize_messages_for_antchat(messages),
                    **profile,
                }
                for attempt in range(self.max_retries):
                    try:
                        response = self.client.chat.completions.create(**payload)
                    except Exception as exc:  # pragma: no cover - depends on runtime env
                        last_error = f"{type(exc).__name__}: {exc}"
                        lowered = last_error.lower()
                        is_rate_limit = "429" in lowered or "rate limit" in lowered
                        is_connection_error = any(token in lowered for token in [
                            "failed to establish a new connection",
                            "connection error",
                            "apiconnectionerror",
                            "winerror 10013",
                            "timed out",
                        ])
                        error_type = "rate_limit" if is_rate_limit else ("connection_exception" if is_connection_error else "request_exception")
                        sleep_seconds = self.rate_limit_retry_sleep if is_rate_limit else self.retry_sleep
                        if is_connection_error:
                            sleep_seconds = min(max(self.retry_sleep * (2 ** attempt), 10.0), self.rate_limit_max_sleep)
                            self.reset_session()
                        elif is_rate_limit:
                            sleep_seconds = min(self.rate_limit_retry_sleep * (2 ** attempt), self.rate_limit_max_sleep)
                        retry_trace = list(last_error_info.get("retry_trace", []))
                        retry_trace.append(
                            {
                                "attempt": attempt + 1,
                                "error_type": error_type,
                                "sleep_seconds": round(sleep_seconds, 2),
                                "message": last_error[-240:],
                                "parameter_profile": profile,
                                "completion_budget": completion_budget,
                                "stream_enabled": False,
                            }
                        )
                        last_error_info = {
                            "status_code": 429 if is_rate_limit else None,
                            "error_type": error_type,
                            "retriable": attempt + 1 < self.max_retries or budget_index + 1 < len(completion_budgets),
                            "retry_trace": retry_trace,
                        }
                        if attempt + 1 < self.max_retries:
                            time.sleep(sleep_seconds)
                            continue
                        if budget_index + 1 < len(completion_budgets):
                            break
                        return "", {}, last_error, last_error_info

                    response_payload = _response_to_dict(response)
                    prediction = _extract_prediction_from_sdk_response(response)
                    usage = _usage_to_dict(getattr(response, "usage", None))
                    if qwen_phase1.should_retry_for_reasoning_exhaustion(prediction, response_payload, completion_budget) and budget_index + 1 < len(completion_budgets):
                        last_error = f"Qwen antchat completion budget exhausted at {completion_budget} tokens without visible answer."
                        retry_trace = list(last_error_info.get("retry_trace", []))
                        retry_trace.append(
                            {
                                "attempt": attempt + 1,
                                "error_type": "reasoning_exhaustion",
                                "sleep_seconds": 0.0,
                                "message": last_error,
                                "parameter_profile": profile,
                                "completion_budget": completion_budget,
                                "stream_enabled": False,
                            }
                        )
                        last_error_info = {
                            "status_code": None,
                            "error_type": "reasoning_exhaustion",
                            "retriable": True,
                            "retry_trace": retry_trace,
                        }
                        break
                    if qwen_phase1.should_retry_for_empty_completion(prediction, response_payload, usage):
                        last_error = "Qwen antchat returned an empty visible completion."
                        retry_trace = list(last_error_info.get("retry_trace", []))
                        retry_trace.append(
                            {
                                "attempt": attempt + 1,
                                "error_type": "empty_completion",
                                "sleep_seconds": 0.0,
                                "message": last_error,
                                "parameter_profile": profile,
                                "completion_budget": completion_budget,
                                "stream_enabled": False,
                            }
                        )
                        last_error_info = {
                            "status_code": None,
                            "error_type": "empty_completion",
                            "retriable": attempt + 1 < self.max_retries or budget_index + 1 < len(completion_budgets),
                            "retry_trace": retry_trace,
                        }
                        if attempt + 1 < self.max_retries:
                            time.sleep(self.retry_sleep)
                            continue
                        if budget_index + 1 < len(completion_budgets):
                            break
                        return "", {}, last_error, last_error_info
                    return prediction, usage, "", {"status_code": None, "error_type": "", "retriable": False, "retry_trace": []}

        return "", {}, last_error, last_error_info


def build_runner(args: Any) -> QwenAntchatSDKPhase1Runner:
    return QwenAntchatSDKPhase1Runner(
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
        stream_mode=args.stream_mode,
    )


def main() -> None:
    qwen_phase1.build_runner = build_runner
    sys.argv = [sys.argv[0], *build_forward_argv(sys.argv[1:])]
    qwen_phase1.main()


if __name__ == "__main__":
    main()
