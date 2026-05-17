from __future__ import annotations

import os
import sys
import time
from typing import Any, Dict, List, Tuple

import run_claude_vision_openai_compatible_phase1 as claude_phase1


DEFAULT_MODEL = "claude-opus-4-7"
DEFAULT_BASE_URL = "https://matrixllm.alipay.com/v1"
DEFAULT_API_KEY_ENV = "MATRIXLLM_API_KEY"


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


def _normalize_base_url(api_url: str) -> str:
    url = (api_url or "").strip().rstrip("/")
    if url.endswith("/chat/completions"):
        return url[: -len("/chat/completions")]
    return url


def build_forward_argv(argv: List[str]) -> List[str]:
    forward = list(argv)
    if not _has_option(forward, "--model"):
        forward.extend(["--model", DEFAULT_MODEL])

    current_api_url = _value_after(forward, "--api-url")
    normalized_api_url = _normalize_base_url(current_api_url or DEFAULT_BASE_URL)
    if _has_option(forward, "--api-url"):
        api_index = forward.index("--api-url")
        if api_index + 1 >= len(forward):
            raise SystemExit("--api-url 后缺少地址值。")
        forward[api_index + 1] = normalized_api_url
    else:
        forward.extend(["--api-url", normalized_api_url])

    if not _has_option(forward, "--api-key-env"):
        forward.extend(["--api-key-env", DEFAULT_API_KEY_ENV])
    return forward


def _lazy_openai_client(api_key: str, base_url: str, timeout: int) -> Any:
    try:
        from openai import OpenAI
    except Exception as exc:  # pragma: no cover - import guard
        raise RuntimeError(
            "当前环境未安装 openai 包，无法使用 Claude OpenAI SDK 路线。"
            "请先安装：pip install openai"
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
    for key in [
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "completion_tokens_details",
        "prompt_tokens_details",
    ]:
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


class ClaudeOpenAISDKPhase1Runner(claude_phase1.OpenAICompatiblePhase1Runner):
    def __init__(
        self,
        *,
        model_name: str,
        api_url: str,
        api_key_env: str,
        timeout: int,
        max_retries: int,
        retry_sleep: float,
        rate_limit_retry_sleep: float,
        rate_limit_max_sleep: float,
        temperature: float,
        top_p: float,
        max_output_tokens: int,
    ) -> None:
        api_key = os.getenv(api_key_env, "").strip()
        if not api_key:
            raise RuntimeError(f"未找到 API key，请先设置环境变量 {api_key_env}")
        self.api_key = api_key
        self.model_name = model_name
        self.api_url = _normalize_base_url(api_url)
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_sleep = retry_sleep
        self.rate_limit_retry_sleep = rate_limit_retry_sleep
        self.rate_limit_max_sleep = rate_limit_max_sleep
        self.temperature = temperature
        self.top_p = top_p
        self.max_output_tokens = max_output_tokens
        self.client = _lazy_openai_client(self.api_key, self.api_url, self.timeout)

    def reset_session(self) -> None:
        self.client = _lazy_openai_client(self.api_key, self.api_url, self.timeout)

    def close(self) -> None:
        return None

    def generate_round(self, messages: List[Dict[str, Any]]) -> Tuple[str, Dict[str, Any], str, Dict[str, Any]]:
        parameter_profiles: List[Dict[str, Any]] = [
            {"temperature": self.temperature, "top_p": self.top_p, "max_tokens": self.max_output_tokens},
            {"temperature": self.temperature, "max_tokens": self.max_output_tokens},
            {"max_tokens": self.max_output_tokens},
            {"max_completion_tokens": self.max_output_tokens},
        ]
        deduped_profiles: List[Dict[str, Any]] = []
        seen_profiles: set[Tuple[Tuple[str, Any], ...]] = set()
        for profile in parameter_profiles:
            key = tuple(sorted(profile.items()))
            if key in seen_profiles:
                continue
            seen_profiles.add(key)
            deduped_profiles.append(profile)

        last_error = ""
        last_error_info: Dict[str, Any] = {"status_code": None, "error_type": "", "retriable": False, "retry_trace": []}

        for profile_index, profile in enumerate(deduped_profiles):
            payload: Dict[str, Any] = {
                "model": self.model_name,
                "messages": messages,
                **profile,
            }

            for attempt in range(self.max_retries):
                try:
                    response = self.client.chat.completions.create(**payload)
                except Exception as exc:  # pragma: no cover - runtime/network dependent
                    last_error = f"{type(exc).__name__}: {exc}"
                    lowered = last_error.lower()
                    status_code = getattr(exc, "status_code", None)
                    is_rate_limit = status_code == 429 or "429" in lowered or "rate limit" in lowered
                    is_connection_error = any(
                        token in lowered
                        for token in [
                            "failed to establish a new connection",
                            "connection error",
                            "apiconnectionerror",
                            "winerror 10013",
                            "timed out",
                        ]
                    )
                    unsupported_parameter = any(
                        token in lowered
                        for token in [
                            "unsupported parameter",
                            "unknown parameter",
                            "not supported",
                            "deprecated",
                            "is deprecated",
                            "extra inputs are not permitted",
                            "unexpected field",
                        ]
                    )
                    retriable_http = isinstance(status_code, int) and status_code >= 500

                    if is_connection_error:
                        sleep_seconds = min(max(self.retry_sleep * (2 ** attempt), 10.0), self.rate_limit_max_sleep)
                        error_type = "connection_exception"
                        self.reset_session()
                    elif is_rate_limit:
                        sleep_seconds = min(self.rate_limit_retry_sleep * (2 ** attempt), self.rate_limit_max_sleep)
                        error_type = "rate_limit"
                    else:
                        sleep_seconds = self.retry_sleep
                        error_type = "http_error" if status_code else "request_exception"

                    retry_trace = list(last_error_info.get("retry_trace", []))
                    retry_trace.append(
                        {
                            "attempt": attempt + 1,
                            "error_type": error_type,
                            "sleep_seconds": round(sleep_seconds, 2),
                            "message": last_error[-240:],
                            "parameter_profile": profile,
                        }
                    )
                    last_error_info = {
                        "status_code": status_code,
                        "error_type": error_type,
                        "retriable": (
                            attempt + 1 < self.max_retries
                            or (unsupported_parameter and profile_index + 1 < len(deduped_profiles))
                        ),
                        "retry_trace": retry_trace,
                    }

                    if unsupported_parameter and profile_index + 1 < len(deduped_profiles):
                        break
                    if (is_rate_limit or is_connection_error or retriable_http) and attempt + 1 < self.max_retries:
                        time.sleep(sleep_seconds)
                        continue
                    return "", {}, last_error, last_error_info

                response_payload = _response_to_dict(response)
                prediction = _extract_prediction_from_sdk_response(response)
                usage = _usage_to_dict(getattr(response, "usage", None))
                return prediction, usage, "", {"status_code": None, "error_type": "", "retriable": False, "retry_trace": []}

        return "", {}, last_error, last_error_info


def build_runner(args: Any) -> ClaudeOpenAISDKPhase1Runner:
    return ClaudeOpenAISDKPhase1Runner(
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


def main() -> None:
    claude_phase1.build_runner = build_runner
    sys.argv = [sys.argv[0], *build_forward_argv(sys.argv[1:])]
    claude_phase1.main()


if __name__ == "__main__":
    main()
