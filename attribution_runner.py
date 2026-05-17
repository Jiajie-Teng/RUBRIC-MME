from __future__ import annotations

from typing import Any

from judge_media import JudgeRequest
from judge_runner import (
    DEFAULT_INTERNAL_API_URL,
    JudgeBackendResult,
    MatrixLLMJudgeBackend,
    OfficialGeminiJudgeBackend,
)


def build_text_only_request(prompt_text: str) -> JudgeRequest:
    content_blueprint = [{"segment_type": "text", "text": prompt_text}]
    internal_messages = [{"role": "user", "content": [{"type": "text", "text": prompt_text}]}]
    return JudgeRequest(
        prompt_text=prompt_text,
        internal_messages=internal_messages,
        content_blueprint=content_blueprint,
        media_refs=[],
        input_mode="text_only",
    )



def build_internal_backend(
    *,
    model_name: str,
    api_url: str = DEFAULT_INTERNAL_API_URL,
    api_key_env: str = "MATRIXLLM_API_KEY",
    timeout: int = 180,
    max_retries: int = 5,
    retry_sleep: float = 3.0,
    rate_limit_retry_sleep: float = 12.0,
    rate_limit_max_sleep: float = 60.0,
    temperature: float = 0.0,
    top_p: float = 0.95,
    max_output_tokens: int = 2048,
) -> MatrixLLMJudgeBackend:
    return MatrixLLMJudgeBackend(
        api_url=api_url,
        api_key_env=api_key_env,
        timeout=timeout,
        top_p=top_p,
        model_name=model_name,
        max_retries=max_retries,
        retry_sleep=retry_sleep,
        rate_limit_retry_sleep=rate_limit_retry_sleep,
        rate_limit_max_sleep=rate_limit_max_sleep,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
    )



def build_official_backend(
    *,
    model_name: str,
    api_key_env: str = "GOOGLE_API_KEY",
    use_response_schema: bool = True,
    timeout_seconds: int = 300,
    poll_interval: float = 3.0,
    max_retries: int = 5,
    retry_sleep: float = 3.0,
    rate_limit_retry_sleep: float = 12.0,
    rate_limit_max_sleep: float = 60.0,
    temperature: float = 0.0,
    max_output_tokens: int = 2048,
) -> OfficialGeminiJudgeBackend:
    return OfficialGeminiJudgeBackend(
        api_key_env=api_key_env,
        use_response_schema=use_response_schema,
        timeout_seconds=timeout_seconds,
        poll_interval=poll_interval,
        model_name=model_name,
        max_retries=max_retries,
        retry_sleep=retry_sleep,
        rate_limit_retry_sleep=rate_limit_retry_sleep,
        rate_limit_max_sleep=rate_limit_max_sleep,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
    )


__all__ = [
    "DEFAULT_INTERNAL_API_URL",
    "JudgeBackendResult",
    "MatrixLLMJudgeBackend",
    "OfficialGeminiJudgeBackend",
    "build_text_only_request",
    "build_internal_backend",
    "build_official_backend",
]
