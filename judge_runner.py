from __future__ import annotations

import base64
import binascii
import json
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

try:
    import google.generativeai as genai
    from google.generativeai.types import HarmBlockThreshold, HarmCategory
except Exception as exc:  # pragma: no cover
    genai = None
    HarmBlockThreshold = None
    HarmCategory = None
    GOOGLE_IMPORT_ERROR = exc
else:
    GOOGLE_IMPORT_ERROR = None

try:
    from PIL import Image
except Exception as exc:  # pragma: no cover
    Image = None
    PIL_IMPORT_ERROR = exc
else:
    PIL_IMPORT_ERROR = None

from judge_media import JudgeRequest

DEFAULT_INTERNAL_API_URL = "https://matrixllm.alipay.com/v1/chat/completions"


@dataclass
class JudgeBackendResult:
    raw_text: str
    raw_response: Any
    usage: Dict[str, Any]
    error: str
    error_type: str
    status_code: Optional[int]
    attempt_count: int
    backend_name: str
    model_name: str
    schema_mode: str


class BaseJudgeBackend:
    backend_name = "base"

    def __init__(
        self,
        *,
        model_name: str,
        max_retries: int,
        retry_sleep: float,
        rate_limit_retry_sleep: float,
        rate_limit_max_sleep: float,
        temperature: float,
        max_output_tokens: int,
    ) -> None:
        self.model_name = model_name
        self.max_retries = max_retries
        self.retry_sleep = retry_sleep
        self.rate_limit_retry_sleep = rate_limit_retry_sleep
        self.rate_limit_max_sleep = rate_limit_max_sleep
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens

    def judge(self, request: JudgeRequest, response_schema: Dict[str, Any]) -> JudgeBackendResult:
        raise NotImplementedError

    def _sleep_for_retry(
        self,
        attempt: int,
        status_code: Optional[int],
        retry_after_seconds: Optional[float] = None,
    ) -> float:
        if status_code == 429:
            base = self.rate_limit_retry_sleep * (2 ** max(0, attempt - 1))
            base = min(base, self.rate_limit_max_sleep)
        else:
            base = self.retry_sleep * max(1, attempt)
        if retry_after_seconds is not None:
            base = max(base, float(retry_after_seconds))
        jitter = random.uniform(0.0, 1.0)
        sleep_seconds = round(base + jitter, 3)
        time.sleep(sleep_seconds)
        return sleep_seconds


class MatrixLLMJudgeBackend(BaseJudgeBackend):
    backend_name = "matrixllm_internal"

    def __init__(
        self,
        *,
        api_url: str,
        api_key_env: str,
        timeout: int,
        top_p: float,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.api_url = api_url
        self.api_key_env = api_key_env
        self.timeout = timeout
        self.top_p = top_p

        token = os.getenv(api_key_env, "").strip()
        if not token:
            raise RuntimeError(f"环境变量 {api_key_env} 未设置，无法调用内部 judge 接口。")
        self.token = token

    def _extract_text(self, response_json: Dict[str, Any]) -> str:
        choices = response_json.get("choices") or []
        if not choices:
            return ""
        message = (choices[0] or {}).get("message") or {}
        content = message.get("content", "")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            fragments = []
            for item in content:
                if isinstance(item, dict):
                    if isinstance(item.get("text"), str):
                        fragments.append(item["text"])
                    elif isinstance(item.get("content"), str):
                        fragments.append(item["content"])
            return "\n".join(part for part in fragments if part)
        return str(content)

    def judge(self, request: JudgeRequest, response_schema: Dict[str, Any]) -> JudgeBackendResult:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.token}",
        }
        last_error = ""
        last_status: Optional[int] = None
        last_error_type = ""
        last_response: Any = None
        retry_after_seconds: Optional[float] = None
        messages = request.internal_messages or [{"role": "user", "content": [{"type": "text", "text": request.prompt_text}]}]
        for attempt in range(1, self.max_retries + 1):
            payload = {
                "stream": False,
                "model": self.model_name,
                "messages": messages,
                "generation_config": {
                    "temperature": self.temperature,
                    "top_p": self.top_p,
                    "max_output_tokens": self.max_output_tokens,
                    "response_mime_type": "application/json",
                    "response_schema": response_schema,
                },
            }
            try:
                response = requests.post(
                    self.api_url,
                    headers=headers,
                    json=payload,
                    timeout=self.timeout,
                )
                last_response = response.text
                response.raise_for_status()
                response_json = response.json()
                return JudgeBackendResult(
                    raw_text=self._extract_text(response_json).strip(),
                    raw_response=response_json,
                    usage=response_json.get("usage", {}) if isinstance(response_json, dict) else {},
                    error="",
                    error_type="",
                    status_code=response.status_code,
                    attempt_count=attempt,
                    backend_name=self.backend_name,
                    model_name=self.model_name,
                    schema_mode="response_schema",
                )
            except requests.HTTPError as exc:  # pragma: no cover
                status_code = exc.response.status_code if exc.response is not None else None
                last_status = status_code
                retry_after_seconds = None
                if status_code == 429:
                    last_error_type = "rate_limit"
                    if exc.response is not None:
                        retry_after_header = str(exc.response.headers.get("Retry-After", "") or "").strip()
                        if retry_after_header:
                            try:
                                retry_after_seconds = float(retry_after_header)
                            except ValueError:
                                retry_after_seconds = None
                elif status_code and status_code >= 500:
                    last_error_type = "server_error"
                else:
                    last_error_type = "http_error"
                last_error = f"{type(exc).__name__}: {exc}; body={last_response}"
            except requests.RequestException as exc:  # pragma: no cover
                last_error = f"{type(exc).__name__}: {exc}"
                last_error_type = "request_exception"
                last_status = None
                retry_after_seconds = None
            if attempt < self.max_retries:
                self._sleep_for_retry(attempt, last_status, retry_after_seconds=retry_after_seconds)
        return JudgeBackendResult(
            raw_text="",
            raw_response=last_response,
            usage={},
            error=last_error,
            error_type=last_error_type,
            status_code=last_status,
            attempt_count=self.max_retries,
            backend_name=self.backend_name,
            model_name=self.model_name,
            schema_mode="response_schema",
        )


class OfficialGeminiJudgeBackend(BaseJudgeBackend):
    backend_name = "gemini_official"

    def __init__(
        self,
        *,
        api_key_env: str,
        use_response_schema: bool,
        timeout_seconds: int = 300,
        poll_interval: float = 3.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if genai is None:
            raise RuntimeError(f"google.generativeai 不可用: {GOOGLE_IMPORT_ERROR}")
        api_key = os.getenv(api_key_env, "").strip()
        if not api_key:
            raise RuntimeError(f"环境变量 {api_key_env} 未设置，无法调用官方 Gemini judge 接口。")
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(self.model_name)
        self.use_response_schema = use_response_schema
        self.timeout_seconds = timeout_seconds
        self.poll_interval = poll_interval
        self.upload_cache: Dict[str, Any] = {}
        self.uploaded_files: List[Any] = []

    def close(self) -> None:
        for uploaded in self.uploaded_files:
            try:
                uploaded.delete()
            except Exception:
                pass
        self.uploaded_files.clear()
        self.upload_cache.clear()

    def _wait_until_ready(self, uploaded: Any) -> Any:
        deadline = time.time() + self.timeout_seconds
        current = uploaded
        while time.time() < deadline:
            state = getattr(getattr(current, "state", None), "name", "")
            state_normalized = str(state).lower()
            if state_normalized in {"", "active", "succeeded", "ready"}:
                return current
            if state_normalized in {"failed", "error"}:
                raise RuntimeError(f"Gemini file processing failed for {getattr(current, 'name', '<unknown>')}")
            time.sleep(self.poll_interval)
            current = genai.get_file(current.name)
        raise TimeoutError(f"Timed out waiting for Gemini file to become ready: {getattr(uploaded, 'name', '<unknown>')}")

    def _upload_file(self, file_path: str) -> Any:
        resolved = str(Path(file_path).resolve())
        cached = self.upload_cache.get(resolved)
        if cached is not None:
            return cached
        uploaded = genai.upload_file(path=resolved)
        uploaded = self._wait_until_ready(uploaded)
        self.upload_cache[resolved] = uploaded
        self.uploaded_files.append(uploaded)
        return uploaded

    def _decode_data_url(self, data_url: str) -> tuple[bytes, str]:
        header, encoded = data_url.split(",", 1)
        mime_type = header.split(";", 1)[0].split(":", 1)[1]
        return base64.b64decode(encoded), mime_type

    def _convert_segment(self, segment: Dict[str, Any]) -> List[Any]:
        segment_type = segment.get("segment_type", "")
        if segment_type == "text":
            text = str(segment.get("text", "") or "").strip()
            return [text] if text else []

        kind = segment.get("kind", "")
        local_path = str(segment.get("local_path", "") or "")
        remote_url = str(segment.get("remote_url", "") or "")
        mime_type = str(segment.get("mime_type", "") or "")
        data_url = str(segment.get("data_url", "") or "")

        if kind == "image":
            if local_path and Path(local_path).exists():
                if Image is not None:
                    with Image.open(local_path) as image:
                        return [image.convert("RGB")]
                return [{"mime_type": mime_type or "image/jpeg", "data": Path(local_path).read_bytes()}]
            if data_url.startswith("data:"):
                raw_bytes, resolved_mime = self._decode_data_url(data_url)
                return [{"mime_type": resolved_mime, "data": raw_bytes}]
            if remote_url:
                return [remote_url]
            return []

        if kind == "video":
            if local_path and Path(local_path).exists():
                return [self._upload_file(local_path)]
            if remote_url:
                return [remote_url]
            return []

        return []

    def _build_contents(self, request: JudgeRequest) -> List[Any]:
        contents: List[Any] = []
        for segment in request.content_blueprint:
            contents.extend(self._convert_segment(segment))
        if not contents:
            contents.append(request.prompt_text)
        return contents

    def _extract_usage(self, response: Any) -> Dict[str, Any]:
        usage = getattr(response, "usage_metadata", None)
        if usage is None:
            return {}
        return {
            "prompt_token_count": int(getattr(usage, "prompt_token_count", 0) or 0),
            "candidates_token_count": int(getattr(usage, "candidates_token_count", 0) or 0),
            "total_token_count": int(getattr(usage, "total_token_count", 0) or 0),
        }

    def _generate_with_mode(self, request: JudgeRequest, response_schema: Dict[str, Any], *, schema_mode: str) -> JudgeBackendResult:
        config_kwargs: Dict[str, Any] = {
            "temperature": self.temperature,
            "max_output_tokens": self.max_output_tokens,
        }
        contents = self._build_contents(request)
        if schema_mode == "response_schema":
            config_kwargs["response_mime_type"] = "application/json"
            config_kwargs["response_schema"] = response_schema
        else:
            contents = list(contents)
            contents.append(
                "\n\n请只返回 JSON，不要输出 markdown。"
                + "\nJSON Schema 如下：\n"
                + json.dumps(response_schema, ensure_ascii=False, indent=2)
            )
        config = genai.GenerationConfig(**config_kwargs)
        response = self.model.generate_content(
            list(contents),
            generation_config=config,
            safety_settings={
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            },
        )
        return JudgeBackendResult(
            raw_text=(getattr(response, "text", "") or "").strip(),
            raw_response=response,
            usage=self._extract_usage(response),
            error="",
            error_type="",
            status_code=None,
            attempt_count=1,
            backend_name=self.backend_name,
            model_name=self.model_name,
            schema_mode=schema_mode,
        )

    def judge(self, request: JudgeRequest, response_schema: Dict[str, Any]) -> JudgeBackendResult:
        last_error = ""
        last_result: Optional[JudgeBackendResult] = None
        for attempt in range(1, self.max_retries + 1):
            modes = ["response_schema", "prompt_only_json"] if self.use_response_schema else ["prompt_only_json"]
            for schema_mode in modes:
                try:
                    result = self._generate_with_mode(request, response_schema, schema_mode=schema_mode)
                    result.attempt_count = attempt
                    return result
                except Exception as exc:  # pragma: no cover
                    message = str(exc)
                    error_type = "official_generation_error"
                    if "Incorrect padding" in message or isinstance(exc, binascii.Error):
                        error_type = "data_url_decode_error"
                    last_error = f"{type(exc).__name__}: {exc}"
                    last_result = JudgeBackendResult(
                        raw_text="",
                        raw_response="",
                        usage={},
                        error=last_error,
                        error_type=error_type,
                        status_code=None,
                        attempt_count=attempt,
                        backend_name=self.backend_name,
                        model_name=self.model_name,
                        schema_mode=schema_mode,
                    )
                    if schema_mode == "response_schema" and ("response_schema" in message or "response_mime_type" in message):
                        continue
                    break
            if attempt < self.max_retries:
                self._sleep_for_retry(attempt, None)
        return last_result or JudgeBackendResult(
            raw_text="",
            raw_response="",
            usage={},
            error=last_error,
            error_type="official_generation_error",
            status_code=None,
            attempt_count=self.max_retries,
            backend_name=self.backend_name,
            model_name=self.model_name,
            schema_mode="prompt_only_json",
        )
