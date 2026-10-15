"""Ollama inference adapter (Phase 5 §2).

A thin, synchronous HTTPX client for a locally running Ollama. It uses
``/api/chat`` with ``stream=false`` and JSON-schema-constrained output, and it
validates every response independently — a model that ignores the schema is
reported as a generation failure, never trusted because it parsed.

Design points:

* Nothing here contacts Ollama at import time; the client is built lazily and the
  model is configurable, never hardcoded.
* Only loopback endpoints are allowed by default; a remote base URL requires a
  deliberate opt-in (``RAG_WORKBENCH_OLLAMA_ALLOW_REMOTE=true``).
* Connection failures are surfaced as :class:`GeneratorUnavailableError`; a
  timeout or malformed reply is returned as a ``GenerationResult`` with
  ``error`` set (which the pipeline maps to the ``generation_error`` state)
  rather than crashing.
"""

from __future__ import annotations

import json
import time
from urllib.parse import urlsplit

import httpx
from pydantic import ValidationError

from rag_workbench.answering.prompt import build_instructions, build_user_message
from rag_workbench.config import Settings
from rag_workbench.domain.errors import GeneratorUnavailableError
from rag_workbench.domain.schemas import (
    AnswerPayload,
    GenerationRequest,
    GenerationResult,
)

_LOOPBACK_HOSTS = {"localhost", "127.0.0.1", "::1", "0.0.0.0"}
_NS_PER_MS = 1_000_000.0


def _is_loopback(base_url: str) -> bool:
    try:
        host = urlsplit(base_url).hostname
    except ValueError:
        return False
    return host in _LOOPBACK_HOSTS


def _extract_json_object(text: str) -> str | None:
    """Return the outermost ``{...}`` slice of ``text``, or ``None``.

    Bounded recovery: models sometimes wrap JSON in prose or a code fence. We
    take the span from the first ``{`` to the last ``}`` and try once more; we do
    not attempt clever repairs, so a genuinely malformed reply stays a failure.
    """

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    return text[start : end + 1]


class OllamaGenerator:
    """Ollama-backed :class:`~rag_workbench.domain.interfaces.AnswerGenerator`."""

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        temperature: float = 0.1,
        num_predict: int = 512,
        timeout_s: float = 120.0,
        allow_remote: bool = False,
        client: httpx.Client | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._temperature = temperature
        self._num_predict = num_predict
        self._timeout_s = timeout_s
        self._allow_remote = allow_remote
        self._client = client

    # ---------------------------------------------------------------- identity

    @property
    def model_id(self) -> str:
        return self._model

    @property
    def base_url(self) -> str:
        return self._base_url

    def _ensure_allowed(self) -> None:
        if not self._allow_remote and not _is_loopback(self._base_url):
            raise GeneratorUnavailableError(
                f"Ollama base URL '{self._base_url}' is not loopback; set "
                "RAG_WORKBENCH_OLLAMA_ALLOW_REMOTE=true to permit a remote endpoint "
                "(document text would then leave this machine)"
            )

    def _request(self, method: str, path: str, **kwargs) -> httpx.Response:
        url = f"{self._base_url}{path}"
        try:
            if self._client is not None:
                return self._client.request(method, url, **kwargs)
            with httpx.Client(timeout=self._timeout_s) as client:
                return client.request(method, url, **kwargs)
        except httpx.ConnectError as exc:
            raise GeneratorUnavailableError(
                f"could not connect to Ollama at {self._base_url}; start it with "
                "`ollama serve` and ensure the model is pulled"
            ) from exc

    # ------------------------------------------------------------- availability

    def list_models(self) -> list[str]:
        """Return the models installed in the connected Ollama instance."""

        self._ensure_allowed()
        response = self._request("GET", "/api/tags", timeout=min(self._timeout_s, 10.0))
        response.raise_for_status()
        payload = response.json()
        return [
            str(item.get("name"))
            for item in payload.get("models", [])
            if item.get("name")
        ]

    def _model_installed(self, models: list[str]) -> bool:
        if self._model in models:
            return True
        base = self._model.split(":", 1)[0]
        return any(name.split(":", 1)[0] == base for name in models)

    def ensure_available(self) -> None:
        """Raise :class:`GeneratorUnavailableError` unless the model is usable."""

        self._ensure_allowed()
        try:
            models = self.list_models()
        except GeneratorUnavailableError:
            raise
        except httpx.HTTPError as exc:
            raise GeneratorUnavailableError(
                f"Ollama at {self._base_url} did not answer a model list request: {exc}"
            ) from exc
        if not models:
            raise GeneratorUnavailableError(
                f"Ollama at {self._base_url} has no models installed; pull one with "
                f"`ollama pull {self._model}`"
            )
        if not self._model_installed(models):
            available = ", ".join(sorted(models))
            raise GeneratorUnavailableError(
                f"model '{self._model}' is not installed in Ollama; available: {available}. "
                "Set RAG_WORKBENCH_OLLAMA_MODEL to one of them."
            )

    # ---------------------------------------------------------------- generation

    def generate(self, request: GenerationRequest) -> GenerationResult:
        """Run one non-streaming chat completion and validate its JSON output."""

        self._ensure_allowed()
        body = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": request.instructions or build_instructions()},
                {
                    "role": "user",
                    "content": build_user_message(request.question, request.evidence),
                },
            ],
            "stream": False,
            "format": request.response_schema or "json",
            "options": {
                "temperature": request.config.temperature,
                "num_predict": request.config.num_predict,
            },
        }
        if request.config.seed is not None:
            body["options"]["seed"] = request.config.seed

        started = time.perf_counter()
        try:
            response = self._request("POST", "/api/chat", json=body)
        except httpx.TimeoutException:
            return self._error_result(
                f"generation timed out after {request.config.timeout_s:.0f}s", started
            )
        except httpx.HTTPError as exc:
            return self._error_result(f"generation request failed: {exc}", started)

        if response.status_code >= 400:
            return self._error_result(
                f"Ollama returned HTTP {response.status_code}: {response.text[:300]}", started
            )

        try:
            data = response.json()
        except ValueError as exc:
            return self._error_result(f"Ollama returned non-JSON: {exc}", started)

        content = str((data.get("message") or {}).get("content", ""))
        timings = self._timings(data, started)
        result = self._parse_content(content)
        return result.model_copy(
            update={
                "timings_ms": timings,
                "done_reason": data.get("done_reason"),
                "prompt_tokens": data.get("prompt_eval_count"),
                "completion_tokens": data.get("eval_count"),
            }
        )

    def _parse_content(self, content: str) -> GenerationResult:
        diagnostics: list[str] = []
        text = content.strip()
        parsed: object | None = None
        for candidate in (text, _extract_json_object(text)):
            if not candidate:
                continue
            try:
                parsed = json.loads(candidate)
                break
            except json.JSONDecodeError:
                continue
        if parsed is None:
            diagnostics.append("the model did not return parseable JSON")
            return GenerationResult(
                model_id=self._model,
                raw_text=content,
                structured=False,
                error="malformed JSON in model output",
                diagnostics=diagnostics,
            )
        try:
            payload = AnswerPayload.model_validate(parsed)
        except ValidationError as exc:
            diagnostics.append("model output did not match the answer schema")
            return GenerationResult(
                model_id=self._model,
                raw_text=content,
                structured=False,
                error=f"schema validation failed: {exc.error_count()} error(s)",
                diagnostics=diagnostics,
            )
        return GenerationResult(
            model_id=self._model, payload=payload, raw_text=content, structured=True
        )

    def _timings(self, data: dict, started: float) -> dict[str, float]:
        timings: dict[str, float] = {"generation": round((time.perf_counter() - started) * 1000, 3)}
        for name in ("total_duration", "load_duration", "prompt_eval_duration", "eval_duration"):
            value = data.get(name)
            if isinstance(value, (int, float)):
                timings[name.replace("_duration", "")] = round(float(value) / _NS_PER_MS, 3)
        return timings

    def _error_result(self, message: str, started: float) -> GenerationResult:
        return GenerationResult(
            model_id=self._model,
            structured=False,
            error=message,
            diagnostics=[message],
            timings_ms={"generation": round((time.perf_counter() - started) * 1000, 3)},
        )


def build_answer_generator(settings: Settings) -> OllamaGenerator:
    """Build the configured Ollama generator from :class:`Settings`."""

    return OllamaGenerator(
        base_url=settings.ollama_base_url,
        model=settings.ollama_model,
        temperature=settings.ollama_temperature,
        num_predict=settings.ollama_num_predict,
        timeout_s=settings.ollama_timeout_s,
        allow_remote=settings.ollama_allow_remote,
    )


def probe_generation(settings: Settings, *, timeout_s: float = 2.0) -> dict[str, object]:
    """Check whether live generation is usable, without raising.

    Used by ``/v1/config`` and the UI so a status banner can say whether Ollama is
    reachable *now*. It is deliberately short-timeout and only runs when
    generation is enabled, so the default install (and the public read-only demo)
    never waits on a model.
    """

    if not settings.generation_enabled:
        return {"available": False, "detail": "live generation is disabled"}
    try:
        with httpx.Client(timeout=timeout_s) as client:
            generator = OllamaGenerator(
                base_url=settings.ollama_base_url,
                model=settings.ollama_model,
                timeout_s=timeout_s,
                allow_remote=settings.ollama_allow_remote,
                client=client,
            )
            generator.ensure_available()
    except GeneratorUnavailableError as exc:
        return {"available": False, "detail": exc.message}
    except Exception as exc:  # noqa: BLE001 - a probe must never break /v1/config
        return {"available": False, "detail": f"probe failed: {exc}"}
    return {"available": True, "detail": "Ollama is reachable and the model is installed"}


__all__ = ["OllamaGenerator", "build_answer_generator", "probe_generation"]
