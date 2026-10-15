"""Unit tests for the Ollama adapter using an in-memory HTTP transport.

These never touch the network: ``httpx.MockTransport`` stands in for Ollama so the
request/response handling, schema validation, timeouts and failure paths are all
exercised deterministically.
"""

from __future__ import annotations

import json

import httpx
import pytest

from rag_workbench.answering.ollama import OllamaGenerator
from rag_workbench.domain.errors import GeneratorUnavailableError
from rag_workbench.domain.schemas import (
    EvidenceItem,
    GenerationConfig,
    GenerationRequest,
)


def _client(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def _generator(handler, *, model: str = "qwen2.5:1.5b-instruct", **kwargs):
    return OllamaGenerator(
        base_url="http://127.0.0.1:11434",
        model=model,
        timeout_s=5.0,
        client=_client(handler),
        **kwargs,
    )


def _request() -> GenerationRequest:
    evidence = [
        EvidenceItem(
            source_id="S1",
            chunk_id="c1",
            document_id="d1",
            source_name="nda.txt",
            rank=1,
            score=0.0,
            text="The term is three years.",
        )
    ]
    return GenerationRequest(
        instructions="system",
        question="How long?",
        evidence=evidence,
        config=GenerationConfig(model="qwen2.5:1.5b-instruct", timeout_s=5.0),
        response_schema={"type": "object"},
    )


def _tags(*names: str) -> httpx.Response:
    return httpx.Response(200, json={"models": [{"name": name} for name in names]})


def test_list_models_reads_names() -> None:
    generator = _generator(lambda request: _tags("qwen2.5:1.5b-instruct", "llama3:8b"))
    assert generator.list_models() == ["qwen2.5:1.5b-instruct", "llama3:8b"]


def test_ensure_available_accepts_base_name_match() -> None:
    generator = _generator(
        lambda request: _tags("qwen2.5:1.5b-instruct"), model="qwen2.5"
    )
    generator.ensure_available()  # does not raise


def test_ensure_available_rejects_missing_model() -> None:
    generator = _generator(lambda request: _tags("llama3:8b"))
    with pytest.raises(GeneratorUnavailableError) as excinfo:
        generator.ensure_available()
    assert "not installed" in excinfo.value.message
    assert "llama3:8b" in excinfo.value.message


def test_ensure_available_rejects_no_models() -> None:
    generator = _generator(lambda request: _tags())
    with pytest.raises(GeneratorUnavailableError):
        generator.ensure_available()


def test_connection_error_is_unavailable() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    generator = _generator(handler)
    with pytest.raises(GeneratorUnavailableError):
        generator.ensure_available()


def test_remote_endpoint_requires_opt_in() -> None:
    generator = OllamaGenerator(
        base_url="http://192.0.2.10:11434",
        model="qwen2.5:1.5b-instruct",
        client=_client(lambda request: _tags("qwen2.5:1.5b-instruct")),
    )
    with pytest.raises(GeneratorUnavailableError):
        generator.ensure_available()


def test_generate_parses_structured_output_and_usage() -> None:
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        content = json.dumps(
            {
                "status": "answered",
                "answer": "Three years [S1].",
                "claims": [{"text": "Three years.", "citations": ["S1"]}],
                "unanswered": [],
            }
        )
        return httpx.Response(
            200,
            json={
                "message": {"content": content},
                "done_reason": "stop",
                "prompt_eval_count": 42,
                "eval_count": 7,
                "total_duration": 1_000_000_000,
            },
        )

    result = _generator(handler).generate(_request())
    assert result.structured is True
    assert result.payload is not None
    assert result.payload.status == "answered"
    assert result.payload.claims[0].citations == ["S1"]
    assert result.prompt_tokens == 42
    assert result.completion_tokens == 7
    assert result.done_reason == "stop"
    assert result.timings_ms["generation"] >= 0
    # The request asks for schema-constrained output and no streaming.
    assert captured["body"]["stream"] is False
    assert captured["body"]["format"] == {"type": "object"}


def test_generate_recovers_json_from_a_code_fence() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        content = "```json\n" + json.dumps(
            {
                "status": "insufficient_evidence",
                "answer": "",
                "claims": [],
                "unanswered": [],
                "reason": "no evidence",
            }
        ) + "\n```"
        return httpx.Response(200, json={"message": {"content": content}})

    result = _generator(handler).generate(_request())
    assert result.structured is True
    assert result.payload is not None
    assert result.payload.status == "insufficient_evidence"


def test_generate_reports_malformed_output() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"message": {"content": "no json"}})

    result = _generator(handler).generate(_request())
    assert result.structured is False
    assert result.payload is None
    assert result.error


def test_generate_reports_schema_violation() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        content = json.dumps({"status": "not-a-real-status", "answer": "x"})
        return httpx.Response(200, json={"message": {"content": content}})

    result = _generator(handler).generate(_request())
    assert result.structured is False
    assert result.error and "schema" in result.error


def test_generate_timeout_is_an_error_result_not_a_crash() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.TimeoutException("timed out")

    result = _generator(handler).generate(_request())
    assert result.structured is False
    assert result.error and "timed out" in result.error


def test_generate_http_error_is_an_error_result() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="boom")

    result = _generator(handler).generate(_request())
    assert result.structured is False
    assert result.error and "HTTP 500" in result.error
