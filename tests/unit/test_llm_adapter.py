from __future__ import annotations

import json
import urllib.request

import pytest

from aerodiagnosis.adapters.llm import OpenAICompatibleLanguageModel
from aerodiagnosis.ports import LanguageModelError


class FakeResponse:
    def __init__(self, payload: object) -> None:
        self._payload = json.dumps(payload).encode()

    def __enter__(self) -> FakeResponse:
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def read(self) -> bytes:
        return self._payload


def test_openai_compatible_adapter_sends_frontend_credential_without_exposing_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, object] = {}

    def fake_urlopen(request: urllib.request.Request, timeout: float) -> FakeResponse:
        observed["url"] = request.full_url
        observed["authorization"] = request.get_header("Authorization")
        observed["timeout"] = timeout
        return FakeResponse(
            {"choices": [{"message": {"content": '{"tools":["search_manual_chunks"]}'}}]}
        )

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    model = OpenAICompatibleLanguageModel(
        base_url="https://provider.example/v1/",
        api_key="user-secret",
        model="user-model",
        timeout_seconds=12,
    )

    result = model.complete_json("plan_evidence", {"question": "EGT rise"})

    assert result == {"tools": ["search_manual_chunks"]}
    assert observed == {
        "url": "https://provider.example/v1/chat/completions",
        "authorization": "Bearer user-secret",
        "timeout": 12,
    }
    assert model.identity.startswith("openai_compatible:user-model:")
    assert model.identity.endswith(":prompts@1")
    assert "provider.example" not in model.identity
    assert "user-secret" not in repr(model.__dict__)


def test_openai_compatible_adapter_rejects_invalid_configuration_and_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        OpenAICompatibleLanguageModel(base_url="", api_key="key", model="model")

    model = OpenAICompatibleLanguageModel(
        base_url="https://provider.example/v1",
        api_key="secret",
        model="model",
    )
    with pytest.raises(ValueError, match="unknown language-model task"):
        model.complete_json("unknown", {})

    monkeypatch.setattr(
        urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: FakeResponse({"choices": []}),
    )
    with pytest.raises(LanguageModelError, match="missing message content"):
        model.complete_json("plan_evidence", {})
