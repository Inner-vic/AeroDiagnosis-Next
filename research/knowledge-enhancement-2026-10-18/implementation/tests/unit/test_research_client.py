import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from aerodiagnosis.adapters.llm.research_client import (
    BudgetExceeded,
    ResearchCallError,
    ResearchClient,
    load_credential,
)

KEY = "sk-" + "a" * 32


def envelope(content: str = '{"ok":true}', prompt: int = 100, output: int = 20) -> dict:
    return {
        "model": "deepseek-flash",
        "usage": {"prompt_tokens": prompt, "completion_tokens": output},
        "choices": [{"message": {"content": content}, "finish_reason": "stop"}],
    }


def call(client: ResearchClient, **kw: object):
    return client.complete(
        task="test", instruction="test contract", payload={"source": "public"}, **kw
    )


def test_exact_input_cache_has_zero_new_fee_and_no_secret(tmp_path: Path) -> None:
    requests = []
    c = ResearchClient(
        api_key=KEY, runtime=tmp_path, transport=lambda body: requests.append(body) or envelope()
    )
    a, b = call(c), call(c)
    assert a.value == b.value == {"ok": True}
    assert len(requests) == 1 and b.receipt["cache_hit"] is True
    assert b.receipt["estimated_peak_cost_cny"] == 0
    assert requests[0]["thinking"] == {"type": "disabled"}
    assert KEY not in next((tmp_path / "cache").glob("*.json")).read_text()
    assert c.summary()["provider_call_count"] == 1


def test_budget_stops_before_transport(tmp_path: Path) -> None:
    sent = []
    c = ResearchClient(
        api_key=KEY,
        runtime=tmp_path,
        cap_cny=0.001,
        transport=lambda body: sent.append(body) or envelope(),
    )
    with pytest.raises(BudgetExceeded):
        call(c)
    assert not sent and c.summary()["provider_call_count"] == 0


def test_concurrent_reservations_do_not_overspend(tmp_path: Path) -> None:
    c = ResearchClient(
        api_key=KEY, runtime=tmp_path, cap_cny=0.019, transport=lambda body: envelope()
    )

    def reserve(i):
        try:
            return c._reserve(str(i), "pilot", "test", 10000)
        except BudgetExceeded:
            return None

    with ThreadPoolExecutor(max_workers=4) as pool:
        values = list(pool.map(reserve, range(8)))
    assert sum(v is not None for v in values) == 1
    assert c.summary()["budget_upper_estimate_cny"] == 0.01


def test_failed_usage_keeps_reservation_and_safe_error(tmp_path: Path) -> None:
    c = ResearchClient(api_key=KEY, runtime=tmp_path, transport=lambda body: {"usage": {}})
    with pytest.raises(ResearchCallError, match="usage"):
        call(c)
    summary = c.summary()
    assert summary["pending_or_unknown_calls"] == 1
    assert summary["budget_upper_estimate_cny"] > 0
    assert KEY not in json.dumps(summary)


def test_invalid_json_still_accounts_billed_usage(tmp_path: Path) -> None:
    c = ResearchClient(api_key=KEY, runtime=tmp_path, transport=lambda body: envelope("not JSON"))
    with pytest.raises(ResearchCallError):
        call(c)
    summary = c.summary()
    assert summary["pending_or_unknown_calls"] == 0
    assert summary["budget_upper_estimate_cny"] == 0.00036
    assert not list((tmp_path / "cache").glob("*.json"))


def test_changed_instruction_cannot_reuse_cache(tmp_path: Path) -> None:
    c = ResearchClient(api_key=KEY, runtime=tmp_path, transport=lambda body: envelope())
    call(c)
    c.complete(task="test", instruction="changed", payload={"source": "public"})
    assert c.summary()["provider_call_count"] == 2


def test_cache_fingerprint_is_verified(tmp_path: Path) -> None:
    c = ResearchClient(api_key=KEY, runtime=tmp_path, transport=lambda body: envelope())
    call(c)
    p = next((tmp_path / "cache").glob("*.json"))
    d = json.loads(p.read_text())
    d["request_sha256"] = "wrong"
    p.write_text(json.dumps(d))
    with pytest.raises(ResearchCallError, match="fingerprint"):
        call(c)


def test_credential_file_requires_one_key(tmp_path: Path) -> None:
    p = tmp_path / "env.txt"
    p.write_text("DEEPSEEK_API_KEY=" + KEY)
    assert load_credential(p) == KEY
    p.write_text(KEY + "\n" + "sk-" + "b" * 32)
    with pytest.raises(ResearchCallError):
        load_credential(p)


def test_secret_shaped_payload_is_not_dispatched(tmp_path: Path) -> None:
    c = ResearchClient(api_key=KEY, runtime=tmp_path, transport=lambda body: envelope())
    with pytest.raises(ResearchCallError, match="Credential-shaped"):
        c.complete(task="test", instruction="test", payload={"bad": KEY})
    assert c.summary()["provider_call_count"] == 0


def test_truncation_is_billed_but_not_cached(tmp_path: Path) -> None:
    def send(body):
        result = envelope()
        result["choices"][0]["finish_reason"] = "length"
        return result

    c = ResearchClient(api_key=KEY, runtime=tmp_path, transport=send)
    with pytest.raises(ResearchCallError, match="truncated"):
        call(c)
    assert c.summary()["budget_upper_estimate_cny"] > 0.0
    assert not list((tmp_path / "cache").glob("*.json"))


@pytest.mark.parametrize("suffix", ["</final>", "</br>"])
def test_known_closing_tag_normalization_preserves_raw_billed_response(tmp_path, suffix):
    def transport(body):
        return {
            "model": "deepseek-flash",
            "usage": {"prompt_tokens": 10, "completion_tokens": 10},
            "choices": [{"finish_reason": "stop", "message": {"content": '{"value":1}' + suffix}}],
        }

    client = ResearchClient(api_key="test", runtime=tmp_path, cap_cny=5, transport=transport)
    reply = client.complete(task="format", instruction="JSON", payload={})
    assert reply.value == {"value": 1}
    assert reply.receipt["format_normalization"]["removed_suffix"] == suffix
    saved = json.loads(next((tmp_path / "format-normalizations").glob("*.json")).read_text())
    assert saved["response_content"] == '{"value":1}' + suffix
    assert client.summary()["budget_upper_estimate_cny"] > 0


@pytest.mark.parametrize("suffix", [" extra fact", '{"value":2}', "</unknown>", "</final>text"])
def test_format_normalization_never_discards_content(tmp_path, suffix):
    def transport(body):
        return {
            "model": "deepseek-flash",
            "usage": {"prompt_tokens": 10, "completion_tokens": 10},
            "choices": [{"finish_reason": "stop", "message": {"content": '{"value":1}' + suffix}}],
        }

    client = ResearchClient(api_key="test", runtime=tmp_path, cap_cny=5, transport=transport)
    with pytest.raises(ResearchCallError):
        client.complete(task="format", instruction="JSON", payload={})
    assert not list((tmp_path / "cache").glob("*.json"))
