import json
from decimal import Decimal
from pathlib import Path

import httpx
import pytest

from change_order_extract.pipeline import extract_text
from change_order_extract.providers import OpenAICompatibleExtractor, build_extractor

SAMPLE = Path(__file__).resolve().parents[1] / "samples" / "co_01_aia_clean.txt"


def test_auto_uses_rules_without_a_key() -> None:
    assert build_extractor("auto").name == "rules"
    result = extract_text(SAMPLE.read_text(encoding="utf-8"))
    assert result.method == "rules"


def test_openai_provider_requires_a_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(RuntimeError):
        build_extractor("openai")


def test_openai_compatible_call_is_grounded(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    payload = {
        "project_name": "Riverside Medical Pavilion",
        "change_order_number": "07",
        "total": 99999.99,
        "currency": "USD",
        "field_confidence": {"total": 0.99, "project_name": 0.95, "change_order_number": 0.9},
        "line_items": [],
    }
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode())
        seen["temperature"] = body["temperature"]
        seen["model"] = body["model"]
        assert request.url.path.endswith("/chat/completions")
        assert request.headers["Authorization"] == "Bearer test-key"
        return httpx.Response(
            200, json={"choices": [{"message": {"content": json.dumps(payload)}}]}
        )

    extractor = OpenAICompatibleExtractor(
        "test-key",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    result = extract_text(SAMPLE.read_text(encoding="utf-8"), extractor=extractor)
    assert seen["temperature"] == 0
    assert result.method == "openai"
    assert result.project_name.value == "Riverside Medical Pavilion"
    assert result.project_name.confidence > 0.7
    assert result.total.value == Decimal("99999.99")
    assert result.total.needs_review
    assert result.total.confidence < 0.4
    assert any(flag == "grounding: total" for flag in result.review_flags)
