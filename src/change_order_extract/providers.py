"""Extraction providers.

``auto`` uses the OpenAI-compatible API when ``OPENAI_API_KEY`` is set and
the rule extractor otherwise. A failed API call falls back to the rules
inside the pipeline, so a reviewer can always run the project locally.
"""

from __future__ import annotations

import json
import os
import re
from datetime import date
from decimal import Decimal
from typing import Protocol

import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from change_order_extract.rules import RuleExtractor
from change_order_extract.schema import ChangeOrderExtraction, LineItem, Scored
from change_order_extract.textutil import parse_date, quantize_money

_LLM_CAP = 0.8
_APPROVAL = {"approved", "pending", "draft", "rejected", "unknown"}
_MAX_CHARS = 24_000

_SYSTEM_PROMPT = """You extract construction change orders into JSON.
Use null for any field that is not explicitly stated. Do not calculate a
missing tax or total, and do not fold handwritten margin notes into the
total or the line items. Copy numbers as printed, including negatives for
credits. Dates must be ISO YYYY-MM-DD or null. approval_status must be one
of approved, pending, draft, rejected, unknown, or null.
Money fields are numbers with no currency symbol.
line_subtotal is the sum of line items before markup, tax, and bond.
Return a single JSON object with these keys:
project_name, project_number, change_order_number, date, contractor, owner,
description, currency, line_items, line_subtotal, markup, markup_percent,
tax, tax_percent, other_fees, total, schedule_impact_days, reason,
approval_status, field_confidence.
line_items is a list of objects with description, quantity, unit, unit_price,
amount. field_confidence maps field names to a number from 0 to 1.
Units should be short codes when obvious (LF, SF, EA, LS, HR, SY, LB).
"""


class Extractor(Protocol):
    name: str

    def extract(self, text: str) -> ChangeOrderExtraction: ...


class _LlmLine(BaseModel):
    model_config = ConfigDict(extra="ignore")

    description: str | None = None
    quantity: Decimal | None = None
    unit: str | None = None
    unit_price: Decimal | None = None
    amount: Decimal | None = None


class _LlmPayload(BaseModel):
    model_config = ConfigDict(extra="ignore")

    project_name: str | None = None
    project_number: str | None = None
    change_order_number: str | None = None
    date: str | None = None
    contractor: str | None = None
    owner: str | None = None
    description: str | None = None
    currency: str | None = None
    line_items: list[_LlmLine] = Field(default_factory=list)
    line_subtotal: Decimal | None = None
    markup: Decimal | None = None
    markup_percent: Decimal | None = None
    tax: Decimal | None = None
    tax_percent: Decimal | None = None
    other_fees: Decimal | None = None
    total: Decimal | None = None
    schedule_impact_days: int | None = None
    reason: str | None = None
    approval_status: str | None = None
    field_confidence: dict[str, float] = Field(default_factory=dict)


class OpenAICompatibleExtractor:
    """Chat-completions client for any OpenAI-compatible base URL."""

    name = "openai"

    def __init__(
        self,
        api_key: str,
        *,
        base_url: str | None = None,
        model: str | None = None,
        client: httpx.Client | None = None,
    ) -> None:
        self.api_key = api_key
        self.base_url = (
            base_url or os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1"
        ).rstrip("/")
        self.model = model or os.environ.get("CHANGE_ORDER_MODEL") or "gpt-4o-mini"
        self._client = client

    def extract(self, text: str) -> ChangeOrderExtraction:
        truncated = False
        payload_text = text
        if len(payload_text) > _MAX_CHARS:
            payload_text = payload_text[:_MAX_CHARS]
            truncated = True
        body = {
            "model": self.model,
            "temperature": 0,
            "max_tokens": 2500,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": payload_text},
            ],
        }
        owns_client = self._client is None
        client = self._client or httpx.Client(timeout=60)
        try:
            response = client.post(
                f"{self.base_url}/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json=body,
            )
            response.raise_for_status()
            content = response.json()["choices"][0]["message"]["content"]
        finally:
            if owns_client:
                client.close()
        draft = _from_llm_json(content)
        if truncated:
            draft.warnings.append("llm_input_truncated")
        return draft


def build_extractor(name: str | None = None) -> RuleExtractor | OpenAICompatibleExtractor:
    choice = (name or os.environ.get("CHANGE_ORDER_PROVIDER") or "auto").lower()
    if choice == "rules":
        return RuleExtractor()
    if choice == "openai":
        key = os.environ.get("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("OPENAI_API_KEY is not set")
        return OpenAICompatibleExtractor(api_key=key)
    if choice == "auto":
        key = os.environ.get("OPENAI_API_KEY")
        if key:
            return OpenAICompatibleExtractor(api_key=key)
        return RuleExtractor()
    raise ValueError(f"unknown provider: {choice}")


def _from_llm_json(content: str) -> ChangeOrderExtraction:
    raw = _strip_fence(content)
    try:
        payload = _LlmPayload.model_validate(json.loads(raw))
    except (json.JSONDecodeError, ValidationError) as exc:
        raise ValueError("LLM response was not valid change-order JSON") from exc
    confidence = payload.field_confidence

    def field(name: str, value: object, *, money: bool = False) -> Scored[object]:
        if value is None or value == "":
            return Scored()
        if money and isinstance(value, Decimal):
            value = quantize_money(value)
        stated = confidence.get(name, 0.75)
        try:
            stated_f = float(stated)
        except (TypeError, ValueError):
            stated_f = 0.75
        return Scored(value=value, confidence=round(min(_LLM_CAP, max(0.0, stated_f)), 3))

    parsed_date = _parse_llm_date(payload.date)
    approval = (payload.approval_status or "").strip().lower()
    if approval not in _APPROVAL:
        approval_value = None
    else:
        approval_value = approval
    items: list[LineItem] = []
    for line in payload.line_items:
        if not any(getattr(line, name) is not None for name in ("description", "amount", "quantity")):
            continue
        items.append(
            LineItem(
                description=field("line_items", _clean(line.description)),
                quantity=field("line_items", line.quantity),
                unit=field("line_items", _clean(line.unit).upper() if line.unit else None),
                unit_price=field("line_items", line.unit_price, money=True),
                amount=field("line_items", line.amount, money=True),
            )
        )
    return ChangeOrderExtraction(
        project_name=field("project_name", _clean(payload.project_name)),
        project_number=field("project_number", _clean(payload.project_number)),
        change_order_number=field("change_order_number", _clean(payload.change_order_number)),
        date=field("date", parsed_date),
        contractor=field("contractor", _clean(payload.contractor)),
        owner=field("owner", _clean(payload.owner)),
        description=field("description", _clean(payload.description)),
        currency=field("currency", _clean(payload.currency).upper() if payload.currency else None),
        line_items=items,
        line_subtotal=field("line_subtotal", payload.line_subtotal, money=True),
        markup=field("markup", payload.markup, money=True),
        markup_percent=field("markup_percent", payload.markup_percent),
        tax=field("tax", payload.tax, money=True),
        tax_percent=field("tax_percent", payload.tax_percent),
        other_fees=field("other_fees", payload.other_fees, money=True),
        total=field("total", payload.total, money=True),
        schedule_impact_days=field("schedule_impact_days", payload.schedule_impact_days),
        reason=field("reason", _clean(payload.reason)),
        approval_status=field("approval_status", approval_value),
        method="openai",
    )


def _parse_llm_date(value: str | None) -> date | None:
    if not value:
        return None
    try:
        return date.fromisoformat(value[:10])
    except ValueError:
        return parse_date(value)


def _clean(value: str | None) -> str | None:
    if value is None:
        return None
    text = " ".join(value.split())
    return text or None


def _strip_fence(content: str) -> str:
    text = content.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return text
