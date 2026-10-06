"""Pydantic schema for a change-order extraction.

Every extracted value is a `Scored` object: the value, a confidence in
``[0, 1]``, whether a person should review it, and a short source quote.
Money is decimal, never float. The document also carries one overall
confidence and the cross-field review flags raised by deterministic checks.
"""

from __future__ import annotations

from datetime import date as Date
from decimal import Decimal
from typing import Generic, TypeVar

from pydantic import BaseModel, ConfigDict, Field

T = TypeVar("T")


class Scored(BaseModel, Generic[T]):
    """One field value plus how much the pipeline trusts it."""

    model_config = ConfigDict(extra="forbid")

    value: T | None = None
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    needs_review: bool = False
    evidence: str | None = None


class LineItem(BaseModel):
    """A single priced row. Each column is scored on its own."""

    model_config = ConfigDict(extra="forbid")

    description: Scored[str] = Field(default_factory=Scored[str])
    quantity: Scored[Decimal] = Field(default_factory=Scored[Decimal])
    unit: Scored[str] = Field(default_factory=Scored[str])
    unit_price: Scored[Decimal] = Field(default_factory=Scored[Decimal])
    amount: Scored[Decimal] = Field(default_factory=Scored[Decimal])


class ChangeOrderExtraction(BaseModel):
    """Validated extraction result returned by the CLI, API, and eval harness."""

    model_config = ConfigDict(extra="forbid")

    project_name: Scored[str] = Field(default_factory=Scored[str])
    project_number: Scored[str] = Field(default_factory=Scored[str])
    change_order_number: Scored[str] = Field(default_factory=Scored[str])
    date: Scored[Date] = Field(default_factory=lambda: Scored[Date]())
    contractor: Scored[str] = Field(default_factory=Scored[str])
    owner: Scored[str] = Field(default_factory=Scored[str])
    description: Scored[str] = Field(default_factory=Scored[str])
    currency: Scored[str] = Field(default_factory=Scored[str])
    line_items: list[LineItem] = Field(default_factory=list)
    # Sum of line items, before markup, tax, and other fees.
    # Forms often label this "Subtotal" or "Net". "Subtotal before tax"
    # on some forms is a later figure; we do not use that label for this field.
    line_subtotal: Scored[Decimal] = Field(default_factory=Scored[Decimal])
    markup: Scored[Decimal] = Field(default_factory=Scored[Decimal])
    markup_percent: Scored[Decimal] = Field(default_factory=Scored[Decimal])
    tax: Scored[Decimal] = Field(default_factory=Scored[Decimal])
    tax_percent: Scored[Decimal] = Field(default_factory=Scored[Decimal])
    other_fees: Scored[Decimal] = Field(default_factory=Scored[Decimal])
    total: Scored[Decimal] = Field(default_factory=Scored[Decimal])
    schedule_impact_days: Scored[int] = Field(default_factory=Scored[int])
    reason: Scored[str] = Field(default_factory=Scored[str])
    approval_status: Scored[str] = Field(default_factory=Scored[str])
    unincorporated_notes: list[Scored[str]] = Field(default_factory=list)
    document_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    review_flags: list[str] = Field(default_factory=list)
    method: str = "rules"
    source_name: str | None = None
    ocr_used: bool = False
    page_count: int | None = None
    warnings: list[str] = Field(default_factory=list)
