"""Versioned rules for allocating daily OpenAI aggregate costs to customers."""

from __future__ import annotations

from collections.abc import Mapping
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from typing import Any


POLICY_VERSION = "openai-provider-cost-allocation-v1"
NANOS_PER_USD = Decimal("1000000000")


def usd_to_nanos(amount_usd: Any) -> int:
    try:
        amount = Decimal(str(amount_usd))
    except (InvalidOperation, ValueError) as exc:
        raise ValueError("OpenAI provider cost is not a valid decimal amount.") from exc
    if not amount.is_finite() or amount < 0:
        raise ValueError("OpenAI provider cost must be a finite nonnegative amount.")
    return int((amount * NANOS_PER_USD).to_integral_value(rounding=ROUND_HALF_UP))


def allocate_integer(total: int, weights: Mapping[str, int]) -> dict[str, int]:
    """Allocate integer nanos exactly, using largest remainder and stable ties."""
    if isinstance(total, bool) or not isinstance(total, int) or total < 0:
        raise ValueError("Allocation total must be a nonnegative integer.")
    clean = {
        str(key): int(value)
        for key, value in weights.items()
        if isinstance(value, int) and not isinstance(value, bool) and value > 0
    }
    denominator = sum(clean.values())
    if not clean or denominator <= 0:
        return {}
    allocations: dict[str, int] = {}
    remainders: list[tuple[int, str]] = []
    allocated = 0
    for key in sorted(clean):
        numerator = total * clean[key]
        quotient, remainder = divmod(numerator, denominator)
        allocations[key] = quotient
        allocated += quotient
        remainders.append((remainder, key))
    for _, key in sorted(remainders, key=lambda item: (-item[0], item[1]))[: total - allocated]:
        allocations[key] += 1
    return allocations


def cost_line_item_weight_kind(line_item: str | None) -> str:
    """Return the v1 customer weight for one OpenAI Costs API line item."""
    name = (line_item or "").casefold().replace("_", " ").replace("-", " ")
    if "web search" in name or "web_search" in name:
        return "web_search_calls"
    if any(term in name for term in ("container", "sandbox", "code interpreter", "command execution")):
        return "container_tool_calls"
    if "cache write" in name or "cache writes" in name or "cached write" in name:
        return "cache_write_proxy_tokens"
    if "cached input" in name or "input cached" in name or "cache read" in name:
        return "input_cached_tokens"
    if "output" in name:
        return "output_tokens"
    if "input" in name:
        return "input_uncached_tokens"
    return "model_cost_fallback"


def line_item_model(line_item: str | None, models: set[str]) -> str | None:
    name = (line_item or "").casefold()
    matches = [model for model in models if model.casefold() in name]
    return max(matches, key=len) if matches else None


def item_weight(user: Mapping[str, Any], kind: str) -> int:
    if kind == "web_search_calls":
        return _number(user.get("web_search_calls"))
    if kind == "container_tool_calls":
        return _number(user.get("container_tool_calls"))
    models = user.get("models")
    if not isinstance(models, Mapping):
        return _number(user.get("estimated_cost_nanos"))
    return sum(
        _model_weight(value, kind)
        for value in models.values()
        if isinstance(value, Mapping)
    )


def model_item_weight(user: Mapping[str, Any], model: str | None, kind: str) -> int:
    models = user.get("models")
    if not isinstance(models, Mapping):
        return _number(user.get("estimated_cost_nanos"))
    if model:
        return _model_weight(models.get(model, {}), kind)
    return sum(
        _model_weight(value, kind)
        for value in models.values()
        if isinstance(value, Mapping)
    )


def _model_weight(value: Any, kind: str) -> int:
    if not isinstance(value, Mapping):
        return 0
    if kind in {"web_search_calls", "container_tool_calls"}:
        return 0
    if kind == "input_uncached_tokens":
        return _number(value.get("input_uncached_tokens"))
    if kind == "input_cached_tokens":
        return _number(value.get("input_cached_tokens"))
    if kind == "output_tokens":
        return _number(value.get("output_tokens"))
    if kind == "cache_write_proxy_tokens":
        # Agents API usage does not expose cache-write counts. V1 assigns this
        # aggregate line item by the user's reported input-token share.
        return _number(value.get("input_tokens"))
    estimated = _number(value.get("estimated_cost_nanos"))
    return estimated or (
        _number(value.get("input_tokens")) + _number(value.get("output_tokens"))
    )


def _number(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return 0
    return value
