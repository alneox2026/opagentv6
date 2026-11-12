from decimal import Decimal

from common.openai_cost_allocation import (
    POLICY_VERSION,
    allocate_integer,
    cost_line_item_weight_kind,
)
from services.agent_persistence_worker_v3.app.services.openai_cost_reconciliation import (
    OpenAICostReconciliationService,
)


def test_largest_remainder_allocation_is_exact_and_stable():
    weights = {"customer-b": 1, "customer-a": 1, "customer-c": 1}

    result = allocate_integer(5, weights)

    assert result == {"customer-a": 2, "customer-b": 2, "customer-c": 1}
    assert sum(result.values()) == 5


def test_policy_classifies_provider_line_items():
    assert POLICY_VERSION == "openai-provider-cost-allocation-v1"
    assert cost_line_item_weight_kind("gpt-6-luna, cache writes") == "cache_write_proxy_tokens"
    assert cost_line_item_weight_kind("gpt-6-luna cached input") == "input_cached_tokens"
    assert cost_line_item_weight_kind("gpt-6-luna output") == "output_tokens"
    assert cost_line_item_weight_kind("web search tool calls") == "web_search_calls"
    assert cost_line_item_weight_kind("OpenAI hosted containers") == "container_tool_calls"
    assert cost_line_item_weight_kind("other provider resource") == "model_cost_fallback"


def test_provider_line_items_allocate_by_usage_categories_and_sum_exactly():
    users = {
        "customer-a": {
            "web_search_calls": 3,
            "container_tool_calls": 1,
            "models": {
                "gpt-6-luna": {
                    "input_uncached_tokens": 100,
                    "input_cached_tokens": 50,
                    "input_tokens": 150,
                    "output_tokens": 10,
                    "estimated_cost_nanos": 1_000,
                }
            },
        },
        "customer-b": {
            "web_search_calls": 1,
            "container_tool_calls": 3,
            "models": {
                "gpt-6-luna": {
                    "input_uncached_tokens": 300,
                    "input_cached_tokens": 150,
                    "input_tokens": 450,
                    "output_tokens": 30,
                    "estimated_cost_nanos": 3_000,
                }
            },
        },
    }
    rows = [
        {"amount_usd": Decimal("0.000004"), "line_item": "gpt-6-luna input tokens"},
        {"amount_usd": Decimal("0.000002"), "line_item": "gpt-6-luna cached input"},
        {"amount_usd": Decimal("0.000001"), "line_item": "gpt-6-luna cache writes"},
        {"amount_usd": Decimal("0.000003"), "line_item": "web search tool calls"},
        {"amount_usd": Decimal("0.000004"), "line_item": "OpenAI hosted containers"},
        {"amount_usd": Decimal("0.000002"), "line_item": "other resource"},
    ]

    shares, fallbacks, unallocated = OpenAICostReconciliationService._allocate_cost_rows(
        rows, users
    )

    assert shares == {"customer-a": 5_500, "customer-b": 10_500}
    assert sum(shares.values()) == 16_000
    assert fallbacks == {}
    assert unallocated == 0


def test_tool_costs_fall_back_to_model_cost_when_call_count_is_missing():
    users = {
        "customer-a": {
            "web_search_calls": 0,
            "container_tool_calls": 0,
            "models": {
                "gpt-6-luna": {"estimated_cost_nanos": 2},
            },
        },
        "customer-b": {
            "web_search_calls": 0,
            "container_tool_calls": 0,
            "models": {
                "gpt-6-luna": {"estimated_cost_nanos": 6},
            },
        },
    }
    rows = [{"amount_usd": Decimal("0.00000008"), "line_item": "web search tool calls"}]

    shares, fallbacks, unallocated = OpenAICostReconciliationService._allocate_cost_rows(
        rows, users
    )

    assert shares == {"customer-a": 20, "customer-b": 60}
    assert fallbacks == {"web_search_calls_to_model_cost_fallback": 1}
    assert unallocated == 0
