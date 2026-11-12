from datetime import datetime, timezone

import httpx
import pytest

from services.agent_persistence_worker_v3.app.core.config import get_settings
from services.agent_persistence_worker_v3.app.core.errors import RetryableWorkerError
from services.agent_persistence_worker_v3.app.services.openai_cost_reconciliation import (
    OpenAICostReconciliationService,
    _normalize_cost_row,
)


class Snapshot:
    def __init__(self, data):
        self.data = data

    def to_dict(self):
        return dict(self.data)


class Query:
    def __init__(self, collection):
        self.collection = collection
        self.filters = []

    def where(self, *args):
        self.filters.append(args)
        return self

    def stream(self):
        result = []
        for data in self.collection.client.entries:
            created_at = data["created_at"]
            if all(
                (op == ">=" and created_at >= value)
                or (op == "<" and created_at < value)
                for field, op, value in self.filters
                if field == "created_at"
            ):
                result.append(Snapshot(data))
        return result


class Document:
    def __init__(self, collection, document_id):
        self.collection = collection
        self.document_id = document_id

    def set(self, data, merge=False):
        if merge and self.document_id in self.collection.client.documents:
            self.collection.client.documents[self.document_id].update(data)
        else:
            self.collection.client.documents[self.document_id] = dict(data)


class Collection:
    def __init__(self, client, name):
        self.client = client
        self.name = name

    def where(self, *args):
        return Query(self).where(*args)

    def stream(self):
        return Query(self).stream()

    def document(self, document_id):
        return Document(self, document_id)


class Firestore:
    def __init__(self):
        self.entries = []
        self.documents = {}

    def collection(self, name):
        return Collection(self, name)


def test_daily_import_records_only_returned_provider_buckets_when_allocation_is_disabled(monkeypatch):
    get_settings.cache_clear()
    monkeypatch.setenv("OPENAI_COST_RECONCILIATION_ENABLED", "true")
    monkeypatch.setenv("OPENAI_COST_RECONCILIATION_LOOKBACK_DAYS", "7")
    monkeypatch.setenv("OPENAI_PROJECT_ID", "proj_dedicated")
    monkeypatch.setenv("OPENAI_ADMIN_API_KEY", "admin-secret")
    get_settings.cache_clear()
    expected_start = int(datetime(2026, 10, 22, tzinfo=timezone.utc).timestamp())
    day_start = int(datetime(2026, 10, 28, tzinfo=timezone.utc).timestamp())
    requests = []

    def handler(request):
        requests.append(request)
        assert request.headers["Authorization"] == "Bearer admin-secret"
        assert request.url.params.get("project_ids") == "proj_dedicated"
        assert request.url.params.get_list("group_by") == ["project_id", "api_source", "line_item"]
        assert int(request.url.params["start_time"]) == expected_start
        return httpx.Response(
            200,
            json={
                "object": "page",
                "has_more": False,
                "data": [
                    {
                        "object": "bucket",
                        "start_time": day_start,
                        "end_time": day_start + 86400,
                        "results": [
                            {
                                "amount": {"value": 0.07, "currency": "usd"},
                                "project_id": "proj_dedicated",
                                "api_source": "agents_api",
                                "line_item": "gpt-6-luna input tokens",
                                "quantity": 10000,
                                "quantity_unit": "tokens",
                            },
                            {
                                "amount": {"value": 0.02, "currency": "usd"},
                                "project_id": "proj_dedicated",
                                "api_source": "agents_api",
                                "line_item": "OpenAI hosted containers",
                                "quantity": 30,
                                "quantity_unit": "duration_minutes",
                            },
                            {
                                "amount": {"value": 0.01, "currency": "usd"},
                                "project_id": "proj_dedicated",
                                "api_source": "unlabeled",
                                "line_item": "unknown",
                            },
                        ],
                    }
                ],
            },
        )

    firestore = Firestore()
    firestore.entries.append(
        {
            "turn_id": "turn-1",
            "created_at": datetime(2026, 10, 28, 12, tzinfo=timezone.utc),
            "estimated_cost_usd_nanos": 50_000_000,
        }
    )
    service = OpenAICostReconciliationService(
        firestore_client_factory=lambda: firestore,
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
        now=lambda: datetime(2026, 10, 29, 4, tzinfo=timezone.utc),
    )

    result = service.reconcile_sync()

    assert len(requests) == 1
    assert result.status == "completed"
    # Do not synthesize zero-cost buckets for days the Costs API omitted.
    assert result.reconciled_days == 1
    assert result.provider_cost_usd == 0.10
    assert result.agents_api_cost_usd == 0.09
    assert result.local_estimate_usd == 0.05
    stored = next(iter(firestore.documents.values()))
    assert stored["provider_actual_cost_usd"] == 0.10
    assert stored["agents_api_actual_cost_usd"] == 0.09
    assert stored["agents_api_minus_local_model_estimate_usd"] == 0.04
    assert stored["customer_attribution_status"] == "aggregate_only_allocation_disabled"
    assert stored["wallet_adjustment_status"] == "disabled"
    assert stored["allocation_source_status"] == "allocation_disabled"
    get_settings.cache_clear()


def test_daily_import_fails_loudly_without_admin_key(monkeypatch):
    get_settings.cache_clear()
    monkeypatch.setenv("OPENAI_COST_RECONCILIATION_ENABLED", "true")
    monkeypatch.setenv("OPENAI_PROJECT_ID", "proj_test")
    monkeypatch.delenv("OPENAI_ADMIN_API_KEY", raising=False)
    get_settings.cache_clear()

    try:
        OpenAICostReconciliationService().reconcile_sync()
    except RetryableWorkerError as exc:
        assert "OPENAI_ADMIN_API_KEY" in str(exc)
    else:
        raise AssertionError("Missing admin key should fail the enabled scheduled reconciliation.")
    get_settings.cache_clear()


def test_cost_reconciliation_rejects_project_display_name_as_api_filter(monkeypatch):
    get_settings.cache_clear()
    monkeypatch.setenv("OPENAI_COST_RECONCILIATION_ENABLED", "true")
    monkeypatch.setenv("OPENAI_PROJECT_ID", "ceosystem1")
    monkeypatch.setenv("OPENAI_ADMIN_API_KEY", "admin-secret")
    get_settings.cache_clear()

    try:
        OpenAICostReconciliationService().reconcile_sync()
    except RetryableWorkerError as exc:
        assert "OPENAI_PROJECT_ID" in str(exc)
        assert "proj_" in str(exc)
    else:
        raise AssertionError("The project display name must not be sent as a Costs API project ID.")
    get_settings.cache_clear()


def test_cost_import_fails_closed_on_malformed_rows():
    with pytest.raises(RetryableWorkerError, match="without a project ID"):
        _normalize_cost_row({"amount": {"value": "0.01", "currency": "usd"}})
    with pytest.raises(RetryableWorkerError, match="valid amount"):
        _normalize_cost_row(
            {
                "amount": {"value": "not-a-number", "currency": "usd"},
                "project_id": "proj_test",
            }
        )


def test_cost_import_rejects_rows_outside_the_dedicated_project(monkeypatch):
    get_settings.cache_clear()
    monkeypatch.setenv("OPENAI_PROJECT_ID", "proj_expected")
    get_settings.cache_clear()
    day_start = int(datetime(2026, 10, 28, tzinfo=timezone.utc).timestamp())
    response = {
        "object": "page",
        "has_more": False,
        "data": [
            {
                "start_time": day_start,
                "end_time": day_start + 86400,
                "results": [
                    {
                        "amount": {"value": 0.01, "currency": "usd"},
                        "project_id": "proj_other",
                        "api_source": "agents_api",
                        "line_item": "gpt-6-luna input tokens",
                    }
                ],
            }
        ],
    }
    client = httpx.Client(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, json=response))
    )
    service = OpenAICostReconciliationService(http_client=client)

    with pytest.raises(RetryableWorkerError, match="outside the configured project filter"):
        service._fetch_buckets(
            client,
            "admin-secret",
            datetime.fromtimestamp(day_start, tz=timezone.utc),
            datetime.fromtimestamp(day_start + 86400, tz=timezone.utc),
        )
    client.close()
    get_settings.cache_clear()


@pytest.mark.parametrize(
    "bucket_case", ["missing_results", "missing_end", "out_of_range", "end_out_of_range"]
)
def test_cost_import_rejects_malformed_or_out_of_range_buckets(monkeypatch, bucket_case):
    get_settings.cache_clear()
    monkeypatch.setenv("OPENAI_PROJECT_ID", "proj_test")
    get_settings.cache_clear()
    day_start = int(datetime(2026, 10, 28, tzinfo=timezone.utc).timestamp())
    buckets = {
        "missing_results": {"start_time": day_start, "end_time": day_start + 86400},
        "missing_end": {"start_time": day_start, "results": []},
        "out_of_range": {
            "start_time": day_start + 86400,
            "end_time": day_start + 2 * 86400,
            "results": [],
        },
        "end_out_of_range": {
            "start_time": day_start,
            "end_time": day_start + 2 * 86400,
            "results": [],
        },
    }
    client = httpx.Client(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(
                200,
                json={"object": "page", "has_more": False, "data": [buckets[bucket_case]]},
            )
        )
    )
    service = OpenAICostReconciliationService(http_client=client)

    with pytest.raises(RetryableWorkerError, match="malformed cost bucket|invalid timestamps"):
        service._fetch_buckets(
            client,
            "admin-secret",
            datetime.fromtimestamp(day_start, tz=timezone.utc),
            datetime.fromtimestamp(day_start + 86400, tz=timezone.utc),
        )
    client.close()
    get_settings.cache_clear()


@pytest.mark.parametrize(
    ("usage_status", "expected"),
    [("pending", "pending_provider_usage"), ("manual_review", "manual_review_provider_usage")],
)
def test_daily_cost_allocation_waits_for_provider_usage_state(
    monkeypatch, usage_status, expected
):
    get_settings.cache_clear()
    monkeypatch.setenv("OPENAI_COST_ALLOCATION_ENABLED", "true")
    monkeypatch.setenv("OPENAI_PROJECT_ID", "proj_test")
    get_settings.cache_clear()
    firestore = Firestore()
    firestore.entries.append(
        {
            "turn_id": "turn-1",
            "uid": "owner-1",
            "owner_uid": "owner-1",
            "billing_subject_id": "customer-1",
            "created_at": datetime(2026, 10, 28, 12, tzinfo=timezone.utc),
            "pricing_model": "gpt-6-luna",
            "billable_tokens": {
                "input_uncached": 10,
                "input_cached": 0,
                "input_cache_write": 0,
                "output_including_reasoning": 1,
            },
            "estimated_cost_usd_nanos": 10,
            "billing_assessed_cost_nanos": 10,
            "usage_reconciliation_status": usage_status,
            "status": "late_settled",
        }
    )
    service = OpenAICostReconciliationService(firestore_client_factory=lambda: firestore)

    _, _, source_status = service._billing_usage_for_day(
        firestore,
        datetime(2026, 10, 28, tzinfo=timezone.utc),
        datetime(2026, 10, 29, tzinfo=timezone.utc),
    )

    assert source_status == expected
    get_settings.cache_clear()


def test_daily_allocation_applies_line_item_shares_after_all_turns_are_settled(monkeypatch):
    get_settings.cache_clear()
    monkeypatch.setenv("OPENAI_COST_RECONCILIATION_ENABLED", "true")
    monkeypatch.setenv("OPENAI_COST_ALLOCATION_ENABLED", "true")
    monkeypatch.setenv("OPENAI_PROJECT_ID", "proj_dedicated")
    monkeypatch.setenv("OPENAI_PROJECT_NAME", "ceosystem1")
    monkeypatch.setenv("OPENAI_ADMIN_API_KEY", "admin-secret")
    get_settings.cache_clear()
    day_start = int(datetime(2026, 10, 28, tzinfo=timezone.utc).timestamp())
    calls = []

    def handler(request):
        return httpx.Response(
            200,
            json={
                "object": "page",
                "has_more": False,
                "data": [
                    {
                        "start_time": day_start,
                        "end_time": day_start + 86400,
                        "results": [
                            {
                                "amount": {"value": "0.00000020", "currency": "usd"},
                                "project_id": "proj_dedicated",
                                "api_source": "agents_api",
                                "line_item": "gpt-6-luna input tokens",
                                "quantity": 400,
                                "quantity_unit": "tokens",
                            },
                            {
                                "amount": {"value": "0.00000010", "currency": "usd"},
                                "project_id": "proj_dedicated",
                                "api_source": "agents_api",
                                "line_item": "web search tool calls",
                                "quantity": 4,
                                "quantity_unit": "calls",
                            },
                            {
                                "amount": {"value": "0.00000020", "currency": "usd"},
                                "project_id": "proj_dedicated",
                                "api_source": "agents_api",
                                "line_item": "OpenAI hosted containers",
                                "quantity": 4,
                                "quantity_unit": "calls",
                            },
                        ],
                    }
                ],
            },
        )

    class WalletAdjuster:
        def apply_sync(self, **kwargs):
            calls.append(kwargs)
            return type(
                "Adjustment",
                (),
                {
                    "status": "adjusted",
                    "charged_nanos": max(0, kwargs["provider_actual_cost_nanos"] - kwargs["provisional_assessed_cost_nanos"]),
                    "debt_added_nanos": 0,
                    "debt_reduced_nanos": 0,
                    "refunded_nanos": max(0, kwargs["provisional_assessed_cost_nanos"] - kwargs["provider_actual_cost_nanos"]),
                },
            )()

    firestore = Firestore()
    created_at = datetime(2026, 10, 28, 12, tzinfo=timezone.utc)
    firestore.entries.extend(
        [
            {
                "turn_id": "turn-a",
                "uid": "owner-a",
                "billing_subject_id": "customer-a",
                "created_at": created_at,
                "pricing_model": "gpt-6-luna",
                "billable_tokens": {
                    "input_uncached": 100,
                    "input_cached": 0,
                    "input_cache_write": 0,
                    "output_including_reasoning": 0,
                },
                "estimated_cost_usd_nanos": 100,
                "usage": {"provider_resource_usage": {"web_search_calls": 3, "container_tool_calls": 1}},
                "status": "settled",
                "billing_assessed_cost_nanos": 100,
                "owner_uid": "owner-a",
            },
            {
                "turn_id": "turn-b",
                "uid": "owner-b",
                "billing_subject_id": "customer-b",
                "created_at": created_at,
                "pricing_model": "gpt-6-luna",
                "billable_tokens": {
                    "input_uncached": 300,
                    "input_cached": 0,
                    "input_cache_write": 0,
                    "output_including_reasoning": 0,
                },
                "estimated_cost_usd_nanos": 200,
                "usage": {"provider_resource_usage": {"web_search_calls": 1, "container_tool_calls": 3}},
                "status": "settled",
                "billing_assessed_cost_nanos": 200,
                "owner_uid": "owner-b",
            },
        ]
    )
    service = OpenAICostReconciliationService(
        firestore_client_factory=lambda: firestore,
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
        now=lambda: datetime(2026, 10, 29, 4, tzinfo=timezone.utc),
        wallet_adjustment_service=WalletAdjuster(),
    )

    result = service.reconcile_sync()

    assert result.reconciled_days == 1
    assert {
        call["billing_subject_id"]: (
            call["provider_actual_cost_nanos"],
            call["provisional_assessed_cost_nanos"],
        )
        for call in calls
    } == {"customer-a": (175, 100), "customer-b": (325, 200)}
    stored = next(iter(firestore.documents.values()))
    assert stored["project_name"] == "ceosystem1"
    assert stored["allocation_policy_version"] == "openai-provider-cost-allocation-v1"
    assert stored["allocated_cost_nanos"] == 500
    assert stored["unallocated_cost_nanos"] == 0
    assert stored["wallet_adjustment_status"] == "applied"
    get_settings.cache_clear()
