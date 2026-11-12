"""Import daily OpenAI provider costs and optionally allocate them to wallets."""

from __future__ import annotations

import asyncio
import hashlib
import os
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, time as datetime_time, timedelta, timezone
from decimal import Decimal, InvalidOperation
from typing import Any

import httpx

from common.openai_cost_allocation import (
    POLICY_VERSION,
    allocate_integer,
    cost_line_item_weight_kind,
    item_weight,
    line_item_model,
    model_item_weight,
    usd_to_nanos,
)
from services.agent_persistence_worker_v3.app.core.config import get_settings
from services.agent_persistence_worker_v3.app.core.errors import RetryableWorkerError
from services.agent_persistence_worker_v3.app.services.firestore_client import get_firestore_client
from services.agent_persistence_worker_v3.app.services.openai_cost_wallet_adjustment import (
    OpenAICostWalletAdjustmentService,
)

OPENAI_COSTS_URL = "https://api.openai.com/v1/organization/costs"


@dataclass(frozen=True)
class OpenAICostReconciliationResult:
    status: str
    reconciled_days: int = 0
    provider_cost_usd: float = 0.0
    agents_api_cost_usd: float = 0.0
    local_estimate_usd: float = 0.0


class OpenAICostReconciliationService:
    """Fetch and persist daily provider actuals alongside local model estimates."""

    def __init__(
        self,
        *,
        firestore_client_factory: Callable[[], Any] | None = None,
        http_client: httpx.Client | None = None,
        now: Callable[[], datetime] | None = None,
        wallet_adjustment_service: OpenAICostWalletAdjustmentService | None = None,
    ) -> None:
        self._settings = get_settings()
        self._firestore_client_factory = firestore_client_factory or get_firestore_client
        self._http_client = http_client
        self._now = now or (lambda: datetime.now(timezone.utc))
        self._wallet_adjustment_service = wallet_adjustment_service or OpenAICostWalletAdjustmentService(
            firestore_client_factory=self._firestore_client_factory
        )

    async def reconcile(self) -> OpenAICostReconciliationResult:
        return await asyncio.to_thread(self.reconcile_sync)

    def reconcile_sync(self) -> OpenAICostReconciliationResult:
        if not self._settings.openai_cost_reconciliation_enabled:
            return OpenAICostReconciliationResult("disabled")
        if not _is_openai_project_id(self._settings.openai_project_id):
            raise RetryableWorkerError(
                "OpenAI Costs API reconciliation requires OPENAI_PROJECT_ID to be the dedicated proj_... API ID."
            )
        api_key = os.getenv("OPENAI_ADMIN_API_KEY", "").strip()
        if not api_key:
            raise RetryableWorkerError(
                "OpenAI daily cost reconciliation is enabled but OPENAI_ADMIN_API_KEY is not configured."
            )

        now = self._now().astimezone(timezone.utc)
        end = datetime.combine(now.date(), datetime_time.min, tzinfo=timezone.utc)
        lookback_days = max(1, min(180, self._settings.openai_cost_reconciliation_lookback_days))
        start = end - timedelta(days=lookback_days)
        client = self._http_client or httpx.Client(timeout=httpx.Timeout(30.0, connect=5.0))
        owns_client = self._http_client is None
        try:
            buckets = self._fetch_buckets(client, api_key, start, end)
        finally:
            if owns_client:
                client.close()
        firestore = self._firestore_client_factory()
        provider_totals = Decimal("0")
        agents_api_totals = Decimal("0")
        local_totals = Decimal("0")
        days_written = 0
        for bucket_start, rows in sorted(buckets.items()):
            day_start = datetime.fromtimestamp(bucket_start, tz=timezone.utc)
            day_end = day_start + timedelta(days=1)
            usage_by_subject, local_nanos, source_status = self._billing_usage_for_day(
                firestore, day_start, day_end
            )
            local_usd = Decimal(local_nanos) / Decimal("1000000000")
            total_usd = sum((row["amount_usd"] for row in rows), Decimal("0"))
            agents_usd = sum(
                (row["amount_usd"] for row in rows if row.get("api_source") == "agents_api"),
                Decimal("0"),
            )
            provider_totals += total_usd
            agents_api_totals += agents_usd
            local_totals += local_usd
            project_scope = self._settings.openai_project_id
            project_hash = hashlib.sha256(project_scope.encode("utf-8")).hexdigest()[:12]
            document_id = f"{day_start:%Y-%m-%d}_{project_hash}"
            allocations: dict[str, int] = {}
            allocation_notes: dict[str, int] = {}
            unallocated_nanos = 0
            if self._settings.openai_cost_allocation_enabled and source_status == "ready":
                allocations, allocation_notes, unallocated_nanos = self._allocate_cost_rows(
                    rows, usage_by_subject
                )
            adjustment_summary = {
                "status": "disabled",
                "adjusted_customers": 0,
                "charged_nanos": 0,
                "debt_added_nanos": 0,
                "debt_reduced_nanos": 0,
                "refunded_nanos": 0,
            }
            if self._settings.openai_cost_allocation_enabled:
                if source_status != "ready":
                    adjustment_summary["status"] = source_status
                elif not allocations and total_usd > 0:
                    adjustment_summary["status"] = "manual_review_no_customer_weight"
                    unallocated_nanos = usd_to_nanos(total_usd)
                else:
                    charged = debt = debt_reduced = refunded = adjusted = 0
                    for subject_id, user_data in usage_by_subject.items():
                        if subject_id not in allocations and not user_data.get("provisional_assessed_cost_nanos"):
                            continue
                        result = self._wallet_adjustment_service.apply_sync(
                            billing_subject_id=subject_id,
                            owner_uid=str(user_data["owner_uid"]),
                            project_id=self._settings.openai_project_id,
                            day_start=day_start,
                            provider_actual_cost_nanos=allocations.get(subject_id, 0),
                            provisional_assessed_cost_nanos=int(
                                user_data.get("provisional_assessed_cost_nanos", 0)
                            ),
                        )
                        charged += result.charged_nanos
                        debt += result.debt_added_nanos
                        debt_reduced += result.debt_reduced_nanos
                        refunded += result.refunded_nanos
                        adjusted += int(result.status == "adjusted")
                    adjustment_summary.update(
                        {
                            "status": "partially_allocated_manual_review" if unallocated_nanos else "applied",
                            "adjusted_customers": adjusted,
                            "charged_nanos": charged,
                            "debt_added_nanos": debt,
                            "debt_reduced_nanos": debt_reduced,
                            "refunded_nanos": refunded,
                        }
                    )
            payload = {
                "schema_version": 1,
                "document_id": document_id,
                "provider": "openai",
                "source": "organization_costs_api",
                "bucket_width": "1d",
                "bucket_start": day_start,
                "bucket_end": day_end,
                "project_scope": project_scope,
                "project_name": self._settings.openai_project_name or None,
                "project_filter_applied": True,
                "currency": "USD",
                "provider_actual_cost_usd": float(total_usd),
                "agents_api_actual_cost_usd": float(agents_usd),
                "local_model_estimate_usd": float(local_usd),
                "agents_api_minus_local_model_estimate_usd": float(agents_usd - local_usd),
                "local_estimate_basis": "per_turn_token_rate_card; excludes aggregate cache-write and non-model OpenAI resources",
                "customer_attribution_status": (
                    adjustment_summary["status"]
                    if self._settings.openai_cost_allocation_enabled
                    else "aggregate_only_allocation_disabled"
                ),
                "wallet_adjustment_status": adjustment_summary["status"],
                "allocation_policy_version": POLICY_VERSION if self._settings.openai_cost_allocation_enabled else None,
                "allocation_source_status": source_status,
                "allocated_customer_count": len(allocations),
                "allocated_cost_nanos": sum(allocations.values()),
                "unallocated_cost_nanos": unallocated_nanos,
                "allocation_fallbacks": allocation_notes,
                "wallet_adjustments": adjustment_summary,
                "line_items": [
                    {
                        "project_id": row.get("project_id"),
                        "api_source": row.get("api_source"),
                        "line_item": row.get("line_item"),
                        "amount_usd": float(row["amount_usd"]),
                        "quantity": row.get("quantity"),
                        "quantity_unit": row.get("quantity_unit"),
                    }
                    for row in rows
                ],
                "reconciled_at": now,
                "reconciliation_window_days": lookback_days,
            }
            firestore.collection(self._settings.openai_costs_collection).document(document_id).set(
                payload,
                merge=True,
            )
            days_written += 1
        return OpenAICostReconciliationResult(
            status="completed",
            reconciled_days=days_written,
            provider_cost_usd=float(provider_totals),
            agents_api_cost_usd=float(agents_api_totals),
            local_estimate_usd=float(local_totals),
        )

    def _fetch_buckets(self, client: httpx.Client, api_key: str, start: datetime, end: datetime) -> dict[int, list[dict[str, Any]]]:
        headers = {"Authorization": f"Bearer {api_key}", "Accept": "application/json"}
        params: dict[str, Any] = {
            "start_time": int(start.timestamp()),
            "end_time": int(end.timestamp()),
            "bucket_width": "1d",
            "limit": 180,
            "group_by": ["project_id", "api_source", "line_item"],
        }
        params["project_ids"] = [self._settings.openai_project_id]
        results: dict[int, list[dict[str, Any]]] = defaultdict(list)
        while True:
            try:
                response = client.get(OPENAI_COSTS_URL, headers=headers, params=params)
            except httpx.RequestError as exc:
                raise RetryableWorkerError(f"OpenAI Costs API request failed: {type(exc).__name__}.") from exc
            if response.status_code >= 500 or response.status_code == 429:
                raise RetryableWorkerError(f"OpenAI Costs API returned retryable HTTP {response.status_code}.")
            if response.status_code != 200:
                raise RetryableWorkerError(f"OpenAI Costs API returned HTTP {response.status_code}.")
            try:
                payload = response.json()
            except ValueError as exc:
                raise RetryableWorkerError("OpenAI Costs API returned invalid JSON.") from exc
            if (
                not isinstance(payload, dict)
                or payload.get("object") != "page"
                or not isinstance(payload.get("data"), list)
                or not isinstance(payload.get("has_more"), bool)
            ):
                raise RetryableWorkerError("OpenAI Costs API response is missing its bucket data.")
            for bucket in payload["data"]:
                if not isinstance(bucket, dict) or not isinstance(bucket.get("results"), list):
                    raise RetryableWorkerError("OpenAI Costs API returned a malformed cost bucket.")
                bucket_start = _as_int(bucket.get("start_time"))
                bucket_end = _as_int(bucket.get("end_time"))
                if (
                    bucket_start is None
                    or bucket_end is None
                    or bucket_start < int(start.timestamp())
                    or bucket_start >= int(end.timestamp())
                    or bucket_end <= bucket_start
                    or bucket_end > int(end.timestamp())
                ):
                    raise RetryableWorkerError("OpenAI Costs API returned a bucket with invalid timestamps.")
                results.setdefault(bucket_start, [])
                for row in bucket["results"]:
                    normalized = _normalize_cost_row(row)
                    if normalized["project_id"] != self._settings.openai_project_id:
                        raise RetryableWorkerError(
                            "OpenAI Costs API returned a row outside the configured project filter."
                        )
                    results[bucket_start].append(normalized)
            if not payload["has_more"]:
                break
            page = payload.get("next_page")
            if not page:
                raise RetryableWorkerError("OpenAI Costs API indicated another page but omitted next_page.")
            params["page"] = str(page)
        return dict(results)

    def _billing_usage_for_day(
        self, client: Any, start: datetime, end: datetime
    ) -> tuple[dict[str, dict[str, Any]], int, str]:
        """Read one day's turn usage and assessed reservations for allocation."""
        users: dict[str, dict[str, Any]] = {}
        ledger_turn_count = 0
        unattributed_turn_count = 0
        local_estimate_nanos = 0
        ledger_collection = client.collection(self._settings.billing_ledger_collection)
        for snapshot in _stream_date_range(ledger_collection, "created_at", start, end):
            ledger = snapshot.to_dict() or {}
            created_at = _as_utc_or_none(ledger.get("created_at"))
            if created_at is None or not start <= created_at < end:
                continue
            ledger_turn_count += 1
            subject_id = str(ledger.get("billing_subject_id") or "").strip()
            owner_uid = str(ledger.get("uid") or "").strip()
            estimate = _as_nonnegative_int(ledger.get("estimated_cost_usd_nanos"))
            local_estimate_nanos += estimate
            if not subject_id or not owner_uid:
                unattributed_turn_count += 1
                continue
            customer = users.setdefault(
                subject_id,
                {
                    "owner_uid": owner_uid,
                    "models": {},
                    "web_search_calls": 0,
                    "container_tool_calls": 0,
                    "ledger_turn_count": 0,
                    "reservation_turn_count": 0,
                    "provisional_assessed_cost_nanos": 0,
                },
            )
            if customer["owner_uid"] != owner_uid:
                raise RetryableWorkerError(
                    f"Multiple wallet owners map to billing subject {subject_id}."
                )
            customer["ledger_turn_count"] += 1
            usage = ledger.get("usage") if isinstance(ledger.get("usage"), dict) else {}
            token_data = ledger.get("billable_tokens")
            if not isinstance(token_data, dict):
                token_data = usage.get("billable_tokens") if isinstance(usage.get("billable_tokens"), dict) else {}
            cached = _as_nonnegative_int(token_data.get("input_cached"))
            uncached = _as_nonnegative_int(token_data.get("input_uncached"))
            cache_write = _as_nonnegative_int(token_data.get("input_cache_write"))
            output = _as_nonnegative_int(token_data.get("output_including_reasoning"))
            model = str(ledger.get("pricing_model") or usage.get("pricing_model") or "unknown").strip()
            model_data = customer["models"].setdefault(
                model,
                {
                    "input_uncached_tokens": 0,
                    "input_cached_tokens": 0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "estimated_cost_nanos": 0,
                },
            )
            model_data["input_uncached_tokens"] += uncached
            model_data["input_cached_tokens"] += cached
            model_data["input_tokens"] += uncached + cached + cache_write
            model_data["output_tokens"] += output
            model_data["estimated_cost_nanos"] += estimate
            resource_usage = usage.get("provider_resource_usage")
            if isinstance(resource_usage, dict):
                customer["web_search_calls"] += _as_nonnegative_int(resource_usage.get("web_search_calls"))
                customer["container_tool_calls"] += _as_nonnegative_int(resource_usage.get("container_tool_calls"))

        if not self._settings.openai_cost_allocation_enabled:
            return users, local_estimate_nanos, "allocation_disabled"

        reservations = client.collection(self._settings.billing_reservations_collection)
        reservation_turn_count = 0
        pending = False
        pending_provider_usage = False
        manual_review_provider_usage = False
        missing_assessment = False
        for snapshot in _stream_date_range(reservations, "created_at", start, end):
            reservation = snapshot.to_dict() or {}
            created_at = _as_utc_or_none(reservation.get("created_at"))
            if created_at is None or not start <= created_at < end:
                continue
            reservation_turn_count += 1
            subject_id = str(reservation.get("billing_subject_id") or "").strip()
            owner_uid = str(reservation.get("owner_uid") or "").strip()
            if not subject_id or not owner_uid or subject_id not in users:
                missing_assessment = True
                continue
            customer = users[subject_id]
            if customer["owner_uid"] != owner_uid:
                raise RetryableWorkerError(
                    f"Reservation owner does not match turn ledger for billing subject {subject_id}."
                )
            customer["reservation_turn_count"] += 1
            state = str(reservation.get("status") or "")
            if state in {"reserved", "completed_pending", "batch_pending", "late_batch_pending", "late_usage_pending"}:
                pending = True
            usage_status = reservation.get("usage_reconciliation_status")
            if usage_status == "pending":
                pending_provider_usage = True
            elif usage_status == "manual_review":
                manual_review_provider_usage = True
            assessed = reservation.get("billing_assessed_cost_nanos")
            if assessed is None and state in {
                "settled", "settled_shortfall", "late_settled", "late_settled_shortfall",
            }:
                assessed = reservation.get("estimated_cost_nanos")
            if assessed is None:
                missing_assessment = True
            else:
                customer["provisional_assessed_cost_nanos"] += _as_nonnegative_int(assessed)

        if ledger_turn_count != reservation_turn_count or unattributed_turn_count:
            return users, local_estimate_nanos, "manual_review_missing_billing_records"
        if any(
            customer["ledger_turn_count"] != customer["reservation_turn_count"]
            for customer in users.values()
        ) or missing_assessment:
            return users, local_estimate_nanos, "manual_review_missing_billing_records"
        if pending:
            return users, local_estimate_nanos, "pending_customer_settlement"
        if pending_provider_usage:
            return users, local_estimate_nanos, "pending_provider_usage"
        if manual_review_provider_usage:
            return users, local_estimate_nanos, "manual_review_provider_usage"
        return users, local_estimate_nanos, "ready"

    @staticmethod
    def _allocate_cost_rows(
        rows: list[dict[str, Any]], users: dict[str, dict[str, Any]]
    ) -> tuple[dict[str, int], dict[str, int], int]:
        allocations: dict[str, int] = defaultdict(int)
        fallback_counts: dict[str, int] = defaultdict(int)
        unallocated_nanos = 0
        model_names = {
            model
            for user in users.values()
            for model in user.get("models", {})
            if model != "unknown"
        }
        for row in rows:
            amount_nanos = usd_to_nanos(row["amount_usd"])
            if amount_nanos == 0:
                continue
            line_item = str(row.get("line_item") or "")
            kind = cost_line_item_weight_kind(line_item)
            model = line_item_model(line_item, model_names)
            weights: dict[str, int] = {}
            for subject_id, user in users.items():
                weight = (
                    item_weight(user, kind)
                    if kind in {"web_search_calls", "container_tool_calls"}
                    else model_item_weight(user, model, kind)
                )
                if weight > 0:
                    weights[subject_id] = weight
            if not weights and kind != "model_cost_fallback":
                fallback_kind = "model_cost_fallback"
                weights = {
                    subject_id: weight
                    for subject_id, user in users.items()
                    if (weight := item_weight(user, fallback_kind)) > 0
                }
                if weights:
                    fallback_counts[f"{kind}_to_{fallback_kind}"] += 1
            if not weights:
                unallocated_nanos += amount_nanos
                fallback_counts["unallocated_no_customer_weight"] += 1
                continue
            for subject_id, amount in allocate_integer(amount_nanos, weights).items():
                allocations[subject_id] += amount
        return dict(allocations), dict(fallback_counts), unallocated_nanos


def _normalize_cost_row(row: Any) -> dict[str, Any]:
    if not isinstance(row, dict):
        raise RetryableWorkerError("OpenAI Costs API returned a malformed cost row.")
    amount = row.get("amount")
    raw_value = amount.get("value") if isinstance(amount, dict) else None
    currency = str(amount.get("currency") or "usd").lower() if isinstance(amount, dict) else "usd"
    if currency != "usd":
        raise RetryableWorkerError(f"OpenAI Costs API returned unsupported currency {currency!r}.")
    try:
        amount_usd = Decimal(str(raw_value))
    except (InvalidOperation, ValueError, TypeError) as exc:
        raise RetryableWorkerError("OpenAI Costs API returned a cost row without a valid amount.") from exc
    if not amount_usd.is_finite() or amount_usd < 0:
        raise RetryableWorkerError("OpenAI Costs API returned a non-finite or negative cost amount.")
    project_id = row.get("project_id")
    if not isinstance(project_id, str) or not project_id.strip():
        raise RetryableWorkerError("OpenAI Costs API returned a cost row without a project ID.")
    return {
        "amount_usd": amount_usd,
        "project_id": project_id,
        "api_source": row.get("api_source"),
        "line_item": row.get("line_item"),
        "quantity": row.get("quantity"),
        "quantity_unit": row.get("quantity_unit"),
    }


def _as_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _stream_date_range(collection: Any, field: str, start: datetime, end: datetime):
    try:
        from google.cloud.firestore_v1.base_query import FieldFilter

        query = collection.where(filter=FieldFilter(field, ">=", start))
        query = query.where(filter=FieldFilter(field, "<", end))
        return query.stream()
    except (AttributeError, ImportError, ModuleNotFoundError, TypeError):
        return collection.stream()


def _as_utc_or_none(value: Any) -> datetime | None:
    if not isinstance(value, datetime):
        return None
    return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value.astimezone(timezone.utc)


def _as_nonnegative_int(value: Any) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else 0


def _is_openai_project_id(value: str) -> bool:
    return bool(value and value.startswith("proj_") and value[5:].isalnum())
