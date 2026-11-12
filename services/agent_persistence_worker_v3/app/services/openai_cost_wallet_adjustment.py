"""Idempotent wallet true-ups for customer shares of OpenAI daily costs."""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from common.billing import customer_billing_period_document_id, customer_wallet_document_id, nonnegative_int
from common.firestore import get_transaction_document_snapshot
from services.agent_persistence_worker_v3.app.core.config import get_settings
from services.agent_persistence_worker_v3.app.core.errors import RetryableWorkerError
from services.agent_persistence_worker_v3.app.services.firestore_client import get_firestore_client


@dataclass(frozen=True)
class OpenAICostWalletAdjustmentResult:
    status: str
    target_cost_nanos: int
    provisional_cost_nanos: int
    delta_nanos: int
    charged_nanos: int = 0
    debt_added_nanos: int = 0
    debt_reduced_nanos: int = 0
    refunded_nanos: int = 0


class OpenAICostWalletAdjustmentService:
    """Applies actual-minus-provisional daily cost through the shared wallet."""

    def __init__(
        self,
        *,
        firestore_client_factory: Callable[[], Any] | None = None,
        transaction_runner: Callable[[Any, Callable[[Any], Any]], Any] | None = None,
    ) -> None:
        self._settings = get_settings()
        self._firestore_client_factory = firestore_client_factory or get_firestore_client
        self._transaction_runner = transaction_runner or _run_firestore_transaction

    def apply_sync(
        self,
        *,
        billing_subject_id: str,
        owner_uid: str,
        project_id: str,
        day_start: datetime,
        provider_actual_cost_nanos: int,
        provisional_assessed_cost_nanos: int,
    ) -> OpenAICostWalletAdjustmentResult:
        if not billing_subject_id.strip() or not owner_uid.strip():
            raise RetryableWorkerError("OpenAI cost allocation requires a customer and wallet owner.")
        target = _nonnegative(provider_actual_cost_nanos, "provider_actual_cost_nanos")
        provisional = _nonnegative(provisional_assessed_cost_nanos, "provisional_assessed_cost_nanos")
        day = _as_utc(day_start)
        month_key, month_start, month_end = _billing_period(day)
        scope = f"{project_id}:{day:%Y-%m-%d}:{billing_subject_id}"
        state_id = hashlib.sha256(scope.encode("utf-8")).hexdigest()
        client = self._firestore_client_factory()
        state_ref = client.collection(self._settings.openai_cost_allocations_collection).document(state_id)
        wallet_id = customer_wallet_document_id(billing_subject_id)
        period_id = customer_billing_period_document_id(billing_subject_id, month_key)
        wallet_ref = client.collection(self._settings.wallets_collection).document(wallet_id)
        period_ref = client.collection(self._settings.customer_billing_periods_collection).document(period_id)
        now = datetime.now(timezone.utc)

        def operation(transaction: Any) -> OpenAICostWalletAdjustmentResult:
            state_snapshot = get_transaction_document_snapshot(transaction, state_ref)
            wallet_snapshot = get_transaction_document_snapshot(transaction, wallet_ref)
            period_snapshot = get_transaction_document_snapshot(transaction, period_ref)
            if not wallet_snapshot.exists:
                raise RetryableWorkerError("Wallet is missing for OpenAI provider cost allocation.")
            state = state_snapshot.to_dict() or {}
            wallet = wallet_snapshot.to_dict() or {}
            period = period_snapshot.to_dict() or {}
            if (
                wallet.get("status") != "active"
                or wallet.get("owner_uid") != owner_uid
                or wallet.get("billing_subject_id") != billing_subject_id
            ):
                raise RetryableWorkerError("Wallet ownership or status does not match OpenAI cost allocation.")
            previous_adjustment = _signed_int(
                state.get("applied_adjustment_nanos", 0), "applied_adjustment_nanos"
            )
            required_adjustment = target - provisional
            delta = required_adjustment - previous_adjustment
            if (
                delta == 0
                and state.get("provider_actual_cost_nanos") == target
                and state.get("provisional_assessed_cost_nanos") == provisional
            ):
                return OpenAICostWalletAdjustmentResult("unchanged", target, provisional, 0)

            try:
                available = nonnegative_int(wallet.get("available_credit_nanos", 0), field_name="available_credit_nanos")
                settled_usage = nonnegative_int(wallet.get("settled_usage_nanos", 0), field_name="settled_usage_nanos")
                usage_debt = nonnegative_int(wallet.get("usage_debt_nanos", 0), field_name="usage_debt_nanos")
                lifetime_debt = nonnegative_int(wallet.get("lifetime_usage_debt_added_nanos", 0), field_name="lifetime_usage_debt_added_nanos")
                previous_period_estimated = nonnegative_int(period.get("usage_estimated_nanos", 0), field_name="usage_estimated_nanos")
                previous_period_collected = nonnegative_int(period.get("collected_usage_nanos", 0), field_name="collected_usage_nanos")
                previous_period_debt = nonnegative_int(period.get("uncollected_usage_nanos", 0), field_name="uncollected_usage_nanos")
            except ValueError as exc:
                raise RetryableWorkerError("Wallet or billing period has invalid values during OpenAI cost allocation.") from exc

            charged = debt_added = debt_reduced = refunded = 0
            new_available = available
            new_settled_usage = settled_usage
            new_usage_debt = usage_debt
            new_lifetime_debt = lifetime_debt
            if delta > 0:
                charged = min(delta, available)
                debt_added = delta - charged
                new_available -= charged
                new_settled_usage += charged
                new_usage_debt += debt_added
                new_lifetime_debt += debt_added
            elif delta < 0:
                reduction = -delta
                debt_reduced = min(reduction, usage_debt)
                refunded = reduction - debt_reduced
                new_available += refunded
                new_settled_usage = max(0, settled_usage - refunded)
                new_usage_debt -= debt_reduced

            revision = _nonnegative(state.get("revision", 0), "revision") + int(delta != 0)
            adjustment_id = f"openai_cost_{state_id[:32]}_{revision:06d}"
            transaction_ref = client.collection(self._settings.wallet_transactions_collection).document(adjustment_id)
            # Read idempotency marker before issuing any writes, as required by Firestore.
            if delta:
                transaction_snapshot = get_transaction_document_snapshot(transaction, transaction_ref)
                if transaction_snapshot.exists:
                    return OpenAICostWalletAdjustmentResult("already_applied", target, provisional, 0)

            if delta:
                transaction.create(
                    transaction_ref,
                    {
                        "schema_version": 1,
                        "transaction_id": adjustment_id,
                        "transaction_type": "openai_provider_cost_adjustment",
                        "status": "adjusted",
                        "billing_subject_id": billing_subject_id,
                        "owner_uid": owner_uid,
                        "wallet_document_id": wallet_id,
                        "currency": "USD",
                        "provider": "openai",
                        "project_id": project_id,
                        "policy_version": "openai-provider-cost-allocation-v1",
                        "cost_day": day,
                        "prior_adjustment_nanos": previous_adjustment,
                        "provider_actual_cost_nanos": target,
                        "provisional_assessed_cost_nanos": provisional,
                        "adjustment_delta_nanos": delta,
                        "amount_nanos": charged,
                        "shortfall_nanos": debt_added,
                        "debt_reduced_nanos": debt_reduced,
                        "refunded_nanos": refunded,
                        "remaining_usage_debt_nanos": new_usage_debt,
                        "period_id": period_id,
                        "created_at": now,
                    },
                )
                transaction.update(
                    wallet_ref,
                    {
                        "available_credit_nanos": new_available,
                        "settled_usage_nanos": new_settled_usage,
                        "usage_debt_nanos": new_usage_debt,
                        "lifetime_usage_debt_added_nanos": new_lifetime_debt,
                        "updated_at": now,
                        "last_settlement_at": now,
                        "last_openai_cost_adjustment_at": now,
                    },
                )
                period_updates = {
                    "usage_estimated_nanos": max(0, previous_period_estimated + delta),
                    "collected_usage_nanos": max(0, previous_period_collected + charged - refunded),
                    "uncollected_usage_nanos": max(0, previous_period_debt + debt_added - debt_reduced),
                    "openai_provider_cost_adjustment_nanos": int(period.get("openai_provider_cost_adjustment_nanos", 0)) + delta,
                    "last_provider_cost_reconciliation_at": now,
                    "updated_at": now,
                }
                if period_snapshot.exists:
                    transaction.update(period_ref, period_updates)
                else:
                    transaction.create(
                        period_ref,
                        {
                            "schema_version": 1,
                            "billing_subject_id": billing_subject_id,
                            "owner_uid": owner_uid,
                            "currency": "USD",
                            "period_key": month_key,
                            "period_start": month_start,
                            "period_end": month_end,
                            "status": "open",
                            "created_at": now,
                            **period_updates,
                        },
                    )

            state_payload = {
                "schema_version": 1,
                "allocation_state_id": state_id,
                "billing_subject_id": billing_subject_id,
                "owner_uid": owner_uid,
                "provider": "openai",
                "project_id": project_id,
                "cost_day": day,
                "policy_version": "openai-provider-cost-allocation-v1",
                "provider_actual_cost_nanos": target,
                "provisional_assessed_cost_nanos": provisional,
                "applied_adjustment_nanos": required_adjustment,
                "revision": revision,
                "updated_at": now,
            }
            if state_snapshot.exists:
                transaction.update(state_ref, state_payload)
            else:
                transaction.create(state_ref, {**state_payload, "created_at": now})
            return OpenAICostWalletAdjustmentResult(
                "adjusted" if delta else "recorded",
                target,
                provisional,
                delta,
                charged,
                debt_added,
                debt_reduced,
                refunded,
            )

        return self._transaction_runner(client, operation)


def _run_firestore_transaction(client: Any, operation: Callable[[Any], Any]) -> Any:
    from google.cloud import firestore

    transaction = client.transaction()

    @firestore.transactional
    def run(transaction: Any) -> Any:
        return operation(transaction)

    return run(transaction)


def _nonnegative(value: Any, field: str) -> int:
    try:
        return nonnegative_int(value, field_name=field)
    except ValueError as exc:
        raise RetryableWorkerError(f"Invalid {field} in OpenAI cost allocation.") from exc


def _signed_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise RetryableWorkerError(f"Invalid {field} in OpenAI cost allocation.")
    return value


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _billing_period(value: datetime) -> tuple[str, datetime, datetime]:
    start = datetime(value.year, value.month, 1, tzinfo=timezone.utc)
    end = datetime(value.year + 1, 1, 1, tzinfo=timezone.utc) if value.month == 12 else datetime(value.year, value.month + 1, 1, tzinfo=timezone.utc)
    return start.strftime("%Y-%m"), start, end
