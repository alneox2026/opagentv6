from datetime import datetime, timezone

from common.billing import customer_billing_period_document_id, customer_wallet_document_id
from services.agent_persistence_worker_v3.app.core.config import get_settings
from services.agent_persistence_worker_v3.app.services.openai_cost_wallet_adjustment import (
    OpenAICostWalletAdjustmentService,
)


class Snapshot:
    def __init__(self, data):
        self.data = data
        self.exists = data is not None

    def to_dict(self):
        return dict(self.data) if self.data is not None else None


class Reference:
    def __init__(self, client, collection, document_id):
        self.client = client
        self.collection = collection
        self.document_id = document_id
        self.key = (collection, document_id)

    def get(self):
        return Snapshot(self.client.documents.get(self.key))


class Collection:
    def __init__(self, client, name):
        self.client = client
        self.name = name

    def document(self, document_id):
        return Reference(self.client, self.name, document_id)


class Transaction:
    def __init__(self, client):
        self.client = client

    def get(self, reference):
        return reference.get()

    def create(self, reference, data):
        if reference.key in self.client.documents:
            raise RuntimeError("document already exists")
        self.client.documents[reference.key] = dict(data)

    def update(self, reference, data):
        self.client.documents[reference.key].update(data)


class Firestore:
    def __init__(self):
        self.documents = {}

    def collection(self, name):
        return Collection(self, name)

    def transaction(self):
        return Transaction(self)


def _run_transaction(client, operation):
    return operation(client.transaction())


def test_daily_provider_adjustment_is_idempotent_and_debt_aware(monkeypatch):
    get_settings.cache_clear()
    monkeypatch.setenv("FIRESTORE_OPENAI_COST_ALLOCATIONS_COLLECTION", "openai_cost_allocations_test")
    monkeypatch.setenv("FIRESTORE_CUSTOMER_WALLETS_COLLECTION", "wallets_test")
    monkeypatch.setenv("FIRESTORE_WALLET_TRANSACTIONS_COLLECTION", "wallet_transactions_test")
    monkeypatch.setenv("FIRESTORE_CUSTOMER_BILLING_PERIODS_COLLECTION", "periods_test")
    get_settings.cache_clear()
    settings = get_settings()
    client = Firestore()
    owner = "user-1"
    subject = "user-1"
    wallet_id = customer_wallet_document_id(subject)
    period_id = customer_billing_period_document_id(subject, "2026-11")
    client.documents[(settings.wallets_collection, wallet_id)] = {
        "owner_uid": owner,
        "billing_subject_id": subject,
        "status": "active",
        "available_credit_nanos": 10,
        "settled_usage_nanos": 100,
        "usage_debt_nanos": 0,
        "lifetime_usage_debt_added_nanos": 0,
    }
    client.documents[(settings.customer_billing_periods_collection, period_id)] = {
        "billing_subject_id": subject,
        "owner_uid": owner,
        "usage_estimated_nanos": 100,
        "collected_usage_nanos": 100,
        "uncollected_usage_nanos": 0,
    }
    service = OpenAICostWalletAdjustmentService(
        firestore_client_factory=lambda: client,
        transaction_runner=_run_transaction,
    )
    day = datetime(2026, 11, 10, tzinfo=timezone.utc)

    positive = service.apply_sync(
        billing_subject_id=subject,
        owner_uid=owner,
        project_id="proj_test",
        day_start=day,
        provider_actual_cost_nanos=130,
        provisional_assessed_cost_nanos=100,
    )
    duplicate = service.apply_sync(
        billing_subject_id=subject,
        owner_uid=owner,
        project_id="proj_test",
        day_start=day,
        provider_actual_cost_nanos=130,
        provisional_assessed_cost_nanos=100,
    )

    wallet = client.documents[(settings.wallets_collection, wallet_id)]
    assert positive.status == "adjusted"
    assert positive.charged_nanos == 10
    assert positive.debt_added_nanos == 20
    assert duplicate.status == "unchanged"
    assert wallet["available_credit_nanos"] == 0
    assert wallet["settled_usage_nanos"] == 110
    assert wallet["usage_debt_nanos"] == 20
    assert wallet["lifetime_usage_debt_added_nanos"] == 20

    downward = service.apply_sync(
        billing_subject_id=subject,
        owner_uid=owner,
        project_id="proj_test",
        day_start=day,
        provider_actual_cost_nanos=100,
        provisional_assessed_cost_nanos=100,
    )
    wallet = client.documents[(settings.wallets_collection, wallet_id)]
    assert downward.delta_nanos == -30
    assert downward.debt_reduced_nanos == 20
    assert downward.refunded_nanos == 10
    assert wallet["available_credit_nanos"] == 10
    assert wallet["settled_usage_nanos"] == 100
    assert wallet["usage_debt_nanos"] == 0

    below_estimate = service.apply_sync(
        billing_subject_id=subject,
        owner_uid=owner,
        project_id="proj_test",
        day_start=day,
        provider_actual_cost_nanos=90,
        provisional_assessed_cost_nanos=100,
    )
    wallet = client.documents[(settings.wallets_collection, wallet_id)]
    assert below_estimate.delta_nanos == -10
    assert below_estimate.refunded_nanos == 10
    assert wallet["available_credit_nanos"] == 20
    assert wallet["settled_usage_nanos"] == 90
    period = client.documents[(settings.customer_billing_periods_collection, period_id)]
    assert period["usage_estimated_nanos"] == 90
    assert period["collected_usage_nanos"] == 90
    get_settings.cache_clear()
