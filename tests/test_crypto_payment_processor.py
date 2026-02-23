"""Tests for BlackRoad Crypto Payment Processor."""

import pytest
from decimal import Decimal
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from crypto_payment_processor import (
    CryptoPaymentDB, CryptoPaymentProcessor,
    Invoice, InvoiceStatus, MOCK_RATES_TO_USD
)


@pytest.fixture
def processor(tmp_path):
    db = CryptoPaymentDB(tmp_path / "test.db")
    return CryptoPaymentProcessor(db)


def test_create_invoice_basic(processor):
    inv = processor.create_invoice(Decimal("100.00"), "USD", "BTC")
    assert inv.amount_fiat == Decimal("100.00")
    assert inv.currency == "USD"
    assert inv.crypto == "BTC"
    assert inv.status == InvoiceStatus.PENDING
    assert inv.crypto_amount > Decimal("0")
    assert inv.address.startswith("bc1q")


def test_create_invoice_eth(processor):
    inv = processor.create_invoice(Decimal("50.00"), "USD", "ETH")
    assert inv.crypto == "ETH"
    assert inv.address.startswith("0x")


def test_create_invoice_eur(processor):
    inv = processor.create_invoice(Decimal("100.00"), "EUR", "BTC")
    assert inv.currency == "EUR"
    assert inv.crypto_amount > Decimal("0")


def test_create_invoice_negative_raises(processor):
    with pytest.raises(ValueError):
        processor.create_invoice(Decimal("-10"), "USD", "BTC")


def test_create_invoice_zero_raises(processor):
    with pytest.raises(ValueError):
        processor.create_invoice(Decimal("0"), "USD", "BTC")


def test_mark_confirmed_full(processor):
    inv = processor.create_invoice(Decimal("200"), "USD", "BTC")
    confirmed = processor.mark_confirmed(inv.id, "abc123txhash", confirmations=6)
    assert confirmed.status == InvoiceStatus.CONFIRMED
    assert confirmed.tx_hash == "abc123txhash"
    assert confirmed.confirmations == 6


def test_mark_confirmed_partial(processor):
    inv = processor.create_invoice(Decimal("100"), "USD", "BTC")
    partial = processor.mark_confirmed(inv.id, "tx789", confirmations=2)
    assert partial.status == InvoiceStatus.AWAITING_CONFIRMATION
    assert partial.confirmations == 2


def test_mark_confirmed_already_confirmed_raises(processor):
    inv = processor.create_invoice(Decimal("100"), "USD", "BTC")
    processor.mark_confirmed(inv.id, "txhash1", confirmations=6)
    with pytest.raises(ValueError, match="already confirmed"):
        processor.mark_confirmed(inv.id, "txhash2")


def test_cancel_invoice(processor):
    inv = processor.create_invoice(Decimal("75"), "USD", "ETH")
    cancelled = processor.cancel_invoice(inv.id)
    assert cancelled.status == InvoiceStatus.CANCELLED


def test_cancel_confirmed_invoice_raises(processor):
    inv = processor.create_invoice(Decimal("100"), "USD", "BTC")
    processor.mark_confirmed(inv.id, "txhash", confirmations=6)
    with pytest.raises(ValueError):
        processor.cancel_invoice(inv.id)


def test_list_pending_invoices(processor):
    processor.create_invoice(Decimal("10"), "USD", "BTC")
    processor.create_invoice(Decimal("20"), "USD", "ETH")
    inv3 = processor.create_invoice(Decimal("30"), "USD", "BTC")
    processor.mark_confirmed(inv3.id, "txhash", confirmations=6)
    pending = processor.list_pending_invoices()
    assert len(pending) == 2
    assert all(i.status == InvoiceStatus.PENDING for i in pending)


def test_exchange_rate_btc_usd(processor):
    rate = processor.calculate_exchange_rate("USD", "BTC", mock=True)
    assert rate.from_currency == "USD"
    assert rate.to_currency == "BTC"
    assert rate.rate > Decimal("0")
    assert rate.source == "mock"


def test_exchange_rate_eur_eth(processor):
    rate = processor.calculate_exchange_rate("EUR", "ETH", mock=True)
    # EUR/ETH = EUR_to_USD / ETH_to_USD
    expected = (MOCK_RATES_TO_USD["EUR"] / MOCK_RATES_TO_USD["ETH"])
    assert abs(rate.rate - expected) < Decimal("0.001")


def test_exchange_rate_unsupported_raises(processor):
    with pytest.raises(ValueError, match="Unsupported"):
        processor.calculate_exchange_rate("XYZ", "BTC", mock=True)


def test_exchange_rate_live_raises(processor):
    with pytest.raises(NotImplementedError):
        processor.calculate_exchange_rate("USD", "BTC", mock=False)


def test_export_accounting_csv_confirmed_only(processor):
    inv1 = processor.create_invoice(Decimal("100"), "USD", "BTC")
    inv2 = processor.create_invoice(Decimal("200"), "USD", "ETH")
    processor.mark_confirmed(inv1.id, "txhash1", confirmations=6)
    csv_out = processor.export_accounting_csv()
    assert inv1.id in csv_out
    assert inv2.id not in csv_out


def test_revenue_summary(processor):
    inv1 = processor.create_invoice(Decimal("100"), "USD", "BTC")
    inv2 = processor.create_invoice(Decimal("50"), "USD", "BTC")
    processor.mark_confirmed(inv1.id, "tx1", confirmations=6)
    processor.mark_confirmed(inv2.id, "tx2", confirmations=6)
    summary = processor.revenue_summary()
    assert summary["total_invoices"] == 2
    assert "USD" in summary["by_fiat_currency"]
    assert "BTC" in summary["by_crypto"]
    assert Decimal(summary["by_fiat_currency"]["USD"]) == Decimal("150.00")


def test_invoice_id_prefix(processor):
    inv = processor.create_invoice(Decimal("10"), "USD", "BTC")
    assert inv.id.startswith("INV-")
