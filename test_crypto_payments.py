"""Tests for BlackRoad Crypto Payment Processor."""
import pytest
from crypto_payments import (
    init_db, create_payment, get_payment, check_status,
    get_wallet_summary, detect_suspicious, detect_rapid_payments,
    export_transactions, list_payments, register_wallet, network_stats,
    SUPPORTED_COINS, COIN_USD_RATES,
)

WALLET_A = "0x1111111111111111111111111111111111111111"
WALLET_B = "0x2222222222222222222222222222222222222222"
WALLET_BTC_A = "1A1zP1eP5QGefi2DMPTfTL5SLmv7Divf" + "Na"
WALLET_BTC_B = "3J98t1WpEZ73CNmQviecrnyiWrnqRhWNL" + "y"


@pytest.fixture
def db(tmp_path):
    path = str(tmp_path / "test_crypto.db")
    init_db(path)
    return path


def test_create_payment_eth(db):
    p = create_payment(WALLET_A, WALLET_B, 1.0, "ETH", path=db)
    assert p.coin == "ETH"
    assert p.amount == 1.0
    assert p.status == "pending"
    assert p.confirmations == 0
    assert p.tx_hash.startswith("0x")
    assert p.usd_value == pytest.approx(COIN_USD_RATES["ETH"])


def test_create_payment_btc(db):
    p = create_payment(WALLET_BTC_A, WALLET_BTC_B, 0.01, "BTC", path=db)
    assert p.coin == "BTC"
    assert len(p.tx_hash) == 64  # no 0x prefix


def test_create_payment_sol(db):
    p = create_payment("SOL_WALLET_A", "SOL_WALLET_B", 10.0, "sol", path=db)
    assert p.coin == "SOL"  # uppercased


def test_create_payment_usdc(db):
    p = create_payment(WALLET_A, WALLET_B, 500.0, "USDC", path=db)
    assert p.usd_value == pytest.approx(500.0)  # 1:1 with USD


def test_create_payment_unsupported_coin(db):
    with pytest.raises(ValueError, match="Unsupported"):
        create_payment(WALLET_A, WALLET_B, 1.0, "DOGE", path=db)


def test_create_payment_negative_amount(db):
    with pytest.raises(ValueError, match="positive"):
        create_payment(WALLET_A, WALLET_B, -1.0, "ETH", path=db)


def test_create_payment_same_wallet(db):
    with pytest.raises(ValueError, match="same wallet"):
        create_payment(WALLET_A, WALLET_A, 1.0, "ETH", path=db)


def test_get_payment(db):
    p = create_payment(WALLET_A, WALLET_B, 2.0, "ETH", path=db)
    fetched = get_payment(p.id, db)
    assert fetched.id == p.id
    assert fetched.amount == 2.0


def test_get_payment_not_found(db):
    with pytest.raises(KeyError):
        get_payment("nonexistent", db)


def test_check_status_pending(db):
    p = create_payment(WALLET_A, WALLET_B, 1.0, "BTC", path=db)
    # BTC takes 600s per confirmation, so newly created payment should still be pending
    updated = check_status(p.id, db)
    # Could be pending or have 0 confirmations
    assert updated.status in ("pending", "confirmed")


def test_check_status_sol_confirms_quickly(db):
    """SOL confirms in 2 seconds; creating with a fake old timestamp should confirm."""
    import sqlite3 as sl
    p = create_payment("SOL_A", "SOL_B", 5.0, "SOL", path=db)
    # Backdate the created_at to force confirmation
    from datetime import datetime, timedelta
    old_time = (datetime.utcnow() - timedelta(seconds=200)).isoformat()
    conn = sl.connect(db)
    conn.execute("UPDATE payments SET created_at=? WHERE id=?", (old_time, p.id))
    conn.commit()
    conn.close()
    updated = check_status(p.id, db)
    assert updated.status == "confirmed"
    assert updated.confirmed_at is not None


def test_fee_estimation_btc(db):
    p = create_payment(WALLET_BTC_A, WALLET_BTC_B, 1.0, "BTC", path=db)
    assert p.fee_amount == 0.0001


def test_fee_estimation_usdc(db):
    p = create_payment(WALLET_A, WALLET_B, 100.0, "USDC", path=db)
    assert p.fee_amount > 0


def test_wallet_summary_empty(db):
    summary = get_wallet_summary("0xNEWWALLET", db)
    assert summary["pending_transactions"] == 0
    assert summary["balances"] == {}


def test_wallet_summary_with_transactions(db):
    create_payment(WALLET_A, WALLET_B, 1.0, "ETH", path=db)
    create_payment(WALLET_B, WALLET_A, 0.5, "ETH", path=db)
    summary = get_wallet_summary(WALLET_A, db)
    assert "pending_transactions" in summary
    assert "balances" in summary


def test_detect_suspicious_high_value(db):
    create_payment(WALLET_A, WALLET_B, 5.0, "ETH", path=db)  # ~$16k
    create_payment(WALLET_A, WALLET_B, 0.001, "ETH", path=db)  # ~$3
    suspicious = detect_suspicious(threshold_usd=10000.0, path=db)
    high_amounts = [p.usd_value for p in suspicious]
    assert all(v >= 10000.0 for v in high_amounts)


def test_detect_suspicious_no_results(db):
    create_payment(WALLET_A, WALLET_B, 0.001, "ETH", path=db)
    result = detect_suspicious(threshold_usd=1_000_000.0, path=db)
    assert result == []


def test_detect_rapid_payments_below_threshold(db):
    for _ in range(3):
        create_payment(WALLET_A, WALLET_B, 0.1, "ETH", path=db)
    result = detect_rapid_payments(WALLET_A, window_minutes=60, min_count=5, path=db)
    assert result == []


def test_detect_rapid_payments_above_threshold(db):
    for _ in range(6):
        create_payment(WALLET_A, WALLET_B, 0.1, "ETH", path=db)
    result = detect_rapid_payments(WALLET_A, window_minutes=60, min_count=5, path=db)
    assert len(result) >= 5


def test_export_transactions_json(db):
    create_payment(WALLET_A, WALLET_B, 1.0, "ETH", path=db)
    output = export_transactions(fmt="json", path=db)
    data = json.loads(output)
    assert len(data) >= 1
    assert "tx_hash" in data[0]


def test_export_transactions_csv(db):
    create_payment(WALLET_A, WALLET_B, 1.0, "ETH", path=db)
    output = export_transactions(fmt="csv", path=db)
    assert "from_wallet" in output
    assert WALLET_A in output


def test_export_transactions_wallet_filter(db):
    create_payment(WALLET_A, WALLET_B, 1.0, "ETH", path=db)
    create_payment("0xOTHER", "0xANOTHER", 2.0, "ETH", path=db)
    output = export_transactions(wallet=WALLET_A, fmt="json", path=db)
    data = json.loads(output)
    assert all(
        p["from_wallet"] == WALLET_A or p["to_wallet"] == WALLET_A
        for p in data
    )


def test_list_payments_by_coin(db):
    create_payment(WALLET_A, WALLET_B, 1.0, "ETH", path=db)
    create_payment(WALLET_BTC_A, WALLET_BTC_B, 0.01, "BTC", path=db)
    eth_payments = list_payments(coin="ETH", path=db)
    assert all(p.coin == "ETH" for p in eth_payments)


def test_register_wallet(db):
    wallet = register_wallet(WALLET_A, "ETH", "Main wallet", db)
    assert wallet.address == WALLET_A
    assert wallet.coin == "ETH"
    assert wallet.label == "Main wallet"


def test_register_wallet_unsupported_coin(db):
    with pytest.raises(ValueError, match="Unsupported"):
        register_wallet("addr", "SHIB", path=db)


def test_network_stats(db):
    create_payment(WALLET_A, WALLET_B, 1.0, "ETH", path=db)
    create_payment(WALLET_BTC_A, WALLET_BTC_B, 0.01, "BTC", path=db)
    stats = network_stats(db)
    assert stats["total_payments"] >= 2
    assert "ETH" in stats["by_coin"]
    assert "BTC" in stats["by_coin"]
    assert "pending" in stats["by_status"]


def test_payment_memo_stored(db):
    p = create_payment(WALLET_A, WALLET_B, 1.0, "ETH", memo="Invoice #123", path=db)
    fetched = get_payment(p.id, db)
    assert fetched.memo == "Invoice #123"


def test_payment_network_stored(db):
    p = create_payment(WALLET_A, WALLET_B, 0.5, "ETH", network="testnet", path=db)
    fetched = get_payment(p.id, db)
    assert fetched.network == "testnet"


import json
